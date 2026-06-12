import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==================== 配置（复用你现有的 API）====================
API_KEY = "sk-Th_ZuQs31rz8M72BwOViKw"
BASE_URL = "https://models.sjtu.edu.cn/api/v1"
CHAT_MODEL = "deepseek-chat"

llm = ChatOpenAI(
    model=CHAT_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.2,  # 解释模型行为需要低温度
    max_tokens=2048,
    timeout=180,
    max_retries=5
)

# ==================== 1. TextCNN 信号提取器 ====================

class TextCNNExplainer:
    """
    从你的 TextCNN 提取可解释信号，无需修改原模型代码。
    基于 Embedding Gradient + Filter Activation。
    """
    
    def __init__(self, model, vocab, tokenizer=None):
        self.model = model
        self.model.eval()
        self.vocab = vocab
        self.tokenizer = tokenizer or (lambda x: x.split())
        self.inv_vocab = {v: k for k, v in vocab.items()}
    
    def _text_to_ids(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = self.tokenizer(text)
        seq = [self.vocab.get(t, self.vocab.get('<UNK>', 1)) for t in tokens[:max_len]]
        if len(seq) < max_len:
            seq += [self.vocab.get('<PAD>', 0)] * (max_len - len(seq))
        return torch.tensor([seq], dtype=torch.long)
    
    def extract(self, text: str, target_class: int = 1, max_len: int = 128) -> Dict:
        """
        提取 CNN 内部决策信号。
        target_class: 1=rumor, 0=non-rumor
        """
        x = self._text_to_ids(text, max_len).to(next(self.model.parameters()).device)
        tokens = self.tokenizer(text)[:max_len]
        
        # 1. 前向传播并记录中间状态
        self.model.zero_grad()
        emb = self.model.embedding(x).unsqueeze(1)  # (1, 1, L, D)
        emb.requires_grad = True
        
        conv_outs = []
        post_relu_maps = []  # 保存 ReLU 后的 feature maps
        
        for conv in self.model.convs:
            c = conv(emb).squeeze(3)           # (1, 100, L-k+1)
            c_relu = torch.relu(c)              # ReLU 后
            post_relu_maps.append(c_relu.detach())
            pooled = torch.max_pool1d(c_relu, c_relu.size(2)).squeeze(2)
            conv_outs.append(pooled)
        
        cat = torch.cat(conv_outs, dim=1)       # (1, 300)
        hidden = cat                            # fc 前的隐藏状态
        out = self.model.fc(self.model.dropout(cat))
        
        # 2. 反向传播获取 embedding 梯度
        score = out[0, target_class]
        score.backward()
        
        emb_grad = emb.grad.squeeze(0).squeeze(0)  # (L, D)
        token_saliency = emb_grad.norm(dim=1).cpu().numpy()  # (L,)
        
        # 3. 归一化显著性
        if token_saliency.max() > token_saliency.min():
            token_saliency = (token_saliency - token_saliency.min()) / \
                             (token_saliency.max() - token_saliency.min() + 1e-8)
        else:
            token_saliency = np.zeros_like(token_saliency)
        
        # 4. 提取显著 n-grams（滑动窗口聚合 token 显著性）
        filter_sizes = [3, 4, 5]
        ngram_scores = []
        
        for k in filter_sizes:
            for i in range(len(tokens) - k + 1):
                if i + k > len(tokens):
                    break
                span_saliency = token_saliency[i:i+k].mean()
                ngram_text = " ".join(tokens[i:i+k])
                ngram_scores.append({
                    "text": ngram_text,
                    "size": k,
                    "start": i,
                    "saliency": round(float(span_saliency), 4),
                    "tokens": tokens[i:i+k]
                })
        
        ngram_scores.sort(key=lambda x: x["saliency"], reverse=True)
        top_ngrams = ngram_scores[:8]
        
        # 5. 决策风格分析
        high_ratio = float((token_saliency > 0.6).mean())
        if high_ratio < 0.08:
            decision_style = "focused"      # 模型极度聚焦少数词
        elif high_ratio > 0.35:
            decision_style = "diffuse"      # 模型广泛分散注意力
        else:
            decision_style = "balanced"
        
        # 6. 检测 shortcut：如果显著性集中在停用词/标点
        stopwords = {"the", "a", "is", "are", "was", "this", "that", "!!!", "..."}
        top_tokens = [tokens[i] for i in np.argsort(token_saliency)[-5:] if i < len(tokens)]
        shortcut_ratio = sum(1 for t in top_tokens if t.lower() in stopwords or t in {"!!!", "???"}) / max(len(top_tokens), 1)
        if shortcut_ratio > 0.4:
            decision_style = "shortcut"
        
        # 7. 隐藏状态向量
        hidden_state = hidden.detach().cpu().numpy()[0]
        
        return {
            "text": text,
            "tokens": tokens,
            "token_saliency": token_saliency[:len(tokens)].tolist(),
            "top_ngrams": top_ngrams,
            "hidden_state": hidden_state.tolist(),
            "decision_style": decision_style,
            "confidence": torch.softmax(out, dim=1)[0, target_class].item(),
            "prediction": target_class,
            "shortcut_ratio": shortcut_ratio
        }


# ==================== 2. 构建 CNN 决策模式库（离线）====================

class CNNPatternDBBuilder:
    """
    用训练/验证集构建 CNN 内部决策模式库。
    每个样本存储其 hidden_state 作为向量，用于后续相似模式检索。
    """
    
    def __init__(self, explainer: TextCNNExplainer, output_path: str = "./chroma_cnn_patterns"):
        self.explainer = explainer
        self.output_path = output_path
        # 使用 dummy embedding function，因为我们将直接传入 hidden_state 向量
        from langchain_huggingface import HuggingFaceEmbeddings
        dummy_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.store = Chroma(
            persist_directory=output_path,
            embedding_function=dummy_emb
        )
    
    def build(self, data: List[Dict], batch_size: int = 32):
        """
        data: [{"text": "...", "label": 0/1}, ...]
        """
        texts, metadatas, embeddings, ids = [], [], [], []
        
        for idx, item in enumerate(data):
            text = item["text"]
            label = item["label"]
            
            # 提取 CNN 信号
            signal = self.explainer.extract(text, target_class=label)
            
            # 生成人类可读的模式描述（用于 LLM 阅读）
            ngrams_str = " | ".join([f"{n['text']}({n['saliency']})" for n in signal["top_ngrams"][:5]])
            
            pattern_doc = f"""
CNN DECISION PATTERN:
The model classified this as {"RUMOR" if label == 1 else "NON-RUMOR"} with confidence {signal['confidence']:.3f}.
Decision style: {signal['decision_style']}.
Top salient n-grams: {ngrams_str}.
The model's attention was {"concentrated on a few key phrases" if signal['decision_style'] == 'focused' else "spread across the text"}.
"""
            
            # 元数据
            meta = {
                "source": "cnn_pattern_db",
                "label": "rumor" if label == 1 else "non-rumor",
                "confidence": signal["confidence"],
                "decision_style": signal["decision_style"],
                "top_ngrams": json.dumps(signal["top_ngrams"][:5]),
                "shortcut_ratio": signal["shortcut_ratio"],
                "pattern_id": f"pattern_{idx}"
            }
            
            texts.append(pattern_doc)
            metadatas.append(meta)
            embeddings.append(signal["hidden_state"])
            ids.append(f"pattern_{idx}")
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} samples...")
        
        # 批量存入 Chroma（直接传入 embedding）
        self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        print(f"[Pattern DB] 构建完成，共 {len(texts)} 条模式")
        return self.store


# ==================== 3. 独立检索器：基于 CNN 内部状态 ====================

class CNNPatternRetriever:
    """
    独立检索器：用当前样本的 hidden_state 检索历史上模型感知相似的样本。
    """
    
    def __init__(self, pattern_store: Chroma, explainer: TextCNNExplainer):
        self.store = pattern_store
        self.explainer = explainer
    
    def retrieve(self, text: str, prediction: str, confidence: float, top_k: int = 5) -> Dict:
        """
        双路检索：
        1. 向量检索：基于 hidden_state 相似度（模型感知相似）
        2. N-gram 重叠：在向量召回结果中，筛选与当前显著 n-gram 共享模式的样本
        """
        target_class = 1 if prediction == "false" else 0
        
        # 1. 提取当前样本 CNN 信号
        current_signal = self.explainer.extract(text, target_class=target_class)
        
        # 2. 向量检索（基于 hidden_state）
        # 使用 similarity_search_by_vector 直接传入 CNN 向量
        vector_results = self.store.similarity_search_by_vector(
            embedding=current_signal["hidden_state"],
            k=top_k * 3  # 多召回一些，供后续过滤
        )
        
        # 3. N-gram 重叠过滤（结构化匹配）
        current_ngrams = set(n["text"].lower() for n in current_signal["top_ngrams"])
        enriched_results = []
        
        for doc in vector_results:
            # 解析历史样本的 n-grams
            hist_ngrams = json.loads(doc.metadata.get("top_ngrams", "[]"))
            hist_ngram_set = set(n["text"].lower() for n in hist_ngrams)
            
            # 计算重叠
            overlap = current_ngrams & hist_ngram_set
            overlap_score = len(overlap) / max(len(current_ngrams), 1)
            
            # 标签一致性
            label_match = (doc.metadata.get("label") == ("rumor" if prediction == "false" else "non-rumor"))
            
            enriched_results.append({
                "doc": doc,
                "overlap_ngrams": list(overlap),
                "overlap_score": overlap_score,
                "label_match": label_match,
                "decision_style": doc.metadata.get("decision_style", "unknown")
            })
        
        # 4. 重排序：优先保留 hidden_state 相似 + n-gram 重叠 + 同标签的样本
        enriched_results.sort(
            key=lambda x: (x["label_match"], x["overlap_score"], 1.0),
            reverse=True
        )
        
        final_results = enriched_results[:top_k]
        
        return {
            "current_signal": current_signal,
            "retrieved_patterns": final_results,
            "current_ngrams": list(current_ngrams)
        }


# ==================== 4. 独立 LLM：模型行为解释器 ====================

CNN_INTERPRETER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Machine Learning Interpretability Expert specializing in TextCNN behavior analysis.
Your task is to explain WHY the TextCNN made its classification decision, not whether the text is actually a rumor.

You have access to:
1. The CNN's internal saliency map (which n-grams it focused on)
2. The CNN's hidden state representation
3. Retrieved historical decision patterns with similar internal activations

Rules:
- Explain the model's decision MECHANISM, not the truth of the claim.
- Use phrases like "The CNN strongly activated on..." or "The model's attention concentrated on..."
- If the model relies on spurious cues (punctuation, stop words), flag this as "shortcut behavior".
- Compare current n-gram focus with historical patterns to identify if this is a learned pattern or an anomaly.
- Assess decision fragility: would the prediction change if the top n-gram were removed?

Output valid JSON only."""),
    ("human", """## Current Sample
Text: {text}
CNN Prediction: {prediction}
Confidence: {confidence}

## CNN Internal Signals
- Decision Style: {decision_style}
- Top Salient N-grams: {current_ngrams}
- Token-level Saliency (top 5): {top_tokens}

## Retrieved Historical Patterns (Similar Internal Activations)
{retrieved_patterns}

## Output Format
{{
    "model_decision_summary": "One sentence summarizing what the CNN did",
    "attention_analysis": {{
        "focused_ngrams": ["list of decisive n-grams"],
        "ignored_regions": ["parts of text the CNN overlooked"],
        "attention_distribution": "focused/diffuse/shortcut"
    }},
    "pattern_alignment": {{
        "historical_similarity": "The CNN treated this similarly to past samples where...",
        "shared_ngrams": ["n-grams that appear in both current and retrieved patterns"],
        "pattern_stability": "stable/moderate/fragile"
    }},
    "mechanism_explanation": "Detailed paragraph explaining the CNN's internal logic...",
    "potential_shortcuts": ["spurious correlations the model may be exploiting"],
    "decision_fragility": "If the top n-gram were removed, the prediction would likely...",
    "confidence_assessment": "The model's confidence is justified/overconfident/uncertain because..."
}}""")
])

def format_retrieved_patterns(patterns: List[Dict]) -> str:
    lines = []
    for i, p in enumerate(patterns, 1):
        doc = p["doc"]
        meta = doc.metadata
        lines.append(f"[Pattern {i}]")
        lines.append(f"  Historical Label: {meta['label']}")
        lines.append(f"  Decision Style: {meta['decision_style']}")
        lines.append(f"  Confidence: {meta['confidence']:.3f}")
        lines.append(f"  N-gram Overlap with Current: {p['overlap_ngrams']}")
        lines.append(f"  Pattern Description: {doc.page_content[:300]}")
        lines.append("")
    return "\n".join(lines)


# ==================== 5. 独立 RAG 管线：CNNInterpreterRAG ====================

class CNNInterpreterRAG:
    """
    完全独立的 CNN 决策解释 RAG。
    与三库 RAG 不共享数据库、检索器或 Prompt。
    """
    
    def __init__(self, model, vocab, pattern_db_path: str = "./chroma_cnn_patterns", tokenizer=None):
        self.explainer = TextCNNExplainer(model, vocab, tokenizer)
        
        # 加载或构建模式库
        from langchain_huggingface import HuggingFaceEmbeddings
        dummy_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.pattern_store = Chroma(
            persist_directory=pattern_db_path,
            embedding_function=dummy_emb
        )
        
        self.retriever = CNNPatternRetriever(self.pattern_store, self.explainer)
    
    def explain(self, text: str, prediction: str, confidence: float) -> Dict:
        """
        主入口：解释 CNN 为什么做出此判断。
        """
        print(f"\n{'='*60}")
        print("[Model RAG] 提取 CNN 内部信号...")
        
        # 1. 检索相似决策模式
        retrieval_result = self.retriever.retrieve(text, prediction, confidence, top_k=5)
        current_signal = retrieval_result["current_signal"]
        
        print(f"  Decision Style: {current_signal['decision_style']}")
        print(f"  Top N-grams: {[n['text'] for n in current_signal['top_ngrams'][:3]]}")
        print(f"  Retrieved {len(retrieval_result['retrieved_patterns'])} similar patterns")
        
        # 2. 格式化 top tokens
        tokens = current_signal["tokens"]
        saliency = np.array(current_signal["token_saliency"])
        top_indices = np.argsort(saliency)[-5:][::-1]
        top_tokens = [(tokens[i] if i < len(tokens) else "<PAD>", round(float(saliency[i]), 3)) 
                      for i in top_indices if i < len(tokens)]
        
        # 3. 组装 Prompt
        prompt_vars = {
            "text": text,
            "prediction": prediction.upper(),
            "confidence": confidence,
            "decision_style": current_signal["decision_style"],
            "current_ngrams": json.dumps(current_signal["top_ngrams"][:5], ensure_ascii=False),
            "top_tokens": json.dumps(top_tokens, ensure_ascii=False),
            "retrieved_patterns": format_retrieved_patterns(retrieval_result["retrieved_patterns"])
        }
        
        messages = CNN_INTERPRETER_PROMPT.format_messages(**prompt_vars)
        
        print("[Model RAG] 调用 LLM 生成模型行为解释...")
        response = llm.invoke(messages)
        
        return {
            "explanation": response.content,
            "current_signal": current_signal,
            "retrieved_patterns": retrieval_result["retrieved_patterns"],
            "token_usage": response.response_metadata.get("token_usage", {})
        }
    
    def build_database(self, data: List[Dict]):
        """离线构建模式库（首次运行或数据更新时调用）"""
        builder = CNNPatternDBBuilder(self.explainer, output_path=self.pattern_store._persist_directory)
        builder.build(data)
        print("[Model RAG] 模式库构建完成")


# ==================== 6. 使用示例 ====================

if __name__ == "__main__":
    # 假设你已加载 model, vocab, tokenizer
    # model.load_state_dict(torch.load('best_textcnn.pt'))
    from sth import TextCNN, build_vocab
    with open('train.json', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('val.json', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('test.json', encoding='utf-8') as f:
        test_data = json.load(f)

    vocab = build_vocab(train_data + val_data + test_data, max_vocab=30000, tokenizer=None)
    model = TextCNN(vocab_size=len(vocab), embedding_dim=128, num_filters=100, filter_sizes=[3,4,5], num_classes=2)
    # 初始化独立 RAG（首次运行需先构建数据库）
    model.load_state_dict(torch.load('best_textcnn.pt'))
    print("model loaded")
    model_rag = CNNInterpreterRAG(model, vocab, pattern_db_path="./chroma_cnn_patterns")
    
    # 构建数据库（离线，只需一次）
    # with open('train.json') as f:
    #     train_data = json.load(f)
    # model_rag.build_database(train_data)
    
    # 在线解释
    test_text = "BREAKING!!! White House under attack!!! Multiple casualties confirmed!!!"
    pred = "false"      # CNN 输出
    conf = 0.94
    
    result = model_rag.explain(test_text, pred, conf)
    
    print(f"\n{'='*60}")
    print("CNN MODEL EXPLANATION:")
    print(f"{'='*60}")
    print(result["explanation"])
    
    # 同时，你的三库 RAG 独立运行
    # evidence_result = evidence_rag.explain(test_text, pred, conf)
