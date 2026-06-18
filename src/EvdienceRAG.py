import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder

# 配置区

API_KEY = "sk-Th_ZuQs31rz8M72BwOViKw"
BASE_URL = "https://models.sjtu.edu.cn/api/v1"
CHAT_MODEL = "deepseek-chat"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

TEMPERATURE = 0.3
MAX_TOKENS = 2048
TIMEOUT = 180
MAX_RETRIES = 5

# 数据库路径
CASE_DB_PATH = "./db/chroma_pheme_cases"
KNOWLEDGE_DB_PATH = "./db/chroma_fever_knowledge"
FEATURE_DB_PATH = "./db/chroma_linguistic_features"

@dataclass
class DetectionResult:
    """CNN 检测结果"""
    text: str
    prediction: str       # "true" 或 "false"
    confidence: float

@dataclass
class RetrievedEvidence:
    """统一证据封装"""
    doc: LangchainDocument
    source_db: str        # case / knowledge / feature
    retrieval_score: float
    final_score: float = 0.0

# 模型加载
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("[1/6] 嵌入模型加载完成")

llm = ChatOpenAI(
    model=CHAT_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    timeout=TIMEOUT,
    max_retries=MAX_RETRIES,
)
print("[2/6] LLM 加载完成")

# 加载三个 Chroma 向量库
case_store = Chroma(
    persist_directory=CASE_DB_PATH,
    embedding_function=embeddings
)
knowledge_store = Chroma(
    persist_directory=KNOWLEDGE_DB_PATH,
    embedding_function=embeddings
)
feature_store = Chroma(
    persist_directory=FEATURE_DB_PATH,
    embedding_function=embeddings
)
print("[3/6] 三个向量库加载完成")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 标签映射系统
class LabelMapper:
    """
    将 CNN 的 true/false 预测映射到各数据库的 label 体系
    """
    @staticmethod
    def to_case_filter(prediction: str) -> Dict:
        """案例库标签映射"""
        if prediction == "false":
            # Pheme: rumour, LIAR: false/barely-true/pants-fire, ISOT: fake
            return {
                "$or": [
                    {"label": {"$in": ["rumour", "false", "barely-true", "pants-fire", "mostly-false", "fake"]}},
                    {"label": {"$eq": 1}}  # 兼容数值型
                ]
            }
        else:
            return {
                "$or": [
                    {"label": {"$in": ["non-rumour", "true", "mostly-true", "half-true", "real"]}},
                    {"label": {"$eq": 0}}
                ]
            }
    
    @staticmethod
    def to_knowledge_filter(prediction: str) -> Dict:
        """知识库 FEVER 标签映射"""
        if prediction == "false":
            return {"claim_label": {"$in": ["REFUTES", "NEI"]}}
        else:
            return {"claim_label": {"$eq": "SUPPORTS"}}
    
    @staticmethod
    def check_consistency(doc: LangchainDocument, prediction: str, source_db: str) -> bool:
        """检查文档标签是否与 CNN 预测一致"""
        meta = doc.metadata
        label = str(meta.get("label", meta.get("claim_label", ""))).lower()
        
        true_labels = ["non-rumour", "true", "mostly-true", "half-true", "real", "supports", "0"]
        false_labels = ["rumour", "false", "barely-true", "pants-fire", "mostly-false", "fake", "refutes", "nei", "1"]
        
        if prediction == "true":
            return label in true_labels
        else:
            return label in false_labels
        return False

# 询改写器 (Query Rewriter)

class QueryRewriter:
    """
    HyDE + 多视角查询生成
    """
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def rewrite(self, detection: DetectionResult) -> Dict[str, any]:
        text = detection.text
        pred = detection.prediction
        features = "quantity/readability/punctuation/emotion/pronoun/congtive_process/specificity/syntactic/inquiry"
        
        # 主张提取
        claim = text[:300]
        
        # HyDE: 生成假设解释文档
        hyde_knowledge_prompt = f"""You are a rumor analysis expert. The following text was classified as {"a RUMOR" if pred == "false" else "TRUE"} by a CNN detector.
Write a 3-sentence expert analysis explaining what deceptive or verifiable characteristics this text might have.

Text: {text}

Expert Analysis:"""
        hyde_feature_prompt = f"""You are a computational linguistics expert writing entries for a deception detection feature database.
Given a text and available linguistic features, generate 3-5 detailed feature descriptions in the following EXACT format:

LINGUISTIC FEATURE - [Feature Name]:
[2-3 sentences describing the observed linguistic phenomenon in this specific text, citing concrete examples from the text.]
Rule: [A clear, actionable rule connecting this feature to deception/rumor detection.]

Requirements:
1. Each description must be grounded in the ACTUAL text provided.
2. Feature names should be specific and technical (e.g., "Cognitive Load Indicators", "Affective Polarization Markers").
3. The "Rule" sentence must follow the exact format: "Rule: [content]"
4. Generate descriptions ONLY for features that are STRONGLY present in the text.
5. Do not hallucinate features not present in the text.
6. Output raw text, no JSON, no markdown code blocks.

Text: {text}

Available Features: {features}

Generate feature descriptions:"""
        try:
            hyde_knowledge_doc = self.llm.invoke(hyde_knowledge_prompt).content
            hyde_feature_doc = self.llm.invoke(hyde_feature_prompt).content
        except Exception:
            hyde_knowledge_doc = claim  # 降级
            hyde_feature_doc = claim
        
        # 生成多视角查询
        queries = {
            "case": f"Similar {'rumor' if pred == 'false' else 'non-rumor'} case: {claim}",
            "feature": hyde_feature_doc,
            "hyde_text": hyde_knowledge_doc,
        }
        
        return queries

# 元数据感知检索器

class MetadataAwareRetriever:
    """
    封装 Chroma 检索器，支持元数据过滤 + MMR
    """
    def __init__(self, vectorstore, db_type: str):
        self.store = vectorstore
        self.db_type = db_type
        
    def retrieve(self, query_text: str, query_vec: Optional[List[float]], 
                 metadata_filter: Optional[Dict], top_k: int = 8) -> List[RetrievedEvidence]:
        
        # 使用 Chroma 的 MMR 检索 + 元数据过滤
        # Chroma 支持 where 参数进行 metadata 过滤
        try:
            docs = self.store.similarity_search_with_score(
                query=query_text,
                k=top_k,
                filter=metadata_filter if metadata_filter else None
            )
        except Exception as e:
            # 如果过滤导致召回为空，降级为无过滤
            docs = self.store.similarity_search_with_score(
                query=query_text,
                k=top_k
            )
        
        results = []
        for doc, score in docs:
            results.append(RetrievedEvidence(
                doc=doc,
                source_db=self.db_type,
                retrieval_score=float(score)
            ))
        return results

# 初始化三个检索器
case_retriever = MetadataAwareRetriever(case_store, "case")
knowledge_retriever = MetadataAwareRetriever(knowledge_store, "knowledge")
feature_retriever = MetadataAwareRetriever(feature_store, "feature")

# RAG-Fusion (RRF)

class RAGFusion:
    @staticmethod
    def reciprocal_rank_fusion(results_per_query: Dict[str, List[RetrievedEvidence]], 
                                k: int = 60) -> List[RetrievedEvidence]:
        """
        多路召回 RRF 融合
        """
        scores = {}
        for source, evidence_list in results_per_query.items():
            for rank, ev in enumerate(evidence_list):
                doc_id = f"{ev.source_db}_{ev.doc.metadata.get('id', ev.doc.page_content[:50])}"
                if doc_id not in scores:
                    scores[doc_id] = {"evidence": ev, "score": 0.0}
                # RRF 公式
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)
        
        # 按分数排序
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        for item in sorted_items:
            item["evidence"].final_score = item["score"]
        
        return [item["evidence"] for item in sorted_items]

# 元数据感知重排序

class MetadataAwareReranker:
    """
    结合 Cross-Encoder 语义相关性与元数据一致性
    """
    def __init__(self, cross_enc=None):
        self.cross_encoder = cross_enc
    
    def rerank(self, candidates: List[RetrievedEvidence], 
               detection: DetectionResult, 
               top_n: int = 12) -> List[RetrievedEvidence]:
        
        query = detection.text
        pred = detection.prediction
        
        # Cross-Encoder 语义分
        if self.cross_encoder and len(candidates) > 0:
            pairs = [(query, ev.doc.page_content) for ev in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
            for i, ev in enumerate(candidates):
                ev.final_score = ce_scores[i] * 0.6  # CE 权重 60%
        else:
            for ev in candidates:
                ev.final_score = ev.retrieval_score * 0.6
        
        # 元数据一致性奖励
        for ev in candidates:
            bonus = 0.0
            meta = ev.doc.metadata
            
            # 标签一致性（最高权重）
            if LabelMapper.check_consistency(ev.doc, pred, ev.source_db):
                bonus += 0.25
            
            # 来源权威性奖励
            source_weights = {
                "pheme_case_db": 0.12,
                "fever": 0.10,
                "liar_case_db": 0.08,
                "isot_case_db": 0.07,
                "linguistic_feature": 0.06
            }
            src = meta.get("source", "")
            bonus += source_weights.get(src, 0)
            
            # 置信度加权：CNN 置信度高时，更强调标签一致性
            if detection.confidence > 0.85:
                bonus *= 1.15
            
            ev.final_score += bonus
        
        # 排序并截断
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_n]

reranker = MetadataAwareReranker(cross_encoder)

# 上下文格式化

class ContextAssembler:
    """
    将不同来源的证据格式化为结构化上下文
    """
    @staticmethod
    def format_case(ev: RetrievedEvidence) -> str:
        doc = ev.doc
        m = doc.metadata
        
        # 适配 Pheme 元数据
        if m.get("source") == "pheme_case_db":
            return (
                f"[CASE:PHEME | ID:{m.get('case_id','?')} | "
                f"Event:{m.get('event','?')} | Label:{m.get('label','?')} | "
                f"Reactions:{m.get('reactions','?')} | Doubt:{m.get('doubt','?')} | "
                f"Support:{m.get('support','?')}]\n"
                f"Text: {doc.page_content[:300]}"
            )
        
        # 适配 LIAR 元数据
        elif m.get("source") == "liar_case_db":
            return (
                f"[CASE:LIAR | ID:{m.get('id','?')} | "
                f"Label:{m.get('label','?')} | Topic:{m.get('topic','?')} | "
                f"Context:{m.get('context','?')}]\n"
                f"Text: {doc.page_content[:300]}"
            )
        
        # 适配 ISOT 元数据
        elif m.get("source") == "isot_case_db":
            return (
                f"[CASE:ISOT | ID:{m.get('id','?')} | "
                f"Label:{m.get('label','?')}]\n"
                f"Text: {doc.page_content[:300]}"
            )
        
        return f"[CASE:UNKNOWN]\n{doc.page_content[:300]}"
    
    @staticmethod
    def format_knowledge(ev: RetrievedEvidence) -> str:
        doc = ev.doc
        m = doc.metadata
        return (
            f"[KNOWLEDGE:FEVER | Claim:{m.get('claim_text','?')} | "
            f"Label:{m.get('claim_label','?')} | "
            f"Wiki:{m.get('wiki_page','?')}:{m.get('wiki_line','?')} | "
            f"Verifiable:{m.get('verifiable','?')}]\n"
            f"Evidence: {doc.page_content[:400]}"
        )
    
    @staticmethod
    def format_feature(ev: RetrievedEvidence) -> str:
        doc = ev.doc
        m = doc.metadata
        return (
            f"[FEATURE:{m.get('category','?')} | "
            f"Feature:{m.get('feature','?')} | "
            f"Priority:{m.get('priority','?')} | "
            f"Cross-cultural:{m.get('cross_cultural','?')}]\n"
            f"Rule: {doc.page_content[:400]}"
        )
    
    @classmethod
    def assemble(cls, evidence_list: List[RetrievedEvidence], max_tokens: int = 3500) -> str:
        # 按来源分组
        cases = [e for e in evidence_list if e.source_db == "case"]
        knowledges = [e for e in evidence_list if e.source_db == "knowledge"]
        features = [e for e in evidence_list if e.source_db == "feature"]
        
        parts = []
        
        # 案例部分（最多 4 条，优先 Pheme）
        if cases:
            parts.append("### A. SIMILAR CASES")
            # Pheme 优先排序
            cases_sorted = sorted(cases, key=lambda x: (
                0 if x.doc.metadata.get("source") == "pheme_case_db" else 1
            ))
            for ev in cases_sorted[:4]:
                parts.append(cls.format_case(ev))
        
        # 知识部分（最多 3 条）
        if knowledges:
            parts.append("\n### B. KNOWLEDGE VERIFICATION")
            for ev in knowledges[:3]:
                parts.append(cls.format_knowledge(ev))
        
        # 特征部分（最多 3 条）
        if features:
            parts.append("\n### C. LINGUISTIC INDICATORS")
            for ev in features[:3]:
                parts.append(cls.format_feature(ev))
        
        context = "\n\n".join(parts)
        # 简单截断（实际应按 token 数截断）
        if len(context) > max_tokens * 4:  # 粗略字符估算
            context = context[:max_tokens * 4]
        
        return context

# LLM Prompt 与解释生成

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Rumor Analysis Assistant. Your task is to explain why a CNN-based detector classified the input text as a rumor or non-rumor, based strictly on the retrieved evidence.

Follow this reasoning chain:
1. Claim Analysis: Identify the core claim in the input text.
2. Case Comparison: Compare with similar cases. Note shared events, topics, or linguistic patterns.
3. Fact Verification: Use knowledge base evidence to verify or refute the core claim.
4. Linguistic Analysis: Reference linguistic indicators. Does the text exhibit known deception patterns?
5. Synthesis: Connect the CNN's prediction to the evidence. Explain WHY the detector likely made this decision.
6. Uncertainty: If evidence is contradictory or insufficient, state this explicitly.

Output must be a valid JSON object with no markdown formatting."""),
    ("human", """## Input Text
{target_text}

## Detector Signal
- Prediction: {predicted_label}
- Confidence: {confidence}
- Model: CNN (trained on Pheme dataset)

## Retrieved Evidence
{context}

## Output Format
{{
    "verdict": "{predicted_label}",
    "confidence": {confidence},
    "explanation": "Detailed explanation paragraph...",
    "key_evidence": [
        {{"type": "case", "source": "pheme_case_db", "relevance": "high", "reason": "..."}},
        {{"type": "knowledge", "source": "fever", "relevance": "medium", "reason": "..."}}
    ],
    "uncertainty": "Low / Medium / High",
    "reasoning_chain": "1. Claim: ... 2. Cases: ... 3. Knowledge: ... 4. Features: ... 5. Synthesis: ..."
}}""")
])

# 主 RAG 管线

class RumorRAGPipeline:
    def __init__(self):
        self.query_rewriter = QueryRewriter(llm)
        self.fusion = RAGFusion()
        self.reranker = MetadataAwareReranker(cross_encoder)
        self.assembler = ContextAssembler()
    
    def explain(self, detection: DetectionResult) -> Dict:
        print(f"\n[Pipeline] 处理文本: {detection.text[:60]}...")
        print(f"[Pipeline] CNN 预测: {detection.prediction} (conf={detection.confidence:.4f})")
        
        # 查询改写
        queries = self.query_rewriter.rewrite(detection)
        print(f"[4/6] 查询改写完成 (HyDE length: {len(queries['hyde_text'])})")
        
        # 元数据过滤条件
        case_filter = LabelMapper.to_case_filter(detection.prediction)
        know_filter = LabelMapper.to_knowledge_filter(detection.prediction)
        feat_filter = None  # 特征库不过滤标签
        
        # 并行检索（三路）
        case_results = case_retriever.retrieve(
            queries["case"], None, case_filter, top_k=8
        )
        # 知识库用 HyDE 文本检索，增强语义匹配
        know_results = knowledge_retriever.retrieve(
            queries["hyde_text"], None, know_filter, top_k=8
        )
        feat_results = feature_retriever.retrieve(
            queries["feature"], None, feat_filter, top_k=6
        )
        
        print(f"[Pipeline] 案例库召回: {len(case_results)} | 知识库: {len(know_results)} | 特征库: {len(feat_results)}")
        
        # RAG-Fusion (RRF)
        fused = self.fusion.reciprocal_rank_fusion({
            "case": case_results,
            "knowledge": know_results,
            "feature": feat_results
        })
        print(f"[5/6] RRF 融合后: {len(fused)} 条证据")
        
        # 重排序
        top_evidence = self.reranker.rerank(fused, detection, top_n=12)
        print(f"[Pipeline] 重排序后保留: {len(top_evidence)} 条")
        
        # 上下文组装
        context = self.assembler.assemble(top_evidence, max_tokens=3500)
        
        # LLM 生成解释
        prompt_vars = {
            "target_text": detection.text,
            "predicted_label": detection.prediction.upper(),
            "confidence": detection.confidence,
            "context": context
        }
        
        messages = EXPLANATION_PROMPT.format_messages(**prompt_vars)
        response = llm.invoke(messages)
        
        # 解析 token 用量
        token_usage = response.response_metadata.get("token_usage", {})
        print(f"[6/6] 生成完成 | Prompt tokens: {token_usage.get('prompt_tokens', '?')} | Completion: {token_usage.get('completion_tokens', '?')}")
        
        return {
            "detection": detection,
            "evidence": top_evidence,
            "context": context,
            "explanation_raw": response.content,
            "token_usage": token_usage
        }

# 运行入口

def main():
    pipeline = RumorRAGPipeline()
    # 读取输入
    try:
        user_input = input("请输入: text label confidence\n> ")
        parts = user_input.strip().split()
        if len(parts) >= 3:
            text = " ".join(parts[:-2])
            label = parts[-2]
            conf = float(parts[-1])
        else:
            # 测试默认值
            text = "BREAKING: Explosion heard near White House. Multiple casualties reported!!!"
            label = "false"
            conf = 0.92
    except Exception:
        text = "BREAKING: Explosion heard near White House. Multiple casualties reported!!!"
        label = "false"
        conf = 0.92
    
    detection = DetectionResult(text=text, prediction=label, confidence=conf)
    result = pipeline.explain(detection)
    
    print("\n" + "="*60)
    print("FINAL EXPLANATION:")
    print("="*60)
    print(result["explanation_raw"])
    print("="*60)
    print(f"\nEvidence sources used: {list(set(e.source_db for e in result['evidence']))}")
    print(f"Top evidence sources detail: {[e.doc.metadata.get('source', '?') for e in result['evidence'][:5]]}")

if __name__ == "__main__":
    main()