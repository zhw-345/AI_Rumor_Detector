import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import torch

# ==================== 配置区 ====================
PHEME_ROOT = "./data/PHEME_veracity/all-rnr-annotated-threads/"      # PHEME 解压后的根目录
CASE_DB_PATH = "../db/chroma_pheme_cases"     # 案例库持久化路径
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

API_KEY = "sk-Th_ZuQs31rz8M72BwOViKw"
BASE_URL = "https://models.sjtu.edu.cn/api/v1"
CHAT_MODEL = "deepseek-chat"

DOUBT_KEYWORDS_2 = [
    # 直接否认
    "fake", "false", "untrue", "not true", "not real", "fabricated", "hoax",
    "fraud", "bogus", "phony", "sham", "counterfeit", "forgery"]
DOUBT_KEYWORDS_1 = [
    # 质疑真实性
    "debunked", "disproved", "refuted", "discredited", "misleading",
    "inaccurate", "incorrect", "wrong", "error", "fallacy"]
DOUBT_KEYWORDS_0 = [
    # 要求证据
    "source", "proof", "evidence", "link", "verify", "confirmation",
    "unverified", "unconfirmed", "alleged", "allegedly", "claim", "claims",
    "rumor", "rumour", "speculation", "conspiracy",
    # 讽刺/反讽
    "really?", "seriously?", "are you kidding", "come on", "nah", "nope",
    "doubt", "dubious", "skeptical", "questionable", "suspect", "sketchy",
    # 修正/辟谣
    "actually", "correction", "update", "clarification", "retracted",
    "retraction", "apologize", "mistake", "misinformation", "disinformation",
    # 情感否定
    "lies", "lying", "liar", "propaganda", "bullshit", "bs", "crap",
    "nonsense", "rubbish", "garbage", "trash"
]
SUPPORT_KEYWORDS_2 = [
    # 直接确认
    "true", "real", "confirmed", "verified", "authentic", "genuine",
    "legitimate", "valid", "accurate", "correct", "factual"]
SUPPORT_KEYWORDS_1 = [
    # 官方/权威背书
    "official", "authorities", "police", "government", "reported by",
    "according to", "sources confirm", "breaking news", "just in",
    # 目击/亲历
    "eyewitness", "witness", "on the scene", "i saw", "we saw",
    "video shows", "photo shows", "evidence shows", "proof"]
SUPPORT_KEYWORDS_0 = [
    # 情感支持
    "pray", "praying", "thoughts and prayers", "rip", "rest in peace",
    "devastating", "tragic", "heartbreaking", "shocking", "unbelievable",
    # 转发传播（中性偏支持）
    "rt", "retweet", "sharing", "spread the word", "pass it on",
    "important", "urgent", "alert", "warning", "be careful", "stay safe",
    # 补充信息
    "update", "developing", "more details", "full story", "read more",
    "learn more", "check this", "see here", "link in bio"
]

# 1. Embedding 初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# ==================== PHEME 解析函数 ====================

def parse_pheme_thread(thread_path: Path) -> Optional[dict]:
    """
    解析单个 thread 文件夹，提取源推文和标签
    返回: {"id": str, "text": str, "label": int, "event": str, "label_detail": str}
    """
    thread_id = thread_path.name
    is_rumour = thread_path.parent.name
    event_name = thread_path.parent.parent.name.replace("-all-rnr-threads", "")
    
    # --- 1. 读取源推文 ---
    source_dir = thread_path / "source-tweets"
    if not source_dir.exists():
        return None
    
    source_files = [f for f in source_dir.glob("*.json") if f.stem.isdigit()]
    if not source_files:
        return None
    
    try:
        with open(source_files[0], "r", encoding="utf-8") as f:
            tweet = json.load(f)
    except Exception:
        return None
    
    text = tweet.get("text", "").strip()
    tweet_id = str(tweet.get("id", thread_id))
    
    if not text:
        return None
    
    label = 1 if is_rumour == "rumours" else 0

        
    # 读取 reactions 统计
    reactions_dir = thread_path / "reactions"
    reaction_stats = {
        "reply_count": 0,
        "doubt_count": 0.0,      # 含 "fake", "false", "not true" 的回复数
        "support_count": 0.0,    # 含 "true", "confirmed" 的回复数
    }
    
    if reactions_dir.exists():
        reaction_files = [f for f in reactions_dir.glob("*.json") if not f.name.startswith(".")]
        reaction_stats["reply_count"] = len(reaction_files)
        
        for f in reaction_files:
            with open(f) as fp:
                r = json.load(fp)
                text_lower = r.get("text", "").lower()
                
                if any(kw in text_lower for kw in DOUBT_KEYWORDS_2):
                    reaction_stats["doubt_count"] += 2
                elif any(kw in text_lower for kw in DOUBT_KEYWORDS_1):
                    reaction_stats["doubt_count"] += 1
                elif any(kw in text_lower for kw in DOUBT_KEYWORDS_0):
                    reaction_stats["doubt_count"] += 0.5
                if any(kw in text_lower for kw in SUPPORT_KEYWORDS_2):
                    reaction_stats["support_count"] += 2
                elif any(kw in text_lower for kw in SUPPORT_KEYWORDS_1):
                    reaction_stats["support_count"] += 1
                elif any(kw in text_lower for kw in SUPPORT_KEYWORDS_0):
                    reaction_stats["support_count"] += 0.5
    
    return {
        "id": tweet_id,
        "text": text,
        "label": label,
        "event": event_name,
        "thread_id": thread_id,
        "reply": reaction_stats["reply_count"],
        "doubt": int(reaction_stats["doubt_count"]),
        "support":int(reaction_stats["support_count"])
    }


def build_pheme_documents(pheme_root: str) -> List[Document]:
    """
    遍历 PHEME 全部事件，提取所有源推文为 Document 列表
    """
    root = Path(pheme_root)
    if not root.exists():
        raise FileNotFoundError(f"PHEME 根目录不存在: {pheme_root}")
    
    records = []
    event_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    print(f"发现 {len(event_dirs)} 个事件文件夹: {[d.name for d in event_dirs]}")
    
    for event_dir in event_dirs:
        # 每个事件下是 thread 文件夹列表
        thread_dirs = [d for d in event_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for rumours in thread_dirs:
            rumour = [d for d in rumours.iterdir() if d.is_dir() and not d.name.startswith(".")]
            for thread_dir in rumour:
                record = parse_pheme_thread(thread_dir)
                if record:
                    records.append(record)
    
    print(f"共解析 {len(records)} 条源推文")
    
    # --- 去重（同一推文可能出现在多个采集批次） ---
    df = pd.DataFrame(records)
    before_drop = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"去重: {before_drop} -> {after_drop}")
    
    # --- 统计 ---
    print("\n=== 数据集统计 ===")
    print(f"总计: {len(df)} 条")
    print(f"谣言(rumour): {len(df[df['label']==1])} 条")
    print(f"非谣言(non-rumour): {len(df[df['label']==0])} 条")
    print("\n事件分布:")
    print(df["event"].value_counts())
    
    # --- 转为 Document ---
    docs = []
    for _, row in df.iterrows():
        docs.append(Document(
            page_content=row["text"],
            metadata={
                "case_id": str(row["id"]),
                "label": "rumour" if row["label"] == 1 else "non-rumour",
                "event": row["event"],
                "thread_id": row["thread_id"],
                "source": "pheme_case_db",
                "reactions": row["reply"],
                "doubt": row["doubt"],
                "support":row["support"]
            }
        ))
    
    return docs


# ==================== 构建 / 加载案例库 ====================

def build_or_load_case_db(docs: List[Document], persist_dir: str):
    """如果已存在则加载，否则构建"""
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"\n加载已有案例库: {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    
    print(f"\n构建新案例库: {persist_dir}")
    # PHEME 推文很短，不需要分块，直接入库
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


# ==================== RAG 解释链（复用之前的架构） ====================

def format_cases(docs):
    """格式化案例，包含 metadata"""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        block = (
            f"[Case ID: {meta.get('case_id')} | "
            f"Event: {meta.get('event')} | "
            f"Label: {meta.get('label')}] |"
            f"Reactions: {meta.get('reactions')}] |"
            f"Doubt: {meta.get('doubt')}] |"
            f"Support: {meta.get('support')}] |"
            f"Text: {doc.page_content}"
        )
        formatted.append(block)
    return "\n\n---\n\n".join(formatted)


# Prompt 模板（谣言解释专用）  {confidence}
prompt = ChatPromptTemplate.from_template("""
You are an expert in rumor detection and explainable AI.
A CNN model has classified the target text as **{predicted_label}**.
Your task is to explain the classification decision based on evidence.

## Target Text to Explain:
{target_text}

## Similar Historical Cases from PHEME Database:
{similar_cases}

## Instructions:
1. Analyze the target text using the similar cases above.
2. Identify specific linguistic features, logical patterns, or factual issues.
3. If RUMOR: explain which patterns it matches and cite Case IDs.
4. If NON-RUMOR: explain why it lacks typical rumor characteristics.
5. Cite specific Case IDs or Event names you referenced.
6. If uncertain, explicitly state it.

## Explanation:
""")


# LLM
llm = ChatOpenAI(
    model=CHAT_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.1,
    max_tokens=2048,
    timeout=600,
    max_retries=5,
)

# 检索器
def get_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": min(k * 4, 50),  # 候选池
            "lambda_mult": 0.7
        }
    )


# 解释函数
def explain_with_pheme(target_text: str, predicted_label: str, confidence: float, retriever):
    similar_cases = retriever.invoke(target_text)
    
    chain_input = {
        "target_text": target_text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "similar_cases": format_cases(similar_cases)
    }
    
    messages = prompt.format_messages(**chain_input)
    response = llm.invoke(messages)
    return response.content, similar_cases


# ==================== 主流程 ====================

if __name__ == "__main__":
    # 1. 解析 PHEME
    print("=" * 50)
    print("Step 1: 解析 PHEME 原始数据")
    docs = build_pheme_documents(PHEME_ROOT)
    
    # 2. 构建/加载向量库
    print("\n" + "=" * 50)
    print("Step 2: 构建案例库")
    case_db = build_or_load_case_db(docs, CASE_DB_PATH)
    
    # 3. 配置检索器
    retriever = get_retriever(case_db, k=5)
    
    # 4. 测试解释
    print("\n" + "=" * 50)
    print("Step 3: 运行解释")
    
    test_cases = [
        ("Bern museum accepts Gurlitt's problematic bequest. Let the litigation begin!", "rumor", 0.91),
        ("Swiss museum confirms it will take on Gurlitt collection", "non-rumor", 0.85),
    ]
    
    for text, label, conf in test_cases:
        print(f"\n--- Target: {text[:60]}... ---")
        explanation, sources = explain_with_pheme(text, label, conf, retriever)
        print(f"Retrieved {len(sources)} cases")
        print("Explanation:")
        print(explanation)
        print("-" * 40)