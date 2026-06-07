import json
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==================== 配置区 ====================
FEVER_JSONL_PATH = "./data/wiki-pages/train.jsonl"           # FEVER 声明文件
KNOWLEDGE_DB_PATH = "./chroma_fever_knowledge"     # 知识库持久化路径

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Embedding 初始化
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# ==================== FEVER 解析函数 ====================

index = {}
for wiki_file in Path("./data/wiki-pages/wiki-pages/").glob("wiki-*.jsonl"):
    with open(wiki_file, "r") as f:
        for line in f:
            page = json.loads(line)
            index[page["id"]] = page.get("lines", "")

def get_evidence(page_name: str, line_num: int) -> str:
    page = index.get(page_name)
    if not page:
        return None
    lines = page.split("\n")
    if line_num < len(lines):
        return lines[line_num].split("\t", 1)[1]
    return None


def parse_fever_claim(claim_item: dict) -> List[Document]:
    """
    解析单条 FEVER 声明，提取证据段落为 Document
    返回多个 Document（每个证据段落一个）
    """
    claim_text = claim_item.get("claim", "")
    label = claim_item.get("label", "UNKNOWN")
    verifiable = claim_item.get("verifiable", "UNKNOWN")
    
    documents = []
    evidence_sets = claim_item.get("evidence", [])
    
    all_evidence = []
    # 遍历所有证据集
    for evidence_set in evidence_sets:
        # 遍历证据集中的每条证据
        all_evidence.extend(evidence_set)
    unique = set()
    for evidence in all_evidence:
        # evidence 格式: [annotation_id, evidence_id, page_name, line_num]
        if len(evidence) < 4:
            continue
            
        _, _, page_name, line_num = evidence
        if (page_name,line_num) in unique:
            continue
        else:
            unique.add((page_name,line_num))

            
        # 尝试加载 Wikipedia 原文
        page_content = get_evidence(page_name, line_num)
        
        if not page_content:
            continue
            
        # 构建 Document
        doc = Document(
            page_content=page_content,
            metadata={
                "source": "fever",
                "claim_text": claim_text,
                "claim_label": label,                    # SUPPORTS / REFUTES / NEI
                "verifiable": verifiable,
                "wiki_page": page_name,
                "wiki_line": line_num,
                "knowledge_type": "fact_verification"
            }
        )
        documents.append(doc)
    
    return documents


def build_fever_knowledge_db(
    fever_jsonl: str,
    persist_dir: str = "./chroma_fever_knowledge",
) -> Chroma:
    """
    从 FEVER 数据集构建知识库
    """
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"加载已有知识库: {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    
    print(f"构建 FEVER 知识库...")
    print(f"  声明文件: {fever_jsonl}")
    
    all_docs = []
    claim_count = 0
    
    with open(fever_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            
            try:
                claim_item = json.loads(line.strip())
                docs = parse_fever_claim(claim_item)
                all_docs.extend(docs)
                claim_count += 1
                
                if claim_count % 1000 == 0:
                    print(f"  已处理 {claim_count} 条声明, {len(all_docs)} 个证据段落")
                    
            except json.JSONDecodeError:
                continue
    
    print(f"\n总计: {claim_count} 条声明, {len(all_docs)} 个证据段落")
    
    # 统计标签分布
    labels = {}
    for doc in all_docs:
        lbl = doc.metadata["claim_label"]
        labels[lbl] = labels.get(lbl, 0) + 1
    print("标签分布:", labels)
    
    # 构建向量库
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"知识库构建完成: {persist_dir}")
    return vectorstore


# ==================== 检索与验证 ====================

def format_fever_evidence(docs: List[Document]) -> str:
    """格式化 FEVER 证据"""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        block = (
            f"[FEVER | Claim ID: {meta.get('claim_id')} | "
            f"Claim_label: {meta.get('claim_label')} | "
            f"Page: {meta.get('wiki_page')}:{meta.get('wiki_line')}]\n"
            f"Original Claim: {meta.get('claim_text', 'N/A')}\n"
            f"Evidence: {doc.page_content}"
        )
        formatted.append(block)
    return "\n\n---\n\n".join(formatted)


def verify_with_fever(target_text: str, fever_retriever, k: int = 3) -> tuple:
    """
    用 FEVER 知识库验证目标文本
    """
    results = fever_retriever.invoke(target_text)
    
    # 分析结果
    supports = sum(1 for d in results if d.metadata.get("claim_label") == "SUPPORTS")
    refutes = sum(1 for d in results if d.metadata.get("claim_label") == "REFUTES")
    nei = sum(1 for d in results if d.metadata.get("claim_label") == "NOT ENOUGH INFO")
    
    verdict = "UNKNOWN"
    if refutes > supports and refutes > nei:
        verdict = "LIKELY FALSE (contradicted by FEVER evidence)"
    elif supports > refutes and supports > nei:
        verdict = "LIKELY TRUE (supported by FEVER evidence)"
    else:
        verdict = "UNCERTAIN (insufficient evidence)"
    
    return verdict, results


# ==================== 主流程 ====================

if __name__ == "__main__":
    # 1. 构建/加载知识库
    fever_db = build_fever_knowledge_db(
        fever_jsonl=FEVER_JSONL_PATH,
        persist_dir=KNOWLEDGE_DB_PATH,
    )
    
    # 2. 配置检索器
    fever_retriever = fever_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 3. 测试检索
    test_claims = [
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "The earth is flat.",
        "Barack Obama was born in Hawaii.",
    ]
    
    print("\n" + "="*60)
    print("FEVER 知识库检索测试")
    
    for claim in test_claims:
        print(f"\n{'='*60}")
        print(f"Claim: {claim}")
        
        verdict, evidence = verify_with_fever(claim, fever_retriever)
        print(f"FEVER Verdict: {verdict}")
        print(f"Evidence ({len(evidence)} items):")
        for i, doc in enumerate(evidence[:3], 1):
            meta = doc.metadata
            print(f"  {i}. [{meta['label']}] {doc.page_content[:100]}...")