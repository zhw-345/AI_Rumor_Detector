import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==================== 配置区 ====================
LIAR_ROOT = "./data/liar_dataset"      #根目录
CASE_DB_PATH = "./chroma_pheme_cases"     # 案例库持久化路径
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

def parse_liar_thread(thread_path:Path):
    docs = []
    ids = []
    datasets = [f for f in thread_path.glob("*.tsv")]
    for dataset in datasets:
        f = pd.read_csv(dataset,sep="\t",header=None,usecols=[0,1,2,3,13])
        for _, row in f.iterrows():
            if row[0] not in ids:
                docs.append(Document(
                    page_content=row[2],
                    metadata={
                        "id":row[0].replace("json","liar"),
                        "label":row[1],
                        "topic":row[3],
                        "context":row[13],
                        "source":"liar_case_db"
                    }
                ))
                ids.append(row[0])
    return docs

def load_database(docs: List[Document], persist_dir: str):
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        persist_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print(f"\n加载已有案例库: {persist_dir}")
        ids = [doc.metadata.get("id") for doc in docs]
        for i in range(0, len(ids), 5000):
            batch_docs = docs[i : i + 5000]
            batch_ids = ids[i : i + 5000]
            persist_db.add_documents(documents=batch_docs, ids=batch_ids)
        return persist_db
    return

if __name__ == "__main__":
    docs = parse_liar_thread(Path(LIAR_ROOT))
    db = load_database(docs, CASE_DB_PATH)
    #db = Chroma(persist_directory=CASE_DB_PATH,embedding_function=embeddings)
    print(f'{db.get(ids=["972.liar"])}')
    print(f"库中总条数: {db._collection.count()}")