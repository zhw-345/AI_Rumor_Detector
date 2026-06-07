import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==================== 配置区 ====================
LIAR_ROOT = "./data/News-_dataset"      #根目录
CASE_DB_PATH = "./chroma_pheme_cases"     # 案例库持久化路径
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def parse_isot_thread(thread_path:Path):
    docs = []
    i = 1
    datasets = [f for f in thread_path.glob("*.csv")]
    for dataset in datasets:
        f = pd.read_csv(dataset,usecols=["title","text"])
        for _, row in f.iterrows():
            label = str(dataset).replace(".csv","")
            docs.append(Document(
                page_content=f"{row['title']}\n\n{row['text']}"[:1500],
                metadata={
                    "id":f"{i}.isot",
                    "label":label,
                    "full_text":row['text'],
                    "source":"isot_case_db"
                }
            ))
            i+=1
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
    docs = parse_isot_thread(Path(LIAR_ROOT))
    db = load_database(docs, CASE_DB_PATH)
    #db = Chroma(persist_directory=CASE_DB_PATH,embedding_function=embeddings)
    print(f'{db.get(ids=["972.isot"])}')
    print(f"库中总条数: {db._collection.count()}")