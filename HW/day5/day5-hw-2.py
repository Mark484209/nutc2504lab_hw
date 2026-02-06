import os
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 切塊邏輯實作
# ==========================================

def fixed_size_chunking(text, size=100):
    """固定大小切塊"""
    return [text[i:i + size] for i in range(0, len(text), size)]

def sliding_window_chunking(text, size=100, overlap=30):
    """滑動視窗切塊 (包含重疊部分)"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
        if end >= len(text): break
    return chunks

# ==========================================
# 2. Qdrant 處理類別
# ==========================================

class QdrantHandler:
    def __init__(self, collection_name="cw_02_collection"):
        # 使用記憶體模式
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = 384
        
        # 初始化 Collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        self.current_id = 0 # 用於生成唯一的整數 ID

    def insert_chunks(self, chunks, method_label):
        """將切塊轉換為向量並存入 VDB (修正 UUID 報錯)"""
        if not chunks: return
        
        vectors = self.model.encode(chunks)
        points = []
        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=self.current_id, # 使用自增整數 ID 避免 UUID 格式錯誤
                vector=vectors[i].tolist(),
                payload={
                    "content": chunk,
                    "method": method_label,
                    "type": "text"
                }
            ))
            self.current_id += 1
            
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ [{method_label}] 成功嵌入 {len(chunks)} 筆切塊")

    def insert_table_data(self, df, filename):
        """處理表格資料 (作業第 6 點)"""
        # 將表格每一列轉為文字描述
        table_texts = []
        for _, row in df.iterrows():
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
            table_texts.append(f"檔案 {filename} 紀錄 - {row_str}")
        
        vectors = self.model.encode(table_texts)
        points = []
        for i, text in enumerate(table_texts):
            points.append(PointStruct(
                id=self.current_id,
                vector=vectors[i].tolist(),
                payload={
                    "content": text,
                    "method": "table_processing",
                    "type": "table",
                    "source": filename
                }
            ))
            self.current_id += 1
            
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"📊 [Table] 成功處理來自 {filename} 的 {len(table_texts)} 筆列資料")

    def search(self, query, limit=3):
        """召回內容"""
        query_vector = self.model.encode(query).tolist()
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points

# ==========================================
# 3. 主執行程序 (包含檔案讀取)
# ==========================================

def main():
    handler = QdrantHandler()

    # --- Step 1: 處理 text.txt ---
    file_path = "text.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # Step 2 & 3: 實作切塊
        fixed_chunks = fixed_size_chunking(raw_text)
        sliding_chunks = sliding_window_chunking(raw_text)

        # Step 4: 嵌入
        handler.insert_chunks(fixed_chunks, "fixed_size")
        handler.insert_chunks(sliding_chunks, "sliding_window")
    else:
        print(f"⚠️ 找不到 {file_path}，跳過文本處理")

    # --- Step 6: 處理表格資料夾 (table/) ---
    table_dir = "table"
    if os.path.exists(table_dir):
        for file in os.listdir(table_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(table_dir, file))
                handler.insert_table_data(df, file)
    else:
        print(f"⚠️ 找不到 {table_dir} 資料夾，跳過表格處理")

    # --- Step 5: 召回並比較 ---
    query = "請告訴我關於檔案中的關鍵資訊"
    print(f"\n🔍 [測試檢索]: {query}")
    print("-" * 50)
    
    results = handler.search(query)
    for hit in results:
        m = hit.payload['method']
        c = hit.payload['content'][:100] # 只印前100字
        s = hit.score
        print(f"結果 (方法:{m}): {c}... (相似度:{s:.4f})")

if __name__ == "__main__":
    main()