import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 建立 Qdrant Collection 並連接
# ==========================================
# 使用記憶體模式（快速測試用），若要連到 Docker 則改為 "http://localhost:6333"
client = QdrantClient(":memory:")
collection_name = "day5_assignment"

# 載入 Embedding 模型 (步驟 3: 使用 API/模型 獲得向量)
# 此模型會輸出 384 維度的向量
model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化 Collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# ==========================================
# 2 & 3. 準備 Points 資料並轉換向量
# ==========================================
# 準備至少 5 個 Point 的內容
raw_data = [
    "Qdrant 是一款高效能的向量資料庫。",
    "Python 是開發 AI 應用的首選語言。",
    "向量檢索比傳統關鍵字搜尋更能理解語義。",
    "今天的天氣非常適合在戶外寫程式。",
    "Github 是程式設計師管理版本的好幫手。",
    "學習新的技術雖然辛苦，但非常有成就感。"
]

# 獲得向量 (API 獲得向量)
vectors = model.encode(raw_data)

# ==========================================
# 4. 嵌入到 VDB (Vector Database)
# ==========================================
points = [
    PointStruct(
        id=i, 
        vector=vectors[i].tolist(), 
        payload={"content": raw_data[i]}
    )
    for i in range(len(raw_data))
]

# 上傳資料
client.upsert(collection_name=collection_name, points=points)
print(f"--- 成功嵌入 {len(points)} 筆資料到 Qdrant ---")


# ==========================================
# 5. 召回內容 (Search / Retrieve)
# ==========================================
query_text = "我想學習怎麼管理代碼"
print(f"\n[檢索查詢]: {query_text}")

# 將問題轉為向量
query_vector = model.encode(query_text).tolist()

# 執行搜尋 (使用 query_points 替代 search 以增加相容性)
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2
).points  # 注意：query_points 回傳的是一個物件，其點在 .points 屬性中

print("-" * 40)
for i, hit in enumerate(search_result):
    # query_points 的結果屬性是 payload
    print(f"結果 {i+1}: {hit.payload['content']} (相似度: {hit.score:.4f})")