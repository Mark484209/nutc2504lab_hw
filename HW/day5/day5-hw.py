from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. 建立 Qdrant 連接 (改為連至 localhost 伺服器)
# ------------------------------------------------
client = QdrantClient(url="http://localhost:6333") 

model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384

# 定義度量模式
metrics = {
    "Dot": Distance.DOT,
    "Cosine": Distance.COSINE,
    "Euclidean": Distance.EUCLID
}

# 建立 Collection 並輸出進度
for mode_name, dist_type in metrics.items():
    coll_name = f"collection_{mode_name}"
    # 檢查若已存在則先刪除，確保每次執行都是新的測試
    if client.collection_exists(coll_name):
        client.delete_collection(coll_name)
    
    print(f">>> 正在建立 {coll_name} (使用 {mode_name} 距離)...")
    client.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=dim, distance=dist_type),
    )

# 2. 建立五個 Point 或更多 (包含分類標籤與內容)
# ------------------------------------------------
raw_data = [
    {"category": "Programming", "content": "Python 是開發 AI 的首選語言"},
    {"category": "AI", "content": "機器學習是未來趨勢"},
    {"category": "Database", "content": "Qdrant 向量資料庫非常高效"},
    {"category": "AI", "content": "深度學習模型需要大量數據"},
    {"category": "Programming", "content": "Git 幫助開發者管理程式碼"}
]

# 3. 使用 API 獲得向量並嵌入到 VDB
# ------------------------------------------------
texts = [d["content"] for d in raw_data]
vectors = model.encode(texts, normalize_embeddings=True)

for mode_name in metrics.keys():
    coll_name = f"collection_{mode_name}"
    points = [
        PointStruct(
            id=i, 
            vector=vectors[i].tolist(), 
            payload=raw_data[i]
        ) for i in range(len(raw_data))
    ]
    client.upsert(collection_name=coll_name, points=points)

# 5. 召回內容 (精準對應截圖格式)
# ------------------------------------------------
print("\n" + "="*60)
query_text = "AI開發，與學習python"
print(f"查詢詞: '{query_text}'")
print("="*60)

query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

for mode_name in metrics.keys():
    print(f"\n【度量模式：{mode_name}】")
    coll_name = f"collection_{mode_name}"
    
    search_results = client.query_points(
        collection_name=coll_name,
        query=query_vector,
        limit=2
    ).points
    
    for res in search_results:
        # 格式化輸出：-> 分數: XXX | 分類: XXX | 內容: XXX
        print(f"-> 分數: {res.score:.4f} | 分類: {res.payload['category']} | 內容: {res.payload['content']}")

print("\n>>> 所有度量比較執行完畢。")