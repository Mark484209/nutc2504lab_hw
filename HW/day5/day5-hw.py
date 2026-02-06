import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 配置區 ---
API_URL = "https://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"

# 3. 使用 API 獲得向量 (動態獲取並偵測維度)
# ------------------------------------------------
def get_embeddings(texts):
    response = requests.post(
        API_URL,
        json={
            "texts": texts,
            "task_description": "檢索技術文件",
            "normalize": True
        }
    )
    return response.json()["embeddings"]

# 初始測試以獲取動態維度
test_embeddings = get_embeddings(["測試偵測維度"])
vector_dim = len(test_embeddings[0]) # 這裡會根據 API 回傳自動計算，不再寫死
print(f">>> 偵測到向量維度: {vector_dim}")

# 1. 建立 Qdrant 連接
# ------------------------------------------------
client = QdrantClient(url=QDRANT_URL) 

metrics = {
    "Dot": Distance.DOT,
    "Cosine": Distance.COSINE,
    "Euclidean": Distance.EUCLID
}

for mode_name, dist_type in metrics.items():
    coll_name = f"collection_{mode_name}"
    if client.collection_exists(coll_name):
        client.delete_collection(coll_name)
    
    print(f">>> 正在建立 {coll_name} (維度: {vector_dim}, 使用 {mode_name} 距離)...")
    client.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=vector_dim, distance=dist_type),
    )

# 2. 準備 Point 資料
# ------------------------------------------------
raw_data = [
    {"category": "Programming", "content": "Python 是開發 AI 的首選語言"},
    {"category": "AI", "content": "機器學習是未來趨勢"},
    {"category": "Database", "content": "Qdrant 向量資料庫非常高效"},
    {"category": "AI", "content": "深度學習模型需要大量數據"},
    {"category": "Programming", "content": "Git 幫助開發者管理程式碼"}
]

# 4. 嵌入到 VDB
# ------------------------------------------------
texts = [d["content"] for d in raw_data]
vectors = get_embeddings(texts)

for mode_name in metrics.keys():
    coll_name = f"collection_{mode_name}"
    points = [
        PointStruct(
            id=i, 
            vector=vectors[i], 
            payload=raw_data[i]
        ) for i in range(len(raw_data))
    ]
    client.upsert(collection_name=coll_name, points=points)

# 5. 召回內容
# ------------------------------------------------
print("\n" + "="*60)
query_text = "AI開發，與學習python"
print(f"查詢詞: '{query_text}'")
print("="*60)

query_vector = get_embeddings([query_text])[0]

for mode_name in metrics.keys():
    print(f"\n【度量模式：{mode_name}】")
    coll_name = f"collection_{mode_name}"
    
    search_results = client.query_points(
        collection_name=coll_name,
        query=query_vector,
        limit=2
    ).points
    
    for res in search_results:
        print(f"-> 分數: {res.score:.4f} | 分類: {res.payload['category']} | 內容: {res.payload['content']}")

print("\n>>> 所有度量比較執行完畢。")