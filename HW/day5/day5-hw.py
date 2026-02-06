import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 定義封裝函數與類別
# ==========================================

class SimpleVDB:
    def __init__(self, collection_name="day5_assignment"):
        """初始化資料庫與模型"""
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = 384
        
        # 建立 Collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def add_documents(self, documents: list):
        """函數：嵌入並上傳文件"""
        vectors = self.model.encode(documents)
        points = [
            PointStruct(
                id=i, 
                vector=vectors[i].tolist(), 
                payload={"content": documents[i]}
            )
            for i in range(len(documents))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ 成功嵌入 {len(points)} 筆資料到 Qdrant")

    def query(self, text: str, top_k: int = 2):
        """函數：搜尋最相關的內容"""
        query_vector = self.model.encode(text).tolist()
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        return search_result

# ==========================================
# 2. 執行主程式
# ==========================================

def run_vdb_demo():
    # 初始化
    vdb = SimpleVDB()

    # 準備資料
    raw_data = [
        "Qdrant 是一款高效能的向量資料庫。",
        "Python 是開發 AI 應用的首選語言。",
        "向量檢索比傳統關鍵字搜尋更能理解語義。",
        "今天的天氣非常適合在戶外寫程式。",
        "Github 是程式設計師管理版本的好幫手。",
        "學習新的技術雖然辛苦，但非常有成就感。"
    ]
    
    # 呼叫新增函數
    vdb.add_documents(raw_data)

    # 測試查詢
    test_queries = ["我想學習怎麼管理代碼", "AI 開發語言"]

    for q in test_queries:
        print(f"\n🔍 [檢索查詢]: {q}")
        print("-" * 40)
        
        # 呼叫查詢函數
        hits = vdb.query(q)
        
        if not hits:
            print("找不到相關結果。")
        else:
            for i, hit in enumerate(hits):
                content = hit.payload['content']
                score = hit.score
                print(f"結果 {i+1}: {content} (相似度: {score:.4f})")

if __name__ == "__main__":
    run_vdb_demo()