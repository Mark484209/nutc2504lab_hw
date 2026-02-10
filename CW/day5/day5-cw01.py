import os
import uuid
import time
import requests
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from langchain_text_splitters import CharacterTextSplitter

# === 全域配置 ===
EMBED_URL = "https://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"

class VectorSearchLab:
    def __init__(self, url):
        self.client = QdrantClient(url=url)
        # 定義實驗模式與對應的度量方式
        self.experiments = {
            "COSINE": {"collection": "lab_cosine", "metric": Distance.COSINE},
            "DOT":    {"collection": "lab_dot_prod", "metric": Distance.DOT},
            "EUCLID": {"collection": "lab_euclidean", "metric": Distance.EUCLID}
        }

    def fetch_embeddings(self, texts):
        """封裝 API 請求邏輯"""
        try:
            res = requests.post(EMBED_URL, json={"texts": texts, "normalize": True, "batch_size": 32})
            res.raise_for_status()
            return res.json().get('embeddings', [])
        except Exception as e:
            print(f"❌ API 連線失敗: {e}")
            return []

    def prepare_collections(self):
        """自動偵測模型維度並初始化三種實驗庫"""
        print("🛠️ 正在初始化實驗環境...")
        sample_vec = self.fetch_embeddings(["init"])
        if not sample_vec: return
        
        dim = len(sample_vec[0])
        for mode, cfg in self.experiments.items():
            name = cfg["collection"]
            # 若已存在則刪除舊資料
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
            
            # 建立新的 Collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=cfg["metric"])
            )
            # 針對分類欄位優化檢索速度
            self.client.create_payload_index(name, "category", "keyword")
            print(f"✅ [{mode}] 集合建立成功 (維度: {dim})")

    def run_ingestion(self, raw_text, category):
        """執行切塊、向量化與批量上傳"""
        # 切塊處理：長度 35, 重疊 5
        splitter = CharacterTextSplitter(separator="\n", chunk_size=35, chunk_overlap=5)
        chunks = splitter.split_text(raw_text)
        print(f"✂️ 文本切割為 {len(chunks)} 個片段")

        vectors = self.fetch_embeddings(chunks)
        if not vectors: return

        # 同步推送到三個不同的資料庫
        for mode, cfg in self.experiments.items():
            points = [
                PointStruct(
                    id=str(uuid.uuid4()), # 使用 UUID 確保唯一性
                    vector=vectors[i],
                    payload={"text": chunks[i], "category": category}
                ) for i in range(len(chunks))
            ]
            self.client.upsert(collection_name=cfg["collection"], points=points)
            print(f"📤 已將數據同步至 [{mode}]")

    def compare_retrieval(self, query_str, filter_cat=None):
        """執行跨庫對比檢索"""
        query_vec = self.fetch_embeddings([query_str])[0]
        
        # 建立過濾器
        q_filter = None
        if filter_cat:
            q_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=filter_cat))])

        print(f"\n" + "🔍" * 20)
        print(f"查詢內容: {query_str} | 過濾條件: {filter_cat or '無'}")
        print("🔍" * 20)

        for mode, cfg in self.experiments.items():
            hits = self.client.query_points(
                collection_name=cfg["collection"],
                query=query_vec,
                query_filter=q_filter,
                limit=3
            ).points
            
            print(f"\n📊 模式: {mode}")
            for hit in hits:
                score = hit.score
                txt = hit.payload['text'].replace('\n', ' ')
                print(f"   [{score:10.4f}] -> {txt}")

# === 主程式執行 ===
if __name__ == "__main__":
    # 初始化實驗物件
    lab = VectorSearchLab(QDRANT_URL)
    
    # 1. 準備資料庫
    lab.prepare_collections()

    # 2. 準備測試數據
    source_content = """
    機器學習是人工智慧的一個子領域，專注於演算法的開發。
    深度學習利用神經網路結構來模擬人類大腦的學習方式。
    大型語言模型如 GPT-4 具備強大的文本理解與生成能力。
    向量資料庫 Qdrant 提供了高效的相似度檢索功能。
    語意搜尋能理解詞句背後的真實意義，而非僅靠關鍵字。
    在 RAG 架構中，切塊技術與向量化是檢索精準度的關鍵。
    """

    # 3. 數據入庫
    lab.run_ingestion(source_content, category="ai_tech")

    # 4. 進行相似度對比測試
    lab.compare_retrieval("大型語言模型與 RAG 技術", filter_cat="ai_tech")