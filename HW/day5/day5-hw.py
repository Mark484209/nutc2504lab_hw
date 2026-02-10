import os
import uuid
import pandas as pd
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# === 修正後的 Import ===
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# === 0. 配置與初始化 ===
API_KEY = "YOUR_API_KEY" 
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 取得程式碼所在目錄，確保路徑正確
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = QdrantClient(url="http://localhost:6333")

class CustomEmbeddings:
    def embed_documents(self, texts): return get_embeddings(texts)
    def embed_query(self, text): return get_embeddings([text])[0]

# === 1. 功能函數 ===

def get_embeddings(texts):
    if not texts: return []
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        response = requests.post(EMBED_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['embeddings']
    except Exception as e:
        print(f"❌ Embedding API 錯誤: {e}")
        return []

def submit_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload, timeout=20)
        return response.json().get("score", 0) if response.status_code == 200 else 0
    except:
        return 0

# === 2. 檔案處理與切塊 ===

def process_files_and_chunk():
    data_files = [f"data_0{i}.txt" for i in range(1, 6)]
    all_chunks_data = {"固定大小": [], "滑動視窗": [], "語義切塊": []}
    embeddings_tool = CustomEmbeddings()
    
    semantic_sub_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    
    print("\n" + "="*20 + " 1. 開始檔案切塊階段 " + "="*20)
    for file_name in data_files:
        # 使用絕對路徑尋找文本檔案
        full_path = os.path.join(BASE_DIR, file_name)
        if not os.path.exists(full_path):
            print(f"⚠️ 找不到檔案: {full_path}")
            continue
            
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"📄 讀取檔案: {file_name} ({len(content)} 字)")
        
        # 1. 固定大小
        f_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0, separator="")
        for c in [d.page_content for d in f_splitter.create_documents([content])]:
            all_chunks_data["固定大小"].append({"text": c, "source": file_name})
        
        # 2. 滑動視窗
        s_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for c in [d.page_content for d in s_splitter.create_documents([content])]:
            all_chunks_data["滑動視窗"].append({"text": c, "source": file_name})
        
        # 3. 語義切塊
        sem_splitter = SemanticChunker(
            embeddings_tool, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        sem_base_docs = sem_splitter.create_documents([content])
        
        for doc in sem_base_docs:
            if len(doc.page_content) > CHUNK_SIZE:
                sub_docs = semantic_sub_splitter.split_text(doc.page_content)
                for sub_c in sub_docs:
                    all_chunks_data["語義切塊"].append({"text": sub_c, "source": file_name})
            else:
                all_chunks_data["語義切塊"].append({"text": doc.page_content, "source": file_name})
        
    return all_chunks_data

# === 3. 向量檢索與評分 ===

def setup_vdb_and_search():
    results_for_csv = []
    
    # --- 修正路徑與欄位名稱問題 ---
    questions_path = os.path.join(BASE_DIR, "questions.csv")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"❌ 找不到題目檔: {questions_path}")
        
    questions_df = pd.read_csv(questions_path)
    # 強制清理欄位名稱
    questions_df.columns = [c.strip().lower() for c in questions_df.columns]
    
    # 彈性匹配欄位名
    target_q_col = next((c for c in ['questions', 'question'] if c in questions_df.columns), None)
    target_id_col = next((c for c in ['q_id', 'id'] if c in questions_df.columns), None)
    
    if not target_q_col or not target_id_col:
        raise KeyError(f"❌ CSV 欄位不符！需要 id 和 questions。目前: {questions_df.columns.tolist()}")

    q_texts = questions_df[target_q_col].astype(str).tolist()
    q_ids = questions_df[target_id_col].tolist()
    
    method_to_coll = {
        "固定大小": "coll_fixed_size",
        "滑動視窗": "coll_sliding_window",
        "語義切塊": "coll_semantic_chunk"
    }
    
    print(f"\n📡 正在批量獲取 {len(q_texts)} 個問題的向量...")
    all_q_vectors = get_embeddings(q_texts)
    
    all_chunks_data = None 

    print("\n" + "="*20 + " 2. 向量檢索與評分階段 " + "="*20)

    for method, coll_name in method_to_coll.items():
        print(f"\n🛠️ 正在處理方法: [{method}]")
        
        if not client.collection_exists(collection_name=coll_name):
            if all_chunks_data is None:
                all_chunks_data = process_files_and_chunk()
            
            chunk_items = all_chunks_data[method]
            texts = [item['text'] for item in chunk_items]
            sources = [item['source'] for item in chunk_items]
            
            chunk_vectors = get_embeddings(texts)
            if not chunk_vectors: continue

            client.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=len(chunk_vectors[0]), distance=Distance.COSINE)
            )
            
            points = [
                PointStruct(
                    id=uuid.uuid4().hex, 
                    vector=chunk_vectors[i], 
                    payload={"text": texts[i], "source": sources[i]}
                ) for i in range(len(texts))
            ]
            client.upsert(collection_name=coll_name, points=points)
            print(f"✅ {coll_name} 初始化完成。")

        for i, q_vec in enumerate(all_q_vectors):
            search_res = client.query_points(
                collection_name=coll_name, query=q_vec, limit=3
            ).points
            
            retrieved_content = "\n".join([h.payload['text'] for h in search_res])
            unique_sources = ",".join(list(set([h.payload['source'] for h in search_res])))
            
            score = submit_and_get_score(q_ids[i], retrieved_content)
            
            if i % 20 == 0:
                print(f"   📝 Q{q_ids[i]} | Score: {score:.4f} | Method: {method}")
            
            results_for_csv.append({
                "q_id": q_ids[i],
                "method": method,
                "retrieve_text": retrieved_content,
                "score": score,
                "source": unique_sources
            })
            
    return results_for_csv

# === 4. 主程式 ===

if __name__ == "__main__":
    start_time = time.time()
    final_results = setup_vdb_and_search()
    
    df_output = pd.DataFrame(final_results)
    
    # 重要：改掉輸出的檔名，避免覆蓋題目
    output_name = os.path.join(BASE_DIR, "hw_results.csv")
    df_output.to_csv(output_name, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*30 + " 3. 執行統計 " + "="*30)
    if not df_output.empty:
        avg_scores = df_output.groupby('method')['score'].mean()
        for m, s in avg_scores.items():
            print(f"   🔹 {m} 平均分: {s:.4f}")
    
    print(f"\n✅ 全部完成！結果已儲存至: {output_name}")