import os
import numpy as np
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# 1. 載入模型 (自動偵測維度)
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = HuggingFaceEmbeddings(model_name=model_name)

def get_metrics(v1, v2):
    """ 動態矩陣運算 """
    v1, v2 = np.array(v1), np.array(v2)
    # Cosine
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Dot Product
    dot = np.dot(v1, v2)
    # Euclidean
    euc = np.linalg.norm(v1 - v2)
    return cos, dot, euc

def run_dynamic_hw():
    print("=== 執行動態維度運算流程 ===\n")

    # --- 步驟 1: 讀取 text.txt ---
    with open("text.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # --- 步驟 2 & 3: 動態切塊 (根據比例設定，不寫死固定數值) ---
    # 設定 chunk 為總長度的 1/10，確保一定會切出多塊
    dynamic_size = max(100, len(content) // 10)
    dynamic_overlap = dynamic_size // 2

    fixed_split = CharacterTextSplitter(separator="。", chunk_size=dynamic_size, chunk_overlap=0)
    fixed_chunks = fixed_split.split_text(content)

    sliding_split = RecursiveCharacterTextSplitter(chunk_size=dynamic_size, chunk_overlap=dynamic_overlap)
    sliding_chunks = sliding_split.split_text(content)

    print(f"✅ 動態切塊：Size={dynamic_size}, Overlap={dynamic_overlap}")
    print(f"✅ 固定切塊: {len(fixed_chunks)} 塊 | 滑動視窗: {len(sliding_chunks)} 塊")

    # --- 步驟 4: 維度偵測 (Dimensions) ---
    sample_vec = model.embed_query("測試")
    dimensions = len(sample_vec)
    print(f"📊 模型動態維度: {dimensions} 維 (Dimensions)")

    # --- 步驟 5: 向量比較 ---
    query = "Graph RAG 如何解決幻覺問題？"
    q_vec = model.embed_query(query)
    
    # 直接從切好的塊中拿數據算
    f_vec = model.embed_query(fixed_chunks[0])
    s_vec = model.embed_query(sliding_chunks[0])

    f_c, f_d, f_e = get_metrics(q_vec, f_vec)
    s_c, s_d, s_e = get_metrics(q_vec, s_vec)

    print(f"\n[固定切塊 0] Cosine: {f_c:.4f} | Dot: {f_d:.2f} | Euc: {f_e:.2f}")
    print(f"[滑動視窗 0] Cosine: {s_c:.4f} | Dot: {s_d:.2f} | Euc: {s_e:.2f}")

    # --- 步驟 6: 表格與維度應用 ---
    if os.path.exists("table_html.html"):
        with open("table_html.html", "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            table_text = soup.find("table").get_text(separator=" ")
            
        t_vec = model.embed_query(table_text)
        t_c, _, _ = get_metrics(q_vec, t_vec)
        print(f"\n✅ 表格文字已轉為 {len(t_vec)} 維向量")
        print(f"✅ 查詢與表格的語義關聯度 (Cosine): {t_c:.4f}")

if __name__ == "__main__":
    run_dynamic_hw()