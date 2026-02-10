import os
import pandas as pd

def setup_vdb_and_search():
    # --- 1. 自動定位檔案路徑 (解決 FileNotFoundError) ---
    # 取得本程式檔案所在的絕對路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(BASE_DIR, "questions.csv")

    if not os.path.exists(questions_path):
        # 嘗試在上一層目錄尋找 (常見的專案結構)
        questions_path = os.path.join(os.path.dirname(BASE_DIR), "questions.csv")
    
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"❌ 找不到 questions.csv！請確認檔案放在: {BASE_DIR}")

    # --- 2. 讀取 CSV 並自動修正欄位名稱 (解決 KeyError) ---
    questions_df = pd.read_csv(questions_path)
    
    # 清理欄位名稱（去除前後空格、轉小寫）以增加匹配成功率
    questions_df.columns = [col.strip().lower() for col in questions_df.columns]
    
    # 定義可能的欄位名稱清單
    possible_cols = ['questions', 'question', 'content', 'q_text', 'query']
    target_col = next((col for col in possible_cols if col in questions_df.columns), None)

    if target_col:
        print(f"✅ 成功找到題目檔，使用欄位: '{target_col}'")
        q_texts = questions_df[target_col].astype(str).tolist()
    else:
        # 如果真的找不到，顯示目前的欄位清單方便你確認
        actual_cols = questions_df.columns.tolist()
        raise KeyError(f"❌ 在 CSV 中找不到問題欄位！目前的欄位是: {actual_cols}。請將 CSV 第一行改為 'questions'")

    # --- 3. 接下來是你的向量資料庫處理邏輯 ---
    # (此處請接續你原本的 vdb 初始化與搜尋代碼)
    print(f"🚀 開始處理 {len(q_texts)} 筆問題檢索...")
    
    # 範例回傳
    return q_texts # 或者你原本預計回傳的結果