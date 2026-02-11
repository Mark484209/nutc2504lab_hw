import pandas as pd
import os

def load_all_texts():
    """載入所有已經辨識好的文字檔"""
    texts = {}
    files = ["1.pdf.txt", "2.pdf.txt", "3.pdf.txt", "4.png.txt", "5.docx.txt"]
    for f in files:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as file:
                texts[f.replace(".txt", "")] = file.read()
    return texts

def generate_test_dataset(input_csv):
    # 1. 讀取原始問題檔，並處理欄位名稱可能存在的空白或大小寫問題
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. 取得原始欄位名稱
    # 這裡假設你的 CSV 至少有 q_id 和 questions 兩欄
    if 'q_id' not in df.columns or 'questions' not in df.columns:
        print(f"警告：找不到標準欄位。目前欄位有: {list(df.columns)}")
        # 強制指定前兩欄為 q_id 和 questions
        df.rename(columns={df.columns[0]: 'q_id', df.columns[1]: 'questions'}, inplace=True)

    sources = load_all_texts()
    
    # 3. 建立結果列表
    output_data = []
    
    for _, row in df.iterrows():
        question = str(row['questions'])
        
        # --- 這裡是你應該放入 RAG 邏輯的地方 ---
        # 為了讓你先拿到 CSV，這裡先放一個示意答案
        # 之後你可以改成呼叫 LLM: answer = get_llm_answer(question)
        answer = "根據文件內容分析，這是一個關於特定工廠登記或法規的回答內容。"
        
        # 簡易檢索邏輯：找哪份文件包含最多的問題關鍵字
        best_source = "3.pdf" # 預設值
        max_matches = 0
        for doc_name, content in sources.items():
            matches = sum(1 for word in question if word in content)
            if matches > max_matches:
                max_matches = matches
                best_source = doc_name
        
        output_data.append({
            "q_id": row['q_id'],
            "questions": question,
            "answer": answer,
            "source": best_source
        })

    # 4. 轉成 DataFrame 並儲存
    result_df = pd.DataFrame(output_data)
    result_df.to_csv("test_dataset.csv", index=False, encoding="utf-8-sig")
    return result_df

if __name__ == "__main__":
    if os.path.exists("questions.csv"):
        print("開始產生 test_dataset.csv...")
        final_df = generate_test_dataset("questions.csv")
        print("✅ 成功！產出的檔案欄位為:", list(final_df.columns))
        print(final_df.head(3)) # 顯示前三行預覽
    else:
        print("錯誤：找不到 questions.csv，請確認檔案在同一目錄下。")