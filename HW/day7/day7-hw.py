import os
import ssl
import docx
import easyocr
import pdfplumber
import pandas as pd
import numpy as np
from pdf2image import convert_from_path

# --- 1. 環境初始化與配置 ---
ssl._create_default_https_context = ssl._create_unverified_context

# 初始化 EasyOCR (支援繁中、英文)
print("正在初始化 EasyOCR 模型...")
reader = easyocr.Reader(['ch_tra', 'en'])

# --- 2. 檔案處理函式 (IDP 技術) ---

def process_pdf(path):
    text = ""
    print(f"解析 PDF: {path}")
    # 優先嘗試提取原生文字層
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    
    # 如果文字太少，判定為掃描檔，啟動 OCR
    if len(text.strip()) < 20:
        print(f"偵測到掃描檔，啟動 OCR 辨識: {path}")
        images = convert_from_path(path)
        for img in images:
            result = reader.readtext(np.array(img), detail=0)
            text += "\n".join(result) + "\n"
    return text

def process_docx(path):
    print(f"解析 Word 檔: {path}")
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_image(path):
    print(f"解析圖片檔: {path}")
    result = reader.readtext(path, detail=0)
    return "\n".join(result)

# --- 3. 主執行邏輯 ---

def main():
    target_files = ["1.pdf", "2.pdf", "3.pdf", "4.png", "5.docx"]
    all_texts = {}

    # 第一階段：提取所有檔案內容
    print("\n=== 階段 1: 提取文檔內容 ===")
    for f in target_files:
        if not os.path.exists(f):
            print(f"找不到檔案 {f}，跳過。")
            continue
        
        ext = os.path.splitext(f)[1].lower()
        if ext == ".pdf":
            content = process_pdf(f)
        elif ext == ".docx":
            content = process_docx(f)
        else:
            content = process_image(f)
            
        all_texts[f] = content
        # 同步存成 txt 方便你檢查惡意提示詞 (作業第 4 題)
        with open(f"{f}.txt", "w", encoding="utf-8") as out:
            out.write(content)

    # 第二階段：產生 test_dataset.csv
    print("\n=== 階段 2: 產生 test_dataset.csv ===")
    questions_file = "questions.csv"
    
    if os.path.exists(questions_file):
        df_q = pd.read_csv(questions_file)
        # 修正欄位名稱可能的空格與大小寫問題
        df_q.columns = df_q.columns.str.strip().str.lower()
        
        # 檢查關鍵欄位
        id_col = 'q_id' if 'q_id' in df_q.columns else df_q.columns[0]
        q_col = 'questions' if 'questions' in df_q.columns else df_q.columns[1]

        results = []
        for _, row in df_q.iterrows():
            question = str(row[q_col])
            
            # 簡易檢索邏輯：尋找最相關的 source
            best_source = "unknown"
            for doc_name, content in all_texts.items():
                # 簡單匹配問題中的前幾個字
                if question[:5] in content:
                    best_source = doc_name
                    break
            
            results.append({
                "q_id": row[id_col],
                "questions": question,
                "answer": "這是一個模擬生成的 RAG 回答內容，請確保此內容符合文件內容。",
                "source": best_source
            })

        output_df = pd.DataFrame(results)
        output_df.to_csv("test_dataset.csv", index=False, encoding="utf-8-sig")
        print("✅ 成功產生 test_dataset.csv！")
    else:
        print(f"❌ 找不到 {questions_file}，無法產生結果 CSV。")

if __name__ == "__main__":
    main()