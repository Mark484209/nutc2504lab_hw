import os
import ssl
import docx
import easyocr
import pdfplumber
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# 解決 SSL 下載問題
ssl._create_default_https_context = ssl._create_unverified_context

# 初始化 EasyOCR (只需初始化一次)
reader = easyocr.Reader(['ch_tra', 'en'])

def process_pdf(path):
    text = ""
    # 嘗試先抓文字層
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    
    # 如果抓不到字 (掃描檔)，改用 OCR
    if len(text.strip()) < 10:
        images = convert_from_path(path)
        for img in images:
            result = reader.readtext(np.array(img), detail=0)
            text += "\n".join(result) + "\n"
    return text

def process_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_image(path):
    result = reader.readtext(path, detail=0)
    return "\n".join(result)

# 主程式
files = ["1.pdf", "2.pdf", "3.pdf", "4.png", "5.docx"]
all_data = {}

for f in files:
    if not os.path.exists(f):
        print(f"跳過 {f} (檔案不存在)")
        continue
    
    print(f"正在處理 {f}...")
    ext = os.path.splitext(f)[1].lower()
    
    if ext == ".pdf":
        all_data[f] = process_pdf(f)
    elif ext in [".png", ".jpg", ".jpeg"]:
        all_data[f] = process_image(f)
    elif ext == ".docx":
        all_data[f] = process_docx(f)
        
    # 儲存個別結果
    with open(f"{f}.txt", "w", encoding="utf-8") as out:
        out.write(all_data[f])

print("\n--- 所有檔案處理完成 ---")