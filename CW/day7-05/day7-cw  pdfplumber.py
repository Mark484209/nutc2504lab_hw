import pdfplumber
import os

def run():
    pdf_path = "example.pdf"
    output_path = "output_plumber.md"
    
    if not os.path.exists(pdf_path):
        print(f"找不到檔案: {pdf_path}")
        return

    print(f"🚀 正在執行 pdfplumber 提取...")
    with pdfplumber.open(pdf_path) as pdf:
        content = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                content.append(f"## Page {i+1}\n\n{text}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(content))
    print(f"✅ 完成！輸出至: {output_path}")

if __name__ == "__main__":
    run()