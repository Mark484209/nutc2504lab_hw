from markitdown import MarkItDown
import os

def run():
    pdf_path = "example.pdf"
    output_path = "output_markitdown.md"
    
    if not os.path.exists(pdf_path):
        print(f"找不到檔案: {pdf_path}")
        return

    print(f"🚀 正在執行 Markitdown 轉換...")
    mid = MarkItDown()
    result = mid.convert(pdf_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f"✅ 完成！輸出至: {output_path}")

if __name__ == "__main__":
    run()