from docling.document_converter import DocumentConverter
import os

def run():
    pdf_path = "example.pdf"
    output_path = "output_docling.md"
    
    if not os.path.exists(pdf_path):
        print(f"找不到檔案: {pdf_path}")
        return

    print(f"🚀 正在執行 Docling 轉換...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md_output = result.document.export_to_markdown()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_output)
    print(f"✅ 完成！輸出至: {output_path}")

if __name__ == "__main__":
    run()