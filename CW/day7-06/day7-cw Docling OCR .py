import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

def run_idp_rapidocr():
    # 1. 設定路徑 (確保能抓到 sample_table.pdf)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "sample_table.pdf")
    output_path = os.path.join(base_dir, "output_rapidocr.md")

    if not os.path.exists(pdf_path):
        print(f"❌ 找不到檔案: {pdf_path}")
        return

    print("🚀 啟動 Docling (RapidOCR) 處理流程...")

    # 2. 配置 OCR 選項 (修正後的寫法)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    
    # 建立 RapidOcrOptions 並指定引擎設定
    # 這裡解決了 ValueError: "OcrAutoOptions" has no field "use_gpu" 的問題
    ocr_options = RapidOcrOptions() 
    pipeline_options.ocr_options = ocr_options 

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 3. 執行轉換與輸出
    try:
        result = converter.convert(pdf_path)
        md_output = result.document.export_to_markdown()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_output)
        
        print(f"✅ IDP 流程完成！結果已儲存至: {output_path}")
    except Exception as e:
        print(f"💥 轉換過程中發生錯誤: {e}")

if __name__ == "__main__":
    run_idp_rapidocr()