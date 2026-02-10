import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# 假設你使用 OpenAI 作為評分員，需設定 API Key
os.environ["OPENAI_API_KEY"] = "你的_API_KEY"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_real_ai_answer(question):
    """
    這裡應接上你真正的 RAG 檢索邏輯。
    目前先回傳模擬內容，但評分會由 Ragas 真正執行。
    """
    # 模擬檢索到的參考內容 (Context)
    retrieved_contexts = ["這是從資料庫檢索出來的原始段落內容..."]
    # 模擬 LLM 生成的答案
    generated_answer = f"根據檢索內容，這題的回答是..."
    
    return generated_answer, retrieved_contexts

def main():
    input_file = os.path.join(BASE_DIR, "day6_HW_questions.csv")
    output_file = os.path.join(BASE_DIR, "questions_evaluated.csv")

    if not os.path.exists(input_file):
        print("❌ 找不到原始檔案")
        return

    df = pd.read_csv(input_file)
    df.columns = [c.lower().strip() for c in df.columns]

    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [] # 如果你有標準答案的話
    }

    # 1. 跑 RAG 流程獲取答案與檢索內容
    for _, row in df.iterrows():
        q_text = row['questions']
        ans, ctx = get_real_ai_answer(q_text)
        
        data_samples["question"].append(q_text)
        data_samples["answer"].append(ans)
        data_samples["contexts"].append(ctx)
        # 如果 csv 裡本來就有正確答案，請填入；若無，這欄會影響 Recall 計算
        data_samples["ground_truth"].append(row.get('ground_truth', "預設標準答案"))

    # 2. 轉換為 Ragas 所需的 Dataset 格式
    dataset = Dataset.from_dict(data_samples)

    # 3. 呼叫 AI 進行真正的評分
    print("🚀 正在調用 LLM 進行指標計算 (這可能需要一點時間並消耗 Token)...")
    score_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
    )

    # 4. 整理結果並存檔
    final_df = score_result.to_pandas()
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✅ 評估完成！自動計算的分數已存至: {output_file}")

if __name__ == "__main__":
    main()