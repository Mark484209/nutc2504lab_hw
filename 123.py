import time
import requests
import operator
from pathlib import Path
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# ==========================================
# 1. ASR 語音辨識部分 (取得 20 秒音檔的完整內容)
# ==========================================
BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
WAV_PATH = "/home/pc-49/Downloads/Podcast_EP14_30s.wav" 
auth = ("nutc2504", "nutc2504")

def get_asr_results():
    print("正在上傳音檔進行辨識...")
    with open(WAV_PATH, "rb") as f:
        r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
    r.raise_for_status()
    task_id = r.json()["id"]
    
    txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT"
    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    def wait_download(url: str):
        for _ in range(600):
            try:
                resp = requests.get(url, timeout=(5, 60), auth=auth)
                if resp.status_code == 200: return resp.text
            except: pass
            time.sleep(2)
        return None

    print(f"等待轉錄完成 (Task ID: {task_id})...")
    return wait_download(srt_url), wait_download(txt_url)

# ==========================================
# 2. LangGraph 設定與定義
# ==========================================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="", 
    model="google/gemma-3-27b-it",
    temperature=0
)

def merge_dict(left: dict, right: dict) -> dict:
    new_dict = left.copy()
    new_dict.update(right)
    return new_dict

class GraphState(TypedDict):
    srt_content: str
    txt_content: str
    results: Annotated[dict, merge_dict]

def asr_node(state: GraphState):
    return {"results": {"status": "Processing"}}

def minutes_taker_node(state: GraphState):
    prompt = f"請將以下 SRT 內容轉為 Markdown 表格 (時間|發言內容):\n\n{state['srt_content']}"
    res = llm.invoke(prompt)
    return {"results": {"minutes": res.content}}

def summarizer_node(state: GraphState):
    prompt = f"請摘要以下內容 (包含決策與待辦事項):\n\n{state['txt_content']}"
    res = llm.invoke(prompt)
    return {"results": {"summary": res.content}}

def writer_node(state: GraphState):
    summary = state["results"].get("summary", "")
    minutes = state["results"].get("minutes", "")
    # 依照圖片 40 格式組合
    report = f"# 📑 智慧會議紀錄報告\n\n## 🎯 重點摘要 (Executive Summary)\n{summary}\n\n---\n## 📝 詳細逐字稿 (Detailed Minutes)\n{minutes}"
    return {"results": {"final_report": report}}

# 建立圖結構
workflow = StateGraph(GraphState)
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)
workflow.set_entry_point("asr")
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)
app = workflow.compile()

# ==========================================
# 3. 執行流程與存檔
# ==========================================
srt_data, txt_data = get_asr_results() # 這裡會取得完整 20 秒內容

if srt_data and txt_data:
    print("--- 智慧會議助理開始分析 ---")
    inputs = {"srt_content": srt_data, "txt_content": txt_data}
    final_output = app.invoke(inputs)
    
    report_content = final_output["results"]["final_report"]
    
    # 印出結果
    print(report_content)
    
    # 存檔 (解決「檔案沒有出來」的問題)
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "meeting_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\n✅ 報告已儲存至: {report_path}")
else:
    print("ASR 轉錄失敗。")