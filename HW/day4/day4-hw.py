import os
import json
import base64
import requests
from typing import List, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from playwright.sync_api import sync_playwright

# --- 1. 設定區域 ---
SEARXNG_URL = "https://puli-8080.huannago.com/search"

# 建議加上 max_retries 與 timeout 以應對之前遇到的 524 超時問題
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="your_api_key_here", 
    model="google/gemma-3-27b-it",
    temperature=0,
    max_retries=2,
    timeout=120
)

# --- 2. 狀態定義 ---
class AgentState(TypedDict):
    question: str
    keywords: str
    knowledge_base: str
    cache_hit: bool
    final_answer: str
    count: int 
    feedback: str

# --- 3. 核心工具函數 ---
def search_searxng(query: str, limit: int = 2):
    params = {"q": query, "format": "json", "language": "zh-TW"}
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        return [r for r in response.json().get('results', []) if 'url' in r][:limit]
    except Exception as e:
        print(f"❌ 搜尋出錯: {e}")
        return []

def vlm_analyze_page(url: str, question: str):
    print(f"📸 [VLM] 啟動視覺閱讀: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 800})
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(2000)
            img_b64 = base64.b64encode(page.screenshot()).decode('utf-8')
            browser.close()
            
            msg = HumanMessage(content=[
                {"type": "text", "text": f"分析此截圖內容並針對問題 '{question}' 提供關鍵資訊。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ])
            return llm.invoke([msg]).content
    except Exception as e:
        return f"網頁閱讀失敗: {e}"

# --- 4. LangGraph 節點實作 ---

def check_cache(state: AgentState):
    print("\n[Node] 1. 檢查快取...")
    return {"cache_hit": False, "knowledge_base": "", "count": 0, "feedback": ""}

def query_gen(state: AgentState):
    new_count = state.get("count", 0) + 1
    fb = f"\n前次思考反饋：{state['feedback']}" if state['feedback'] else ""
    print(f"🔄 [Node] 2. 第 {new_count}/3 次搜尋 - 生成關鍵字...")
    
    prompt = f"問題：'{state['question']}'{fb}\n請產出一個精準的搜尋關鍵字（僅輸出字串內容）。"
    keyword = llm.invoke(prompt).content.strip().replace('"', '')
    return {"keywords": keyword, "count": new_count}

def search_tool(state: AgentState):
    print(f"🔍 [Node] 3. 執行檢索: {state['keywords']}")
    results = search_searxng(state['keywords'])
    info = ""
    for r in results:
        analysis = vlm_analyze_page(r['url'], state['question'])
        info += f"\n[來源: {r['title']}]\n{analysis}\n"
    return {"knowledge_base": state['knowledge_base'] + info}

# ⭐ 新增節點：資訊精煉 (Research Refiner)
def research_refiner(state: AgentState):
    print("🧹 [Node] 4. 資訊精煉 - 過濾雜訊...")
    if not state['knowledge_base']:
        return {"knowledge_base": "尚未取得有效資訊"}
    
    prompt = f"""
    你是一個資料處理專家。請根據問題整理目前的搜尋資訊。
    問題：{state['question']}
    
    原始資料：
    {state['knowledge_base']}
    
    請移除廣告、重複內容，將事實以條列式摘要整理。如果資訊衝突，請並列說明。
    """
    refined_info = llm.invoke(prompt).content
    return {"knowledge_base": refined_info}

def planner(state: AgentState):
    print(f"🧠 [Node] 5. Planner 評估中...")
    prompt = f"""
    評估現有資訊是否足以完整回答問題。
    問題：{state['question']}
    現有精煉資訊：{state['knowledge_base']}
    
    請以 JSON 格式回傳：
    {{
        "sufficient": "YES" 或 "NO",
        "feedback": "若為 NO，請說明還缺少什麼關鍵資訊？"
    }}
    """
    res = llm.invoke(prompt).content
    try:
        data = json.loads(res[res.find("{"):res.rfind("}")+1])
        decision = data.get("sufficient", "NO")
        feedback = data.get("feedback", "資訊仍不足")
    except:
        decision = "NO"
        feedback = "無法解析思考內容"

    # 將決策暫存在 final_answer 欄位供路徑判斷使用
    return {"feedback": feedback, "final_answer": decision}

def final_answer(state: AgentState):
    print("📢 [Node] 6. 生成最終報告...")
    prompt = f"請根據以下查證事實，為使用者寫一份專業、客觀的報告：\n{state['knowledge_base']}\n問題：{state['question']}"
    res = llm.invoke(prompt).content
    return {"final_answer": res}

# --- 5. 構建流程圖 ---
workflow = StateGraph(AgentState)

workflow.add_node("check_cache", check_cache)
workflow.add_node("query_gen", query_gen)
workflow.add_node("search_tool", search_tool)
workflow.add_node("research_refiner", research_refiner) # <-- 加入新節點
workflow.add_node("planner", planner)
workflow.add_node("final_answer", final_answer)

workflow.set_entry_point("check_cache")

# 設定路徑邏輯
workflow.add_conditional_edges(
    "check_cache",
    lambda x: "final_answer" if x["cache_hit"] else "query_gen",
    {"final_answer": "final_answer", "query_gen": "query_gen"}
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "research_refiner") # 檢索完先精煉
workflow.add_edge("research_refiner", "planner")    # 精煉後才給 Planner 評估

def route_logic(state: AgentState):
    # 如果 Planner 說夠了 (YES) 或者次數到了 (>=3) 就結案
    if state.get("count", 0) >= 3 or "YES" in state.get("final_answer", ""):
        return "final_answer"
    return "query_gen"

workflow.add_conditional_edges(
    "planner",
    route_logic,
    {"final_answer": "final_answer", "query_gen": "query_gen"}
)

workflow.add_edge("final_answer", END)
app = workflow.compile()

# --- 6. 輸出流程圖與執行 ---
print("\n" + "="*20 + " 系統架構圖 " + "="*20)
# 在程式執行前先列印 ASCII 流程圖
app.get_graph().print_ascii()
print("="*55 + "\n")



if __name__ == "__main__":
    q = input("請輸入查證問題：")
    # 開始串流執行
    for output in app.stream({"question": q, "knowledge_base": "", "cache_hit": False, "count": 0}):
        for node, data in output.items():
            # 當運行到 final_answer 節點完成時，輸出結果
            if node == "final_answer" and "final_answer" in data:
                # 確保我們拿到的是最終報告字串，而非 Planner 的 YES/NO
                if len(data["final_answer"]) > 5: 
                    print("\n" + "✨"*10 + " 查證報告 " + "✨"*10)
                    print(data["final_answer"])
                    