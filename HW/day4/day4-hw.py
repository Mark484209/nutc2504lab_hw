import os
import re
import base64
import requests
import json
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from playwright.sync_api import sync_playwright

# --- 1. 初始化 LLM (設定超時與重試防止 SSL 崩潰) ---
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="", 
    model="google/gemma-3-27b-it",
    temperature=0,
    timeout=45,
    max_retries=2
)

SEARXNG_URL = "https://puli-8080.huannago.com/search"

# --- 2. 狀態定義 ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    knowledge_base: str
    is_hit: bool
    loop_count: int
    target_url: str

# --- 3. 核心工具函式 ---

def search_searxng(query: str):
    """搜尋工具：獲取最相關網址"""
    params = {"q": query, "format": "json", "language": "zh-TW"}
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        results = response.json().get('results', [])
        return results[0].get('url') if results else None
    except:
        return None

def vlm_read_website(url: str) -> str:
    """視覺工具：Playwright 截圖 + VLM 分析"""
    print(f"📸 [VLM] 啟動視覺閱讀: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1280, 'height': 1200})
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(2000)
            img_b64 = base64.b64encode(page.screenshot()).decode('utf-8')
            browser.close()

        msg = [
            {"type": "text", "text": "請根據網頁截圖摘要核心內容，包含數據、日期與事實。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
        res = llm.invoke([HumanMessage(content=msg)])
        return re.sub(r'<.*?>', '', res.content).strip()
    except Exception as e:
        return f"視覺分析失敗: {e}"

# --- 4. LangGraph 節點實作 ---

def check_cache_node(state: AgentState):
    """優化方式：快取檢查 (調鬆判斷條件)"""
    query = state["messages"][-1].content.lower()
    # 只要包含 langchain 相關字眼就命中
    if any(kw in query for kw in ["langchain", "基礎", "概念"]):
        return {"is_hit": True, "knowledge_base": "快取命中：LangChain 是建立 LLM 應用程式的框架，支援 Chain 與 Agent 結構。", "loop_count": 0}
    return {"is_hit": False, "knowledge_base": "", "loop_count": 0}

def planner_node(state: AgentState):
    """決策節點"""
    kb = state.get("knowledge_base", "")
    if not kb: return {"messages": [AIMessage(content="NO")]} # 沒資料直接說 NO
    
    prompt = f"問題：{state['messages'][0].content}\n資料：{kb}\n資料是否足以回答？只需回 YES 或 NO。"
    res = llm.invoke([HumanMessage(content=prompt)])
    clean = re.sub(r'<.*?>', '', res.content).strip().upper()
    return {"messages": [AIMessage(content="YES" if "YES" in clean else "NO")]}

def query_gen_node(state: AgentState):
    """生成關鍵字"""
    user_q = state["messages"][0].content
    res = llm.invoke([HumanMessage(content=f"為此問題產出一個搜尋關鍵字：{user_q}")])
    kw = re.sub(r'<.*?>', '', res.content).strip()
    return {"messages": [AIMessage(content=f"關鍵字：{kw}")], "loop_count": state["loop_count"] + 1}

def search_tool_node(state: AgentState):
    """執行搜尋"""
    kw = state["messages"][-1].content.replace("關鍵字：", "")
    url = search_searxng(kw)
    return {"target_url": url}

def vlm_processing_node(state: AgentState):
    """VLM 處理節點"""
    url = state.get("target_url")
    if not url: return {"knowledge_base": "找不到相關網頁。"}
    result = vlm_read_website(url)
    return {"knowledge_base": result}

def final_answer_node(state: AgentState):
    """產出最終回答"""
    kb = state.get("knowledge_base", "")
    res = llm.invoke([HumanMessage(content=f"根據資料：{kb}\n回答問題：{state['messages'][0].content}")])
    return {"messages": [AIMessage(content=re.sub(r'<.*?>', '', res.content).strip())]}

# --- 5. 構建圖與路由 ---

workflow = StateGraph(AgentState)
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("vlm_processing", vlm_processing_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

# 路由判斷
workflow.add_conditional_edges("check_cache", lambda x: "hit" if x["is_hit"] else "miss", {"hit": "final_answer", "miss": "planner"})

def decision_router(state):
    if state["loop_count"] >= 2: return "y"
    return "y" if "YES" in state["messages"][-1].content.upper() else "n"

workflow.add_conditional_edges("planner", decision_router, {"y": "final_answer", "n": "query_gen"})

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "vlm_processing")
workflow.add_edge("vlm_processing", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()

# --- 6. 互動介面 ---
if __name__ == "__main__":
    print("\n--- 🤖 大作業：自動查證 AI 啟動 ---")
    while True:
        user_input = input("\n請輸入問題 (q 離開): ")
        if user_input.lower() == 'q': break
        
        inputs = {"messages": [HumanMessage(content=user_input)], "knowledge_base": "", "is_hit": False, "loop_count": 0, "target_url": ""}
        
        for event in app.stream(inputs):
            for node, data in event.items():
                print(f"📍 [節點]: {node}")
                if node == "final_answer":
                    print(f"\n📢 最終回答：\n{data['messages'][-1].content}")