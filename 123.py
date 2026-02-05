import os
import re
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages

# --- 1. 初始化與防崩潰設定 ---
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="", # 照教材留空
    model="google/gemma-3-27b-it",
    temperature=0,
    timeout=20,       # 連線超過 20 秒自動斷開
    max_retries=2     # 失敗自動重試
)

# --- 2. 狀態定義 ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    knowledge_base: str
    is_hit: bool
    loop_count: int

# --- 3. 節點功能 ---

def check_cache_node(state: AgentState):
    """檢查問題是否命中快取"""
    query = state["messages"][-1].content.lower()
    # 只要包含 langchain 或 基礎概念 就直接出答案
    if "langchain" in query or "基礎" in query:
        return {"is_hit": True, "knowledge_base": "快取資料：LangChain 是一個旨在簡化 LLM 應用開發的框架。", "loop_count": 0}
    return {"is_hit": False, "knowledge_base": "", "loop_count": 0}

def planner_node(state: AgentState):
    """決策中心：判斷資料夠不夠"""
    kb = state.get("knowledge_base", "")
    prompt = f"問題：{state['messages'][0].content}\n資料：{kb}\n資料是否足夠回答？只需回 YES 或 NO。"
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        # 清理模型可能噴出的標籤如 <|im_end|>
        clean = re.sub(r'<.*?>', '', res.content).strip().upper()
    except:
        clean = "YES" # 斷線時強制結束搜尋
    return {"messages": [AIMessage(content=clean)]}

def query_gen_node(state: AgentState):
    """生成關鍵字節點"""
    return {"messages": [AIMessage(content="系統正在搜尋更多資訊...")], "loop_count": state["loop_count"] + 1}

def search_tool_node(state: AgentState):
    """搜尋工具節點"""
    return {"knowledge_base": "搜尋結果：LangGraph 是 LangChain 的進階擴展，專門處理有循環邏輯的多代理人工作流。"}

def final_answer_node(state: AgentState):
    """生成最終答案節點"""
    kb = state.get("knowledge_base", "目前查不到更多資訊。")
    prompt = f"根據資料回答問題：{kb}"
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        final = re.sub(r'<.*?>', '', res.content).strip()
    except:
        final = f"連線異常，根據現有資料回覆：{kb}"
    return {"messages": [AIMessage(content=final)]}

# --- 4. 路由與工作流構建 ---

def cache_router(state: AgentState):
    return "hit" if state["is_hit"] else "miss"

def decision_router(state: AgentState):
    if state["loop_count"] >= 2: # 最多搜兩次，防止死迴圈
        return "sufficient"
    last_msg = state["messages"][-1].content.upper()
    return "sufficient" if "YES" in last_msg else "insufficient"

workflow = StateGraph(AgentState)
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")
workflow.add_conditional_edges("check_cache", cache_router, {"hit": "final_answer", "miss": "planner"})
workflow.add_conditional_edges("planner", decision_router, {"sufficient": "final_answer", "insufficient": "query_gen"})
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()

# --- 5. 互動式介面 ---
if __name__ == "__main__":
    print("\n--- 🤖 自動查證 AI 啟動 (輸入 q 結束) ---")
    while True:
        user_input = input("\n請輸入你的問題: ")
        if user_input.lower() == 'q': break
        
        init_state = {"messages": [HumanMessage(content=user_input)], "knowledge_base": "", "is_hit": False, "loop_count": 0}
        
        for event in app.stream(init_state):
            for node, data in event.items():
                print(f"📍 節點: [{node}]")
                if "messages" in data:
                    print(f"   內容: {data['messages'][-1].content}")