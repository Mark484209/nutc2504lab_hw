import random
import json
import os
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

# ================= 配置區 =================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",  #
    api_key="",                        # 請填入你的 API KEY
    model="google/gemma-3-27b-it",            #
    temperature=0
)

# 1. 定義工具 (模擬 50% 失敗率)
@tool
def get_weather(city: str):
    """查詢指定城市的天氣。"""
    # 故意模擬出錯
    if random.random() < 0.5:
        return "系統錯誤：天氣資料庫連線失敗，請再試一次。"
    
    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    elif "台中" in city:
        return "台中晴天，氣溫 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

# 2. 定義狀態與節點
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot_node(state: AgentState):
    """思考節點"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node_executor = ToolNode(tools)

def fallback_node(state: AgentState):
    """備援節點：當重試次數過多時執行"""
    last_message = state["messages"][-1]
    tool_call_id = last_message.tool_calls[0]["id"]
    
    error_message = ToolMessage(
        content="系統提示：已達到最大重試次數 (Max Retries Reached)。請停止嘗試，並告知使用者服務暫時無法使用。",
        tool_call_id=tool_call_id
    )
    return {"messages": [error_message]}

# 3. 路由邏輯 (關鍵：判斷是否重試)
def router(state: AgentState) -> Literal["tools", "fallback", "end"]:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"

    # 計算歷史紀錄中的連續錯誤次數
    retry_count = 0
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            if "系統錯誤" in msg.content:
                retry_count += 1
            else:
                break
        elif isinstance(msg, HumanMessage):
            break
    
    print(f"DEBUG: 目前連續重試次數: {retry_count}")
    
    if retry_count >= 3: # 設定上限為 3 次
        return "fallback"
    
    return "tools"

# 4. 建構 LangGraph 工作流
workflow = StateGraph(AgentState)

workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("agent")

# 設定條件分支
workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "fallback": "fallback",
        "end": END
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("fallback", "agent")

app = workflow.compile()

# 5. 執行對話
if __name__ == "__main__":
    print("--- 天氣機器人已啟動 (具備重試機制) ---")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]: break

        # 使用 stream 模式查看執行過程
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if msg.tool_calls:
                        print(f" -> [Agent]: 決定呼叫工具 (判斷中...)")
                    else:
                        print(f" -> [Agent]: {msg.content}")
                elif key == "tools":
                    # 檢查工具執行結果是否包含錯誤字眼
                    tool_res = value["messages"][-1].content
                    if "系統錯誤" in tool_res:
                        print(f" -> [Tools]: 🔴 系統故障，準備重試...")
                    else:
                        print(f" -> [Tools]: ✅ 成功取得資料")
                elif key == "fallback":
                    print(f" -> [Fallback]: ⚠️ 觸發熔斷，停止重試")