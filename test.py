import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化模型 (對應 ch4-1 實作)
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="YOUR_API_KEY", # 記得填入你的 API Key
    model="google/gemma-3-27b-it",
    temperature=0
)

# 2. 定義 Tool (對應 Function Calling 原理中的「好的定義」)
@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }

# 3. 綁定工具
llm_with_tools = llm.bind_tools([extract_order_data])

# 4. 定義後處理函數 (對應 ch4-2 的改進版邏輯)
def extract_tool_args(ai_message):
    # 如果模型決定調用工具，回傳工具參數 (Dict)
    if ai_message.tool_calls:
        return ai_message.tool_calls[0]['args']
    # 如果沒有調用工具，直接回傳模型的文字回覆
    else:
        return ai_message.content

# 5. 建立 Prompt 與 Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("user", "{user_input}")
])

# 組合 Chain: 提示詞 -> 模型 -> 參數提取
chain = prompt | llm_with_tools | extract_tool_args

# 6. 啟動對話迴圈
print("--- 訂單機器人已啟動 (輸入 exit 或 q 退出) ---")
while True:
    user_input = input("User: ")
    
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    # 執行並獲取結果
    result = chain.invoke({"user_input": user_input})

    # 輸出結果 (確保中文正常顯示)
    print(json.dumps(result, ensure_ascii=False, indent=2))