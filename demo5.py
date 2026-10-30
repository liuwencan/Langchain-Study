import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# =============== 工具调用 ===================
@tool
def lookup_policy(employee_level: str, city: str) -> str:
    """查询员工在指定城市的差旅住宿限额。"""
    # 实际项目中这里调用数据库或规则服务
    return f"{employee_level} 在 {city} 的住宿限额：800 元/晚"


model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0,
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

agent = create_agent(
    model=model,
    tools=[lookup_policy], # 传入工具
    system_prompt="根据差旅政策回答问题。需要查询限额时使用工具。",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我在成都出差，住宿限额是多少？我的级别是P7, 我想查询的是在北京"}]
})
print(result["messages"][-1].content)