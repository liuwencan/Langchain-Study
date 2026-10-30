import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0,
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气；当前返回演示数据。"""
    return f"{city}：晴，25℃（演示数据）"


agent = create_agent(model=model, tools=[get_weather], system_prompt="你是天气助手，查询天气时必须使用工具，并注明演示数据。")
result = agent.invoke({"messages": [{"role": "user", "content": "杭州天气怎么样？"}]})
print(result["messages"][-1].content)