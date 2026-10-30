import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0,
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# invoke 是同步调用
response = model.invoke("用一句话解释什么是 LangChain。")
print(response.content)