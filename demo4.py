import asyncio
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ============= 流式输出 =====================

model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0,
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# invoke 是同步调用
response = model.invoke("用一句话解释什么是 LangChain。")
print(response.content)


# ✅ 异步部分包进 async 函数
async def stream_demo():
    async for chunk in model.astream("用三句话介绍 LangChain"):
        if chunk.content:
            # print(chunk.content, end="", flush=True)
            print(chunk.content.replace("\n", ""), end="", flush=True)

# ✅ 在同步环境里启动事件循环来跑它
asyncio.run(stream_demo())