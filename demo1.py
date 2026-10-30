import os

from langchain.agents import create_agent
from openai import OpenAI

# ==================== 利用Python实现LLM最小调用 =====================

# 创建一个客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


response = client.chat.completions.create(
    model="qwen3.8-flash",
    messages=[
        {"role": "system", "content": "你是一个全能小助手"},
        {"role": "user", "content": "请介绍一下Langchain框架"},
    ],
)

print("test")
print(response)
print("单词模型调用用量信息：" + response.usage)
print("调用的模型为： " + response.model)