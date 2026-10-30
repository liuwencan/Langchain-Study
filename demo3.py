import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ============== 结构化输出 ==================
# 注意需要导入的是 Pydantic 模型
# 定义结构化输出格式, 有几个字段, 每个字段什么类型, 对应字段的中文含义
class Expense(BaseModel):
    category: str = Field(description="费用类别")
    amount: float = Field(description="金额")
    confidence: float = Field(ge=0, le=1)

model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0,
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 使用model提供的结构化输出, 传入定义好的结构
extractor = model.with_structured_output(Expense)
expense = extractor.invoke("住宿费用 680 元")
print(expense.category, expense.amount, expense.confidence)