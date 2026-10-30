import os

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0,
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义 embedding模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 定义向量数据库, 并且指定embedding模型
store = InMemoryVectorStore(embedding=embeddings)

# 定义文档
docs = [
    Document(
        page_content="成都住宿标准：员工住宿费上限为每晚 800 元。",
        metadata={"source": "travel-policy.md", "section": "住宿", "city": "成都"},
    ),
    Document(
        page_content="出差报销应在行程结束后 30 天内提交。",
        metadata={"source": "travel-policy.md", "section": "报销期限"},
    ),
]
# 把文档添加到向量数据库里面
store.add_documents(docs)


# 用户问题
question = "我在成都出差，住宿费每晚最多能报多少？"
matches = store.similarity_search(question, k=3)
context = "\n\n".join(
    f"来源：{doc.metadata.get('source')}；内容：{doc.page_content}"
    for doc in matches
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "只根据给定资料回答。资料没有答案时明确说不知道，并引用资料来源。"),
    ("human", "资料：\n{context}\n\n问题：{question}"),
])

answer = model.invoke(qa_prompt.invoke({"context": context, "question": question}))
print(answer.content)
