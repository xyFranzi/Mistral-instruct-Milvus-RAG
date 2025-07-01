from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="mistral:7b-instruct-v0.3-q5_0")

def search_vector_db(query: str) -> str:
    return f"SS result: “{query}”"

def ask_external_qa(query: str) -> str:
    return f"QA result: “{query}”"

tools = [
    Tool(
        name="VectorSearch",
        func=search_vector_db,
        description=""
    ),
    Tool(
        name="AskQA",
        func=ask_external_qa,
        description="QA"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

print("\n test 1 SS")
response_1 = agent.run("question11111")
print("model final answer", response_1)

# print("\n test 2 QA")
# response_2 = agent.run("question22222")
# print("model final answer", response_2)