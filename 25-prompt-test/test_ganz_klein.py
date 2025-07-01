from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="mistral:7b-instruct-v0.3-q5_0",
)
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

messages_with_system = [
    SystemMessage(content="Du bist ein hilfreicher Assistent, der bei der Auswahl von Werkzeugen hilft."),
    HumanMessage(content="Hier sind zwei Werkzeuge verfügbar:\n"
                        "1. VectorSearch: Eine semantische Suchmaschine\n"
                        "2. AskQA: Ein externes QA-System\n"
                        "Wenn du gefragt wirst 'Was ist künstliche Intelligenz?', welches Werkzeug würdest du verwenden?\n"
                        "Bitte antworte nur mit dem Namen des Werkzeugs oder 'kein Werkzeug'.")
]

response3 = llm.invoke(messages_with_system)
print("\n" + "="*50 + "\n")
print("使用系统消息 + 用户消息:")
print(response3)