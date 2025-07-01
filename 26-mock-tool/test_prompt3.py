from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(
    model="mistral:7b-instruct-v0.3-q5_0",
    temperature=0,  # 降低随机性，提高一致性
)

# 测试1: 更详细的推理提示
detailed_prompt = """[INST]Du hast Zugang zu zwei Werkzeugen:

1. VectorSearch: Eine semantische Suchmaschine - gut für konzeptuelle Fragen und Definitionen
2. AskQA: Ein externes QA-System - gut für spezifische Fakten und direkte Antworten

Frage: "Was ist künstliche Intelligenz?"

Denke Schritt für Schritt:
- Dies ist eine konzeptuelle/definitorische Frage
- VectorSearch ist besser für semantische/konzeptuelle Suchen
- AskQA ist besser für spezifische Fakten

Antwort (nur Werkzeugname):[/INST]"""

print("测试1 - 详细推理提示:")
response1 = llm.invoke(detailed_prompt)
print(f"Antwort: {response1.content.strip()}")
print()

# 测试2: 多次运行检验一致性
print("测试2 - 一致性检验 (5次运行):")
simple_prompt = """[INST]Werkzeuge:
1. VectorSearch: semantische Suche
2. AskQA: QA-System

Frage: "Was ist künstliche Intelligenz?"
Welches Werkzeug? (nur Name)[/INST]"""

results = []
for i in range(5):
    response = llm.invoke(simple_prompt)
    answer = response.content.strip()
    results.append(answer)
    print(f"Lauf {i+1}: {answer}")

print(f"\nErgebnisse: {results}")
print(f"Eindeutig: {'Ja' if len(set(results)) == 1 else 'Nein'}")
print()

# 测试3: 对比不同问题类型
test_questions = [
    "Was ist künstliche Intelligenz?",  # 概念性
    "Wer hat Python erfunden?",         # 具体事实
    "Wie funktioniert maschinelles Lernen?",  # 解释性
    "Was ist 2+2?"                      # 简单事实
]

print("测试3 - 不同问题类型:")
for question in test_questions:
    prompt = f"""[INST]Werkzeuge:
1. VectorSearch: semantische Suche für Konzepte
2. AskQA: direkte Antworten für Fakten

Frage: "{question}"
Werkzeug:[/INST]"""
    
    response = llm.invoke(prompt)
    print(f"'{question}' -> {response.content.strip()}")