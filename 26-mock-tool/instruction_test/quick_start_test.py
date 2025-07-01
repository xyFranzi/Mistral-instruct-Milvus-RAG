# 运行这个简化版本快速开始

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def quick_test():
    llm = ChatOllama(
        model="mistral:7b-instruct-v0.3-q5_0",
        temperature=0,
    )
    
    # 快速测试 - 只测试几个关键案例
    system_prompt = """Du hast zwei Werkzeuge:
1. SimilaritySearch: für konzeptuelle Fragen über AI/ML
2. QA: für spezifische Anweisungen und Fakten

Antworte nur mit "SimilaritySearch" oder "QA"."""

    test_cases = [
        ("Was ist künstliche Intelligenz?", "SimilaritySearch"),
        ("Wie installiere ich Python?", "QA"),
        ("Was ist Deep Learning?", "SimilaritySearch"),
        ("Wo finde ich die API-Dokumentation?", "QA"),
    ]
    
    print("快速测试结果:")
    print("-" * 40)
    
    correct = 0
    for question, expected in test_cases:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        response = llm.invoke(messages)
        actual = response.content.strip().lstrip()
        
        is_correct = actual == expected
        if is_correct:
            correct += 1
            
        status = "✅" if is_correct else "❌"
        print(f"{status} {question}")
        print(f"   期望: {expected}")
        print(f"   实际: '{actual}'")
        print(f"   Token: {response.usage_metadata.get('total_tokens', 'N/A')}")
        print()
    
    accuracy = correct / len(test_cases)
    print(f"准确率: {accuracy:.2%} ({correct}/{len(test_cases)})")
    
    if accuracy >= 0.75:
        print("🎉 初步测试通过！可以继续深入测试")
    else:
        print("⚠️  需要调整prompt或考虑其他模型")

if __name__ == "__main__":
    quick_test()