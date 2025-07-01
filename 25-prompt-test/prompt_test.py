from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="mistral:7b-instruct-v0.3-q5_0",
    temperature=0.0,
    num_predict=512
)

system_message = (
    "Du bist ein hilfreicher deutschsprachiger Assistent. "
    "Verwende nur die folgenden Werkzeuge, wenn es nötig ist:\n"
    "- VectorSearch: semantische Suche in unserer Vektor-Datenbank\n"
    "- AskQA: präzise Beantwortung durch unser externes QA-System\n"
)

def search_vector_db(query: str) -> str:
    return f"SS-Ergebnis: „{query}“"

def ask_external_qa(query: str) -> str:
    return f"QA-Ergebnis: „{query}“"

tools = [
    Tool(
        name="VectorSearch",
        func=search_vector_db,
        description=(
            "Führe eine semantische Suche in der Vektor-Datenbank durch. "
            "Eingabe: eine natürliche Sprachabfrage; Ausgabe: die besten Treffer."
        )
    ),
    Tool(
        name="AskQA",
        func=ask_external_qa,
        description=(
            "Stelle eine Frage an das externe QA-System. "
            "Eingabe: Frage; Ausgabe: beste Antwort."
        )
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_message}
)

test_cases = [
    {
        "name": "Semantische Suche auslösen",
        "prompt": "Bitte nutze VectorSearch, um Informationen über ‚künstliche Intelligenz‘ zu finden und antworte in Deutsch.",
        "check_func": lambda out: "SS-Ergebnis:" in out
    },
    {
        "name": "AskQA auslösen",
        "prompt": "Ich habe eine Frage: ‚Was ist ein Transformer-Modell?‘ Bitte rufe AskQA auf und fasse die Antwort auf Deutsch zusammen.",
        "check_func": lambda out: "QA-Ergebnis:" in out
    },
    {
        "name": "Keine Tool-Nutzung",
        "prompt": "Sag mir auf Deutsch, welches Datum heute ist.",
        "check_func": lambda out: "heute" in out or "Datum" in out
    }
]

results = []
for case in test_cases:
    print(f"\n=== {case['name']} ===")
    output = agent.run(case["prompt"])
    success = case["check_func"](output)
    results.append(success)
    status = "✅ Erfolg" if success else "❌ Misserfolg"
    print(f"Prompt: {case['prompt']}")
    print(f"Output: {output}")
    print(f"Test {status}")

total = len(results)
passed = sum(results)
print(f"\nTest-Erfolgsrate: {passed} / {total} = {passed/total:.0%}")
