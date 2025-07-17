import re
import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from db_ss import MilvusSimilaritySearchTool  # Schritt 2: dein Tool

# --- 1) LLM und Tool initialisieren ---
llm = ChatOllama(model="mistral-small:24b", temperature=0)
similarity_tool = MilvusSimilaritySearchTool(
    collection_name="german_docs",
    db_path=os.path.expanduser("~/mypython/Mistral-instruct-Milvus-RAG/milvus_german_docs.db"),
    embedding_model="aari1995/German_Semantic_V3b",
    embedding_device="cpu",
    k=3
)

# --- 2) Prompt-Vorlagen auf Deutsch (System-Prompt bleibt immer gleich) ---
system_message = SystemMessage(
    content=(
        "Du bist ein intelligenter Assistent. "
        "Wenn du ein externes Tool aufrufen möchtest, antworte **nur** mit einer "
        "gültigen JSON-Struktur, z.B.:\n"
        "{\n"
        '  "name": "<tool_name>",\n'
        '  "arguments": { /* Parameter als JSON-Objekt */ }\n'
        "}\n"
        "Schreibe keine anderen Texte oder Erklärungen in dieser Ausgabe."
    )
)

def run_tool_calling_for_query(query: str) -> str:
    """
    Führt den Tool-Calling-Loop für eine einzelne User-Frage (query) synchron aus.
    """
    messages = [system_message, HumanMessage(content=query)]
    final_answer = None

    while True:
        # 1) LLM synchron aufrufen
        assistant_msg = llm.invoke(messages)
        messages.append(assistant_msg)

        text = (assistant_msg.content or "").strip()
        # 2) Prüfen, ob die komplette Antwort ein JSON-Tool-Call ist
        match = re.fullmatch(r"\s*(\{[\s\S]*\})\s*", text)
        if not match:
            # kein Tool-Call → das ist die Endantwort
            final_answer = text
            break

        # 3) JSON parsen
        try:
            call = json.loads(match.group(1))
            tool_name = call["name"]
            args = call["arguments"]
        except (json.JSONDecodeError, KeyError) as e:
            final_answer = f"Fehler beim Parsen des Tool-Calls: {e}"
            break

        # 4) Tool ausführen
        if tool_name == similarity_tool.name:
            # wir erwarten args["query"]
            tool_output = similarity_tool._run(args.get("query", ""))
        else:
            tool_output = json.dumps(
                {"error": f"Unbekanntes Tool: {tool_name}"}, ensure_ascii=False
            )

        # 5) Ergebnis zurück in die Konversation
        messages.append(
            HumanMessage(
                content=json.dumps(
                    {"tool": tool_name, "output": json.loads(tool_output)},
                    ensure_ascii=False
                )
            )
        )
        # Schleife für nächsten Tool-Call

    return final_answer

if __name__ == "__main__":
    # --- 3 Abfragen gleichzeitig testen ---
    user_queries = [
        "Wer ist Raymond?",
        "Wer ist die Hauptfigur dieses Romans?",
        "Welche Schutzmaßnahmen gibt es gegen Klimawandel?"
    ]

    results = {}
    for q in user_queries:
        answer = run_tool_calling_for_query(q)
        results[q] = answer

    # Ausgabe formatieren
    for q, ans in results.items():
        print(f"\nFrage: {q}\nAntwort: {ans}\n{'-'*40}")
