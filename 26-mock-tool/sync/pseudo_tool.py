from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List, Dict, Any
import json
import time
import random

# Milvus SS tool
class MockSimilaritySearchTool(BaseTool):
    name: str = "similarity_search"
    description: str = """Semantic similarity search tool. Use this to find relevant document chunks 
    based on semantic similarity. Good for conceptual questions, explanations, and general knowledge queries.
    Input should be the user's question as a string."""
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """imitate SS"""
        # fake documents for testing
        mock_documents = [
            {
                "content": "Künstliche Intelligenz (KI) ist ein Bereich der Informatik, der sich mit der Entwicklung von Computersystemen beschäftigt, die Aufgaben ausführen können, die normalerweise menschliche Intelligenz erfordern.",
                "source": "ai_basics.pdf",
                "similarity": 0.95
            },
            {
                "content": "Machine Learning ist ein Teilbereich der künstlichen Intelligenz, bei dem Algorithmen aus Daten lernen und Vorhersagen treffen können, ohne explizit programmiert zu werden.",
                "source": "ml_introduction.pdf", 
                "similarity": 0.88
            },
            {
                "content": "Deep Learning verwendet künstliche neuronale Netze mit mehreren Schichten, um komplexe Muster in großen Datenmengen zu erkennen und zu lernen.",
                "source": "deep_learning_guide.pdf",
                "similarity": 0.82
            }
        ]
        
        # time.sleep(0.5)
        
        # choose relevant doc (fake! no embedding)
        query_lower = query.lower() 
        relevant_docs = [] 
        
        if "künstliche intelligenz" in query_lower or "ki" in query_lower or "artificial intelligence" in query_lower:
            relevant_docs.append(mock_documents[0])
        if "machine learning" in query_lower or "maschinelles lernen" in query_lower:
            relevant_docs.append(mock_documents[1])
        if "deep learning" in query_lower:
            relevant_docs.append(mock_documents[2])
            
        # if no specific match, return first two docs
        if not relevant_docs:
            relevant_docs = mock_documents[:2]
            
        result = {
            "tool": "similarity_search",
            "query": query,
            "found_documents": len(relevant_docs),
            "documents": relevant_docs
        }
        
        return json.dumps(result, ensure_ascii=False)

# QA tool
class MockQATool(BaseTool):
    name: str = "qa_search"
    description: str = """Question-Answer search tool. Use this to find specific answers to direct questions.
    Good for factual queries, how-to questions, and specific information lookup.
    Input should be the user's question as a string."""
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """imitate QA ?"""
        mock_qa_pairs = [
            {
                "question": "Was ist künstliche Intelligenz?",
                "answer": "KI ist die Simulation menschlicher Intelligenz in Maschinen, die programmiert sind, um zu denken und zu lernen.",
                "source": "faq_ai.pdf",
                "confidence": 0.92
            },
            {
                "question": "Wie installiere ich Python?", 
                "answer": "1. Besuchen Sie python.org 2. Laden Sie die neueste Version herunter 3. Führen Sie den Installer aus 4. Fügen Sie Python zum PATH hinzu",
                "source": "installation_guide.pdf",
                "confidence": 0.95
            },
            {
                "question": "Was sind die Systemanforderungen?",
                "answer": "Mindestens 4GB RAM, 10GB freier Speicherplatz, Python 3.8+, und eine moderne CPU.",
                "source": "system_requirements.pdf", 
                "confidence": 0.88
            }
        ]
        
        # time.sleep(0.3)
        
        # simulate query matching (also fake logic)
        query_lower = query.lower()
        matched_qa = []
        
        for qa in mock_qa_pairs:
            if any(word in query_lower for word in qa["question"].lower().split()):
                matched_qa.append(qa)
                
        # if no specific match, return general info
        if not matched_qa:
            matched_qa = [{
                "question": "General query",
                "answer": "Für spezifische Antworten konsultieren Sie bitte die Dokumentation.",
                "source": "general.pdf",
                "confidence": 0.5
            }]
            
        result = {
            "tool": "qa_search", 
            "query": query,
            "found_answers": len(matched_qa),
            "qa_pairs": matched_qa
        }
        
        return json.dumps(result, ensure_ascii=False)

class RAGToolIntegrationTester:
    def __init__(self, model_name: str = "mistral:7b-instruct-v0.3-q5_0"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0, # set to 0 for deterministic output
        )
        self.tools = [MockSimilaritySearchTool(), MockQATool()]
        
        # prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Du bist ein intelligenter RAG-Assistent. Du hast Zugang zu zwei Werkzeugen:

            1. similarity_search: Für semantische Suche in Dokumenten
            2. qa_search: Für spezifische Fragen und Antworten

            WICHTIG: Für die meisten Fragen solltest du BEIDE Werkzeuge verwenden, um umfassende Antworten zu liefern.
            - Verwende similarity_search für konzeptuelle und erklärende Inhalte
            - Verwende qa_search für spezifische Fakten und direkte Antworten

            Antworte basierend auf den Ergebnissen beider Werkzeuge und gib eine strukturierte, hilfreiche Antwort."""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
    def test_single_tool_usage(self):
        """test single tool usage without LLM"""
        print("=== Test 1: single tool usage ===")
        
        similarity_tool = MockSimilaritySearchTool()
        qa_tool = MockQATool()
        
        test_queries = [
            "Was ist künstliche Intelligenz?",
            "Wie installiere ich Python?",
            "Was ist Deep Learning?"
        ]
        
        for query in test_queries:
            print(f"\nQ: {query}")
            print("-" * 40)
            
            # test SS
            try:
                sim_result = similarity_tool._run(query)
                sim_data = json.loads(sim_result)
                print(f"✅ SimilaritySearch: found {sim_data['found_documents']} documents")
            except Exception as e:
                print(f"❌ SimilaritySearch error: {e}")
            
            # test QA
            try:
                qa_result = qa_tool._run(query)
                qa_data = json.loads(qa_result)
                print(f"✅ QA Search: found {qa_data['found_answers']} answers")
            except Exception as e:
                print(f"❌ QA Search error: {e}")
    
    def test_tool_calling_capability(self):
        """test LLM's tool calling capability"""
        print("\n=== Test 2: LLM tool usage ===")
        
        # 注意：mistral 7b 可能不支持原生工具调用，我们用函数调用格式测试
        test_cases = [
            {
                "question": "Was ist künstliche Intelligenz?",
                "expected_tools": ["similarity_search", "qa_search"],
                "reasoning": "was Frage, sollte beide SS-Werkzeug und QA-Werkzeug verwenden"
            },
            {
                "question": "Wie installiere ich Python?", 
                "expected_tools": ["qa_search"],
                "reasoning": "wie Frage, sollte QA-Werkzeug verwenden"
            }
        ]
        
        for test_case in test_cases:
            print(f"\nQuestion: {test_case['question']}")
            print(f"Expected tool: {test_case['expected_tools']}")
            print(f"Reasoning: {test_case['reasoning']}")
            
            function_prompt = f"""Du hast Zugang zu diesen Funktionen:

            1. similarity_search(query: str) - Semantische Suche in Dokumenten
            2. qa_search(query: str) - Spezifische QA-Suche

            Frage: "{test_case['question']}"

            Welche Funktionen würdest du aufrufen? Antworte im Format:
            Funktionen: [function_name1, function_name2, ...]
            Begründung: Warum diese Funktionen?"""

            response = self.llm.invoke(function_prompt)
            print(f"LLM Antwort: {response.content}")
            
            # check if expected tools are mentioned
            mentioned_tools = []
            for tool in test_case['expected_tools']:
                if tool in response.content.lower():
                    mentioned_tools.append(tool)
            
            coverage = len(mentioned_tools) / len(test_case['expected_tools'])
            print(f"Tool coverage {coverage:.1%} ({len(mentioned_tools)}/{len(test_case['expected_tools'])})")
            
    def test_weighted_approach(self):
        print("\n=== Test 3: tool calling with weighted approach ===")
        
        test_questions = [
            "Was ist künstliche Intelligenz?",
            "Wie löse ich einen Python ImportError?", 
            "Erklären Sie den Unterschied zwischen AI und ML"
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            print("-" * 50)
            
            start_time = time.time()
            
            tool_selection_prompt = f"""Du bist ein intelligenter RAG-Assistent mit Zugang zu zwei Werkzeugen:

                1. similarity_search: Für semantische Suche und konzeptuelle Inhalte
                2. qa_search: Für spezifische Antworten und Fakten

                Frage: "{question}"

                Analysiere die Frage und bestimme:
                1. Welche Werkzeuge du verwenden solltest (beide empfohlen für beste Ergebnisse)
                2. Welche Gewichtung die Ergebnisse haben sollten

                Antworte im folgenden JSON-Format:
                {{
                    "tools_to_use": ["similarity_search", "qa_search"],
                    "similarity_weight": 0.6,
                    "qa_weight": 0.4,
                    "reasoning": "Begründung für die Auswahl"
                }}"""

            # LLM entscheidet über Werkzeugauswahl
            try:
                selection_response = self.llm.invoke(tool_selection_prompt)
                print(f"LLM Werkzeugauswahl: {selection_response.content[:200]}...")
                
                # Versuche JSON zu extrahieren (vereinfacht)
                content = selection_response.content
                if "similarity_search" in content.lower():
                    use_similarity = True
                else:
                    use_similarity = False
                    
                if "qa_search" in content.lower():
                    use_qa = True
                else:
                    use_qa = False
                    
                # Gewichte aus dem Text extrahieren (vereinfacht)
                ss_weight = 0.6 if use_similarity else 0.0
                qa_weight = 0.4 if use_qa else 0.0
                
                # Normalisierung falls nur ein Tool verwendet wird
                if use_similarity and not use_qa:
                    ss_weight = 1.0
                elif use_qa and not use_similarity:
                    qa_weight = 1.0
                
                print(f"📊 Bestimmte Gewichte: Similarity={ss_weight}, QA={qa_weight}")
                
            except Exception as e:
                print(f"⚠️ LLM Werkzeugauswahl fehlgeschlagen, verwende Standard: {e}")
                use_similarity, use_qa = True, True
                ss_weight, qa_weight = 0.6, 0.4
            
            # Werkzeuge basierend auf LLM-Entscheidung aufrufen
            similarity_tool = MockSimilaritySearchTool()
            qa_tool = MockQATool()
            
            ss_result = None
            qa_result = None
            
            if use_similarity:
                try:
                    ss_result = similarity_tool._run(question)
                    ss_data = json.loads(ss_result)
                    print(f"✅ Similarity Search: {ss_data['found_documents']} Dokumente gefunden")
                except Exception as e:
                    print(f"❌ Similarity Search Fehler: {e}")
                    ss_data = {"documents": [], "found_documents": 0}
            
            if use_qa:
                try:
                    qa_result = qa_tool._run(question)
                    qa_data = json.loads(qa_result)
                    print(f"✅ QA Search: {qa_data['found_answers']} Antworten gefunden")
                except Exception as e:
                    print(f"❌ QA Search Fehler: {e}")
                    qa_data = {"qa_pairs": [], "found_answers": 0}
            
            call_time = time.time() - start_time
            print(f"Gesamte LLM+Tools Verarbeitungszeit: {call_time:.2f}seconds")
            
            # LLM für Ergebnisfusion verwenden
            if ss_result or qa_result:
                fusion_prompt = f"""Du bist ein RAG-Assistent. Analysiere und fusioniere die folgenden Suchergebnisse für die Frage: "{question}"

                    Similarity Search Ergebnisse:
                    {ss_result if ss_result else "Nicht verwendet"}

                    QA Search Ergebnisse:
                    {qa_result if qa_result else "Nicht verwendet"}

                    Gewichtung: Similarity={ss_weight}, QA={qa_weight}

                    Erstelle eine strukturierte Antwort, die:
                    1. Die wichtigsten Informationen kombiniert
                    2. Die Qualität der Ergebnisse bewertet
                    3. Eine Vertrauensbewertung (0-1) abgibt

                    Format:
                    ANTWORT: [Deine zusammengefasste Antwort]
                    VERTRAUEN: [0.0-1.0]
                    QUALITÄT: [Hoch/Mittel/Niedrig]"""

                try:
                    fusion_response = self.llm.invoke(fusion_prompt)
                    print(f"\n🔄 LLM Fusion Ergebnis:")
                    print("-" * 30)
                    print(fusion_response.content)
                    
                    # Versuche Vertrauenswert zu extrahieren
                    content = fusion_response.content.lower()
                    confidence = 0.5  # Standard
                    if "vertrauen:" in content:
                        try:
                            confidence_part = content.split("vertrauen:")[1].split("\n")[0]
                            confidence = float(''.join(filter(lambda x: x.isdigit() or x == '.', confidence_part)))
                            if confidence > 1.0:
                                confidence = confidence / 100  # Falls als Prozent angegeben
                        except:
                            pass
                    
                    print(f"\n📈 Extrahiertes Vertrauen: {confidence:.2f}")
                    
                except Exception as e:
                    print(f"❌ LLM Fusion fehlgeschlagen: {e}")
                    print("📋 Fallback: Verwende grundlegende Ergebniskombination")
                    
                    # Fallback ohne LLM
                    if ss_data and qa_data:
                        basic_confidence = (
                            sum(doc.get('similarity', 0.5) for doc in ss_data.get('documents', [])) / max(len(ss_data.get('documents', [])), 1) * ss_weight +
                            sum(qa.get('confidence', 0.5) for qa in qa_data.get('qa_pairs', [])) / max(len(qa_data.get('qa_pairs', [])), 1) * qa_weight
                        )
                        print(f"📊 Basis-Vertrauenswert: {basic_confidence:.2f}")
            
            print("-" * 50)
    
    def run_integration_test(self):
        print("START TEST")
        print("=" * 60)
        
        start_time = time.time()
        
        # run all tests
        self.test_single_tool_usage()
        self.test_tool_calling_capability()
        self.test_weighted_approach()
            
        end_time = time.time()
            
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print("=" * 60)
        

if __name__ == "__main__":
    tester = RAGToolIntegrationTester("mistral:7b-instruct-v0.3-q5_0")
    tester.run_integration_test()