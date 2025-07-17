import time
import json
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage
import os
from pymilvus import MilvusClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import Field

class MilvusSimilaritySearchTool(BaseTool):
    """Enhanced Milvus tool for RAG system evaluation"""
    name: str = "similarity_search"
    description: str = (
        "Führe semantische Ähnlichkeitssuche in der deutschen Dokumentensammlung durch. "
        "Verwende dieses Tool für: literarische Analysen, Themenuntersuchungen, "
        "konzeptuelle Fragen zu 'Der Fremde' von Camus."
    )
    collection_name: str = Field(..., description="Name of the Milvus collection")
    k: int = Field(..., description="Number of top results to retrieve")

    def __init__(self, collection_name: str = "german_docs", k: int = 3):
        super().__init__()
        self.collection_name = collection_name
        self.k = k
        
        # Database setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(current_dir)
        db_path = os.path.join(workspace_root, "milvus_german_docs.db")
        db_path = os.path.abspath(db_path)
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Milvus database not found: {db_path}")
        
        self._client = MilvusClient(db_path)
        
        # Initialize embeddings
        self._embedder = HuggingFaceEmbeddings(
            model_name="aari1995/German_Semantic_V3b",
            model_kwargs={"device": "cpu"}
        )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute semantic search and return formatted results"""
        query_embedding = self._embedder.embed_query(query)
        
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=self.k,
            output_fields=["source", "text"]
        )

        docs_out = []
        if results and len(results) > 0:
            for hit in results[0]:
                docs_out.append({
                    "score": float(hit.get("distance", 0.0)),
                    "source": hit.get("entity", {}).get("source", "Unknown"),
                    "content": hit.get("entity", {}).get("text", "")
                })

        return json.dumps({
            "found_documents": len(docs_out),
            "documents": docs_out
        }, ensure_ascii=False, indent=2)

class WebSearchTool(BaseTool):
    """Mock web search tool for testing"""
    name: str = "web_search"
    description: str = (
        "Durchsuche das Internet nach aktuellen Informationen zu Literatur, "
        "Philosophie oder anderen Themen. Verwende bei Bedarf für zusätzliche Kontextinformationen."
    )
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Mock web search - in real implementation, use actual web search"""
        # Simulate web search delay
        time.sleep(1)
        
        mock_results = [
            {
                "title": "Albert Camus - Der Fremde: Analyse und Interpretation",
                "url": "https://example.com/camus-analysis",
                "snippet": "Eine tiefgreifende Analyse von Camus' existentialistischem Meisterwerk...",
                "relevance": 0.95
            },
            {
                "title": "Existentialismus in der französischen Literatur",
                "url": "https://example.com/existentialisme",
                "snippet": "Untersuchung der existentialistischen Themen in der französischen Literatur...",
                "relevance": 0.87
            }
        ]
        
        return json.dumps({
            "query": query,
            "results": mock_results,
            "total_found": len(mock_results)
        }, ensure_ascii=False, indent=2)

class RAGSystemEvaluator:
    """Comprehensive RAG system evaluator for LLM assessment"""
    
    def __init__(self, model_name: str = "mistral:7b-instruct-v0.3-q5_0"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            base_url="http://localhost:11434"
        )

        # Initialize tools
        self.tools = [
            MilvusSimilaritySearchTool(collection_name="german_docs", k=3),
            WebSearchTool()
        ]
        
        # Create LangChain agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _create_agent(self):
        """Create LangChain tool-calling agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sie sind ein intelligenter Assistent für deutsche Literatur, 
            spezialisiert auf Albert Camus' 'Der Fremde'. Sie haben Zugang zu folgenden Werkzeugen:
            
            1. similarity_search: Für semantische Suche in deutschen Dokumenten
            2. web_search: Für aktuelle Informationen aus dem Internet
            
            Verwenden Sie die Werkzeuge strategisch:
            - Für literarische Analysen: similarity_search
            - Für aktuelle Informationen: web_search
            - Für komplexe Fragen: beide Werkzeuge kombinieren
            
            Antworten Sie immer auf Deutsch und stützen Sie sich auf die Werkzeugergebnisse."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        return create_tool_calling_agent(self.llm, self.tools, prompt)
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        print(f"\n=== RAG System Evaluation: {self.model_name} ===")
        
        test_cases = [
            {
                "category": "Factual Query",
                "question": "Wer ist der Hauptcharakter in 'Der Fremde' und was sind seine wichtigsten Eigenschaften?",
                "expected_tools": ["similarity_search"],
                "complexity": "niedrig",
                "evaluation_criteria": ["character_identification", "trait_description", "german_quality"]
            },
            {
                "category": "Analytical Query", 
                "question": "Wie zeigt Camus das Thema der Absurdität in 'Der Fremde'? Analysieren Sie mit Beispielen.",
                "expected_tools": ["similarity_search"],
                "complexity": "hoch",
                "evaluation_criteria": ["theme_analysis", "examples", "philosophical_understanding", "german_quality"]
            },
            {
                "category": "Multi-tool Query",
                "question": "Welche Bedeutung hat 'Der Fremde' in der modernen Literatur? Berücksichtigen Sie sowohl historische als auch aktuelle Perspektiven.",
                "expected_tools": ["similarity_search", "web_search"],
                "complexity": "hoch",
                "evaluation_criteria": ["historical_context", "modern_relevance", "synthesis", "german_quality"]
            },
            {
                "category": "Instruction Following",
                "question": "Fassen Sie die Handlung von 'Der Fremde' in genau 3 Absätzen zusammen. Jeder Absatz soll einen anderen Teil des Romans behandeln.",
                "expected_tools": ["similarity_search"],
                "complexity": "mittel",
                "evaluation_criteria": ["structure_adherence", "content_accuracy", "german_quality"]
            }
        ]
        
        results = {
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": [],
            "overall_scores": {}
        }
        
        total_scores = {
            "tool_usage": 0,
            "instruction_following": 0,
            "german_quality": 0,
            "response_accuracy": 0,
            "response_time": 0
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {test_case['category']} ---")
            print(f"Frage: {test_case['question']}")
            print(f"Erwartete Werkzeuge: {test_case['expected_tools']}")
            
            # Execute query
            start_time = time.time()
            try:
                response = self.agent_executor.invoke({"input": test_case['question']})
                execution_time = time.time() - start_time
                
                # Evaluate response
                evaluation = self._evaluate_response(
                    test_case, response, execution_time
                )
                
                results["test_results"].append({
                    "test_case": test_case,
                    "response": response,
                    "evaluation": evaluation,
                    "execution_time": execution_time
                })
                
                # Update total scores
                for key in total_scores:
                    if key in evaluation:
                        total_scores[key] += evaluation[key]
                
                print(f"✅ Test abgeschlossen in {execution_time:.2f}s")
                self._print_evaluation(evaluation)
                
            except Exception as e:
                print(f"❌ Test fehlgeschlagen: {e}")
                results["test_results"].append({
                    "test_case": test_case,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                })
        
        # Calculate overall scores
        num_tests = len([r for r in results["test_results"] if "evaluation" in r])
        if num_tests > 0:
            for key in total_scores:
                results["overall_scores"][key] = total_scores[key] / num_tests
        
        self._print_final_results(results)
        return results
    
    def _evaluate_response(self, test_case: Dict, response: Dict, execution_time: float) -> Dict[str, float]:
        """Evaluate response quality across multiple dimensions"""
        evaluation = {
            "tool_usage": 0,
            "instruction_following": 0,
            "german_quality": 0,
            "response_accuracy": 0,
            "response_time": min(1.0, 10 / execution_time)  # Faster responses get higher scores
        }
        
        output_text = response.get("output", "").lower()
        
        # Tool usage evaluation
        expected_tools = test_case.get("expected_tools", [])
        tools_used = []
        
        # Check which tools were actually used (this is simplified)
        if "similarity_search" in str(response.get("intermediate_steps", [])):
            tools_used.append("similarity_search")
        if "web_search" in str(response.get("intermediate_steps", [])):
            tools_used.append("web_search")
        
        if expected_tools:
            tool_overlap = len(set(expected_tools) & set(tools_used))
            evaluation["tool_usage"] = tool_overlap / len(expected_tools)
        
        # Instruction following (basic checks)
        if test_case["category"] == "Instruction Following":
            # Check for 3 paragraphs
            paragraphs = output_text.split('\n\n')
            if len(paragraphs) >= 3:
                evaluation["instruction_following"] = 1.0
            else:
                evaluation["instruction_following"] = 0.5
        else:
            evaluation["instruction_following"] = 0.8 if len(output_text) > 50 else 0.3
        
        # German quality (basic checks)
        german_indicators = [
            "der", "die", "das", "und", "ist", "von", "zu", "auf", "mit", "für",
            "meursault", "camus", "fremde", "absurd", "roman"
        ]
        german_score = sum(1 for word in german_indicators if word in output_text)
        evaluation["german_quality"] = min(1.0, german_score / 10)
        
        # Response accuracy (length and content relevance)
        if len(output_text) > 100 and any(keyword in output_text for keyword in ["camus", "fremde", "meursault"]):
            evaluation["response_accuracy"] = 0.8
        else:
            evaluation["response_accuracy"] = 0.4
        
        return evaluation
    
    def _print_evaluation(self, evaluation: Dict[str, float]):
        """Print evaluation results"""
        print("Bewertung:")
        for criterion, score in evaluation.items():
            print(f"  {criterion}: {score:.2f} ({score*100:.0f}%)")
    
    def _print_final_results(self, results: Dict):
        """Print final evaluation results"""
        print(f"\n{'='*50}")
        print(f"FINAL EVALUATION: {self.model_name}")
        print(f"{'='*50}")
        
        if results["overall_scores"]:
            print("Gesamtbewertung:")
            for criterion, score in results["overall_scores"].items():
                print(f"  {criterion}: {score:.2f} ({score*100:.0f}%)")
            
            overall_avg = sum(results["overall_scores"].values()) / len(results["overall_scores"])
            print(f"\nGesamtdurchschnitt: {overall_avg:.2f} ({overall_avg*100:.0f}%)")
            
            # Recommendation
            if overall_avg >= 0.8:
                print("✅ EMPFEHLUNG: Ausgezeichnet für RAG-System geeignet")
            elif overall_avg >= 0.6:
                print("⚠️  EMPFEHLUNG: Gut geeignet, kleinere Anpassungen empfohlen")
            else:
                print("❌ EMPFEHLUNG: Nicht optimal für RAG-System")

def compare_models():
    """Compare multiple models for RAG system suitability"""
    models_to_test = [
        "mistral:7b-instruct-v0.3-q5_0",
        "qwen2:7b-instruct",
        "llama3:8b-instruct",
        # Add more models as needed
    ]
    
    all_results = []
    
    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model}")
        print(f"{'='*60}")
        
        try:
            evaluator = RAGSystemEvaluator(model)
            results = evaluator.evaluate_model()
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")
            continue
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        if "overall_scores" in result and result["overall_scores"]:
            overall_avg = sum(result["overall_scores"].values()) / len(result["overall_scores"])
            print(f"{result['model']}: {overall_avg:.2f} ({overall_avg*100:.0f}%)")
    
    return all_results

if __name__ == "__main__":
    # Test single model
    evaluator = RAGSystemEvaluator("mistral:7b-instruct-v0.3-q5_0")
    results = evaluator.evaluate_model()
    
    # # Or compare multiple models
    # compare_models()