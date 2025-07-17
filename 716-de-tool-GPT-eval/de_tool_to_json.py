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
import re
import os
from datetime import datetime
from pymilvus import connections, utility, MilvusClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

MILVUS_HOST = "milvus"
MILVUS_PORT = "19530"
DEFAULT_COLLECTION = "german_docs"
EMBEDDING_MODEL = "aari1995/German_Semantic_V3b"

class MilvusSimilaritySearchTool(BaseTool):
    """
    Milvus Lite 语义相似性搜索工具
    使用 MilvusClient 直接连接到本地 .db 文件
    """
    name: str = "similarity_search"
    description: str = (
        "Führe semantische Ähnlichkeitssuche in der Milvus Lite-Kollektion durch. "
        "Eingabe: Query-String; Ausgabe: JSON mit Dokumentfragmenten und Scores."
    )

    _client: Any = None
    _embedder: Any = None
    collection_name: str = "german_docs"
    k: int = 3

    def __init__(self,
                 collection_name: str = "german_docs",
                 db_path: str = None,
                 embedding_model: str = "aari1995/German_Semantic_V3b",
                 embedding_device: str = "cpu",
                 k: int = 3):
        super().__init__()
        
        # Set default db_path to the actual location of the database
        if db_path is None:
            # Get the absolute path to the database file in the workspace root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(current_dir)  # Go up one level from 711-edit-logic
            db_path = os.path.join(workspace_root, "milvus_german_docs.db")
        
        # Ensure the path is absolute
        db_path = os.path.abspath(db_path)
        
        # Check if database file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Milvus database file not found: {db_path}")
        
        print(f"Using Milvus database: {db_path}")
        
        # Initialize MilvusClient directly
        self._client = MilvusClient(db_path)
        self.collection_name = collection_name
        
        # Check if collection exists
        collections = self._client.list_collections()
        if collection_name not in collections:
            raise ValueError(f"Milvus Lite collection '{collection_name}' not found in {db_path}. Available: {collections}")

        # Load Embeddings
        self._embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": embedding_device}
        )
        self.k = k

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        使用 MilvusClient 进行相似性检索，格式化返回 JSON 字符串
        """
        # Embed the query
        query_embedding = self._embedder.embed_query(query)
        
        # Execute search using MilvusClient
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=self.k,
            output_fields=["source", "text"]  # Specify which fields to return
        )

        docs_out: List[Dict[str, Any]] = []
        
        # Process results
        if results and len(results) > 0:
            for hit in results[0]:  # results[0] contains the hits for our single query
                docs_out.append({
                    "score": float(hit.get("distance", 0.0)),  # Milvus returns distance, lower is better
                    "source": hit.get("entity", {}).get("source", "Unknown"),
                    "content": hit.get("entity", {}).get("text", "")
                })

        output = {
            "tool": self.name,
            "query": query,
            "found_documents": len(docs_out),
            "documents": docs_out,
            "search_type": "semantic_similarity"
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

class MockQATool(BaseTool):
    """
    模拟问答搜索工具
    设计思路：
    1. 模拟预构建的问答对数据库
    2. 基于问题相似性匹配已有答案
    3. 返回高置信度的直接答案
    4. 测试LLM对结构化知识的理解
    """
    name: str = "qa_search"
    description: str = """Frage-Antwort-Suchwerkzeug für Camus 'Der Fremde'.
    Eingabe: Spezifische Frage; Ausgabe: JSON mit Antwort und Metadaten."""
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        模拟问答搜索过程
        1. 接收问题
        2. 匹配预构建的问答对
        3. 返回最相关的答案
        """
        
        mock_qa_pairs = [
    {
        "question": "Wer ist der Protagonist von 'Der Fremde'?",
        "answer": "'Der Fremde' hat einen Hauptprotagonisten: Meursault, ein emotional distanzierter Algerienfranzose, der durch seine Gleichgültigkeit gegenüber gesellschaftlichen Normen auffällt.",
        "source": "der_fremde_charaktere.pdf",
        "confidence": 0.95,
        "category": "Charaktere"
    },
    {
        "question": "An welchem Tag beginnt die Handlung von 'Der Fremde'?",
        "answer": "Die Handlung beginnt am Tag, an dem Meursault erfährt, dass seine Mutter gestorben ist. Im berühmten ersten Satz heißt es: 'Heute ist Mama gestorben. Vielleicht auch gestern, ich weiß nicht.'",
        "source": "der_fremde_zeitlinie.pdf",
        "confidence": 0.98,
        "category": "Handlung"
    },
    {
        "question": "Wie ist die Struktur von 'Der Fremde' aufgebaut?",
        "answer": "'Der Fremde' ist in zwei Teile gegliedert: Der erste Teil beschreibt Meursaults Alltag und den Mord am Strand. Der zweite Teil behandelt den Prozess, die Verurteilung und Meursaults Reflexionen über das Leben und den Tod.",
        "source": "der_fremde_struktur.pdf",
        "confidence": 0.92,
        "category": "Struktur"
    },
    {
        "question": "Welche literarischen Techniken verwendet Camus in 'Der Fremde'?",
        "answer": "Camus verwendet eine nüchterne, einfache und distanzierte Erzählweise. Die Ich-Perspektive und der knappe, emotionslose Stil verstärken die Absurdität des Geschehens.",
        "source": "der_fremde_techniken.pdf",
        "confidence": 0.88,
        "category": "Technik"
    },
    {
        "question": "Wo spielt 'Der Fremde'?",
        "answer": "'Der Fremde' spielt in Algier, Algerien, während der französischen Kolonialzeit. Schauplätze sind u.a. Meursaults Wohnung, das Altenheim, der Strand und das Gerichtsgebäude.",
        "source": "der_fremde_schauplatz.pdf",
        "confidence": 0.94,
        "category": "Schauplatz"
    },
    {
        "question": "Warum wurde 'Der Fremde' kontrovers diskutiert?",
        "answer": "'Der Fremde' wurde kontrovers diskutiert wegen Meursaults Gleichgültigkeit gegenüber dem Tod seiner Mutter, dem scheinbar grundlosen Mord und seiner Ablehnung religiöser Trostangebote. Das Werk provozierte durch die Darstellung der Absurdität des Lebens.",
        "source": "der_fremde_kontroversen.pdf",
        "confidence": 0.90,
        "category": "Geschichte"
    }
]

        
        # 问题匹配逻辑
        query_lower = query.lower()
        matched_qa = []
        
        # 精确匹配和模糊匹配
        for qa in mock_qa_pairs:
            question_lower = qa["question"].lower()
            
            # 计算问题相似度
            similarity_score = 0
            
            # 关键词匹配
            query_words = set(query_lower.split())
            question_words = set(question_lower.split())
            
            # 计算词汇重叠度
            overlap = len(query_words.intersection(question_words))
            total_words = len(query_words.union(question_words))
            
            if total_words > 0:
                similarity_score = overlap / total_words
            
            # 特定关键词加权
            key_patterns = {
    "protagonist|charakter|figur|meursault": "Charaktere",
    "zeit|datum|tag|beginn|beerdigung": "Handlung", 
    "kapitel|struktur|aufbau|gliederung": "Struktur",
    "technik|methode|stil|erzählweise|sprache": "Technik",
    "schauplatz|ort|algier|algerien|strand|gericht": "Schauplatz",
    "verbot|zensur|kontrovers|kritik|provokation": "Geschichte"
}

            
            for pattern, category in key_patterns.items():
                if re.search(pattern, query_lower) and qa["category"] == category:
                    similarity_score += 0.3
            
            # 如果相似度足够高，添加到匹配结果
            if similarity_score > 0.2:
                qa_copy = qa.copy()
                qa_copy["match_score"] = similarity_score
                matched_qa.append(qa_copy)
        
        # 如果没有匹配，返回通用答案
        if not matched_qa:
            matched_qa = [{
        "question": "Allgemeine Anfrage",
        "answer": (
            "Für spezifische Informationen zu Camus' 'Der Fremde' konsultieren Sie bitte entsprechende Literatur. "
            "Das Werk behandelt zentrale Themen des Existentialismus und der Absurdität. "
            "Eine detaillierte Analyse erfordert eine tiefergehende Auseinandersetzung mit philosophischen und literarischen Aspekten."
        ),
        "source": "allgemeine_referenz_der_fremde.pdf",
        "confidence": 0.3,
        "category": "Allgemein",
        "match_score": 0.1
    }]

        
        # 按匹配分数排序
        matched_qa.sort(key=lambda x: x["match_score"], reverse=True)
        matched_qa = matched_qa[:2]  # 最多返回2个答案
        
        result = {
            "tool": "qa_search",
            "query": query,
            "found_answers": len(matched_qa),
            "qa_pairs": matched_qa,
            "search_type": "question_answering"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)

class EnhancedRAGToolIntegrationTester:
    """
    增强的RAG工具集成测试器
    设计目标：
    1. 测试LLM对工具的理解能力
    2. 测试LLM的工具选择策略
    3. 测试LLM的结果融合能力
    4. 评估整体RAG系统性能
    """
    
    def __init__(self, model_name: str = "mistral:7b-instruct-v0.3-q5_0"):
        """
        初始化测试器
        参数：
        - model_name: 要测试的LLM模型名称
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,  # 稍微提高温度以获得更自然的响应
            num_predict=512,  # 限制输出长度
            base_url="http://localhost:11434"  # Use localhost instead of ollama hostname
        )
        self.tools = [MilvusSimilaritySearchTool(collection_name="german_docs"), MockQATool()]
        self.model_name = model_name
        
        # Initialize test results storage
        self.test_results = {
            "model_name": model_name,
            "test_timestamp": datetime.now().isoformat(),
            "test_config": {
                "temperature": 0.1,
                "num_predict": 512,
                "tools_used": ["similarity_search", "qa_search"]
            },
            "test_cases": []
        }
        
        # 改进的系统提示
        self.system_prompt = """Sie sind ein intelligenter Assistent, der sich auf Albert Camus' "Der Fremde" spezialisiert hat. Ihnen stehen zwei leistungsstarke Werkzeuge zur Verfügung:


                1. **similarity_search**: Semantische Ähnlichkeitssuche
                - Verwendung für: konzeptuelle Fragen, literarische Analysen, Themenuntersuchungen, Charakterstudien
                - Liefert: relevante Dokumentfragmente und Analyseinhalte

                2. **qa_search**: Frage-Antwort-Suche  
                - Verwendung für: spezifische Faktenfragen, direkte Informationsabfragen
                - Liefert: präzise Antworten und hochwertige Informationen

                **Wichtige Strategien**:
                - Bei komplexen Fragen empfiehlt sich die gleichzeitige Nutzung beider Werkzeuge für umfassende Informationen
                - Bei einfachen Faktenfragen sollten Sie qa_search bevorzugt verwenden
                - Bei analytischen Fragen sollten Sie similarity_search priorisieren
                - Stützen Sie Ihre Antworten stets auf die von den Werkzeugen gelieferten Ergebnisse"""

        # 融合后的三个预定义测试问题
        self.test_questions = [
    {
        "question": "Wer ist Meursault und welche Rolle spielt er in 'Der Fremde'?",
        "complexity": "mittel",
        "type": "Charakter-Faktenfrage",
        "expected_primary": "qa_search",
        "expected_secondary": "similarity_search",
        "reasoning": "Kombiniert Faktenfrage mit Charakteranalyse"
    },
    {
        "question": "Wie zeigt Camus den Existentialismus und das Gefühl der Absurdität in 'Der Fremde'?",
        "complexity": "hoch",
        "type": "Analytische Frage",
        "expected_primary": "similarity_search",
        "expected_secondary": "qa_search",
        "reasoning": "Erfordert tiefgehende literarische Analyse und thematische Interpretation"
    },
    # {
    #     "question": "An welchem Tag beginnt die Handlung von 'Der Fremde' und warum ist dieser Tag wichtig?",
    #     "complexity": "niedrig",
    #     "type": "Faktenfrage mit Kontext",
    #     "expected_primary": "qa_search",
    #     "expected_secondary": "similarity_search",
    #     "reasoning": "Direkte Faktenfrage mit Kontextbezug zum berühmten ersten Satz"
    # },
    {
        "question": "Was ist der Unterschied zwischen einem Hund und einer Katze?",
        "complexity": "niedrig",
        "type": "Irrelevante Faktenfrage",
        "expected_primary": " ",
        "expected_secondary": " ",
        "reasoning": "Direkte Faktenfrage ohne Relevanz für die Handlung"
    }
]



    def test_integrated_understanding_and_workflow(self):
        """
        融合测试：LLM工具理解和集成工作流测试
        目的：测试LLM的工具理解能力和完整的RAG工作流程
        """
        print("\n=== Integrierter Test: Werkzeugverständnis und Arbeitsablauf ===")
        print("Designidee: Kombinierte Prüfung der Werkzeugkognition und vollständigen RAG-Szenarios")
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"\n{i}. Integrierter Test: {test_case['complexity']} Komplexität")
            print(f"   Frage: {test_case['question']}")
            print(f"   Typ: {test_case['type']}")
            print(f"   Erwartete Strategie: {test_case['expected_primary']} → {test_case['expected_secondary']}")
            print("-" * 60)
            
            # Initialize test case result
            test_case_result = {
                "test_case_id": i,
                "question": test_case['question'],
                "question_type": test_case['type'],
                "complexity": test_case['complexity'],
                "expected_strategy": {
                    "primary_tool": test_case['expected_primary'],
                    "secondary_tool": test_case['expected_secondary']
                },
                "reasoning": test_case['reasoning'],
                "execution_log": [],
                "strategy_analysis": {},
                "tool_results": {},
                "final_response": {},
                "performance_metrics": {},
                "evaluation_ready": True
            }
            
            workflow_start = time.time()
            
            # 步骤1: LLM工具理解和策略制定
            strategy_prompt = f"""{self.system_prompt}

                        Folgende Frage soll beantwortet werden: "{test_case['question']}"

                        Analysieren Sie zunächst diese Frage und entwickeln Sie eine Werkzeugstrategie:
                        1. Bestimmen Sie die Art und Komplexität der Frage
                        2. Wählen Sie die zu verwendenden Werkzeuge aus
                        3. Erläutern Sie die Priorität und Gewichtung der Werkzeugnutzung

                        Antwortformat:
                        **Frageanalyse**: [Art und Komplexität der Frage]
                        **Werkzeugstrategie**: [Liste der zu verwendenden Werkzeuge]
                        **Verwendungsreihenfolge**: [Welches Werkzeug zuerst und warum]
                        **Erwartetes Ergebnis**: [Welche Informationen Sie zu erhalten hoffen]"""

            try:
                strategy_response = self.llm.invoke(strategy_prompt)
                strategy_time = time.time() - workflow_start
                
                test_case_result["strategy_analysis"] = {
                    "strategy_prompt": strategy_prompt,
                    "strategy_response": strategy_response.content,
                    "execution_time": strategy_time,
                    "success": True
                }
                
                print(f"   📋 Strategieentwicklung ({strategy_time:.2f}s):")
                print(f"   {strategy_response.content}")
                print()
                
                # 从策略响应中提取工具使用计划
                content = strategy_response.content.lower()
                planned_tools = []
                
                if "similarity_search" in content or "ähnlichkeit" in content or "semantisch" in content:
                    planned_tools.append("similarity_search")
                if "qa_search" in content or "frage-antwort" in content or "qa" in content:
                    planned_tools.append("qa_search")
                
                test_case_result["strategy_analysis"]["planned_tools"] = planned_tools
                print(f"   🎯 Geplante Werkzeugnutzung: {planned_tools}")
                
                # 评估工具选择准确性
                strategy_accuracy = 0
                if test_case['expected_primary'] in planned_tools:
                    strategy_accuracy += 0.6
                    print(f"   ✅ Hauptwerkzeug korrekt erkannt: {test_case['expected_primary']}")
                else:
                    print(f"   ❌ Hauptwerkzeug nicht erkannt: {test_case['expected_primary']}")
                
                if test_case['expected_secondary'] and test_case['expected_secondary'] in planned_tools:
                    strategy_accuracy += 0.4
                    print(f"   ✅ Hilfswerkzeug korrekt erkannt: {test_case['expected_secondary']}")
                elif not test_case['expected_secondary'] and len(planned_tools) == 1:
                    strategy_accuracy += 0.4
                    print(f"   ✅ Korrekt erkannt: nur ein Werkzeug benötigt")
                
                test_case_result["strategy_analysis"]["strategy_accuracy"] = strategy_accuracy
                print(f"   📊 Strategiegenauigkeit: {strategy_accuracy*100:.0f}%")
                
            except Exception as e:
                strategy_time = time.time() - workflow_start
                test_case_result["strategy_analysis"] = {
                    "strategy_prompt": strategy_prompt,
                    "strategy_response": None,
                    "execution_time": strategy_time,
                    "success": False,
                    "error": str(e)
                }
                print(f"   ❌ Strategieentwicklung fehlgeschlagen: {e}")
                # Use both tools as fallback to ensure we test both
                planned_tools = ["qa_search", "similarity_search"]
                test_case_result["strategy_analysis"]["planned_tools"] = planned_tools
                print(f"   🔄 Fallback: Beide Werkzeuge werden getestet")
            
            # 步骤2: 执行工具调用
            tool_results = {}
            
            similarity_tool = MilvusSimilaritySearchTool(collection_name="german_docs")
            qa_tool = MockQATool()
            
            if "similarity_search" in planned_tools:
                try:
                    sim_start = time.time()
                    sim_result = similarity_tool._run(test_case['question'])
                    sim_time = time.time() - sim_start
                    
                    tool_results['similarity_search'] = {
                        'result': json.loads(sim_result),
                        'execution_time': sim_time,
                        'success': True
                    }
                    print(f"   🔍 Ähnlichkeitssuche abgeschlossen ({sim_time:.2f}s): {tool_results['similarity_search']['result']['found_documents']} Dokumente")
                    
                except Exception as e:
                    sim_time = time.time() - sim_start if 'sim_start' in locals() else 0
                    tool_results['similarity_search'] = {
                        'result': None,
                        'execution_time': sim_time,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ Ähnlichkeitssuche fehlgeschlagen: {e}")
            
            if "qa_search" in planned_tools:
                try:
                    qa_start = time.time()
                    qa_result = qa_tool._run(test_case['question'])
                    qa_time = time.time() - qa_start
                    
                    tool_results['qa_search'] = {
                        'result': json.loads(qa_result),
                        'execution_time': qa_time,
                        'success': True
                    }
                    print(f"   💡 Q&A-Suche abgeschlossen ({qa_time:.2f}s): {tool_results['qa_search']['result']['found_answers']} Antworten")
                    
                except Exception as e:
                    qa_time = time.time() - qa_start if 'qa_start' in locals() else 0
                    tool_results['qa_search'] = {
                        'result': None,
                        'execution_time': qa_time,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ Q&A-Suche fehlgeschlagen: {e}")
            
            test_case_result["tool_results"] = tool_results
            
            # 步骤3: 结果融合和最终回答
            if tool_results:
                fusion_prompt = f"""{self.system_prompt}

                            Ursprüngliche Frage: "{test_case['question']}"

                            Suchergebnisse der Werkzeuge:
                            """
                
                for tool_name, tool_data in tool_results.items():
                    if tool_data['success'] and tool_data['result']:
                        fusion_prompt += f"\n{tool_name.upper()} Ergebnisse:\n{json.dumps(tool_data['result'], ensure_ascii=False, indent=2)}\n"
                
                fusion_prompt += f"""
                            Basierend auf den obigen Suchergebnissen geben Sie bitte eine umfassende und präzise Antwort. Anforderungen:

                            1. Integrieren Sie alle relevanten Informationen
                            2. Heben Sie wichtige Inhalte hervor
                            3. Wahren Sie wissenschaftlichen Anspruch und Genauigkeit
                            4. Falls Informationen unzureichend sind, weisen Sie darauf hin
                            5. Geben Sie Quellenverweise an

                            Antwortformat:
                            **Antwort**: [Ihre vollständige Antwort]
                            **Informationsquellen**: [Welche Werkzeuge und Dokumente wurden verwendet]
                            **Vertrauensgrad**: [0-100%, Ihr Vertrauen in die Antwort]
                            **Zusätzliche Anmerkungen**: [Falls weitere Erläuterungen nötig sind]"""

                fusion_start = time.time()
                try:
                    final_response = self.llm.invoke(fusion_prompt)
                    fusion_time = time.time() - fusion_start
                    
                    test_case_result["final_response"] = {
                        "fusion_prompt": fusion_prompt,
                        "response_content": final_response.content,
                        "execution_time": fusion_time,
                        "success": True,
                        "model_name": self.model_name
                    }
                        
                    print(f"Ergebnisfusion abgeschlossen ({fusion_time:.2f}s)")
                    print(f"Endgültige Antwort:")
                    print("-" * 30)
                    print(final_response.content)
                    print("-" * 30)
                except Exception as e:
                    fusion_time = time.time() - fusion_start
                    test_case_result["final_response"] = {
                        "fusion_prompt": fusion_prompt,
                        "response_content": None,
                        "execution_time": fusion_time,
                        "success": False,
                        "error": str(e),
                        "model_name": self.model_name
                    }
                    print(f"❌ Ergebnisfusion fehlgeschlagen ({fusion_time:.2f}s): {e}")
                    print("📊 Werkzeugergebnisse ohne LLM-Fusion:")
                    for tool_name, tool_data in tool_results.items():
                        if tool_data['success'] and tool_data['result']:
                            print(f"\n{tool_name.upper()}:")
                            print(json.dumps(tool_data['result'], ensure_ascii=False, indent=2))
            
            total_time = time.time() - workflow_start
            test_case_result["performance_metrics"] = {
                "total_execution_time": total_time,
                "strategy_time": test_case_result["strategy_analysis"].get("execution_time", 0),
                "tools_time": sum([tool_data.get("execution_time", 0) for tool_data in tool_results.values()]),
                "fusion_time": test_case_result["final_response"].get("execution_time", 0)
            }
            
            print(f"Gesamtverarbeitungszeit: {total_time:.2f}s")
            
            # Add test case result to the overall results
            self.test_results["test_cases"].append(test_case_result)
    
    def run_comprehensive_test(self):
        print("START")
        print("="*80)
        print(f"CONFIG")
        print(f"   - Model: {self.model_name}")
        print(f"   - Tool: SS + QA")
        print(f"   - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        overall_start = time.time()
        
        self.test_integrated_understanding_and_workflow()
        
        total_time = time.time() - overall_start
        
        # Add summary statistics to test results
        self.test_results["summary"] = {
            "total_execution_time": total_time,
            "total_test_cases": len(self.test_results["test_cases"]),
            "successful_test_cases": sum(1 for tc in self.test_results["test_cases"] if tc.get("final_response", {}).get("success", False)),
            "average_response_time": total_time / len(self.test_results["test_cases"]) if self.test_results["test_cases"] else 0
        }
        
        # Save results to JSON file
        self._save_test_results()
        
        print(f"\nTest done! Gesamtzeit: {total_time:.2f} Sekunden")
        print("="*80)
    
    def _save_test_results(self):
        """Save test results to JSON file in test_results folder"""
        try:
            # Create test_results directory if it doesn't exist
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(current_dir)
            test_results_dir = os.path.join(workspace_root, "test_results")
            
            if not os.path.exists(test_results_dir):
                os.makedirs(test_results_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            # Clean model name for filename
            clean_model_name = self.model_name.replace(":", "_").replace("/", "_")
            filename = f"{clean_model_name}_test_results_{timestamp}.json"
            filepath = os.path.join(test_results_dir, filename)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n📁 Test results saved to: {filepath}")
            
            # Also create a summary for quick evaluation
            summary = {
                "model_name": self.model_name,
                "test_timestamp": self.test_results["test_timestamp"],
                "summary_stats": self.test_results["summary"],
                "evaluation_data": []
            }
            
            for tc in self.test_results["test_cases"]:
                eval_data = {
                    "question": tc["question"],
                    "question_type": tc["question_type"],
                    "complexity": tc["complexity"],
                    "expected_tools": [tc["expected_strategy"]["primary_tool"], tc["expected_strategy"]["secondary_tool"]],
                    "actual_tools": tc.get("strategy_analysis", {}).get("planned_tools", []),
                    "strategy_accuracy": tc.get("strategy_analysis", {}).get("strategy_accuracy", 0),
                    "final_response": tc.get("final_response", {}).get("response_content", ""),
                    "response_success": tc.get("final_response", {}).get("success", False),
                    "total_time": tc.get("performance_metrics", {}).get("total_execution_time", 0),
                    "tools_used_successfully": [tool for tool, data in tc.get("tool_results", {}).items() if data.get("success", False)],
                    "evaluation_ready": True
                }
                summary["evaluation_data"].append(eval_data)
            
            # Save evaluation-ready summary
            eval_filename = f"{clean_model_name}_evaluation_ready_{timestamp}.json"
            eval_filepath = os.path.join(test_results_dir, eval_filename)
            
            with open(eval_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"📊 Evaluation-ready summary saved to: {eval_filepath}")
            
        except Exception as e:
            print(f"❌ Error saving test results: {e}")
            # Print results to console as fallback
            print("\n" + "="*50)
            print("FALLBACK: Test results printed to console")
            print("="*50)
            print(json.dumps(self.test_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 可以测试不同的模型
    models_to_test = [
        "mistral:7b-instruct-v0.3-q5_0",
        # "mistral-small:24b",
        # "qwen3:8b",
    ]
    
    for model in models_to_test:
        print(f"\n{'='*20} Testing model: {model} {'='*20}")
        tester = EnhancedRAGToolIntegrationTester(model)
        tester.run_comprehensive_test()
        
        if len(models_to_test) > 1:
            print(f"\n...")
            time.sleep(5)