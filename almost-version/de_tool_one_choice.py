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
from pymilvus import connections, utility, MilvusClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

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

class OptimizedSingleToolRAGTester:
    """
    优化的单工具RAG测试器
    设计目标：
    1. 测试LLM精确的单工具选择能力
    2. 测试LLM的工具适配策略
    3. 测试高效的RAG工作流程
    4. 评估优化后的RAG系统性能
    """
    
    def __init__(self, model_name: str = "mistral:7b-instruct-v0.3-q5_0"):
        """
        初始化单工具策略测试器
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
        
        # 改进的系统提示 - Single Tool Strategy
        self.system_prompt = """Sie sind ein intelligenter Assistent, der sich auf Albert Camus' "Der Fremde" spezialisiert hat. Ihnen stehen zwei leistungsstarke Werkzeuge zur Verfügung:


                1. **similarity_search**: Semantische Ähnlichkeitssuche
                - Verwendung für: konzeptuelle Fragen, literarische Analysen, Themenuntersuchungen, Charakterstudien
                - Liefert: relevante Dokumentfragmente und Analyseinhalte

                2. **qa_search**: Frage-Antwort-Suche  
                - Verwendung für: spezifische Faktenfragen, direkte Informationsabfragen
                - Liefert: präzise Antworten und hochwertige Informationen

                **WICHTIGE STRATEGIE - Ein Werkzeug pro Anfrage**:
                - Wählen Sie für jede Frage NUR EIN Werkzeug aus, das am besten geeignet ist
                - Bei spezifischen Faktenfragen: Verwenden Sie ausschließlich qa_search
                - Bei analytischen/konzeptuellen Fragen: Verwenden Sie ausschließlich similarity_search
                - Bei irrelevanten Fragen: Verwenden Sie kein Werkzeug
                - Vermeiden Sie die gleichzeitige Nutzung beider Werkzeuge
                - Stützen Sie Ihre Antworten auf das gewählte Werkzeugergebnis"""

        # 更新后的测试问题 - Single Tool Strategy
        self.test_questions = [
            {
                "question": "Wer ist Meursault und welche Rolle spielt er in 'Der Fremde'?",
                "complexity": "mittel",
                "type": "Charakter-Faktenfrage",
                "expected_primary": "qa_search",
                "expected_secondary": None,
                "reasoning": "Spezifische Faktenfrage über Hauptcharakter - qa_search optimal"
            },
            {
                "question": "Wie zeigt Camus den Existentialismus und das Gefühl der Absurdität in 'Der Fremde'?",
                "complexity": "hoch",
                "type": "Analytische Frage",
                "expected_primary": "similarity_search",
                "expected_secondary": None,
                "reasoning": "Tiefgehende literarische Analyse - similarity_search optimal"
            },
            {
                "question": "Was ist der Unterschied zwischen einem Hund und einer Katze?",
                "complexity": "niedrig",
                "type": "Irrelevante Faktenfrage",
                "expected_primary": None,
                "expected_secondary": None,
                "reasoning": "Irrelevante Frage - kein Werkzeug sollte verwendet werden"
            }
        ]



    def test_integrated_understanding_and_workflow(self):
        """
        Single Tool Strategy Test: LLM工具理解和优化工作流测试
        目的：测试LLM的单工具选择能力和高效的RAG工作流程
        """
        print("\n=== Single Tool Strategy Test: Werkzeugverständnis und optimierter Arbeitsablauf ===")
        print("Designidee: Prüfung der präzisen Werkzeugauswahl und effizienten RAG-Prozesse")
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"\n{i}. Single Tool Test: {test_case['complexity']} Komplexität")
            print(f"   Frage: {test_case['question']}")
            print(f"   Typ: {test_case['type']}")
            print(f"   Erwartetes Werkzeug: {test_case['expected_primary'] or 'Kein Werkzeug'}")
            print("-" * 60)
            
            workflow_start = time.time()
            
            # 步骤1: LLM工具理解和单一策略制定
            strategy_prompt = f"""{self.system_prompt}

                        Folgende Frage soll beantwortet werden: "{test_case['question']}"

                        Analysieren Sie diese Frage und wählen Sie EIN optimales Werkzeug aus:
                        1. Bestimmen Sie die Art und Komplexität der Frage
                        2. Wählen Sie NUR EIN Werkzeug aus (oder keines bei irrelevanten Fragen)
                        3. Begründen Sie Ihre Werkzeugwahl eindeutig

                        Antwortformat:
                        **Frageanalyse**: [Art und Komplexität der Frage]
                        **Gewähltes Werkzeug**: [similarity_search ODER qa_search ODER kein Werkzeug]
                        **Begründung**: [Warum dieses spezifische Werkzeug optimal ist]
                        **Erwartetes Ergebnis**: [Welche Informationen Sie zu erhalten hoffen]"""

            try:
                strategy_response = self.llm.invoke(strategy_prompt)
                strategy_time = time.time() - workflow_start
                
                print(f"   📋 Strategieentwicklung ({strategy_time:.2f}s):")
                print(f"   {strategy_response.content}")
                print()
                
                # 从策略响应中提取单一工具使用计划
                content = strategy_response.content.lower()
                planned_tools = []
                
                # Check for specific tool mentions
                if "similarity_search" in content or "ähnlichkeit" in content or "semantisch" in content:
                    if "qa_search" not in content and "frage-antwort" not in content:  # Only if qa_search is not mentioned
                        planned_tools.append("similarity_search")
                elif "qa_search" in content or "frage-antwort" in content or "qa" in content:
                    if "similarity_search" not in content and "ähnlichkeit" not in content:  # Only if similarity_search is not mentioned
                        planned_tools.append("qa_search")
                elif "kein werkzeug" in content or "no tool" in content or "nicht relevant" in content:
                    planned_tools = []  # No tools
                
                # If both tools mentioned, prioritize based on keywords
                if len(planned_tools) == 0 and ("similarity_search" in content and "qa_search" in content):
                    if "analytisch" in content or "konzeptuell" in content or "existentialismus" in content:
                        planned_tools.append("similarity_search")
                    elif "faktenfrage" in content or "spezifisch" in content or "direkt" in content:
                        planned_tools.append("qa_search")
                
                print(f"   🎯 Geplante Werkzeugnutzung: {planned_tools if planned_tools else ['Kein Werkzeug']}")
                
                # 评估单工具选择准确性
                strategy_accuracy = 0
                expected_tool = test_case['expected_primary']
                
                if expected_tool is None:  # No tool expected
                    if len(planned_tools) == 0:
                        strategy_accuracy = 1.0
                        print(f"   ✅ Korrekt erkannt: Kein Werkzeug benötigt")
                    else:
                        print(f"   ❌ Werkzeug fälschlicherweise gewählt: {planned_tools}")
                elif len(planned_tools) == 1 and expected_tool in planned_tools:
                    strategy_accuracy = 1.0
                    print(f"   ✅ Korrektes Werkzeug gewählt: {expected_tool}")
                elif len(planned_tools) == 0 and expected_tool:
                    print(f"   ❌ Werkzeug nicht erkannt: {expected_tool} erwartet")
                elif len(planned_tools) > 1:
                    strategy_accuracy = 0.5  # Partial credit for mentioning correct tool but using multiple
                    print(f"   ⚠️  Mehrere Werkzeuge gewählt (Single Tool erwartet): {planned_tools}")
                else:
                    print(f"   ❌ Falsches Werkzeug gewählt: {planned_tools}, erwartet: {expected_tool}")
                
                print(f"   📊 Strategiegenauigkeit: {strategy_accuracy*100:.0f}%")
                
            except Exception as e:
                print(f"   ❌ Strategieentwicklung fehlgeschlagen: {e}")
                # For fallback, use the expected primary tool or no tool
                if test_case['expected_primary']:
                    planned_tools = [test_case['expected_primary']]
                    print(f"   🔄 Fallback: Erwartetes Werkzeug wird getestet: {test_case['expected_primary']}")
                else:
                    planned_tools = []
                    print(f"   🔄 Fallback: Kein Werkzeug wird getestet")
            
            # 步骤2: 执行单一工具调用
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
                        'execution_time': sim_time
                    }
                    print(f"   🔍 Ähnlichkeitssuche abgeschlossen ({sim_time:.2f}s): {tool_results['similarity_search']['result']['found_documents']} Dokumente")
                    
                except Exception as e:
                    print(f"   ❌ Ähnlichkeitssuche fehlgeschlagen: {e}")
            
            if "qa_search" in planned_tools:
                try:
                    qa_start = time.time()
                    qa_result = qa_tool._run(test_case['question'])
                    qa_time = time.time() - qa_start
                    
                    tool_results['qa_search'] = {
                        'result': json.loads(qa_result),
                        'execution_time': qa_time
                    }
                    print(f"   💡 Q&A-Suche abgeschlossen ({qa_time:.2f}s): {tool_results['qa_search']['result']['found_answers']} Antworten")
                    
                except Exception as e:
                    print(f"   ❌ Q&A-Suche fehlgeschlagen: {e}")
            
            # 步骤3: 单一工具结果处理和最终回答
            if tool_results:
                fusion_prompt = f"""{self.system_prompt}

                            Ursprüngliche Frage: "{test_case['question']}"

                            Suchergebnis des gewählten Werkzeugs:
                            """
                
                for tool_name, tool_data in tool_results.items():
                    fusion_prompt += f"\n{tool_name.upper()} Ergebnisse:\n{json.dumps(tool_data['result'], ensure_ascii=False, indent=2)}\n"
                
                fusion_prompt += f"""
                            Basierend auf dem obigen Suchergebnis geben Sie bitte eine präzise Antwort. Anforderungen:

                            1. Nutzen Sie die verfügbaren Informationen optimal
                            2. Heben Sie wichtige Inhalte hervor
                            3. Wahren Sie wissenschaftlichen Anspruch und Genauigkeit
                            4. Falls Informationen unzureichend sind, weisen Sie darauf hin
                            5. Geben Sie Quellenverweise an

                            Antwortformat:
                            **Antwort**: [Ihre vollständige Antwort]
                            **Informationsquelle**: [Welches Werkzeug und welche Dokumente wurden verwendet]
                            **Vertrauensgrad**: [0-100%, Ihr Vertrauen in die Antwort]
                            **Zusätzliche Anmerkungen**: [Falls weitere Erläuterungen nötig sind]"""

                fusion_start = time.time()
                try:
                    final_response = self.llm.invoke(fusion_prompt)
                    fusion_time = time.time() - fusion_start
                        
                    print(f"   📋 Ergebnisverarbeitung abgeschlossen ({fusion_time:.2f}s)")
                    print(f"   Endgültige Antwort:")
                    print("-" * 30)
                    print(final_response.content)
                    print("-" * 30)
                except Exception as e:
                    fusion_time = time.time() - fusion_start
                    print(f"   ❌ Ergebnisverarbeitung fehlgeschlagen ({fusion_time:.2f}s): {e}")
                    print("   📊 Werkzeugergebnis ohne LLM-Verarbeitung:")
                    for tool_name, tool_data in tool_results.items():
                        print(f"\n   {tool_name.upper()}:")
                        print(json.dumps(tool_data['result'], ensure_ascii=False, indent=2))
            else:
                # No tool was used - handle irrelevant questions
                if test_case['expected_primary'] is None:
                    print(f"   ✅ Korrekt: Kein Werkzeug verwendet für irrelevante Frage")
                    print(f"   📋 Antwort ohne Werkzeug:")
                    no_tool_response = f"Die Frage '{test_case['question']}' ist nicht relevant für Albert Camus' 'Der Fremde' und erfordert keine Suche in der Literaturdatenbank."
                    print(f"   {no_tool_response}")
                else:
                    print(f"   ❌ Fehler: Kein Werkzeug verwendet, aber {test_case['expected_primary']} erwartet")
                    print(f"   📋 Fallback-Antwort:")
                    fallback_response = f"Entschuldigung, ich konnte kein geeignetes Werkzeug für die Frage '{test_case['question']}' identifizieren."
                    print(f"   {fallback_response}")
                    
                    # 评估回答质量
                # response_content = final_response.content.lower()
                # quality_score = 0
                    
                # for criterion in test_case['evaluation_criteria']:
                    # if criterion.lower() in response_content:
                        # quality_score += 1
                    
                # quality_percentage = (quality_score / len(test_case['evaluation_criteria'])) * 100
                # print(f"   📊 Antwortqualitätsbewertung: {quality_percentage:.0f}% ({quality_score}/{len(test_case['evaluation_criteria'])} Kriterien)")
                    
            
            total_time = time.time() - workflow_start
            print(f"Gesamtverarbeitungszeit: {total_time:.2f}s")
    
    def run_comprehensive_test(self):
        print("START - Single Tool Strategy Test")
        print("="*80)
        print(f"CONFIG")
        print(f"   - Model: {self.model_name}")
        print(f"   - Strategy: Single Tool Selection")
        print(f"   - Tools: similarity_search OR qa_search (one at a time)")
        print(f"   - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        overall_start = time.time()
        
        self.test_integrated_understanding_and_workflow()
        
        total_time = time.time() - overall_start
        print(f"\nSingle Tool Strategy Test Completed! Gesamtzeit: {total_time:.2f} Sekunden")
        print("="*80)


if __name__ == "__main__":
    # Single Tool Strategy Testing - 测试不同模型的单工具选择能力
    models_to_test = [
        "mistral:7b-instruct-v0.3-q5_0",
        # "mistral-small:24b",
        # "qwen3:8b",
    ]
    
    for model in models_to_test:
        print(f"\n{'='*20} Testing model: {model} {'='*20}")
        tester = OptimizedSingleToolRAGTester(model)
        tester.run_comprehensive_test()
        
        if len(models_to_test) > 1:
            print(f"\n...")
            time.sleep(5)