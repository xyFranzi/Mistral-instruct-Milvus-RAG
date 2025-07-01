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
from pymilvus import connections, utility
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

MILVUS_HOST = "milvus"
MILVUS_PORT = "19530"
DEFAULT_COLLECTION = "der_fremde_docs"
EMBEDDING_MODEL = "aari1995/German_Semantic_V3b"

class MilvusSimilaritySearchTool(BaseTool):
    """
    真实 Milvus 语义相似性搜索工具
    使用 LangChain Milvus 封装，直接在 Milvus 向量库上运行向量检索
    """
    name: str = "similarity_search"
    description: str = (
        "Führe semantische Ähnlichkeitssuche in der Milvus-Kollektion durch. "
        "Eingabe: Query-String; Ausgabe: JSON mit Dokumentfragmenten und Scores."
    )

    vs: Any = None
    k: int = 3

    def __init__(self,
                 collection_name: str = "der_fremde_docs",
                 host: str = "milvus",
                 port: str = "19530",
                 embedding_model: str = "aari1995/German_Semantic_V3b",
                 embedding_device: str = "cpu",
                 k: int = 3):
        super().__init__()
        # 1) 连接 Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        # 2) 检查集合是否存在
        cols = utility.list_collections()
        if collection_name not in cols:
            raise ValueError(f"Milvus collection '{collection_name}' not found. Available: {cols}")

        # 3) 加载 Embeddings
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": embedding_device}
        )
        # 4) 初始化向量存储
        self.vs = Milvus(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args={"host": host, "port": port}
        )
        self.k = k

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        调用 Milvus 向量库进行相似性检索，格式化返回 JSON 字符串
        """
        # 执行检索
        results: List[tuple[Document, float]] = self.vs.similarity_search_with_score(query, k=self.k)

        docs_out: List[Dict[str, Any]] = []
        for doc, score in results:
            docs_out.append({
                "score": float(score),
                "source": doc.metadata.get("source", None),
                "content": doc.page_content
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
            base_url="http://ollama:11434"  
        )
        self.tools = [MilvusSimilaritySearchTool(collection_name="der_fremde_docs"), MockQATool()]
        self.model_name = model_name
        
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
    {
        "question": "An welchem Tag beginnt die Handlung von 'Der Fremde' und warum ist dieser Tag wichtig?",
        "complexity": "niedrig",
        "type": "Faktenfrage mit Kontext",
        "expected_primary": "qa_search",
        "expected_secondary": "similarity_search",
        "reasoning": "Direkte Faktenfrage mit Kontextbezug zum berühmten ersten Satz"
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
                
                print(f"   📊 Strategiegenauigkeit: {strategy_accuracy*100:.0f}%")
                
            except Exception as e:
                print(f"   ❌ Strategieentwicklung fehlgeschlagen: {e}")
                planned_tools = [test_case['expected_primary']]  # 使用预期工具作为fallback
            
            # 步骤2: 执行工具调用
            tool_results = {}
            
            similarity_tool = MilvusSimilaritySearchTool(collection_name="der_fremde_docs")
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
            
            # 步骤3: 结果融合和最终回答
            if tool_results:
                fusion_prompt = f"""{self.system_prompt}

                            Ursprüngliche Frage: "{test_case['question']}"

                            Suchergebnisse der Werkzeuge:
                            """
                
                for tool_name, tool_data in tool_results.items():
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
                final_response = self.llm.invoke(fusion_prompt)
                fusion_time = time.time() - fusion_start
                    
                print(f"Ergebnisfusion abgeschlossen ({fusion_time:.2f}s)")
                print(f"Endgültige Antwort:")
                print("-" * 30)
                print(final_response.content)
                print("-" * 30)
                    
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
        print(f"\nTest done! Gesamtzeit: {total_time:.2f} Sekunden")
        print("="*80)


if __name__ == "__main__":
    # 可以测试不同的模型
    models_to_test = [
        "mistral:7b-instruct-v0.3-q5_0",
        # "xxx",  # 取消注释以测试其他模型
        # "xxx",
    ]
    
    for model in models_to_test:
        print(f"\n{'='*20} Testing model: {model} {'='*20}")
        tester = EnhancedRAGToolIntegrationTester(model)
        tester.run_comprehensive_test()
        
        if len(models_to_test) > 1:
            print(f"\n...")
            time.sleep(5)