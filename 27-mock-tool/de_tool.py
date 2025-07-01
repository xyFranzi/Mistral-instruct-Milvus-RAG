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

class MockSimilaritySearchTool(BaseTool):
    """
    模拟语义相似性搜索工具
    设计思路：
    1. 模拟真实的Milvus向量搜索行为
    2. 基于关键词匹配（简化的语义搜索）
    3. 返回带有相似度分数的文档片段
    4. 测试LLM是否能正确理解和使用语义搜索
    """
    name: str = "similarity_search"
    description: str = """Semantisches Ähnlichkeitssuchwerkzeug. Wird verwendet, um semantisch ähnliche Textpassagen aus Dokumenten über James Joyces "Ulysses" zu finden.
    Anwendbar für: Begriffsfragen, literarische Analysen, Themenuntersuchungen, Charakterstudien usw.
    Eingabe: Die Benutzeranfrage als String."""
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        模拟语义搜索过程
        1. 接收查询
        2. 基于关键词匹配相关文档
        3. 返回结构化结果
        """
        # 模拟《尤利西斯》相关文档数据库
        mock_documents = [
            {
                "content": "Leopold Bloom ist einer der Protagonisten in Joyce' 'Ulysses'. Er ist ein jüdischer Mann mittleren Alters, der in Dublin im Werbegeschäft tätig ist. Seine innere Welt ist komplex und reich, repräsentiert die Einsamkeit und das Nachdenken des modernen Stadtmenschen.",
                "source": "ulysses_charaktere.pdf",
                "chapter": "Kapitel 4: Kalypso",
                "similarity": 0.95,
                "themes": ["Charakteranalyse", "Modernität", "Stadtleben"]
            },
            {
                "content": "Der Bewusstseinsstrom ist eine wichtige literarische Technik, die Joyce in 'Ulysses' anwendet. Durch die Darstellung der kontinuierlichen Denkprozesse der Charaktere durchbricht Joyce traditionelle Erzählstrukturen und schafft eine völlig neue Romanform.",
                "source": "modernistische_techniken.pdf", 
                "chapter": "Analyse literarischer Techniken",
                "similarity": 0.88,
                "themes": ["Bewusstseinsstrom", "Modernismus", "Erzähltechnik"]
            },
            {
                "content": "Dublin ist nicht nur der Schauplatz von 'Ulysses', sondern auch ein wichtiger 'Charakter' des Romans. Joyce schafft durch präzise geografische Beschreibungen und die Integration von Dublins Straßen, Gebäuden und Kultur in die Erzählung das detaillierteste Stadtporträt der Literaturgeschichte.",
                "source": "dublin_schauplatz.pdf",
                "chapter": "Raumstudien",
                "similarity": 0.82,
                "themes": ["Dublin", "Raumerzählung", "Realismus"]
            },
            {
                "content": "Die Entsprechungen zwischen 'Ulysses' und Homers Epos 'Odyssee' sind der Schlüssel zum Verständnis dieses Werkes. Bloom entspricht Odysseus, Stephen entspricht Telemachos, Molly entspricht Penelope - diese Entsprechungen durchziehen das gesamte Buch.",
                "source": "homerische_parallelen.pdf",
                "chapter": "Intertextualitätsstudien",
                "similarity": 0.90,
                "themes": ["Intertextualität", "Mythologie", "Klassische Literatur"]
            },
            {
                "content": "Joyce zeigt in 'Ulysses' komplexe und vielfältige Frauenfiguren. Molly Blooms innerer Monolog repräsentiert das Erwachen des weiblichen Bewusstseins und verkörpert Joyce' Herausforderung und Neudefinition traditioneller Frauenbilder.",
                "source": "weibliche_charaktere.pdf",
                "chapter": "Kapitel 18: Penelope",
                "similarity": 0.87,
                "themes": ["Frauenfiguren", "Geschlechterstudien", "Bewusstseinsstrom"]
            }
        ]
        
        # 简化的语义匹配逻辑
        query_lower = query.lower()
        relevant_docs = []
        
        # 基于关键词和主题的匹配逻辑
        keyword_mapping = {
            "bloom": ["bloom", "protagonist", "jüdisch", "werbung"],
            "bewusstseinsstrom": ["bewusstseinsstrom", "innere", "denken", "modernismus"],
            "dublin": ["dublin", "stadt", "geografisch", "schauplatz"],
            "homer": ["homer", "odyssee", "mythologie", "klassisch"],
            "frauen": ["molly", "frauen", "geschlecht", "penelope"],
            "joyce": ["joyce", "autor", "modernismus", "literatur"],
            "technik": ["technik", "erzählung", "struktur", "form"],
            "thema": ["thema", "symbol", "bedeutung", "analyse"]
        }
        
        # 匹配相关文档
        for doc in mock_documents:
            doc_score = 0
            
            # 检查查询与文档内容的匹配度
            for keyword, variations in keyword_mapping.items():
                if keyword in query_lower:
                    for variation in variations:
                        if variation in doc["content"].lower():
                            doc_score += 0.2
                            break
            
            # 检查主题匹配
            for theme in doc["themes"]:
                if theme in query_lower or any(word in theme.lower() for word in query_lower.split()):
                    doc_score += 0.1
            
            # 如果有匹配，添加到结果中
            if doc_score > 0:
                doc_copy = doc.copy()
                doc_copy["similarity"] = min(doc_copy["similarity"], 0.95)  # 调整相似度
                relevant_docs.append(doc_copy)
        
        # 如果没有特定匹配，返回最相关的文档
        if not relevant_docs:
            relevant_docs = mock_documents[:2]
        
        # 按相似度排序
        relevant_docs.sort(key=lambda x: x["similarity"], reverse=True)
        relevant_docs = relevant_docs[:3]  # 最多返回3个文档
        
        result = {
            "tool": "similarity_search",
            "query": query,
            "found_documents": len(relevant_docs),
            "documents": relevant_docs,
            "search_type": "semantic_similarity"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)

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
    description: str = """Frage-Antwort-Suchwerkzeug. Wird verwendet, um präzise Antworten auf spezifische Fragen zu James Joyces "Ulysses" zu finden.
    Anwendbar für: Faktische Fragen, literarische Details, Werkbezogene Informationen usw.
    Eingabe: Die spezifische Benutzerfrage als String."""
    
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
        # 模拟《尤利西斯》相关问答数据库
        mock_qa_pairs = [
            {
                "question": "Wer sind die Protagonisten von 'Ulysses'?",
                "answer": "'Ulysses' hat drei Hauptfiguren: Leopold Bloom, Stephen Dedalus und Molly Bloom.",
                "source": "ulysses_faq.pdf",
                "confidence": 0.95,
                "category": "Charaktere"
            },
            {
                "question": "An welchem Tag spielt 'Ulysses'?",
                "answer": "Die Handlung von 'Ulysses' spielt am 16. Juni 1904, dieser Tag wird als 'Bloomsday' bezeichnet.",
                "source": "ulysses_zeitlinie.pdf",
                "confidence": 0.98,
                "category": "Handlung"
            },
            {
                "question": "Wie viele Kapitel hat 'Ulysses'?",
                "answer": "'Ulysses' besteht aus 18 Kapiteln, jedes entspricht einem Abschnitt von Homers 'Odyssee'.",
                "source": "strukturanalyse.pdf",
                "confidence": 0.92,
                "category": "Struktur"
            },
            {
                "question": "Welche Schreibtechniken verwendet Joyce?",
                "answer": "Joyce verwendet in 'Ulysses' Bewusstseinsstrom, innere Monologe und freie indirekte Rede als modernistische Schreibtechniken.",
                "source": "literarische_techniken.pdf",
                "confidence": 0.88,
                "category": "Technik"
            },
            {
                "question": "Wo spielt 'Ulysses'?",
                "answer": "'Ulysses' spielt in Dublin, Irland. Joyce beschreibt detailliert die Straßen und Wahrzeichen Dublins.",
                "source": "schauplatz_analyse.pdf",
                "confidence": 0.94,
                "category": "Schauplatz"
            },
            {
                "question": "Warum wurde 'Ulysses' verboten?",
                "answer": "'Ulysses' wurde wegen seiner gewagten sexuellen Beschreibungen und experimentellen Erzählweise in mehreren Ländern verboten, bis es in den 1960er Jahren in den USA legal veröffentlicht wurde.",
                "source": "zensur_geschichte.pdf",
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
                "protagonist|charakter|figur": "Charaktere",
                "zeit|datum|tag": "Handlung", 
                "kapitel|struktur": "Struktur",
                "technik|methode|stil": "Technik",
                "schauplatz|ort|dublin": "Schauplatz",
                "verbot|zensur|banned": "Geschichte"
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
                "answer": "Für spezifische Informationen zu 'Ulysses' konsultieren Sie bitte entsprechende Literaturforschung. Dies ist ein komplexer modernistischer Roman, der eingehende literarische Analyse erfordert.",
                "source": "allgemeine_referenz.pdf",
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
        )
        self.tools = [MockSimilaritySearchTool(), MockQATool()]
        self.model_name = model_name
        
        # 改进的系统提示
        self.system_prompt = """Sie sind ein intelligenter Assistent, der sich auf James Joyce' "Ulysses" spezialisiert hat. Ihnen stehen zwei leistungsstarke Werkzeuge zur Verfügung:


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
                "question": "Wer ist Leopold Bloom und welche Rolle spielt er in 'Ulysses'?",
                "complexity": "mittel",
                "type": "Charakter-Faktenfrage",
                "expected_primary": "qa_search",
                "expected_secondary": "similarity_search",
                "reasoning": "Kombiniert Faktenfrage mit Charakteranalyse"
            },
            {
                "question": "Wie setzt Joyce den Bewusstseinsstrom in 'Ulysses' ein und welche Wirkung erzielt er damit?",
                "complexity": "hoch",
                "type": "Analytische Frage",
                "expected_primary": "similarity_search",
                "expected_secondary": "qa_search",
                "reasoning": "Erfordert tiefgehende literarische Analyse"
            },
            {
                "question": "An welchem Tag spielt 'Ulysses' und warum ist dieser Tag bedeutsam?",
                "complexity": "niedrig",
                "type": "Faktenfrage mit Kontext",
                "expected_primary": "qa_search",
                "expected_secondary": "similarity_search",
                "reasoning": "Direkte Faktenfrage mit zusätzlichem Kontext"
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
            
            similarity_tool = MockSimilaritySearchTool()
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