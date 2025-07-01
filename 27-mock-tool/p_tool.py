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
    description: str = """语义相似性搜索工具。用于在乔伊斯《尤利西斯》相关文档中查找语义相似的内容片段。
    适用于：概念性问题、文学分析、主题探讨、人物研究等。
    输入：用户的查询问题（字符串格式）"""
    
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
                "content": "利奥波德·布卢姆是乔伊斯《尤利西斯》的主人公之一，他是一个中年犹太人，在都柏林从事广告业务。他的内心世界复杂而丰富，代表了现代都市人的孤独和思考。",
                "source": "ulysses_characters.pdf",
                "chapter": "第四章：迦勒底",
                "similarity": 0.95,
                "themes": ["人物分析", "现代性", "都市生活"]
            },
            {
                "content": "意识流是乔伊斯在《尤利西斯》中运用的重要文学技巧。通过展现人物内心的连续思维过程，乔伊斯打破了传统叙事结构，创造了一种全新的小说形式。",
                "source": "modernist_techniques.pdf", 
                "chapter": "文学技巧分析",
                "similarity": 0.88,
                "themes": ["意识流", "现代主义", "叙事技巧"]
            },
            {
                "content": "都柏林不仅是《尤利西斯》的背景，更是小说的一个重要'角色'。乔伊斯通过精确的地理描述，将都柏林的街道、建筑、文化融入到叙事中，创造了文学史上最详细的城市肖像。",
                "source": "dublin_setting.pdf",
                "chapter": "空间研究",
                "similarity": 0.82,
                "themes": ["都柏林", "空间叙事", "现实主义"]
            },
            {
                "content": "《尤利西斯》与荷马史诗《奥德赛》的对应关系是理解这部作品的关键。布卢姆对应奥德修斯，斯蒂芬对应忒勒马科斯，茉莉对应佩涅洛佩，这种对应关系贯穿全书。",
                "source": "homeric_parallels.pdf",
                "chapter": "互文性研究",
                "similarity": 0.90,
                "themes": ["互文性", "神话", "古典文学"]
            },
            {
                "content": "乔伊斯在《尤利西斯》中展现的女性形象复杂多样。茉莉·布卢姆的内心独白代表了女性意识的觉醒，同时也体现了乔伊斯对传统女性形象的挑战和重新定义。",
                "source": "female_characters.pdf",
                "chapter": "第十八章：佩涅洛佩",
                "similarity": 0.87,
                "themes": ["女性形象", "性别研究", "意识流"]
            }
        ]
        
        # 简化的语义匹配逻辑
        query_lower = query.lower()
        relevant_docs = []
        
        # 基于关键词和主题的匹配逻辑
        keyword_mapping = {
            "布卢姆": ["布卢姆", "主人公", "犹太人", "广告"],
            "意识流": ["意识流", "内心", "思维", "现代主义"],
            "都柏林": ["都柏林", "城市", "地理", "背景"],
            "荷马": ["荷马", "奥德赛", "神话", "古典"],
            "女性": ["茉莉", "女性", "性别", "佩涅洛佩"],
            "乔伊斯": ["乔伊斯", "作者", "现代主义", "文学"],
            "技巧": ["技巧", "叙事", "结构", "形式"],
            "主题": ["主题", "象征", "意义", "分析"]
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
    description: str = """问答搜索工具。用于查找《尤利西斯》相关的特定问题答案。
    适用于：直接的事实性问题、具体的文学细节、作品信息查询等。
    输入：用户的具体问题（字符串格式）"""
    
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
                "question": "《尤利西斯》的主人公是谁？",
                "answer": "《尤利西斯》有三个主要人物：利奥波德·布卢姆（Leopold Bloom）、斯蒂芬·迪达勒斯（Stephen Dedalus）和茉莉·布卢姆（Molly Bloom）。",
                "source": "ulysses_faq.pdf",
                "confidence": 0.95,
                "category": "人物"
            },
            {
                "question": "《尤利西斯》发生在哪一天？",
                "answer": "《尤利西斯》的故事发生在1904年6月16日，这一天被称为'布卢姆日'（Bloomsday）。",
                "source": "ulysses_timeline.pdf",
                "confidence": 0.98,
                "category": "情节"
            },
            {
                "question": "《尤利西斯》有多少章？",
                "answer": "《尤利西斯》共有18章，每章对应荷马史诗《奥德赛》的一个章节。",
                "source": "structure_analysis.pdf",
                "confidence": 0.92,
                "category": "结构"
            },
            {
                "question": "乔伊斯使用了什么写作技巧？",
                "answer": "乔伊斯在《尤利西斯》中使用了意识流、内心独白、自由间接叙述等现代主义写作技巧。",
                "source": "literary_techniques.pdf",
                "confidence": 0.88,
                "category": "技巧"
            },
            {
                "question": "《尤利西斯》的背景是哪里？",
                "answer": "《尤利西斯》的背景设定在爱尔兰都柏林，乔伊斯详细描述了都柏林的街道和地标。",
                "source": "setting_analysis.pdf",
                "confidence": 0.94,
                "category": "背景"
            },
            {
                "question": "为什么《尤利西斯》被禁？",
                "answer": "《尤利西斯》因其大胆的性描写和实验性的叙事风格，在多个国家被禁止出版，直到1960年代才在美国合法出版。",
                "source": "censorship_history.pdf",
                "confidence": 0.90,
                "category": "历史"
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
                "主人公|角色|人物": "人物",
                "时间|日期|哪天": "情节", 
                "章节|结构": "结构",
                "技巧|方法|手法": "技巧",
                "背景|地点|都柏林": "背景",
                "禁止|审查|banned": "历史"
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
                "question": "通用查询",
                "answer": "关于《尤利西斯》的具体信息，请参考相关文学研究资料。这是一部复杂的现代主义小说，需要深入的文学分析。",
                "source": "general_reference.pdf",
                "confidence": 0.3,
                "category": "通用",
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
        self.system_prompt = """你是一个专门研究詹姆斯·乔伊斯《尤利西斯》的智能助手。你有两个强大的工具：

1. **similarity_search**: 语义相似性搜索
   - 用于：概念性问题、文学分析、主题探讨、人物研究
   - 返回：相关的文档片段和分析内容

2. **qa_search**: 问答搜索  
   - 用于：具体的事实性问题、直接的信息查询
   - 返回：精确的答案和高置信度信息

**重要策略**：
- 对于复杂问题，建议同时使用两个工具以获得全面的信息
- 对于简单的事实查询，优先使用qa_search
- 对于分析性问题，优先使用similarity_search
- 始终基于工具返回的结果来构建你的回答"""

    def test_individual_tools(self):
        """
        测试1：单独工具功能测试
        目的：验证每个工具的基本功能是否正常
        """
        print("=== 测试1：单独工具功能验证 ===")
        print("设计思路：隔离测试每个工具，确保基础功能正常")
        
        similarity_tool = MockSimilaritySearchTool()
        qa_tool = MockQATool()
        
        test_queries = [
            "布卢姆是谁？",
            "《尤利西斯》使用了什么文学技巧？", 
            "都柏林在小说中的作用是什么？",
            "《尤利西斯》发生在哪一天？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 测试查询: {query}")
            print("-" * 40)
            
            # 测试相似性搜索工具
            try:
                start_time = time.time()
                sim_result = similarity_tool._run(query)
                sim_time = time.time() - start_time
                
                sim_data = json.loads(sim_result)
                print(f"✅ 相似性搜索: 找到 {sim_data['found_documents']} 个文档 ({sim_time:.2f}s)")
                
                # 显示第一个结果的简要信息
                if sim_data['documents']:
                    first_doc = sim_data['documents'][0]
                    print(f"   📄 最相关文档: {first_doc['source']} (相似度: {first_doc['similarity']:.2f})")
                
            except Exception as e:
                print(f"❌ 相似性搜索错误: {e}")
            
            # 测试问答搜索工具
            try:
                start_time = time.time()
                qa_result = qa_tool._run(query)
                qa_time = time.time() - start_time
                
                qa_data = json.loads(qa_result)
                print(f"✅ 问答搜索: 找到 {qa_data['found_answers']} 个答案 ({qa_time:.2f}s)")
                
                # 显示第一个答案的简要信息
                if qa_data['qa_pairs']:
                    first_qa = qa_data['qa_pairs'][0]
                    print(f"   💡 最佳答案置信度: {first_qa['confidence']:.2f}")
                
            except Exception as e:
                print(f"❌ 问答搜索错误: {e}")
    
    def test_llm_tool_understanding(self):
        """
        测试2：LLM工具理解能力测试
        目的：测试LLM是否理解每个工具的用途和适用场景
        """
        print("\n=== 测试2：LLM工具理解能力 ===")
        print("设计思路：测试LLM对工具功能的认知，不涉及实际调用")
        
        test_scenarios = [
            {
                "question": "谁是《尤利西斯》的主人公？",
                "type": "事实查询",
                "expected_primary": "qa_search",
                "expected_secondary": "similarity_search",
                "reasoning": "直接的事实性问题，应优先使用QA搜索"
            },
            {
                "question": "分析《尤利西斯》中意识流技巧的运用",
                "type": "分析性问题", 
                "expected_primary": "similarity_search",
                "expected_secondary": "qa_search",
                "reasoning": "需要深入分析，应优先使用语义搜索"
            },
            {
                "question": "《尤利西斯》与《奥德赛》的关系如何？",
                "type": "比较分析",
                "expected_primary": "similarity_search",
                "expected_secondary": "qa_search", 
                "reasoning": "复杂的文学比较，需要概念性搜索"
            },
            {
                "question": "《尤利西斯》有多少章？",
                "type": "具体信息",
                "expected_primary": "qa_search",
                "expected_secondary": None,
                "reasoning": "简单的数字信息，只需QA搜索"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. 场景测试: {scenario['type']}")
            print(f"   问题: {scenario['question']}")
            print(f"   预期策略: {scenario['expected_primary']} -> {scenario['expected_secondary']}")
            
            # 创建工具选择提示
            tool_selection_prompt = f"""作为《尤利西斯》研究助手，你需要为以下问题选择合适的工具：

问题: "{scenario['question']}"

可用工具:
1. similarity_search - 语义相似性搜索，适合概念性和分析性问题
2. qa_search - 问答搜索，适合具体的事实性问题

请分析这个问题的性质，并选择最合适的工具策略。

回答格式:
主要工具: [工具名称]
辅助工具: [工具名称或"无"]
选择理由: [简要说明]
问题类型: [事实性/分析性/比较性/其他]"""

            try:
                start_time = time.time()
                response = self.llm.invoke(tool_selection_prompt)
                response_time = time.time() - start_time
                
                print(f"   🤖 LLM响应 ({response_time:.2f}s):")
                print(f"   {response.content}")
                
                # 简单的响应分析
                content_lower = response.content.lower()
                mentioned_tools = []
                
                if "similarity_search" in content_lower:
                    mentioned_tools.append("similarity_search")
                if "qa_search" in content_lower:
                    mentioned_tools.append("qa_search")
                
                # 评估工具选择的准确性
                if scenario['expected_primary'] in mentioned_tools:
                    print(f"   ✅ 正确识别主要工具: {scenario['expected_primary']}")
                else:
                    print(f"   ❌ 未能识别主要工具: {scenario['expected_primary']}")
                
                if scenario['expected_secondary'] and scenario['expected_secondary'] in mentioned_tools:
                    print(f"   ✅ 正确识别辅助工具: {scenario['expected_secondary']}")
                elif not scenario['expected_secondary'] and len(mentioned_tools) == 1:
                    print(f"   ✅ 正确识别仅需一个工具")
                
            except Exception as e:
                print(f"   ❌ LLM响应错误: {e}")
    
    def test_integrated_workflow(self):
        """
        测试3：集成工作流测试
        目的：测试完整的RAG工作流程
        """
        print("\n=== 测试3：集成工作流程 ===")
        print("设计思路：模拟真实RAG场景，测试工具调用、结果融合和最终输出")
        
        complex_questions = [
            {
                "question": "《尤利西斯》中布卢姆这个角色有什么特点？",
                "complexity": "中等",
                "expected_tools": ["similarity_search", "qa_search"],
                "evaluation_criteria": ["人物特征", "文学分析", "具体信息"]
            },
            {
                "question": "乔伊斯在《尤利西斯》中如何运用意识流技巧？",
                "complexity": "高",
                "expected_tools": ["similarity_search"],
                "evaluation_criteria": ["技巧分析", "文学理论", "具体例子"]
            },
            {
                "question": "《尤利西斯》的故事发生在什么时候？",
                "complexity": "简单",
                "expected_tools": ["qa_search"],
                "evaluation_criteria": ["准确的时间信息"]
            }
        ]
        
        for i, test_case in enumerate(complex_questions, 1):
            print(f"\n{i}. 集成测试: {test_case['complexity']}复杂度")
            print(f"   问题: {test_case['question']}")
            print("-" * 50)
            
            workflow_start = time.time()
            
            # 步骤1: LLM决策工具使用策略
            strategy_prompt = f"""{self.system_prompt}

现在需要回答这个问题: "{test_case['question']}"

请制定工具使用策略:
1. 分析问题类型和复杂度
2. 选择需要使用的工具
3. 说明工具使用的优先级和权重

回答格式:
问题分析: [问题的性质和复杂度]
工具策略: [要使用的工具列表]
使用顺序: [先用哪个工具，为什么]
预期结果: [期望获得什么信息]"""

            try:
                strategy_response = self.llm.invoke(strategy_prompt)
                strategy_time = time.time() - workflow_start
                
                print(f"   📋 策略制定 ({strategy_time:.2f}s):")
                print(f"   {strategy_response.content[:300]}...")
                
                # 从策略响应中提取工具使用计划
                content = strategy_response.content.lower()
                planned_tools = []
                
                if "similarity_search" in content or "语义" in content or "相似" in content:
                    planned_tools.append("similarity_search")
                if "qa_search" in content or "问答" in content or "qa" in content:
                    planned_tools.append("qa_search")
                
                print(f"   🎯 计划使用工具: {planned_tools}")
                
            except Exception as e:
                print(f"   ❌ 策略制定失败: {e}")
                planned_tools = test_case['expected_tools']  # 使用预期工具作为fallback
            
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
                    print(f"   🔍 语义搜索完成 ({sim_time:.2f}s): {tool_results['similarity_search']['result']['found_documents']} 文档")
                    
                except Exception as e:
                    print(f"   ❌ 语义搜索失败: {e}")
            
            if "qa_search" in planned_tools:
                try:
                    qa_start = time.time()
                    qa_result = qa_tool._run(test_case['question'])
                    qa_time = time.time() - qa_start
                    
                    tool_results['qa_search'] = {
                        'result': json.loads(qa_result),
                        'execution_time': qa_time
                    }
                    print(f"   💡 问答搜索完成 ({qa_time:.2f}s): {tool_results['qa_search']['result']['found_answers']} 答案")
                    
                except Exception as e:
                    print(f"   ❌ 问答搜索失败: {e}")
            
            # 步骤3: 结果融合和最终回答
            if tool_results:
                fusion_prompt = f"""{self.system_prompt}

原始问题: "{test_case['question']}"

工具搜索结果:
"""
                
                for tool_name, tool_data in tool_results.items():
                    fusion_prompt += f"\n{tool_name.upper()} 结果:\n{json.dumps(tool_data['result'], ensure_ascii=False, indent=2)}\n"
                
                fusion_prompt += f"""
基于以上搜索结果，请提供一个全面、准确的回答。要求：

1. 综合所有相关信息
2. 突出重点内容
3. 保持学术性和准确性
4. 如果信息不足，请说明
5. 提供信息来源引用

回答格式：
**回答**: [你的完整回答]
**信息来源**: [使用了哪些工具和文档]
**置信度**: [0-100%，你对答案的信心程度]
**补充说明**: [如有需要的额外说明]"""

                try:
                    fusion_start = time.time()
                    final_response = self.llm.invoke(fusion_prompt)
                    fusion_time = time.time() - fusion_start
                    
                    print(f"   🔄 结果融合完成 ({fusion_time:.2f}s)")
                    print(f"   📝 最终回答:")
                    print("-" * 30)
                    print(final_response.content)
                    print("-" * 30)
                    
                    # 评估回答质量
                    response_content = final_response.content.lower()
                    quality_score = 0
                    
                    for criterion in test_case['evaluation_criteria']:
                        if criterion.lower() in response_content:
                            quality_score += 1
                    
                    quality_percentage = (quality_score / len(test_case['evaluation_criteria'])) * 100
                    print(f"   📊 回答质量评分: {quality_percentage:.0f}% ({quality_score}/{len(test_case['evaluation_criteria'])} 标准)")
                    
                except Exception as e:
                    print(f"   ❌ 结果融合失败: {e}")
            
            total_time = time.time() - workflow_start
            print(f"   ⏱️ 总处理时间: {total_time:.2f}s")
    
    def test_edge_cases(self):
        """
        测试4：边界情况测试
        目的：测试系统在异常情况下的表现
        """
        print("\n=== 测试4：边界情况和异常处理 ===")
        print("设计思路：测试系统的鲁棒性和错误处理能力")
        
        edge_cases = [
            {
                "name": "空查询",
                "query": "",
                "expected_behavior": "graceful handling"
            },
            {
                "name": "超长查询",
                "query": "关于乔伊斯尤利西斯的" * 50,
                "expected_behavior": "truncation or error handling"
            },
            {
                "name": "无关查询",
                "query": "今天天气怎么样？",
                "expected_behavior": "redirect to relevant topics"
            },
            {
                "name": "模糊查询",
                "query": "那个那个那个是什么？",
                "expected_behavior": "clarification request"
            },
            {
                "name": "专业术语查询",
                "query": "《尤利西斯》中的epiphany概念是什么？",
                "expected_behavior": "technical explanation"
            }
        ]
        
        similarity_tool = MockSimilaritySearchTool()
        qa_tool = MockQATool()
        
        for i, case in enumerate(edge_cases, 1):
            print(f"\n{i}. 边界测试: {case['name']}")
            print(f"   查询: '{case['query']}'")
            print(f"   预期行为: {case['expected_behavior']}")
            
            # 测试工具对边界情况的处理
            try:
                sim_result = similarity_tool._run(case['query'])
                sim_data = json.loads(sim_result)
                print(f"   ✅ 语义搜索处理: {sim_data['found_documents']} 文档")
            except Exception as e:
                print(f"   ⚠️ 语义搜索异常: {e}")
            
            try:
                qa_result = qa_tool._run(case['query'])
                qa_data = json.loads(qa_result)
                print(f"   ✅ 问答搜索处理: {qa_data['found_answers']} 答案")
            except Exception as e:
                print(f"   ⚠️ 问答搜索异常: {e}")
            
            # 测试LLM对边界情况的处理
            if case['query']:  # 非空查询才测试LLM
                edge_prompt = f"""{self.system_prompt}

用户查询: "{case['query']}"

请分析这个查询并给出适当的响应。如果查询不合适或不相关，请礼貌地引导用户提出关于《尤利西斯》的问题。"""

                try:
                    llm_response = self.llm.invoke(edge_prompt)
                    print(f"   🤖 LLM处理: {llm_response.content[:100]}...")
                except Exception as e:
                    print(f"   ❌ LLM处理失败: {e}")
    
    def generate_performance_report(self):
        """
        生成性能报告
        """
        print("\n=== 性能和能力总结报告 ===")
        print("基于以上测试结果的综合分析")
        
        report_prompt = f"""基于刚才进行的RAG工具集成测试，请生成一份关于{self.model_name}模型性能的总结报告。

报告应包括：
1. **工具理解能力**: 模型对工具功能和使用场景的理解程度
2. **策略制定能力**: 模型选择合适工具的能力
3. **结果整合能力**: 模型融合多个工具结果的能力  
4. **异常处理能力**: 模型处理边界情况的能力
5. **整体RAG适用性**: 该模型是否适合作为RAG系统的核心

请给出客观的评估和建议。"""

        try:
            report_response = self.llm.invoke(report_prompt)
            print("\n" + "="*60)
            print("📊 MODEL PERFORMANCE REPORT")
            print("="*60)
            print(report_response.content)
            print("="*60)
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
    
    def run_comprehensive_test(self):
        """
        运行完整的综合测试套件
        """
        print("🚀 启动《尤利西斯》RAG工具集成测试")
        print("="*80)
        print(f"📋 测试配置:")
        print(f"   - 模型: {self.model_name}")
        print(f"   - 主题: 詹姆斯·乔伊斯《尤利西斯》")
        print(f"   - 工具: 语义搜索 + 问答搜索")
        print(f"   - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        overall_start = time.time()
        
        try:
            # 运行所有测试模块
            self.test_individual_tools()
            self.test_llm_tool_understanding() 
            self.test_integrated_workflow()
            self.test_edge_cases()
            self.generate_performance_report()
            
        except KeyboardInterrupt:
            print("\n⚠️ 测试被用户中断")
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
        
        finally:
            total_time = time.time() - overall_start
            print(f"\n🏁 测试完成")
            print("="*80)
            print(f"📊 测试统计:")
            print(f"   - 总耗时: {total_time:.2f} 秒")
            print(f"   - 平均每项测试: {total_time/4:.2f} 秒")
            print(f"   - 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # 给出使用建议
            print("\n💡 使用建议:")
            print("1. 如果模型在工具理解测试中表现良好，说明适合RAG场景")
            print("2. 如果策略制定能力较弱，可能需要更详细的提示工程")
            print("3. 如果结果整合有问题，考虑简化融合逻辑或使用后处理")
            print("4. 注意观察模型的响应时间和资源消耗")


if __name__ == "__main__":
    # 可以测试不同的模型
    models_to_test = [
        "mistral:7b-instruct-v0.3-q5_0",
        # "llama2:7b-instruct",  # 取消注释以测试其他模型
        # "codellama:7b-instruct",
    ]
    
    for model in models_to_test:
        print(f"\n{'='*20} 测试模型: {model} {'='*20}")
        tester = EnhancedRAGToolIntegrationTester(model)
        tester.run_comprehensive_test()
        
        if len(models_to_test) > 1:
            print(f"\n⏳ 等待5秒后测试下一个模型...")
            time.sleep(5)