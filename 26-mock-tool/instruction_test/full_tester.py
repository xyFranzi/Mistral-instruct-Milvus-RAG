from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json
import time
from typing import List, Dict, Tuple
from collections import Counter

class RAGLLMTester:
    def __init__(self, model_name: str = "mistral:7b-instruct-v0.3-q5_0"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,  # set temperature to 0 for consistency in responses
        )
        self.results = []
        
    def test_tool_selection(self) -> Dict:
        """测试工具选择能力"""
        print("=== 测试1: 工具选择能力 ===")
        
        # 定义测试用例
        test_cases = [
            # (问题, 期望的工具, 问题类型)
            ("Was ist künstliche Intelligenz?", "SimilaritySearch", "概念定义"),
            ("Was ist maschinelles Lernen?", "SimilaritySearch", "概念定义"), 
            ("Erklären Sie Deep Learning", "SimilaritySearch", "概念解释"),
            ("Wie installiere ich Python?", "QA", "具体步骤"),
            ("Was sind die Systemanforderungen?", "QA", "具体信息"),
            ("Wo finde ich die Dokumentation?", "QA", "具体位置"),
            ("Was ist der Unterschied zwischen AI und ML?", "SimilaritySearch", "概念比较"),
            ("Wie löse ich Fehler XYZ?", "QA", "问题解决"),
        ]
        
        system_prompt = """Du bist ein intelligenter Assistent für ein RAG-System. Du hast Zugang zu zwei Werkzeugen:

        1. SimilaritySearch: Für semantische Suche in Dokumenten. Verwende dies für:
        - Konzeptuelle Fragen ("Was ist...?")
        - Erklärungen ("Wie funktioniert...?") 
        - Vergleiche ("Unterschied zwischen...")
        - Allgemeine Wissensfragen

        2. QA: Für spezifische Fragen mit direkten Antworten. Verwende dies für:
        - Spezifische Anweisungen ("Wie installiere ich...?")
        - Konkrete Informationen ("Wo finde ich...?")
        - Problemlösungen ("Wie löse ich...?")
        - Faktische Details

        Antworte nur mit dem Namen des Werkzeugs: "SimilaritySearch" oder "QA"."""

        results = []
        for question, expected, question_type in test_cases:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Frage: {question}")
            ]
            
            response = self.llm.invoke(messages)
            actual = response.content.strip()
            
            # 清理输出（移除可能的前导空格）
            if actual.startswith(' '):
                actual = actual[1:]
                
            correct = actual == expected
            results.append({
                'question': question,
                'expected': expected,
                'actual': actual,
                'correct': correct,
                'question_type': question_type,
                'tokens_used': response.usage_metadata['total_tokens'] if hasattr(response, 'usage_metadata') else 0
            })
            
            status = "✅" if correct else "❌"
            print(f"{status} {question}")
            print(f"   期望: {expected} | 实际: {actual} | 类型: {question_type}")
            print()
            
        return self._analyze_tool_selection_results(results)
    
    def test_instruction_following(self) -> Dict:
        """测试指令遵循能力"""
        print("=== 测试2: 指令遵循能力 ===")
        
        test_cases = [
            {
                'instruction': "Antworte nur mit 'JA' oder 'NEIN'",
                'question': "Ist Python eine Programmiersprache?",
                'valid_responses': ['JA', 'NEIN'],
                'expected': 'JA'
            },
            {
                'instruction': "Antworte nur mit einer Zahl zwischen 1-5",
                'question': "Wie schwer ist Python zu lernen? (1=sehr leicht, 5=sehr schwer)",
                'valid_responses': ['1', '2', '3', '4', '5'],
                'expected': None  # 主观问题，只要格式正确
            },
            {
                'instruction': "Antworte nur mit dem Werkzeugnamen",
                'question': "Für die Frage 'Was ist AI?' - SimilaritySearch oder QA?",
                'valid_responses': ['SimilaritySearch', 'QA'],
                'expected': 'SimilaritySearch'
            }
        ]
        
        results = []
        for test_case in test_cases:
            prompt = f"{test_case['instruction']}\n\nFrage: {test_case['question']}"
            response = self.llm.invoke(prompt)
            actual = response.content.strip()
            
            if actual.startswith(' '):
                actual = actual[1:]
            
            follows_format = actual in test_case['valid_responses']
            correct_answer = test_case['expected'] is None or actual == test_case['expected']
            
            results.append({
                'instruction': test_case['instruction'],
                'question': test_case['question'],
                'expected': test_case['expected'],
                'actual': actual,
                'follows_format': follows_format,
                'correct_answer': correct_answer,
                'tokens_used': response.usage_metadata['total_tokens'] if hasattr(response, 'usage_metadata') else 0
            })
            
            status = "✅" if follows_format and correct_answer else "❌"
            print(f"{status} {test_case['instruction']}")
            print(f"   问题: {test_case['question']}")
            print(f"   期望: {test_case['expected']} | 实际: {actual}")
            print(f"   格式正确: {follows_format} | 答案正确: {correct_answer}")
            print()
            
        return self._analyze_instruction_following_results(results)
    
    def test_consistency(self, num_runs: int = 5) -> Dict:
        """测试一致性"""
        print(f"=== 测试3: 一致性测试 ({num_runs}次运行) ===")
        
        test_prompt = """Du hast zwei Werkzeuge:
        1. SimilaritySearch: für konzeptuelle Fragen
        2. QA: für spezifische Antworten

        Frage: "Was ist künstliche Intelligenz?"
        Antworte nur mit dem Werkzeugnamen."""
        
        results = []
        for i in range(num_runs):
            response = self.llm.invoke(test_prompt)
            actual = response.content.strip()
            if actual.startswith(' '):
                actual = actual[1:]
            results.append(actual)
            print(f"运行 {i+1}: {actual}")
        
        # 分析一致性
        counter = Counter(results)
        most_common = counter.most_common(1)[0]
        consistency_rate = most_common[1] / num_runs
        
        print(f"\n结果分布: {dict(counter)}")
        print(f"一致性率: {consistency_rate:.2%}")
        
        return {
            'results': results,
            'consistency_rate': consistency_rate,
            'most_common_answer': most_common[0]
        }
    
    def _analyze_tool_selection_results(self, results: List[Dict]) -> Dict:
        """分析工具选择结果"""
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total
        
        # 按问题类型分析
        by_type = {}
        for result in results:
            q_type = result['question_type']
            if q_type not in by_type:
                by_type[q_type] = {'correct': 0, 'total': 0}
            by_type[q_type]['total'] += 1
            if result['correct']:
                by_type[q_type]['correct'] += 1
        
        print(f"总体准确率: {accuracy:.2%} ({correct}/{total})")
        print("\n按问题类型分析:")
        for q_type, stats in by_type.items():
            type_accuracy = stats['correct'] / stats['total']
            print(f"  {q_type}: {type_accuracy:.2%} ({stats['correct']}/{stats['total']})")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'by_type': by_type,
            'details': results
        }
    
    def _analyze_instruction_following_results(self, results: List[Dict]) -> Dict:
        """分析指令遵循结果"""
        total = len(results)
        format_correct = sum(1 for r in results if r['follows_format'])
        answer_correct = sum(1 for r in results if r['correct_answer'])
        
        format_rate = format_correct / total
        answer_rate = answer_correct / total
        
        print(f"格式遵循率: {format_rate:.2%} ({format_correct}/{total})")
        print(f"答案正确率: {answer_rate:.2%} ({answer_correct}/{total})")
        
        return {
            'format_rate': format_rate,
            'answer_rate': answer_rate,
            'format_correct': format_correct,
            'answer_correct': answer_correct,
            'total': total,
            'details': results
        }
    
    def run_full_test(self) -> Dict:
        """运行完整测试套件"""
        print("开始RAG LLM替代方案测试...")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行所有测试
        tool_results = self.test_tool_selection()
        instruction_results = self.test_instruction_following()
        consistency_results = self.test_consistency()
        
        end_time = time.time()
        
        # # 综合评估
        # overall_score = self._calculate_overall_score(
        #     tool_results, instruction_results, consistency_results
        # )
        
        final_results = {
            # 'overall_score': overall_score,
            'tool_selection': tool_results,
            'instruction_following': instruction_results,
            'consistency': consistency_results,
            'test_duration': end_time - start_time
        }
        
        self._print_final_report(final_results)
        return final_results
    
    # def _calculate_overall_score(self, tool_results, instruction_results, consistency_results) -> float:
    #     """计算综合评分"""
    #     # 权重分配
    #     tool_weight = 0.4      # 工具选择能力40%
    #     instruction_weight = 0.3  # 指令遵循30%
    #     consistency_weight = 0.3  # 一致性30%
        
    #     tool_score = tool_results['accuracy']
    #     instruction_score = (instruction_results['format_rate'] + instruction_results['answer_rate']) / 2
    #     consistency_score = consistency_results['consistency_rate']
        
    #     overall = (tool_score * tool_weight + 
    #               instruction_score * instruction_weight + 
    #               consistency_score * consistency_weight)
        
    #     return overall
    
    def _print_final_report(self, results: Dict):
        """打印最终报告"""
        # print("\n" + "=" * 60)
        # print("最终测试报告")
        # print("=" * 60)
        
        # score = results['overall_score']
        # print(f"综合评分: {score:.2%}")
        
        # if score >= 0.8:
        #     recommendation = "✅ 推荐使用 - 模型表现优秀"
        # elif score >= 0.6:
        #     recommendation = "⚠️  谨慎使用 - 模型表现一般，需要更多调优"
        # else:
        #     recommendation = "❌ 不推荐使用 - 模型表现不佳，考虑其他模型"
        
        # print(f"推荐结果: {recommendation}")
        print(f"测试耗时: {results['test_duration']:.2f}秒")
        
        print("\n详细指标:")
        print(f"  工具选择准确率: {results['tool_selection']['accuracy']:.2%}")
        print(f"  指令格式遵循率: {results['instruction_following']['format_rate']:.2%}")
        print(f"  指令答案正确率: {results['instruction_following']['answer_rate']:.2%}")
        print(f"  回答一致性: {results['consistency']['consistency_rate']:.2%}")

# 使用示例
if __name__ == "__main__":
    # 创建测试器
    tester = RAGLLMTester("mistral:7b-instruct-v0.3-q5_0")
    
    # 运行完整测试
    results = tester.run_full_test()
    
    # # 保存结果
    # with open("llm_test_results.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    
    # print(f"\n测试结果已保存到 llm_test_results.json")