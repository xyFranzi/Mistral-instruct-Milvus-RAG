import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from openai import OpenAI
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

class LLMResponseEvaluator:
    def __init__(self):
        self.test_results_dir = Path(__file__).parent.parent / "test_results"
        self.eval_results_dir = Path(__file__).parent.parent / "eval_results"
        self.eval_results_dir.mkdir(exist_ok=True)
        
    def evaluate_response(self, query: str, documents: List[Dict], llm_response: str, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single LLM response using GPT-4
        """
        # Prepare context from retrieved documents
        context = "\n\n".join([f"Document {i+1}: {doc['content']}" for i, doc in enumerate(documents)])
        
        evaluation_prompt = f"""
        You are an expert evaluator for German-language RAG (Retrieval-Augmented Generation) systems. Please evaluate the following German LLM response based on the German query and provided German context documents.

        **Deutsche Anfrage:** {query}

        **Deutsche Kontext-Dokumente:**
        {context}

        **LLM-Antwort:** {llm_response}

        **Modell:** {model_name}

        Please evaluate the German response on the following criteria and provide scores from 1-5 (1=Poor, 5=Excellent):

        1. **Relevanz** (1-5): How well does the response address the German query? Does it understand the question correctly?
        2. **Genauigkeit** (1-5): How factually correct is the response based on the provided German documents? Are there any factual errors?
        3. **Vollständigkeit** (1-5): How complete is the response in answering the German query? Does it cover all important aspects?
        4. **Kohärenz** (1-5): How well-structured and logical is the German response? Is the German grammar and syntax correct?
        5. **Kontextnutzung** (1-5): How effectively does the response utilize the provided German documents? Does it properly reference the source material?
        6. **Deutsche Sprachqualität** (1-5): How is the quality of the German language? Consider grammar, vocabulary, style, and natural flow.
        7. **Kulturelle Angemessenheit** (1-5): Is the response culturally appropriate for German-speaking contexts?

        Special considerations for German language evaluation:
        - German grammar complexity (cases, verb positions, compound words)
        - Appropriate use of formal/informal language (Sie/Du)
        - Correct German vocabulary and idioms
        - Natural German sentence structure
        - Understanding of German cultural context when relevant

        Please provide your evaluation in the following JSON format:
        {{
            "relevance": <score>,
            "accuracy": <score>,
            "completeness": <score>,
            "coherence": <score>,
            "use_of_context": <score>,
            "german_language_quality": <score>,
            "cultural_appropriateness": <score>,
            "overall_score": <average_score>,
            "explanation": "<detailed explanation of your evaluation in English>",
            "german_language_assessment": "<specific assessment of German language quality>",
            "strengths": ["<list of strengths>"],
            "weaknesses": ["<list of weaknesses>"],
            "suggestions": "<suggestions for improvement>",
            "grammar_issues": ["<list any German grammar issues found>"],
            "vocabulary_assessment": "<assessment of German vocabulary usage>"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator for AI-generated responses. Provide detailed, objective evaluations."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        evaluation_text = response.choices[0].message.content
        print(f"Raw GPT response: {evaluation_text[:200]}...")  # Debug output
        
        # Find JSON in the response
        start_idx = evaluation_text.find('{')
        end_idx = evaluation_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            print("No JSON found in response")
            evaluation_result = {
                "relevance": 0,
                "accuracy": 0,
                "completeness": 0,
                "coherence": 0,
                "use_of_context": 0,
                "german_language_quality": 0,
                "cultural_appropriateness": 0,
                "overall_score": 0,
                "explanation": evaluation_text,
                "german_language_assessment": "Could not parse evaluation",
                "strengths": [],
                "weaknesses": ["GPT evaluation could not be parsed"],
                "suggestions": "Review the evaluation manually",
                "grammar_issues": [],
                "vocabulary_assessment": "Could not assess"
            }
        else:
            json_str = evaluation_text[start_idx:end_idx]
            print(f"Extracted JSON: {json_str[:100]}...")  # Debug output
            evaluation_result = json.loads(json_str)
            
        return evaluation_result
    
    def process_test_results_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single test results JSON file
        """
        print(f"Processing: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        evaluated_results = {
            "source_file": file_path.name,
            "test_run_timestamp": test_data.get("test_run_timestamp"),
            "total_queries": test_data.get("total_queries", 0),
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluations": []
        }
        
        for result in test_data.get("results", []):
            query = result.get("query", "")
            documents = result.get("similarity_search", {}).get("documents", [])
            llm_response = result.get("llm_response", {}).get("content", "")
            model_name = result.get("llm_response", {}).get("model", "unknown")
            
            print(f"  Evaluating query {result.get('query_id', 'unknown')}: {query[:50]}...")
            
            evaluation = self.evaluate_response(query, documents, llm_response, model_name)
            
            evaluated_result = {
                "query_id": result.get("query_id"),
                "query": query,
                "model": model_name,
                "found_documents": result.get("similarity_search", {}).get("found_documents", 0),
                "llm_response": llm_response,
                "llm_tokens": result.get("llm_response", {}).get("tokens", {}),
                "llm_timing": result.get("llm_response", {}).get("timing", {}),
                "gpt_evaluation": evaluation,
                "timestamp": result.get("timestamp")
            }
            
            evaluated_results["evaluations"].append(evaluated_result)
        
        return evaluated_results
    
    def calculate_summary_statistics(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """
        Calculate summary statistics for all evaluations
        """
        if not evaluations:
            return {}
        
        metrics = ["relevance", "accuracy", "completeness", "coherence", "use_of_context", "german_language_quality", "cultural_appropriateness", "overall_score"]
        summary = {}
        
        for metric in metrics:
            scores = [eval_data["gpt_evaluation"][metric] for eval_data in evaluations 
                     if eval_data["gpt_evaluation"].get(metric, 0) > 0]
            
            if scores:
                summary[metric] = {
                    "average": round(sum(scores) / len(scores), 2),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
            else:
                summary[metric] = {
                    "average": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0
                }
        
        return summary
    
    def save_evaluation_results(self, evaluated_results: Dict[str, Any]):
        """
        Save evaluation results to a JSON file
        """
        # Add summary statistics
        evaluated_results["summary_statistics"] = self.calculate_summary_statistics(
            evaluated_results["evaluations"]
        )
        
        # Generate output filename
        source_file = evaluated_results["source_file"]
        base_name = source_file.replace(".json", "")
        output_filename = f"{base_name}_gpt_evaluation.json"
        output_path = self.eval_results_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluated_results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to: {output_path}")
        return output_path
    
    def process_all_test_results(self):
        """
        Process all JSON files in the test_results directory
        """
        json_files = list(self.test_results_dir.glob("*.json"))
        
        if not json_files:
            print("No JSON files found in test_results directory")
            return
        
        print(f"Found {len(json_files)} test result files to process")
        
        for json_file in json_files:
            evaluated_results = self.process_test_results_file(json_file)
            self.save_evaluation_results(evaluated_results)
            print(f"✓ Successfully processed {json_file.name}")
            print()
    
    def generate_comparison_report(self):
        """
        Generate a comparison report across all evaluated models
        """
        eval_files = list(self.eval_results_dir.glob("*_gpt_evaluation.json"))
        
        if not eval_files:
            print("No evaluation files found to compare")
            return
        
        comparison_data = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models_compared": [],
            "overall_comparison": {}
        }
        
        all_summaries = {}
        
        for eval_file in eval_files:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # Extract model name from first evaluation
            if eval_data["evaluations"]:
                model_name = eval_data["evaluations"][0]["model"]
                all_summaries[model_name] = eval_data["summary_statistics"]
                comparison_data["models_compared"].append({
                    "model": model_name,
                    "file": eval_file.name,
                    "total_queries": eval_data["total_queries"]
                })
        
        comparison_data["overall_comparison"] = all_summaries
        
        # Save comparison report
        comparison_file = self.eval_results_dir / "model_comparison_report.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"Comparison report saved to: {comparison_file}")

def main():
    evaluator = LLMResponseEvaluator()
    
    print("=== LLM Response Evaluator using GPT-4o ===")
    print(f"Test results directory: {evaluator.test_results_dir}")
    print(f"Evaluation results directory: {evaluator.eval_results_dir}")
    print()
    
    # Process all test result files
    evaluator.process_all_test_results()
    
    # Generate comparison report
    print("Generating comparison report...")
    evaluator.generate_comparison_report()
    
    print("=== Evaluation Complete ===")

if __name__ == "__main__":
    main()