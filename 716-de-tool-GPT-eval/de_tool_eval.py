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

def find_evaluation_ready_files(test_results_dir: str = "test_results") -> List[str]:
    """
    Find all evaluation_ready JSON files in the test_results directory and its subfolders
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir)
    test_results_path = os.path.join(workspace_root, test_results_dir)
    
    evaluation_files = []
    
    if os.path.exists(test_results_path):
        # Check files in the root test_results directory
        for file in os.listdir(test_results_path):
            file_path = os.path.join(test_results_path, file)
            if os.path.isfile(file_path) and "evaluation_ready" in file and file.endswith(".json"):
                evaluation_files.append(file_path)
        
        # Check subfolders for model-specific folders
        for item in os.listdir(test_results_path):
            subfolder_path = os.path.join(test_results_path, item)
            if os.path.isdir(subfolder_path):
                # Look for evaluation_ready files in subfolders
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    if os.path.isfile(file_path) and "evaluation_ready" in file and file.endswith(".json"):
                        evaluation_files.append(file_path)
    
    return evaluation_files

def load_test_results(file_path: str) -> Dict[str, Any]:
    """
    Load test results from JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_evaluation_prompt(test_case: Dict[str, Any]) -> str:
    """
    Create a prompt for GPT to evaluate a single test case
    """
    prompt = f"""
Als GPT-4o Evaluator bewerten Sie bitte die Qualität der folgenden Antwort eines RAG-Systems (Retrieval-Augmented Generation) auf eine Frage zu Albert Camus' "Der Fremde".

FRAGE: {test_case['question']}
FRAGENTYP: {test_case['question_type']}
KOMPLEXITÄT: {test_case['complexity']}

ERWARTETE WERKZEUGE: {', '.join(test_case['expected_tools'])}
TATSÄCHLICH VERWENDETE WERKZEUGE: {', '.join(test_case['actual_tools'])}
WERKZEUGSTRATEGIE-GENAUIGKEIT: {test_case['strategy_accuracy'] * 100:.0f}%

ANTWORT DES SYSTEMS:
{test_case['final_response']}

BEWERTUNGSKRITERIEN:
Bewerten Sie die Antwort auf einer Skala von 1-10 in den folgenden Kategorien:

1. FAKTISCHE RICHTIGKEIT (1-10): Ist die Antwort faktisch korrekt bezüglich "Der Fremde"?

2. VOLLSTÄNDIGKEIT (1-10): Beantwortet die Antwort die gestellte Frage vollständig?

3. RELEVANZ (1-10): Ist die Antwort relevant zur gestellten Frage?

4. STRUKTUR UND KLARHEIT (1-10): Ist die Antwort gut strukturiert und verständlich?

5. VERWENDUNG DER WERKZEUGE (1-10): Wurden die richtigen Werkzeuge verwendet und deren Ergebnisse angemessen integriert?

6. QUELLENVERWEIS (1-10): Sind die Quellenverweise angemessen und nachvollziehbar?

7. GESAMTQUALITÄT (1-10): Wie bewerten Sie die Antwort insgesamt?

SPEZIELLE BEWERTUNG:
- Falls die Frage irrelevant zu "Der Fremde" ist (wie die Hund-Katze-Frage): Bewerten Sie, ob das System angemessen reagiert hat, indem es die Irrelevanz erkannt und trotzdem im Kontext geantwortet hat.

Antworten Sie im folgenden JSON-Format:
{{
    "faktische_richtigkeit": <Bewertung 1-10>,
    "vollständigkeit": <Bewertung 1-10>,
    "relevanz": <Bewertung 1-10>,
    "struktur_klarheit": <Bewertung 1-10>,
    "werkzeug_verwendung": <Bewertung 1-10>,
    "quellenverweis": <Bewertung 1-10>,
    "gesamtqualität": <Bewertung 1-10>,
    "kommentar": "<Detaillierte Begründung der Bewertung>",
    "stärken": ["<Liste der Stärken>"],
    "schwächen": ["<Liste der Schwächen>"],
    "verbesserungsvorschläge": ["<Liste von Verbesserungsvorschlägen>"]
}}
"""
    return prompt

def evaluate_with_gpt4(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use GPT-4o to evaluate a test case
    """
    try:
        prompt = create_evaluation_prompt(test_case)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sie sind ein Experte für Literaturanalyse und RAG-Systembewertung. Bewerten Sie die Antworten objektiv und konstruktiv."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        # Parse the JSON response
        evaluation_text = response.choices[0].message.content
        
        # Extract JSON from the response
        json_start = evaluation_text.find('{')
        json_end = evaluation_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = evaluation_text[json_start:json_end]
            evaluation = json.loads(json_str)
            return evaluation
        else:
            print(f"Could not parse JSON from GPT response: {evaluation_text}")
            return None
            
    except Exception as e:
        print(f"Error evaluating with GPT-4o: {e}")
        return None

def calculate_summary_stats(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from all evaluations
    """
    if not evaluations:
        return {}
    
    # Extract numeric scores
    scores = {
        'faktische_richtigkeit': [],
        'vollständigkeit': [],
        'relevanz': [],
        'struktur_klarheit': [],
        'werkzeug_verwendung': [],
        'quellenverweis': [],
        'gesamtqualität': []
    }
    
    for eval_data in evaluations:
        if eval_data and 'gpt_evaluation' in eval_data:
            gpt_eval = eval_data['gpt_evaluation']
            for key in scores.keys():
                if key in gpt_eval and isinstance(gpt_eval[key], (int, float)):
                    scores[key].append(gpt_eval[key])
    
    # Calculate averages
    averages = {}
    for key, values in scores.items():
        if values:
            averages[f'avg_{key}'] = sum(values) / len(values)
            averages[f'min_{key}'] = min(values)
            averages[f'max_{key}'] = max(values)
    
    return averages

def save_evaluation_results(original_data: Dict[str, Any], evaluations: List[Dict[str, Any]], output_file: str):
    """
    Save evaluation results to JSON file in eval_results folder
    """
    summary_stats = calculate_summary_stats(evaluations)
    
    result = {
        "evaluation_metadata": {
            "evaluator": "GPT-4o",
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_file": original_data.get("model_name", "unknown"),
            "original_test_timestamp": original_data.get("test_timestamp", "unknown")
        },
        "original_summary_stats": original_data.get("summary_stats", {}),
        "evaluation_summary_stats": summary_stats,
        "evaluated_test_cases": evaluations
    }
    
    # Ensure eval_results directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation results saved to: {output_file}")

def create_model_comparison_report(all_evaluations: Dict[str, Dict], output_dir: str):
    """
    Create a comprehensive comparison report across all models
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    comparison_report = {
        "comparison_metadata": {
            "report_timestamp": datetime.now().isoformat(),
            "models_evaluated": list(all_evaluations.keys()),
            "total_models": len(all_evaluations)
        },
        "model_performance_summary": {},
        "question_type_analysis": {},
        "detailed_comparison": all_evaluations
    }
    
    # Create performance summary for each model
    for model_name, model_data in all_evaluations.items():
        eval_stats = model_data.get("evaluation_summary_stats", {})
        comparison_report["model_performance_summary"][model_name] = {
            "avg_gesamtqualität": eval_stats.get("avg_gesamtqualität", 0),
            "avg_faktische_richtigkeit": eval_stats.get("avg_faktische_richtigkeit", 0),
            "avg_vollständigkeit": eval_stats.get("avg_vollständigkeit", 0),
            "avg_relevanz": eval_stats.get("avg_relevanz", 0),
            "avg_struktur_klarheit": eval_stats.get("avg_struktur_klarheit", 0),
            "avg_werkzeug_verwendung": eval_stats.get("avg_werkzeug_verwendung", 0),
            "avg_quellenverweis": eval_stats.get("avg_quellenverweis", 0),
            "total_test_cases": len(model_data.get("evaluated_test_cases", []))
        }
    
    # Analyze performance by question type
    question_types = set()
    for model_data in all_evaluations.values():
        for test_case in model_data.get("evaluated_test_cases", []):
            question_types.add(test_case.get("question_type", "unknown"))
    
    for q_type in question_types:
        comparison_report["question_type_analysis"][q_type] = {}
        for model_name, model_data in all_evaluations.items():
            scores = []
            for test_case in model_data.get("evaluated_test_cases", []):
                if test_case.get("question_type") == q_type:
                    gpt_eval = test_case.get("gpt_evaluation", {})
                    if "gesamtqualität" in gpt_eval:
                        scores.append(gpt_eval["gesamtqualität"])
            
            if scores:
                comparison_report["question_type_analysis"][q_type][model_name] = {
                    "avg_score": sum(scores) / len(scores),
                    "count": len(scores)
                }
    
    # Find best performing model overall
    best_model = None
    best_score = 0
    for model_name, stats in comparison_report["model_performance_summary"].items():
        avg_score = stats.get("avg_gesamtqualität", 0)
        if avg_score > best_score:
            best_score = avg_score
            best_model = model_name
    
    comparison_report["performance_ranking"] = {
        "best_overall_model": best_model,
        "best_overall_score": best_score,
        "ranking": sorted(
            comparison_report["model_performance_summary"].items(),
            key=lambda x: x[1].get("avg_gesamtqualität", 0),
            reverse=True
        )
    }
    
    # Save comparison report
    comparison_file = os.path.join(output_dir, f"model_comparison_report_{timestamp}.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)
    
    print(f"Model comparison report saved to: {comparison_file}")
    return comparison_report

def main():
    """
    Main function to run GPT evaluation on all evaluation_ready files and create model comparison
    """
    print("Starting GPT-4o evaluation of test results...")
    
    # Find all evaluation_ready files
    evaluation_files = find_evaluation_ready_files()
    
    if not evaluation_files:
        print("No evaluation_ready files found in test_results folder.")
        return
    
    print(f"Found {len(evaluation_files)} evaluation_ready files:")
    for file in evaluation_files:
        print(f"  - {os.path.relpath(file)}")
    
    # Create eval_results directory structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir)
    eval_results_dir = os.path.join(workspace_root, "eval_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    timestamped_eval_dir = os.path.join(eval_results_dir, timestamp)
    os.makedirs(timestamped_eval_dir, exist_ok=True)
    
    all_evaluations = {}
    
    # Process each file
    for file_path in evaluation_files:
        print(f"\nProcessing: {os.path.relpath(file_path)}")
        
        # Load test results
        test_data = load_test_results(file_path)
        if not test_data:
            continue
        
        model_name = test_data.get("model_name", "unknown_model")
        print(f"  Model: {model_name}")
        
        # Evaluate each test case
        evaluations = []
        evaluation_data = test_data.get("evaluation_data", [])
        
        for i, test_case in enumerate(evaluation_data, 1):
            print(f"  Evaluating test case {i}/{len(evaluation_data)}: {test_case.get('question_type', 'unknown')}")
            
            gpt_evaluation = evaluate_with_gpt4(test_case)
            
            # Combine original test case with GPT evaluation
            evaluated_case = test_case.copy()
            evaluated_case["gpt_evaluation"] = gpt_evaluation
            evaluations.append(evaluated_case)
            
            # Small delay to avoid rate limiting
            import time
            time.sleep(1)
        
        # Create output filename in eval_results folder
        base_name = os.path.basename(file_path)
        base_name = base_name.replace("evaluation_ready", "gpt_evaluation")
        output_file = os.path.join(timestamped_eval_dir, f"{base_name.split('.')[0]}_{timestamp}.json")
        
        # Save individual model results
        save_evaluation_results(test_data, evaluations, output_file)
        
        # Store for comparison
        summary_stats = calculate_summary_stats(evaluations)
        all_evaluations[model_name] = {
            "evaluation_metadata": {
                "evaluator": "GPT-4o",
                "evaluation_timestamp": datetime.now().isoformat(),
                "original_file": model_name,
                "original_test_timestamp": test_data.get("test_timestamp", "unknown")
            },
            "original_summary_stats": test_data.get("summary_stats", {}),
            "evaluation_summary_stats": summary_stats,
            "evaluated_test_cases": evaluations
        }
        
        # Print summary
        if summary_stats:
            print(f"  Average overall quality: {summary_stats.get('avg_gesamtqualität', 'N/A'):.2f}/10")
            print(f"  Average tool usage: {summary_stats.get('avg_werkzeug_verwendung', 'N/A'):.2f}/10")
            print(f"  Average factual accuracy: {summary_stats.get('avg_faktische_richtigkeit', 'N/A'):.2f}/10")
    
    # Create model comparison report
    if all_evaluations:
        print(f"\nCreating model comparison report...")
        comparison_report = create_model_comparison_report(all_evaluations, timestamped_eval_dir)
        
        print(f"\n=== MODEL COMPARISON SUMMARY ===")
        print(f"Total models evaluated: {len(all_evaluations)}")
        
        ranking = comparison_report.get("performance_ranking", {}).get("ranking", [])
        if ranking:
            print(f"\nPerformance Ranking (by overall quality):")
            for i, (model_name, stats) in enumerate(ranking, 1):
                score = stats.get("avg_gesamtqualität", 0)
                print(f"  {i}. {model_name}: {score:.2f}/10")
        
        print(f"\nAll results saved to: {timestamped_eval_dir}")
    else:
        print("No evaluations completed successfully.")

if __name__ == "__main__":
    main()