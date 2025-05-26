import json
import re
from statistics import mean

def extract_answer_from_prediction(prediction):
    """Extract the answer letter (A, B, C, D) from a prediction string."""
    # Look for pattern like "Answer: B" or "Answer:B"
    pattern = r"(?i)Answer\s*:\s*([A-D])"
    match = re.search(pattern, prediction)
    if match:
        return match.group(1).upper()
    return None

def analyze_results(json_file_path):
    """Analyze the JSON results file and calculate accuracy metrics."""
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'total_questions': len(data),
        'zero_shot_correct': 0,
        'final_correct': 0,
        'ensemble_correct': 0,
        'improvements': 0,
        'degradations': 0,
        'no_change': 0,
        'question_details': []
    }
    
    print("Analyzing results...")
    print("=" * 50)
    
    for question, info in data.items():
        predictions = info['predictions']
        correct_answer = info['answer']
        
        # Extract answers from predictions
        extracted_answers = []
        for pred in predictions:
            answer = extract_answer_from_prediction(pred)
            extracted_answers.append(answer)
        
        # Zero-shot (first prediction)
        zero_shot_answer = extracted_answers[0]
        zero_shot_correct = (zero_shot_answer == correct_answer)
        
        # Final answer (last prediction)
        final_answer = extracted_answers[-1]
        final_correct = (final_answer == correct_answer)
        
        # Ensemble answer (second to last, based on your code structure)
        ensemble_answer = extracted_answers[-2] if len(extracted_answers) > 1 else final_answer
        ensemble_correct = (ensemble_answer == correct_answer)
        
        # Track overall statistics
        if zero_shot_correct:
            results['zero_shot_correct'] += 1
        if final_correct:
            results['final_correct'] += 1
        if ensemble_correct:
            results['ensemble_correct'] += 1
            
        # Track improvements/degradations
        if final_correct and not zero_shot_correct:
            results['improvements'] += 1
            change = "Improved"
        elif not final_correct and zero_shot_correct:
            results['degradations'] += 1
            change = "Degraded"
        else:
            results['no_change'] += 1
            change = "No Change"
        
        # Store question details
        question_detail = {
            'question': question[:100] + "..." if len(question) > 100 else question,
            'correct_answer': correct_answer,
            'zero_shot': zero_shot_answer,
            'final': final_answer,
            'ensemble': ensemble_answer,
            'zero_shot_correct': zero_shot_correct,
            'final_correct': final_correct,
            'ensemble_correct': ensemble_correct,
            'change': change
        }
        results['question_details'].append(question_detail)
    
    return results

def print_analysis(results):
    """Print the analysis results in a formatted way."""
    total = results['total_questions']
    
    print(f"ACCURACY ANALYSIS")
    print("=" * 50)
    print(f"Total Questions: {total}")
    print()
    
    # Accuracy calculations
    zero_shot_acc = results['zero_shot_correct'] / total * 100
    final_acc = results['final_correct'] / total * 100
    ensemble_acc = results['ensemble_correct'] / total * 100
    
    print(f"Zero-shot Accuracy: {results['zero_shot_correct']}/{total} = {zero_shot_acc:.2f}%")
    print(f"Final Accuracy: {results['final_correct']}/{total} = {final_acc:.2f}%")
    print(f"Ensemble Accuracy: {results['ensemble_correct']}/{total} = {ensemble_acc:.2f}%")
    print()
    
    # Improvement rates
    improvement_rate = (final_acc - zero_shot_acc)
    relative_improvement = ((final_acc - zero_shot_acc) / zero_shot_acc * 100) if zero_shot_acc > 0 else 0
    
    print(f"IMPROVEMENT ANALYSIS")
    print("=" * 50)
    print(f"Absolute Improvement: {improvement_rate:.2f} percentage points")
    print(f"Relative Improvement: {relative_improvement:.2f}%")
    print()
    print(f"Questions Improved: {results['improvements']}")
    print(f"Questions Degraded: {results['degradations']}")
    print(f"Questions Unchanged: {results['no_change']}")
    print()
    
    # Success rate of optimization
    optimization_success_rate = results['improvements'] / total * 100
    print(f"Optimization Success Rate: {optimization_success_rate:.2f}%")
    
    return {
        'zero_shot_accuracy': zero_shot_acc,
        'final_accuracy': final_acc,
        'ensemble_accuracy': ensemble_acc,
        'absolute_improvement': improvement_rate,
        'relative_improvement': relative_improvement,
        'optimization_success_rate': optimization_success_rate
    }

def main():
    # Replace with your actual JSON file path
    json_file = "evaluation/solution_optimization/results/tg_MMLU_machine_learning_predictions.json"  # Update this path
    
    try:
        results = analyze_results(json_file)
        metrics = print_analysis(results)
        
        # Print some example question details
        print(f"\nEXAMPLE QUESTIONS:")
        print("=" * 50)
        for i, detail in enumerate(results['question_details'][:3]):  # Show first 3
            print(f"Question {i+1}: {detail['question']}")
            print(f"  Correct: {detail['correct_answer']}")
            print(f"  Zero-shot: {detail['zero_shot']} ({'✓' if detail['zero_shot_correct'] else '✗'})")
            print(f"  Final: {detail['final']} ({'✓' if detail['final_correct'] else '✗'})")
            print(f"  Status: {detail['change']}")
            print()
            
    except FileNotFoundError:
        print(f"Error: Could not find the JSON file '{json_file}'")
        print("Please update the file path in the main() function")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main()