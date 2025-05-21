import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import textgrad as tg
from textgrad.tasks import load_instance_task
from textgrad.optimizer import TextualGradientDescent, VerifiedTextualGradientDescent

def config():
    parser = argparse.ArgumentParser(description="Compare verification strategies for TextGrad optimization")
    parser.add_argument("--task", type=str, default="GPQA_diamond", 
                        help="Task to evaluate optimization on")
    parser.add_argument("--engine", type=str, default="gemini-1.5-pro", 
                        help="Engine for evaluation and optimization")
    parser.add_argument("--max_iterations", type=int, default=3, 
                        help="Maximum iterations for optimization")
    parser.add_argument("--verification_strategies", type=str, 
                        default="none,process,outcome", 
                        help="Comma-separated list of verification strategies to test")
    parser.add_argument("--confidence_thresholds", type=str, 
                        default="0.7,0.8,0.9", 
                        help="Comma-separated list of confidence thresholds to test")
    parser.add_argument("--output_dir", type=str, default="results/verification", 
                        help="Directory to save results")
    parser.add_argument("--num_threads", type=int, default=8, 
                        help="Number of threads for parallel processing")
    parser.add_argument("--test_size", type=int, default=50, 
                        help="Number of test samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    return parser.parse_args()


# Global setup - initialize args and related variables
args = config()
llm_engine = tg.get_engine(engine_name=args.engine)
tg.set_backward_engine(llm_engine, override=True)
test_set = load_instance_task(args.task, evaluation_api=llm_engine)
# test_subset = test_set[:args.test_size] if args.test_size < len(test_set) else test_set
test_subset = [test_set[0], test_set[1]]

# Parse global arguments
strategies = args.verification_strategies.split(",")
thresholds = [float(t) for t in args.confidence_thresholds.split(",")]

def get_zeroshot_answer(question):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    # The system prompt is from: https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    STARTING_SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False, role_description="system prompt to the language model")
    model = tg.BlackboxLLM(llm_engine, system_prompt)
    response = model(tg.Variable(question, requires_grad=False, role_description="question to the language model"))
    return response

def run_optimization_with_verification(sample, strategy, threshold, engine, max_iterations):
    """Run optimization with a specific verification strategy and threshold"""
    question, answer, test_time_objective, instance_eval_fn = sample
    
    # Get initial solution
    zero_shot_response = get_zeroshot_answer(question)
    
    # Create variable to optimize
    instance_var = tg.Variable(
        zero_shot_response.value,
        requires_grad=True,
        role_description="solution to the question"
    )
    
    # Record metrics
    performance_history = []
    verification_metrics = []
    
    # Initial evaluation
    performance_history.append(int(instance_eval_fn(instance_var)))
    
    # Choose optimizer based on strategy
    if strategy == "none":
        optimizer = TextualGradientDescent(
            parameters=[instance_var],
            engine=engine
        )
    else:
        optimizer = VerifiedTextualGradientDescent(
            parameters=[instance_var],
            verification_strategy=strategy,
            verification_threshold=threshold,
            engine=engine
        )
    
    # Optimization loop
    for _ in range(max_iterations):
        optimizer.zero_grad()
        loss = test_time_objective(instance_var)
        loss.backward()
        
        # If using verification, capture metrics before step
        if strategy != "none":
            verification_applied = False
            verification_confidence = 0.0
            # Add logic to track verification metrics from the optimizer
        
        optimizer.step()
        
        # Record results
        score = int(instance_eval_fn(instance_var))
        performance_history.append(score)
        
        if strategy != "none":
            verification_metrics.append({
                "iteration": len(performance_history) - 1,
                "verification_applied": verification_applied,
                "verification_confidence": verification_confidence,
                "score_after_update": score
            })
    
    return {
        "performance_history": performance_history,
        "verification_metrics": verification_metrics,
        "final_solution": instance_var.value,
        "answer": answer
    }

def main():
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store results
    results = []
    
    # Run experiments for each strategy and threshold
    for strategy in strategies:
        for threshold in thresholds:
            if strategy == "none" and threshold != thresholds[0]:
                continue  # Skip redundant runs for baseline
            
            print(f"Running experiments with {strategy} verification (threshold={threshold})")
            
            for i, sample in enumerate(tqdm(test_subset)):
                result = run_optimization_with_verification(
                    sample, strategy, threshold, llm_engine, args.max_iterations
                )
                
                # Add metadata
                result["sample_id"] = i
                result["verification_strategy"] = strategy
                result["verification_threshold"] = threshold
                result["task"] = args.task
                
                results.append(result)
                
                # Save incremental results
                if (i + 1) % 10 == 0:
                    with open(f"{args.output_dir}/verification_results_partial.json", 'w') as f:
                        json.dump(results, f, indent=2)
    
    # Save final results
    with open(f"{args.output_dir}/verification_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()