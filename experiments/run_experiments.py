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
    zero_shot_response = get_zeroshot_answer(question, engine)
    
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
    args = config()
    
    # Parse arguments
    strategies = args.verification_strategies.split(",")
    thresholds = [float(t) for t in args.confidence_thresholds.split(",")]
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup engine
    engine = tg.get_engine(args.engine)
    tg.set_backward_engine(engine, override=True)
    
    # Load dataset
    test_set = load_instance_task(args.task, evaluation_api=engine)
    if args.test_size > 0:
        test_subset = test_set[:args.test_size]
    else:
        test_subset = test_set
    
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
                    sample, strategy, threshold, engine, args.max_iterations
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
    
    # Generate summary statistics and visualizations
    analyze_results(results, args.output_dir)

def analyze_results(results, output_dir):
    """Analyze verification results and generate visualizations"""
    # Create summary dataframe
    summary = []
    
    # Group results by strategy and threshold
    results_df = pd.DataFrame(results)
    groupby_cols = ['verification_strategy', 'verification_threshold', 'task']
    grouped = results_df.groupby(groupby_cols)
    
    # Calculate metrics
    for (strategy, threshold, task), group in grouped:
        # Success rate by iteration
        success_rates = []
        for i in range(group['performance_history'].iloc[0] + 1):
            success_rate = np.mean([row['performance_history'][i] for _, row in group.iterrows()])
            success_rates.append(success_rate)
        
        # Verification metrics if applicable
        verification_applied_rate = 0
        avg_confidence = 0
        correction_rate = 0
        
        if strategy != "none":
            vm_all = []
            for _, row in group.iterrows():
                if 'verification_metrics' in row and row['verification_metrics']:
                    vm_all.extend(row['verification_metrics'])
            
            if vm_all:
                vm_df = pd.DataFrame(vm_all)
                verification_applied_rate = vm_df['verification_applied'].mean()
                avg_confidence = vm_df['verification_confidence'].mean()
                # Add logic for correction rate calculation
        
        summary.append({
            'verification_strategy': strategy,
            'verification_threshold': threshold,
            'task': task,
            'initial_success_rate': success_rates[0],
            'final_success_rate': success_rates[-1],
            'improvement': success_rates[-1] - success_rates[0],
            'verification_applied_rate': verification_applied_rate,
            'avg_confidence': avg_confidence,
            'correction_rate': correction_rate
        })
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/verification_summary.csv", index=False)
    
    # Generate visualizations
    plot_success_rates_by_strategy(summary_df, output_dir)
    plot_verification_metrics(summary_df, output_dir)

def plot_success_rates_by_strategy(summary_df, output_dir):
    """Plot success rates by verification strategy"""
    plt.figure(figsize=(10, 6))
    
    # Group by strategy
    strategies = summary_df['verification_strategy'].unique()
    
    for strategy in strategies:
        strategy_data = summary_df[summary_df['verification_strategy'] == strategy]
        plt.bar(
            strategy, 
            strategy_data['improvement'].mean(), 
            yerr=strategy_data['improvement'].std(),
            label=f"{strategy} (threshold={strategy_data['verification_threshold'].iloc[0]})"
        )
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Verification Strategy')
    plt.ylabel('Average Improvement in Success Rate')
    plt.title('Performance Improvement by Verification Strategy')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rates_by_strategy.png", dpi=300)

def plot_verification_metrics(summary_df, output_dir):
    """Plot verification metrics by strategy and threshold"""
    # Filter out 'none' strategy
    df = summary_df[summary_df['verification_strategy'] != 'none']
    
    if df.empty:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create subplot for verification application rate
    plt.subplot(2, 2, 1)
    for strategy in df['verification_strategy'].unique():
        strategy_data = df[df['verification_strategy'] == strategy]
        plt.plot(
            strategy_data['verification_threshold'], 
            strategy_data['verification_applied_rate'],
            marker='o',
            label=strategy
        )
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Verification Application Rate')
    plt.title('Verification Application Rate by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create subplot for average confidence
    plt.subplot(2, 2, 2)
    for strategy in df['verification_strategy'].unique():
        strategy_data = df[df['verification_strategy'] == strategy]
        plt.plot(
            strategy_data['verification_threshold'], 
            strategy_data['avg_confidence'],
            marker='o',
            label=strategy
        )
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Average Confidence')
    plt.title('Average Verification Confidence by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create subplot for correction rate
    plt.subplot(2, 2, 3)
    for strategy in df['verification_strategy'].unique():
        strategy_data = df[df['verification_strategy'] == strategy]
        plt.plot(
            strategy_data['verification_threshold'], 
            strategy_data['correction_rate'],
            marker='o',
            label=strategy
        )
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Correction Rate')
    plt.title('Correction Rate by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create subplot for improvement vs verification rate
    plt.subplot(2, 2, 4)
    for strategy in df['verification_strategy'].unique():
        strategy_data = df[df['verification_strategy'] == strategy]
        plt.scatter(
            strategy_data['verification_applied_rate'],
            strategy_data['improvement'],
            alpha=0.7,
            label=strategy
        )
    plt.xlabel('Verification Application Rate')
    plt.ylabel('Performance Improvement')
    plt.title('Improvement vs. Verification Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification_metrics.png", dpi=300)

if __name__ == "__main__":
    main()

args = config()
llm_engine = tg.get_engine(engine_name=args.engine)