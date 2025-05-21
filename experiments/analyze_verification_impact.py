# experiments/analyze_verification_impact.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats

def analyze_verification_impact(results_file, output_dir):
    """Generate comprehensive analysis of verification impact"""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Extract performance metrics
    performance_data = []
    for _, row in results_df.iterrows():
        strategy = row['verification_strategy']
        threshold = row['verification_threshold']
        task = row['task']
        
        # Calculate metrics for each iteration
        for i, score in enumerate(row['performance_history']):
            performance_data.append({
                'verification_strategy': strategy,
                'verification_threshold': threshold,
                'task': task,
                'iteration': i,
                'score': score,
                'sample_id': row['sample_id']
            })
    
    performance_df = pd.DataFrame(performance_data)
    
    # 1. Performance improvement analysis
    improvement_analysis(performance_df, output_dir)
    
    # 2. Verification behavior analysis
    if 'verification_metrics' in results_df.columns:
        verification_df = pd.DataFrame()
        for _, row in results_df.iterrows():
            if 'verification_metrics' in row and row['verification_metrics']:
                vm_df = pd.DataFrame(row['verification_metrics'])
                vm_df['verification_strategy'] = row['verification_strategy']
                vm_df['verification_threshold'] = row['verification_threshold']
                vm_df['task'] = row['task']
                vm_df['sample_id'] = row['sample_id']
                verification_df = pd.concat([verification_df, vm_df])
        
        verification_analysis(verification_df, output_dir)
    
    # 3. Error analysis
    error_analysis(results_df, output_dir)
    
    # 4. Statistical significance testing
    significance_testing(performance_df, output_dir)

def improvement_analysis(performance_df, output_dir):
    """Analyze performance improvement across strategies and iterations"""
    
    # Calculate average performance by strategy, threshold, and iteration
    avg_performance = performance_df.groupby(['verification_strategy', 'verification_threshold', 'iteration'])['score'].mean().reset_index()
    
    # Plot performance over iterations for each strategy
    plt.figure(figsize=(12, 8))
    
    for strategy in avg_performance['verification_strategy'].unique():
        for threshold in avg_performance[avg_performance['verification_strategy'] == strategy]['verification_threshold'].unique():
            if strategy == 'none' and threshold != avg_performance[avg_performance['verification_strategy'] == 'none']['verification_threshold'].min():
                continue
            
            data = avg_performance[(avg_performance['verification_strategy'] == strategy) & 
                                  (avg_performance['verification_threshold'] == threshold)]
            
            label = f"{strategy}" if strategy == 'none' else f"{strategy} (threshold={threshold})"
            plt.plot(data['iteration'], data['score'], marker='o', label=label)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Score')
    plt.title('Performance Improvement Over Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/performance_over_iterations.png", dpi=300)
    
    # Calculate relative improvement
    first_iter = performance_df[performance_df['iteration'] == 0].groupby(['verification_strategy', 'verification_threshold'])['score'].mean()
    last_iter = performance_df[performance_df['iteration'] == performance_df['iteration'].max()].groupby(['verification_strategy', 'verification_threshold'])['score'].mean()
    
    improvement = (last_iter - first_iter).reset_index()
    improvement.columns = ['verification_strategy', 'verification_threshold', 'improvement']
    
    # Plot improvement by strategy
    plt.figure(figsize=(10, 6))
    
    for strategy in improvement['verification_strategy'].unique():
        strategy_data = improvement[improvement['verification_strategy'] == strategy]
        
        if strategy == 'none':
            plt.axhline(
                y=strategy_data['improvement'].iloc[0],
                color='gray',
                linestyle='--',
                label=f"{strategy} (baseline)"
            )
        else:
            plt.plot(
                strategy_data['verification_threshold'],
                strategy_data['improvement'],
                marker='o',
                label=strategy
            )
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Improvement (Final - Initial Score)')
    plt.title('Performance Improvement by Verification Strategy and Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/improvement_by_strategy.png", dpi=300)
    
    # Save summary stats
    improvement.to_csv(f"{output_dir}/improvement_summary.csv", index=False)

def verification_analysis(verification_df, output_dir):
    """Analyze verification behavior metrics"""
    
    # Calculate verification application rate
    verification_rate = verification_df.groupby(['verification_strategy', 'verification_threshold'])['verification_applied'].mean().reset_index()
    
    # Calculate average confidence when verification is applied
    confidence_when_applied = verification_df[verification_df['verification_applied']].groupby(['verification_strategy', 'verification_threshold'])['verification_confidence'].mean().reset_index()
    
    # Calculate impact of verification on score
    verification_impact = []
    for (strategy, threshold, iteration), group in verification_df.groupby(['verification_strategy', 'verification_threshold', 'iteration']):
        # Calculate average score change when verification is applied vs. not applied
        if len(group[group['verification_applied']]) > 0 and len(group[~group['verification_applied']]) > 0:
            score_with_verification = group[group['verification_applied']]['score_after_update'].mean()
            score_without_verification = group[~group['verification_applied']]['score_after_update'].mean()
            
            verification_impact.append({
                'verification_strategy': strategy,
                'verification_threshold': threshold,
                'iteration': iteration,
                'score_with_verification': score_with_verification,
                'score_without_verification': score_without_verification,
                'impact': score_with_verification - score_without_verification
            })
    
    verification_impact_df = pd.DataFrame(verification_impact)
    
    # Plot verification application rate
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for strategy in verification_rate['verification_strategy'].unique():
        if strategy == 'none':
            continue
        
        strategy_data = verification_rate[verification_rate['verification_strategy'] == strategy]
        plt.plot(
            strategy_data['verification_threshold'],
            strategy_data['verification_applied'],
            marker='o',
            label=strategy
        )
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Verification Application Rate')
    plt.title('Verification Application Rate by Strategy and Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot average confidence
    plt.subplot(2, 2, 2)
    for strategy in confidence_when_applied['verification_strategy'].unique():
        if strategy == 'none':
            continue
        
        strategy_data = confidence_when_applied[confidence_when_applied['verification_strategy'] == strategy]
        plt.plot(
            strategy_data['verification_threshold'],
            strategy_data['verification_confidence'],
            marker='o',
            label=strategy
        )
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Average Confidence When Applied')
    plt.title('Verification Confidence by Strategy and Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot verification impact
    if not verification_impact_df.empty:
        plt.subplot(2, 2, 3)
        
        # Average impact across iterations
        avg_impact = verification_impact_df.groupby(['verification_strategy', 'verification_threshold'])['impact'].mean().reset_index()
        
        for strategy in avg_impact['verification_strategy'].unique():
            if strategy == 'none':
                continue
            
            strategy_data = avg_impact[avg_impact['verification_strategy'] == strategy]
            plt.plot(
                strategy_data['verification_threshold'],
                strategy_data['impact'],
                marker='o',
                label=strategy
            )
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score Impact (With - Without Verification)')
        plt.title('Impact of Verification on Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification_metrics.png", dpi=300)
    
    # Save detailed metrics
    verification_rate.to_csv(f"{output_dir}/verification_rate.csv", index=False)
    confidence_when_applied.to_csv(f"{output_dir}/confidence_when_applied.csv", index=False)
    if not verification_impact_df.empty:
        verification_impact_df.to_csv(f"{output_dir}/verification_impact.csv", index=False)

def error_analysis(results_df, output_dir):
    """Analyze error patterns with and without verification"""
    
    # Extract error patterns
    error_patterns = []
    
    for _, row in results_df.iterrows():
        strategy = row['verification_strategy']
        threshold = row['verification_threshold']
        final_solution = row['final_solution']
        
        # This is where you'd analyze the final solution for common error patterns
        # For demonstration, we'll use placeholder categories
        
        # Check for hallucinations
        has_hallucination = False  # Add logic to detect hallucinations
        
        # Check for logical errors
        has_logical_error = False  # Add logic to detect logical errors
        
        # Check for factual errors
        has_factual_error = False  # Add logic to detect factual errors
        
        error_patterns.append({
            'verification_strategy': strategy,
            'verification_threshold': threshold,
            'has_hallucination': has_hallucination,
            'has_logical_error': has_logical_error,
            'has_factual_error': has_factual_error
        })
    
    error_df = pd.DataFrame(error_patterns)
    
    # Calculate error rates by strategy
    error_rates = error_df.groupby(['verification_strategy', 'verification_threshold']).mean().reset_index()
    
    # Plot error rates
    plt.figure(figsize=(12, 6))
    
    # Set up bar positions
    strategies = error_rates['verification_strategy'].unique()
    x = np.arange(len(strategies))
    width = 0.25
    
    # Plot hallucination rates
    plt.bar(
        x - width,
        [error_rates[error_rates['verification_strategy'] == s]['has_hallucination'].mean() for s in strategies],
        width,
        label='Hallucinations'
    )
    
    # Plot logical error rates
    plt.bar(
        x,
        [error_rates[error_rates['verification_strategy'] == s]['has_logical_error'].mean() for s in strategies],
        width,
        label='Logical Errors'
    )
    
    # Plot factual error rates
    plt.bar(
        x + width,
        [error_rates[error_rates['verification_strategy'] == s]['has_factual_error'].mean() for s in strategies],
        width,
        label='Factual Errors'
    )
    
    plt.xlabel('Verification Strategy')
    plt.ylabel('Error Rate')
    plt.title('Error Rates by Verification Strategy')
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_rates.png", dpi=300)
    
    # Save error rates
    error_rates.to_csv(f"{output_dir}/error_rates.csv", index=False)

def significance_testing(performance_df, output_dir):
    """Perform statistical significance testing on performance improvements"""
    
    # Get baseline performance (no verification)
    baseline = performance_df[performance_df['verification_strategy'] == 'none']
    
    # Initialize results
    significance_results = []
    
    # Compare each verification strategy to baseline
    for strategy in performance_df['verification_strategy'].unique():
        if strategy == 'none':
            continue
        
        for threshold in performance_df[performance_df['verification_strategy'] == strategy]['verification_threshold'].unique():
            # Get final scores for this strategy
            final_iter = performance_df['iteration'].max()
            
            strategy_scores = performance_df[
                (performance_df['verification_strategy'] == strategy) &
                (performance_df['verification_threshold'] == threshold) &
                (performance_df['iteration'] == final_iter)
            ]['score']
            
            baseline_scores = baseline[
                baseline['iteration'] == final_iter
            ]['score']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(strategy_scores, baseline_scores)
            
            # Record results
            significance_results.append({
                'verification_strategy': strategy,
                'verification_threshold': threshold,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_difference': strategy_scores.mean() - baseline_scores.mean()
            })
    
    significance_df = pd.DataFrame(significance_results)
    
    # Plot significant differences
    plt.figure(figsize=(10, 6))
    
    # Create a mask for significant results
    significant = significance_df['significant']
    
    # Set up the plot
    strategies = significance_df['verification_strategy'].unique()
    for strategy in strategies:
        strategy_data = significance_df[significance_df['verification_strategy'] == strategy]
        
        # Use different markers for significant vs non-significant
        sig_data = strategy_data[strategy_data['significant']]
        nonsig_data = strategy_data[~strategy_data['significant']]
        
        plt.scatter(
            sig_data['verification_threshold'],
            sig_data['mean_difference'],
            marker='o',
            s=100,
            label=f"{strategy} (significant)",
            alpha=0.8
        )
        
        plt.scatter(
            nonsig_data['verification_threshold'],
            nonsig_data['mean_difference'],
            marker='x',
            s=100,
            label=f"{strategy} (not significant)",
            alpha=0.5
        )
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Mean Difference from Baseline')
    plt.title('Statistical Significance of Verification Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/statistical_significance.png", dpi=300)
    
    # Save significance results
    significance_df.to_csv(f"{output_dir}/significance_tests.csv", index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze verification impact")
    parser.add_argument("--results_file", type=str, required=True, help="Path to verification results JSON file")
    parser.add_argument("--output_dir", type=str, default="results/analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    analyze_verification_impact(args.results_file, args.output_dir)