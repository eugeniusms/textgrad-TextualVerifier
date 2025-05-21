# experiments/analysis/success_rate_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def analyze_success_rate_improvement(results_dir):
    """Analyze success rate improvements across verification strategies."""
    
    # Load results
    results = pd.read_csv(f"{results_dir}/combined_results.csv")
    
    # Create directory for figures if it doesn't exist
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    
    # Get list of tasks and verification strategies
    tasks = results['task'].unique()
    strategies = results['verification_strategy'].unique()
    
    # Initialize dataframe for success rate improvement
    success_improvement = pd.DataFrame(
        index=tasks,
        columns=[s for s in strategies if s != 'none']
    )
    
    # Calculate improvement for each task and strategy
    for task in tasks:
        # Get baseline success rate (no verification)
        baseline = results[(results['task'] == task) & 
                          (results['verification_strategy'] == 'none')]['accuracy'].mean()
        
        # Calculate improvement for each verification strategy
        for strategy in [s for s in strategies if s != 'none']:
            verified = results[(results['task'] == task) & 
                              (results['verification_strategy'] == strategy)]['accuracy'].mean()
            success_improvement.loc[task, strategy] = verified - baseline
    
    # Calculate statistical significance
    significance = pd.DataFrame(
        index=tasks,
        columns=[s for s in strategies if s != 'none']
    )
    
    for task in tasks:
        baseline_data = results[(results['task'] == task) & 
                               (results['verification_strategy'] == 'none')]['accuracy']
        
        for strategy in [s for s in strategies if s != 'none']:
            strategy_data = results[(results['task'] == task) & 
                                  (results['verification_strategy'] == strategy)]['accuracy']
            
            # Perform t-test
            if len(baseline_data) > 0 and len(strategy_data) > 0:
                t_stat, p_value = stats.ttest_ind(strategy_data, baseline_data)
                significance.loc[task, strategy] = p_value < 0.05
    
    # Calculate overall average improvement
    overall_improvement = success_improvement.mean()
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot success rate improvement by task and strategy
    x = np.arange(len(tasks))
    width = 0.8 / len([s for s in strategies if s != 'none'])
    
    for i, strategy in enumerate([s for s in strategies if s != 'none']):
        improvements = [success_improvement.loc[task, strategy] 
                        if not pd.isna(success_improvement.loc[task, strategy]) else 0 
                        for task in tasks]
        
        bars = plt.bar(x + (i - len([s for s in strategies if s != 'none'])/2 + 0.5) * width, 
                improvements, width, label=strategy)
        
        # Add stars for statistical significance
        for j, task in enumerate(tasks):
            if task in significance.index and strategy in significance.columns:
                if significance.loc[task, strategy]:
                    plt.text(x[j] + (i - len([s for s in strategies if s != 'none'])/2 + 0.5) * width, 
                            improvements[j] + 0.01, '*', ha='center', fontsize=12)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Task')
    plt.ylabel('Success Rate Improvement (percentage points)')
    plt.title('Success Rate Improvement by Verification Strategy')
    plt.xticks(x, tasks)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/figures/success_rate_improvement.png", dpi=300)
    
    # Create summary table
    summary = pd.DataFrame({
        'Average_Improvement': overall_improvement
    })
    summary.to_csv(f"{results_dir}/success_rate_improvement_summary.csv")
    
    return success_improvement, overall_improvement