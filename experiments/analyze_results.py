import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results_from_json(filepath):
    """Load verification results from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        print(f"Successfully loaded results from {filepath}")
        print(f"Loaded {len(results)} experiment results")
        return results
    except FileNotFoundError:
        print(f"Results file not found: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}")
        return []
    except Exception as e:
        print(f"Unexpected error loading results: {e}")
        return []

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
        
        # Get the length of performance_history from the first row
        # The +1 is needed because we're getting number of iterations
        num_iterations = len(group['performance_history'].iloc[0])
        
        for i in range(num_iterations):
            # Calculate success rate for this iteration across all samples
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
                # Calculate correction rate if needed
                if 'corrections_applied' in vm_df.columns:
                    correction_rate = vm_df['corrections_applied'].mean()
        
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
    results = load_results_from_json("results/verification/verification_results.json")
    output_dir = "results/verification/"
    # Generate summary statistics and visualizations
    analyze_results(results, output_dir)