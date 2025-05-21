# experiments/analysis/error_analysis.py

def analyze_error_reduction(results_dir):
    """Analyze error reduction by verification strategies."""
    
    # Load error data
    error_data = pd.read_csv(f"{results_dir}/error_analysis.csv")
    
    # Define error categories
    error_categories = [
        'hallucination',
        'logical_error',
        'mathematical_error',
        'factual_error',
        'syntax_error',
        'other_error'
    ]
    
    # Group by task and verification strategy
    grouped = error_data.groupby(['task', 'verification_strategy'])
    
    # Calculate error reduction rates
    error_reduction = {}
    for task in error_data['task'].unique():
        baseline_errors = error_data[(error_data['task'] == task) & 
                                    (error_data['verification_strategy'] == 'none')]
        
        for strategy in [s for s in error_data['verification_strategy'].unique() if s != 'none']:
            strategy_errors = error_data[(error_data['task'] == task) & 
                                        (error_data['verification_strategy'] == strategy)]
            
            # Calculate reduction percentage for each error type
            reduction_rates = {}
            for error_type in error_categories:
                baseline_rate = baseline_errors[error_type].mean()
                strategy_rate = strategy_errors[error_type].mean()
                
                if baseline_rate > 0:
                    reduction = (baseline_rate - strategy_rate) / baseline_rate * 100
                else:
                    reduction = 0
                    
                reduction_rates[error_type] = reduction
                
            error_reduction[(task, strategy)] = reduction_rates
    
    # Convert to DataFrame for easier analysis
    error_df = pd.DataFrame(error_reduction).T
    error_df.index.names = ['task', 'strategy']
    
    # Visualize error reduction
    plt.figure(figsize=(14, 10))
    
    # Create a heatmap of error reduction by strategy and error type
    error_pivot = error_df.reset_index().melt(
        id_vars=['task', 'strategy'], 
        value_vars=error_categories,
        var_name='error_type',
        value_name='reduction_percentage'
    )
    
    for task in error_data['task'].unique():
        task_data = error_pivot[error_pivot['task'] == task]
        
        # Create a pivot table for the heatmap
        heatmap_data = task_data.pivot(index='strategy', columns='error_type', values='reduction_percentage')
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Error Reduction %'})
        plt.title(f'Error Reduction by Verification Strategy: {task}')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/figures/error_reduction_{task}.png", dpi=300)
    
    return error_df