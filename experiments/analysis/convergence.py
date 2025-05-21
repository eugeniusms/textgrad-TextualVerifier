# experiments/analysis/convergence_analysis.py

def analyze_convergence(results_dir):
    """Analyze convergence behavior with different verification strategies."""
    
    # Load convergence data
    convergence_data = pd.read_csv(f"{results_dir}/convergence_data.csv")
    
    # Group by task, verification strategy, and iteration
    grouped = convergence_data.groupby(['task', 'verification_strategy', 'iteration'])
    
    # Calculate metrics per iteration
    iter_metrics = grouped.agg({
        'loss': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'verification_applied': ['mean'],  # % of cases where verification was applied
        'verification_confidence': ['mean', 'std']  # Confidence scores
    })
    
    # Plot convergence curves
    for task in convergence_data['task'].unique():
        plt.figure(figsize=(14, 8))
        
        # Plot loss curves
        plt.subplot(2, 2, 1)
        for strategy in convergence_data['verification_strategy'].unique():
            task_strat_data = iter_metrics.loc[(task, strategy)]
            plt.plot(task_strat_data.index, 
                     task_strat_data[('loss', 'mean')], 
                     marker='o', label=strategy)
            
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Loss Convergence for {task}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy curves
        plt.subplot(2, 2, 2)
        for strategy in convergence_data['verification_strategy'].unique():
            task_strat_data = iter_metrics.loc[(task, strategy)]
            plt.plot(task_strat_data.index, 
                     task_strat_data[('accuracy', 'mean')], 
                     marker='o', label=strategy)
            
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Convergence for {task}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot verification application rate
        plt.subplot(2, 2, 3)
        for strategy in [s for s in convergence_data['verification_strategy'].unique() if s != 'none']:
            task_strat_data = iter_metrics.loc[(task, strategy)]
            plt.plot(task_strat_data.index, 
                     task_strat_data[('verification_applied', 'mean')], 
                     marker='o', label=strategy)
            
        plt.xlabel('Iteration')
        plt.ylabel('Verification Rate')
        plt.title(f'Verification Application Rate for {task}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot verification confidence
        plt.subplot(2, 2, 4)
        for strategy in [s for s in convergence_data['verification_strategy'].unique() if s != 'none']:
            task_strat_data = iter_metrics.loc[(task, strategy)]
            plt.plot(task_strat_data.index, 
                     task_strat_data[('verification_confidence', 'mean')], 
                     marker='o', label=strategy)
            
        plt.xlabel('Iteration')
        plt.ylabel('Verification Confidence')
        plt.title(f'Verification Confidence for {task}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/figures/convergence_{task}.png", dpi=300)
        
    return iter_metrics