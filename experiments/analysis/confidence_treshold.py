# experiments/analysis/threshold_analysis.py

def analyze_threshold_impact(results_dir):
    """Analyze the impact of different confidence thresholds."""
    
    # Load results with different thresholds
    results = pd.read_csv(f"{results_dir}/threshold_results.csv")
    
    # Group by threshold and verification strategy
    grouped = results.groupby(['threshold', 'verification_strategy'])
    
    # Calculate metrics
    metrics_summary = grouped.agg({
        'accuracy': ['mean', 'std'],
        'verification_rate': ['mean'],  # % of updates where verification was applied
        'correction_rate': ['mean'],    # % of updates where corrections were applied
        'rejection_rate': ['mean']      # % of updates rejected by verification
    })
    
    # Visualize threshold impact
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    plt.subplot(2, 2, 1)
    for strategy in results['verification_strategy'].unique():
        if strategy == 'none':
            continue
        strat_data = results[results['verification_strategy'] == strategy]
        plt.plot(strat_data['threshold'], strat_data['accuracy'], 
                 marker='o', label=strategy)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for strategy in results['verification_strategy'].unique():
        if strategy == 'none':
            continue
        strat_data = results[results['verification_strategy'] == strategy]
        plt.plot(strat_data['threshold'], strat_data['verification_rate'], 
                 marker='o', label=strategy)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Verification Rate')
    plt.title('Verification Rate vs. Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    for strategy in results['verification_strategy'].unique():
        if strategy == 'none':
            continue
        strat_data = results[results['verification_strategy'] == strategy]
        plt.plot(strat_data['threshold'], strat_data['correction_rate'], 
                 marker='o', label=strategy)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Correction Rate')
    plt.title('Correction Rate vs. Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    for strategy in results['verification_strategy'].unique():
        if strategy == 'none':
            continue
        strat_data = results[results['verification_strategy'] == strategy]
        plt.plot(strat_data['threshold'], strat_data['iterations_to_convergence'], 
                 marker='o', label=strategy)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Iterations to Convergence')
    plt.title('Convergence vs. Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/figures/threshold_analysis.png", dpi=300)
    
    return metrics_summary