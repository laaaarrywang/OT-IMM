"""
Analyze and compare hyperparameter tuning results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_results(results_dir="results0929"):
    """Load all experiment results."""
    results_dir = Path(results_dir)

    all_results = []

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        config_file = exp_dir / "config.json"
        metrics_file = exp_dir / "metrics.npz"

        if not config_file.exists() or not metrics_file.exists():
            print(f"Skipping {exp_dir.name} - missing files")
            continue

        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Load metrics
        metrics = np.load(metrics_file)

        result = {
            'name': config['name'],
            'config': config,
            'v_losses': metrics['v_losses'],
            'flow_losses': metrics['flow_losses'],
            'v_grads': metrics['v_grads'],
            'flow_grads': metrics['flow_grads'],
            'v_squared_norm': metrics['v_squared_norm'],
        }

        all_results.append(result)

    print(f"Loaded {len(all_results)} experiments")
    return all_results


def create_comparison_plots(results, save_path="results0929/comparison.png"):
    """Create comparison plots for all experiments."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for result in results:
        name = result['name']
        epochs = np.arange(len(result['v_losses']))

        # Plot v loss
        axes[0, 0].plot(epochs, result['v_losses'], label=name, alpha=0.7)

        # Plot flow loss
        axes[0, 1].plot(epochs, result['flow_losses'], label=name, alpha=0.7)

        # Plot transport cost
        axes[0, 2].plot(epochs, result['v_squared_norm'], label=name, alpha=0.7, linewidth=2)

        # Plot v gradient
        axes[1, 0].plot(epochs, result['v_grads'], label=name, alpha=0.7)

        # Plot flow gradient
        axes[1, 1].plot(epochs, result['flow_grads'], label=name, alpha=0.7)

    # Configure axes
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('v Loss')
    axes[0, 0].set_title('Velocity Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Flow Loss')
    axes[0, 1].set_title('Flow Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('E[|v|²]')
    axes[0, 2].set_title('Transport Cost E[|v|²] (Lower is Better)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('v Gradient Norm')
    axes[1, 0].set_title('v Gradient Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Flow Gradient Norm')
    axes[1, 1].set_title('Flow Gradient Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary table in last subplot
    axes[1, 2].axis('off')

    # Create summary table
    summary_data = []
    for result in results:
        final_cost = result['v_squared_norm'][-1]
        summary_data.append({
            'Config': result['name'],
            'Final E[|v|²]': f"{final_cost:.4f}",
            'lr_v': f"{result['config']['lr_v']:.2e}",
            'lr_flow': f"{result['config']['lr_flow']:.2e}",
            'inner/outer': f"{result['config']['n_inner']}/{result['config']['n_outer']}",
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('Final E[|v|²]')

    table_text = df.to_string(index=False)
    axes[1, 2].text(0.1, 0.5, table_text, fontsize=9, family='monospace',
                    verticalalignment='center')
    axes[1, 2].set_title('Summary (sorted by Final E[|v|²])', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")

    return df


def print_summary(results):
    """Print detailed summary of results."""

    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*60)

    # Sort by final transport cost
    results_sorted = sorted(results, key=lambda x: x['v_squared_norm'][-1])

    print(f"\nRanked by Final E[|v|²] (lower is better):")
    print("-"*60)

    for i, result in enumerate(results_sorted, 1):
        config = result['config']
        final_cost = result['v_squared_norm'][-1]
        initial_cost = result['v_squared_norm'][0]
        improvement = (initial_cost - final_cost) / initial_cost * 100

        print(f"\n{i}. {result['name']}")
        print(f"   Final E[|v|²]:    {final_cost:.6f}")
        print(f"   Improvement:      {improvement:.2f}%")
        print(f"   lr_v/lr_flow:     {config['lr_v']:.2e} / {config['lr_flow']:.2e}")
        print(f"   Inner/Outer:      {config['n_inner']} / {config['n_outer']}")
        print(f"   Flow arch:        {config['flow_num_layers']} layers, {config['flow_hidden']} hidden")
        print(f"   Spectral norm:    {config['use_spectral_norm']}")
        print(f"   Log scale clamp:  {config['log_scale_clamp']}")

    print("\n" + "="*60)
    print(f"Best configuration: {results_sorted[0]['name']}")
    print(f"Best E[|v|²]: {results_sorted[0]['v_squared_norm'][-1]:.6f}")
    print("="*60)


if __name__ == "__main__":
    # Load results
    results = load_results()

    if len(results) == 0:
        print("No results found. Make sure experiments have completed.")
        exit(1)

    # Create comparison plots
    df = create_comparison_plots(results)

    # Print summary
    print_summary(results)

    # Save summary table
    df.to_csv("results0929/summary.csv", index=False)
    print(f"\nSummary table saved to: results/summary.csv")