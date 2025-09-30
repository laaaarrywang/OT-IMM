#!/usr/bin/env python
"""
Generate hyperparameter configurations for checker experiments.
Based on the analysis from chat_session_stochastic_interpolants.md
"""

import json
import os

def generate_configs():
    """Generate all hyperparameter configurations for sweep."""

    # Base configuration that stays constant
    base_config = {
        "N_epoch": 100,
        "batch_size": 1000,
        "N_warmup": 50,
        "scheduler_step_size": 1500,
        "scheduler_gamma": 0.8,
        "plot_freq": 1,  # Save plot every epoch
        "v_hidden_sizes": [256, 256, 256, 256],
        "activation": "gelu",
        "use_layernorm": True,  # Keep for stability
        "use_permutation": True,
    }

    configs = []

    # Config 1: Original baseline
    config1 = base_config.copy()
    config1.update({
        "config_name": "baseline",
        "num_layers": 2,
        "hidden": 128,
        "mlp_blocks": 2,
        "fourier_min_freq": 1.0,
        "fourier_max_freq": 10.0,
        "lr_v": 2e-3,
        "lr_flow": 2e-4,
        "n_inner": 500,
        "n_outer": 5,
        "vector_use_spectral_norm": True,
        "vector_log_scale_clamp": 1.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 128,
    })
    configs.append(config1)

    # Config 2: Enhanced for rectangular (recommended in chat)
    config2 = base_config.copy()
    config2.update({
        "config_name": "enhanced_rect",
        "num_layers": 4,
        "hidden": 512,
        "mlp_blocks": 3,
        "fourier_min_freq": 0.5,
        "fourier_max_freq": 30.0,
        "lr_v": 2e-3,
        "lr_flow": 1e-4,  # Lower for stability
        "n_inner": 500,
        "n_outer": 10,  # More outer steps
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 3.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config2)

    # Config 3: High frequency focus
    config3 = base_config.copy()
    config3.update({
        "config_name": "high_freq",
        "num_layers": 4,
        "hidden": 256,
        "mlp_blocks": 3,
        "fourier_min_freq": 0.5,
        "fourier_max_freq": 50.0,  # Very high
        "lr_v": 1e-3,  # Lower LR for stability
        "lr_flow": 1e-4,
        "n_inner": 500,
        "n_outer": 5,
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 2.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config3)

    # Config 4: Deep network
    config4 = base_config.copy()
    config4.update({
        "config_name": "deep_net",
        "num_layers": 6,
        "hidden": 256,
        "mlp_blocks": 4,
        "fourier_min_freq": 0.5,
        "fourier_max_freq": 30.0,
        "lr_v": 1e-3,
        "lr_flow": 1e-4,
        "n_inner": 100,  # Less inner steps
        "n_outer": 10,  # More outer steps
        "vector_use_spectral_norm": True,
        "vector_log_scale_clamp": 2.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config4)

    # Config 5: Wide network
    config5 = base_config.copy()
    config5.update({
        "config_name": "wide_net",
        "num_layers": 3,
        "hidden": 512,
        "mlp_blocks": 2,
        "fourier_min_freq": 1.0,
        "fourier_max_freq": 30.0,
        "lr_v": 2e-3,
        "lr_flow": 2e-4,
        "n_inner": 500,
        "n_outer": 5,
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 2.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config5)

    # Config 6: Balanced with more outer steps
    config6 = base_config.copy()
    config6.update({
        "config_name": "balanced_outer",
        "num_layers": 3,
        "hidden": 256,
        "mlp_blocks": 3,
        "fourier_min_freq": 0.5,
        "fourier_max_freq": 30.0,
        "lr_v": 2e-3,
        "lr_flow": 1e-4,
        "n_inner": 100,
        "n_outer": 20,  # Many outer steps
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 2.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config6)

    # Config 7: Conservative approach
    config7 = base_config.copy()
    config7.update({
        "config_name": "conservative",
        "num_layers": 2,
        "hidden": 256,
        "mlp_blocks": 2,
        "fourier_min_freq": 1.0,
        "fourier_max_freq": 20.0,
        "lr_v": 1e-3,
        "lr_flow": 5e-5,  # Very low
        "n_inner": 500,
        "n_outer": 1,
        "vector_use_spectral_norm": True,
        "vector_log_scale_clamp": 1.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 128,
    })
    configs.append(config7)

    # Config 8: Aggressive without constraints
    config8 = base_config.copy()
    config8.update({
        "config_name": "aggressive",
        "num_layers": 4,
        "hidden": 512,
        "mlp_blocks": 4,
        "fourier_min_freq": 0.5,
        "fourier_max_freq": 50.0,
        "lr_v": 2e-3,
        "lr_flow": 2e-4,
        "n_inner": 100,
        "n_outer": 10,
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 3.0,
        "vector_use_soft_clamp": False,  # No soft clamp
        "time_embed_dim": 256,
    })
    configs.append(config8)

    # Config 9: Very high frequency
    config9 = base_config.copy()
    config9.update({
        "config_name": "very_high_freq",
        "num_layers": 3,
        "hidden": 256,
        "mlp_blocks": 3,
        "fourier_min_freq": 0.1,  # Very low min
        "fourier_max_freq": 100.0,  # Very high max
        "lr_v": 1e-3,
        "lr_flow": 1e-4,
        "n_inner": 500,
        "n_outer": 5,
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 2.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config9)

    # Config 10: Focus on outer optimization
    config10 = base_config.copy()
    config10.update({
        "config_name": "outer_focused",
        "num_layers": 4,
        "hidden": 256,
        "mlp_blocks": 3,
        "fourier_min_freq": 0.5,
        "fourier_max_freq": 30.0,
        "lr_v": 2e-3,
        "lr_flow": 2e-4,  # Higher flow LR
        "n_inner": 50,  # Much less inner
        "n_outer": 50,  # Much more outer
        "vector_use_spectral_norm": False,
        "vector_log_scale_clamp": 2.0,
        "vector_use_soft_clamp": True,
        "time_embed_dim": 256,
    })
    configs.append(config10)

    return configs

def save_configs():
    """Save configurations to individual JSON files."""
    configs = generate_configs()

    # Create configs directory
    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)

    for i, config in enumerate(configs):
        config_name = config.get("config_name", f"config_{i:03d}")
        config["output_dir"] = f"results/{config_name}"

        filename = os.path.join(config_dir, f"{config_name}.json")
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved {filename}")

    print(f"\nTotal configurations generated: {len(configs)}")
    return configs

if __name__ == "__main__":
    configs = save_configs()

    # Print summary
    print("\nConfiguration Summary:")
    print("-" * 60)
    for config in configs:
        name = config.get("config_name", "unnamed")
        print(f"\n{name}:")
        print(f"  Network: layers={config['num_layers']}, hidden={config['hidden']}, blocks={config['mlp_blocks']}")
        print(f"  Fourier: min={config['fourier_min_freq']}, max={config['fourier_max_freq']}")
        print(f"  Learning: lr_v={config['lr_v']}, lr_flow={config['lr_flow']}")
        print(f"  Training: n_inner={config['n_inner']}, n_outer={config['n_outer']}")
        print(f"  Constraints: spectral_norm={config['vector_use_spectral_norm']}, "
              f"log_clamp={config['vector_log_scale_clamp']}")