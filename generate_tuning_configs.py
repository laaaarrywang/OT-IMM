"""
Generate hyperparameter configurations for tuning
Focus: Network structure and learning rate exploration
Fixed: n_inner=500, n_outer=10
"""

import json
from pathlib import Path


def generate_configs():
    """Generate 10 configuration files for hyperparameter search."""

    # Create configs directory
    configs_dir = Path("tune_configs")
    configs_dir.mkdir(exist_ok=True)

    # Fixed settings for all configs
    fixed_settings = {
        "n_epochs": 100,
        "batch_size": 1000,
        "n_inner": 500,  # Fixed
        "n_outer": 10,   # Fixed
        "sched_step_size": 5000,
        "v_hidden_sizes": [256, 256, 256, 256],
        "time_embed_dim": 128,
    }

    # 10 configurations exploring network structure and learning rates
    configs = []

    # Config 1: Baseline (from notebook, but with n_outer=10)
    configs.append({
        **fixed_settings,
        "name": "baseline_500vs10",
        "lr_v": 2e-3,
        "lr_flow": 2e-4,
        "flow_num_layers": 2,
        "flow_hidden": 128,
        "mlp_blocks": 2,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 2: Increase flow LR (2x)
    configs.append({
        **fixed_settings,
        "name": "flow_lr_2x",
        "lr_v": 2e-3,
        "lr_flow": 4e-4,
        "flow_num_layers": 2,
        "flow_hidden": 128,
        "mlp_blocks": 2,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 3: Increase flow LR (5x)
    configs.append({
        **fixed_settings,
        "name": "flow_lr_5x",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 2,
        "flow_hidden": 128,
        "mlp_blocks": 2,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 4: Match flow LR to v LR
    configs.append({
        **fixed_settings,
        "name": "flow_lr_match",
        "lr_v": 2e-3,
        "lr_flow": 2e-3,
        "flow_num_layers": 2,
        "flow_hidden": 128,
        "mlp_blocks": 2,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 5: Deeper flow (4 layers) with moderate LR
    configs.append({
        **fixed_settings,
        "name": "deeper_4layer",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 4,
        "flow_hidden": 128,
        "mlp_blocks": 2,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 6: Wider flow (256 hidden) with moderate LR
    configs.append({
        **fixed_settings,
        "name": "wider_256h",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 2,
        "flow_hidden": 256,
        "mlp_blocks": 2,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 7: Deeper + Wider (4 layers, 256 hidden)
    configs.append({
        **fixed_settings,
        "name": "deeper_wider",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 4,
        "flow_hidden": 256,
        "mlp_blocks": 3,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": True,
        "log_scale_clamp": 1.0,
    })

    # Config 8: Deeper + Wider without spectral norm
    configs.append({
        **fixed_settings,
        "name": "no_specnorm",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 4,
        "flow_hidden": 256,
        "mlp_blocks": 3,
        "fourier_max_freq": 10.0,
        "use_spectral_norm": False,
        "log_scale_clamp": 1.0,
    })

    # Config 9: Relaxed constraints (higher clamp + more frequencies)
    configs.append({
        **fixed_settings,
        "name": "relaxed_constraints",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 4,
        "flow_hidden": 256,
        "mlp_blocks": 3,
        "fourier_max_freq": 50.0,
        "use_spectral_norm": False,
        "log_scale_clamp": 3.0,
    })

    # Config 10: Most expressive (6 layers, 256 hidden, relaxed)
    configs.append({
        **fixed_settings,
        "name": "most_expressive",
        "lr_v": 2e-3,
        "lr_flow": 1e-3,
        "flow_num_layers": 6,
        "flow_hidden": 256,
        "mlp_blocks": 4,
        "fourier_max_freq": 50.0,
        "use_spectral_norm": False,
        "log_scale_clamp": 3.0,
    })

    # Save all configs
    config_files = []
    for config in configs:
        config_path = configs_dir / f"{config['name']}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        config_files.append(str(config_path))

        # Print summary
        print(f"{config['name']:25s} | "
              f"lr_flow={config['lr_flow']:.2e} | "
              f"layers={config['flow_num_layers']} | "
              f"hidden={config['flow_hidden']} | "
              f"mlp_blocks={config['mlp_blocks']} | "
              f"specnorm={config['use_spectral_norm']}")

    print(f"\nTotal configs generated: {len(configs)}")
    print(f"Fixed settings: n_inner=500, n_outer=10")
    print(f"Saved to: {configs_dir}/")

    return config_files


if __name__ == "__main__":
    generate_configs()