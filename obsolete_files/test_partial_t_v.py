#!/usr/bin/env python3
"""
Automated debugging script for testing E[|∂_t v_t|²] computation
Runs in conda environment OT_IMM
"""

import os
import sys
import torch
import numpy as np
import traceback
from typing import Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
import interflow as itf
import interflow.prior as prior
import interflow.fabrics_extra
import interflow.stochastic_interpolant as stochastic_interpolant

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    itf.util.set_torch_device('cuda')

def grab(var):
    """Take a tensor off the gpu and convert it to a numpy array on the CPU."""
    return var.detach().cpu().numpy()

def create_test_setup():
    """Create minimal test setup with all required components."""
    print("\n=== Creating Test Setup ===")

    # Define dimensions
    ndim = 2

    # Create base distribution (Gaussian)
    base_loc = torch.zeros(ndim, device=device, dtype=torch.float32)
    base_var = torch.ones(ndim, device=device, dtype=torch.float32)
    base = prior.SimpleNormal(base_loc, 3*base_var)
    print("✓ Base distribution created")

    # Create target distribution (simple rectangular for testing)
    def target_rect(bs, device=None):
        if device is None:
            device = torch.device('cpu')
        x = torch.rand(bs, 2, device=device) * 20 - 10  # Range [-10, 10]
        return x

    print("✓ Target distribution created")

    # Create interpolant
    path = "nonlinear"
    flow_config = {
        "num_layers": 2,
        "time_embed_dim": 64,  # Smaller for testing
        "hidden": 64,
        "mlp_blocks": 2,
        "activation": "gelu",
        "use_layernorm": False,
        "use_permutation": True,
        "fourier_min_freq": 1.0,
        "fourier_max_freq": 10.0,
        "vector_use_spectral_norm": True,
        "vector_log_scale_clamp": 1.0,
        "vector_use_soft_clamp": True,
    }

    interpolant = stochastic_interpolant.Interpolant(
        path=path,
        gamma_type=None,
        flow_config=flow_config,
        data_type="vector",
        data_dim=ndim
    )
    print("✓ Interpolant created")

    # Create velocity model v
    v = itf.fabrics_extra.make_resnet(
        hidden_sizes=[64, 64],  # Smaller for testing
        in_size=ndim+1,
        out_size=ndim,
        inner_act='gelu',
        use_layernorm=True,
        dropout=0.0
    )
    print("✓ Velocity model created")

    return base, target_rect, interpolant, v

def test_gradient_computation(v, interpolant, base, target, device):
    """Test basic gradient computation with minimal setup."""
    print("\n=== Testing Basic Gradient Computation ===")

    v.eval()
    interpolant.flow_model.eval()

    try:
        # Single sample test
        t = torch.tensor(0.5, device=device, requires_grad=True).unsqueeze(0)
        x0 = base(1).to(device)
        x1 = target(1, device=device)

        print(f"t shape: {t.shape}, x0 shape: {x0.shape}, x1 shape: {x1.shape}")

        # Compute interpolated point
        xt = interpolant.calc_xt(t, x0, x1)
        if isinstance(xt, tuple):
            xt = xt[0]
        print(f"xt shape: {xt.shape}")

        # Compute velocity
        vt = v(xt, t)
        print(f"vt shape: {vt.shape}, values: {vt}")

        # Test gradient computation
        grad = torch.autograd.grad(
            outputs=vt[0, 0],  # Single output
            inputs=t,
            create_graph=False,
            retain_graph=False
        )[0]
        print(f"✓ Gradient computed successfully: {grad}")
        return True

    except Exception as e:
        print(f"✗ Error in basic gradient computation: {e}")
        traceback.print_exc()
        return False

def estimate_partial_t_v_squared_norm_debug(
    v: torch.nn.Module,
    interpolant,
    base,
    target,
    n_samples: int = 100,
    batch_size: int = 10,
    device=None,
    verbose=True
):
    """
    Debug version of E[|∂_t v_t|²] estimation with detailed logging.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n=== Computing E[|∂_t v_t|²] ===")
        print(f"Samples: {n_samples}, Batch size: {batch_size}")

    v.eval()
    interpolant.flow_model.eval()

    partial_t_v_squared_norms = []

    # Determine sampling strategy
    n_time_samples = max(1, n_samples // batch_size)
    samples_per_t = min(batch_size, n_samples)

    if verbose:
        print(f"Time samples: {n_time_samples}, Samples per t: {samples_per_t}")

    ts = torch.linspace(0.1, 0.9, n_time_samples, device=device)

    for idx, t_val in enumerate(ts):
        if verbose and idx % 5 == 0:
            print(f"  Processing time point {idx+1}/{n_time_samples} (t={t_val:.2f})")

        try:
            # Make t require gradients
            t = t_val.unsqueeze(0).requires_grad_(True)

            # Sample data
            x0s = base(samples_per_t).to(device)
            x1s = target(samples_per_t, device=device)

            # Compute interpolated points
            xts = interpolant.calc_xt(t, x0s, x1s)
            if isinstance(xts, tuple):
                xts = xts[0]

            # Evaluate velocity
            vts = v(xts, t)

            # Method 1: Vectorized gradient computation (more efficient)
            try:
                # Use vmap if available, otherwise fall back to loop
                from functorch import vmap

                def compute_grad_for_sample(v_sample):
                    return torch.autograd.grad(
                        outputs=v_sample.sum(),
                        inputs=t,
                        create_graph=False,
                        retain_graph=True
                    )[0]

                # This would be ideal but may not work with current setup
                # partial_t_v_batch = vmap(compute_grad_for_sample)(vts)

                # Fallback to loop
                raise ImportError("Using loop method")

            except (ImportError, RuntimeError):
                # Method 2: Loop-based computation (more reliable)
                partial_t_v_list = []

                for i in range(vts.shape[0]):
                    v_i = vts[i:i+1]

                    # Compute gradient using Jacobian-vector product
                    grad_outputs = torch.ones_like(v_i)
                    grad_i = torch.autograd.grad(
                        outputs=v_i,
                        inputs=t,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        retain_graph=(i < vts.shape[0]-1)  # Retain for all but last
                    )[0]

                    # grad_i has shape [1], broadcast to match dimensions
                    partial_t_v_list.append(grad_i.expand(vts.shape[-1]))

                partial_t_v_batch = torch.stack(partial_t_v_list, dim=0)

            # Compute squared norm
            partial_t_v_squared = torch.sum(partial_t_v_batch ** 2, dim=-1)
            partial_t_v_squared_norms.extend(partial_t_v_squared.detach().cpu().numpy())

        except Exception as e:
            print(f"  ✗ Error at t={t_val:.2f}: {e}")
            if verbose:
                traceback.print_exc()
            continue

    v.train()
    interpolant.flow_model.train()

    if len(partial_t_v_squared_norms) == 0:
        print("✗ No successful computations!")
        return None

    result = float(np.mean(partial_t_v_squared_norms))
    if verbose:
        print(f"✓ E[|∂_t v_t|²] = {result:.4f}")
        print(f"  Min: {np.min(partial_t_v_squared_norms):.4f}")
        print(f"  Max: {np.max(partial_t_v_squared_norms):.4f}")
        print(f"  Std: {np.std(partial_t_v_squared_norms):.4f}")

    return result

def estimate_v_squared_norm_debug(
    v: torch.nn.Module,
    interpolant,
    base,
    target,
    n_samples: int = 100,
    batch_size: int = 10,
    device=None,
    verbose=True
):
    """
    Debug version of E[|v_t|²] estimation for comparison.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n=== Computing E[|v_t|²] for comparison ===")

    v.eval()
    interpolant.flow_model.eval()

    v_squared_norms = []

    with torch.no_grad():
        n_time_samples = max(1, n_samples // batch_size)
        samples_per_t = min(batch_size, n_samples)

        ts = torch.linspace(0, 1, n_time_samples, device=device)
        for t in ts:
            t = t.unsqueeze(0)

            x0s = base(samples_per_t).to(device)
            x1s = target(samples_per_t, device=device)

            xts = interpolant.calc_xt(t, x0s, x1s)
            if isinstance(xts, tuple):
                xts = xts[0]

            vts = v(xts, t)

            v_squared = torch.sum(vts ** 2, dim=-1)
            v_squared_norms.extend(v_squared.cpu().numpy())

    v.train()
    interpolant.flow_model.train()

    result = float(np.mean(v_squared_norms))
    if verbose:
        print(f"✓ E[|v_t|²] = {result:.4f}")

    return result

def run_full_test():
    """Run complete test suite."""
    print("="*60)
    print("AUTOMATED DEBUGGING FOR E[|∂_t v_t|²] COMPUTATION")
    print("="*60)

    # Create test setup
    base, target_rect, interpolant, v = create_test_setup()

    # Test 1: Basic gradient computation
    success = test_gradient_computation(v, interpolant, base, target_rect, device)
    if not success:
        print("\n✗ Basic gradient test failed. Stopping.")
        return False

    # Test 2: Small scale computation
    print("\n=== Test 2: Small Scale Computation ===")
    try:
        result = estimate_partial_t_v_squared_norm_debug(
            v, interpolant, base, target_rect,
            n_samples=20, batch_size=5, device=device, verbose=True
        )
        if result is not None:
            print(f"✓ Small scale test passed: E[|∂_t v_t|²] = {result:.4f}")
        else:
            print("✗ Small scale test failed")
            return False
    except Exception as e:
        print(f"✗ Small scale test failed: {e}")
        traceback.print_exc()
        return False

    # Test 3: Compare with E[|v_t|²]
    print("\n=== Test 3: Comparison Test ===")
    try:
        v_norm = estimate_v_squared_norm_debug(
            v, interpolant, base, target_rect,
            n_samples=100, batch_size=10, device=device, verbose=True
        )
        partial_t_v_norm = estimate_partial_t_v_squared_norm_debug(
            v, interpolant, base, target_rect,
            n_samples=100, batch_size=10, device=device, verbose=True
        )

        print(f"\nComparison:")
        print(f"  E[|v_t|²] = {v_norm:.4f}")
        print(f"  E[|∂_t v_t|²] = {partial_t_v_norm:.4f}")
        print(f"  Ratio = {partial_t_v_norm/v_norm:.4f}")

    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        traceback.print_exc()
        return False

    # Test 4: Larger scale
    print("\n=== Test 4: Larger Scale Test ===")
    try:
        result = estimate_partial_t_v_squared_norm_debug(
            v, interpolant, base, target_rect,
            n_samples=1000, batch_size=100, device=device, verbose=False
        )
        print(f"✓ Large scale test: E[|∂_t v_t|²] = {result:.4f}")
    except Exception as e:
        print(f"✗ Large scale test failed: {e}")
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)