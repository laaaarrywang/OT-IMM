"""
Differentiability testing functions for RealNVP flows.
Tests gradients with respect to both time (t) and input (x).
"""

import torch
import numpy as np

def check_differentiability_t(model, x, name, eps_list=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3)):
    """Check differentiability w.r.t. time using autograd vs finite differences"""
    model.eval()
    B = x.shape[0]
    device = x.device
    D = x.view(B, -1).shape[1]  # Total dimension
    
    # random t in [0,1]
    t = torch.rand(B, device=device, dtype=x.dtype, requires_grad=True)

    # Autograd gradient of sum(y_i) w.r.t. t_i (per-sample)
    y, _ = model(x, t)
    f = y.view(B, -1).sum(dim=1)          # [B]
    g_aut, = torch.autograd.grad(f.sum(), t, create_graph=False)  # [B]
    
    # Per-element gradients (normalized by dimension)
    g_aut_per_elem = g_aut / D

    print(f"\n[{name}] Differentiability in time: autograd vs finite-difference")
    print(f"  Dimension: {D:,}")
    print(f"  Total grad stats: mean {g_aut.mean().item():.3e} | std {g_aut.std().item():.3e}")
    print(f"  Per-elem grad:    mean {g_aut_per_elem.mean().item():.3e} | std {g_aut_per_elem.std().item():.3e}")

    # Finite differences
    with torch.no_grad():
        for eps in eps_list:
            t_plus = (t.detach() + eps).clamp(0.0, 1.0)
            y_plus, _ = model(x, t_plus)
            fd = (y_plus - y.detach()) / eps        # same shape as y
            fd_sum = fd.view(B, -1).sum(dim=1)      # [B]

            abs_err = (fd_sum - g_aut.detach()).abs()
            rel_err = abs_err / (g_aut.detach().abs() + 1e-12)
            
            # Per-element errors
            abs_err_per_elem = abs_err / D
            rel_err_per_elem = rel_err  # relative error doesn't change with normalization

            print(f"  eps={eps:>7.1e} | total_err {abs_err.mean():.3e} | "
                  f"per_elem_err {abs_err_per_elem.mean():.3e} | rel_err {rel_err_per_elem.mean():.3e}")

def check_differentiability_x(model, x_template, name, eps_list=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
    """
    Check differentiability w.r.t. input x using autograd vs finite differences.
    
    Args:
        model: The flow model
        x_template: Template tensor for x shape [B, ...] 
        name: Test name for printing
        eps_list: List of epsilon values for finite differences
    """
    model.eval()
    B = x_template.shape[0]
    device = x_template.device
    D = x_template.view(B, -1).shape[1]  # Total dimension
    
    # Create input x that requires grad
    x = x_template.clone().detach().requires_grad_(True)
    t = torch.rand(B, device=device, dtype=x.dtype)  # Fixed time for this test
    
    # Forward pass
    y, logdet = model(x, t)
    
    # Compute scalar output for differentiation (sum of all outputs)
    f = y.view(B, -1).sum(dim=1).sum()  # Single scalar
    
    # Autograd gradient w.r.t. x
    g_aut = torch.autograd.grad(f, x, create_graph=False)[0]  # Same shape as x
    g_aut_flat = g_aut.view(B, -1)  # [B, D]
    
    # Per-element gradient statistics
    g_aut_per_elem = g_aut_flat.abs().mean()

    print(f"\n[{name}] Differentiability in input: autograd vs finite-difference")
    print(f"  Input shape: {x.shape}")
    print(f"  Total dimensions: {D:,}")
    print(f"  Grad stats: mean {g_aut_flat.mean().item():.3e} | std {g_aut_flat.std().item():.3e}")
    print(f"  Grad magnitude: mean |∇x| {g_aut_per_elem.item():.3e}")

    # Finite differences - test a few random directions
    with torch.no_grad():
        # Create random perturbation directions (normalized)
        directions = torch.randn_like(x)
        directions = directions / directions.view(B, -1).norm(dim=1, keepdim=True).view(B, *([1] * (x.dim()-1)))
        
        for eps in eps_list:
            # Perturb input
            x_plus = x.detach() + eps * directions
            
            # Forward pass with perturbed input
            y_plus, logdet_plus = model(x_plus, t)
            f_plus = y_plus.view(B, -1).sum(dim=1).sum()
            
            # Finite difference approximation
            fd_scalar = (f_plus - f.detach()) / eps
            
            # Directional derivative from autograd: ∇f · direction
            directional_grad = (g_aut.detach() * directions).view(B, -1).sum(dim=1).sum()
            
            # Compare
            abs_err = abs(fd_scalar - directional_grad)
            rel_err = abs_err / (abs(directional_grad) + 1e-12)
            
            print(f"  eps={eps:>7.1e} | abs_err {abs_err:.3e} | rel_err {rel_err:.3e}")

def check_jacobian_determinant_consistency(model, x, name):
    """
    Check that the Jacobian determinant computed by the model is consistent 
    with numerical approximation (for small models only due to computational cost).
    """
    model.eval()
    B, *rest_dims = x.shape
    D = x.view(B, -1).shape[1]
    
    # Only test for small dimensions to avoid computational explosion
    if D > 100:
        print(f"\n[{name}] Jacobian determinant check: SKIPPED (D={D} too large)")
        return
    
    print(f"\n[{name}] Jacobian determinant consistency check")
    print(f"  Computing numerical Jacobian for {D}D input...")
    
    device = x.device
    t = torch.rand(B, device=device, dtype=x.dtype)
    
    # Model's log-determinant
    x_flat = x.view(B, -1).requires_grad_(True)
    y_flat, logdet_model = model.transform.forward(x_flat, context=model.time_embed(t))
    
    if B > 1:
        print("  Using first sample only for numerical Jacobian")
        x_flat_single = x_flat[0:1].requires_grad_(True)
        t_single = t[0:1]
        y_single, logdet_single = model.transform.forward(x_flat_single, context=model.time_embed(t_single))
        
        # Compute numerical Jacobian
        jac = torch.zeros(D, D, device=device, dtype=x.dtype)
        for i in range(D):
            grad_i = torch.autograd.grad(y_single[0, i], x_flat_single, 
                                       create_graph=False, retain_graph=True)[0]
            jac[i] = grad_i[0]
        
        # Compute determinant
        logdet_numerical = torch.logdet(jac).item()
        logdet_model_single = logdet_single[0].item()
        
        error = abs(logdet_numerical - logdet_model_single)
        rel_error = error / (abs(logdet_model_single) + 1e-12)
        
        print(f"  Model logdet:     {logdet_model_single:.6f}")
        print(f"  Numerical logdet: {logdet_numerical:.6f}")
        print(f"  Absolute error:   {error:.6e}")
        print(f"  Relative error:   {rel_error:.6e}")
        
        if rel_error < 1e-4:
            print("  ✅ PASS: Jacobian determinant consistent")
        else:
            print("  ⚠️  MARGINAL: Large relative error in logdet")
    else:
        print("  Need batch size > 1 for this test")

def run_comprehensive_differentiability_tests(model, x, name):
    """Run all differentiability tests for a model"""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE DIFFERENTIABILITY TESTS: {name}")
    print(f"{'='*60}")
    
    # Test 1: Differentiability w.r.t. time
    check_differentiability_t(model, x, name)
    
    # Test 2: Differentiability w.r.t. input
    check_differentiability_x(model, x, name)
    
    # Test 3: Jacobian determinant consistency (for small models)
    check_jacobian_determinant_consistency(model, x, name)

# Example usage functions for different model types
def test_vector_differentiability(device="cpu"):
    """Test differentiability for vector flow"""
    from realnvp import create_vector_flow
    
    model = create_vector_flow(dim=8).to(device)  # Small for Jacobian test
    x = torch.randn(4, 8, device=device, dtype=torch.float64)
    
    run_comprehensive_differentiability_tests(model, x, "Vector (8D)")

def test_mnist_differentiability(device="cpu"):
    """Test differentiability for MNIST flow (skip Jacobian due to size)"""
    from realnvp import create_mnist_flow
    
    model = create_mnist_flow().to(device)
    x = torch.randn(2, 1, 28, 28, device=device, dtype=torch.float64)
    
    # Only test grad w.r.t. t and x (skip Jacobian for 784D)
    check_differentiability_t(model, x, "MNIST (784D)")
    check_differentiability_x(model, x, "MNIST (784D)")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    test_vector_differentiability(device)
    test_mnist_differentiability(device) 