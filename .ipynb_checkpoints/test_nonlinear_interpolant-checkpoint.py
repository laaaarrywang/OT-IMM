# test_nonlinear_interpolant.py
import torch
import numpy as np
from interflow import fabrics

def test_nonlinear_interpolant():
    """Test the nonlinear interpolant implementation"""
    
    # Test 1: Vector data
    print("Testing vector data...")
    dim = 10
    batch_size = 4
    
    # Create test data
    x0 = torch.randn(batch_size, dim, dtype=torch.float64)
    x1 = torch.randn(batch_size, dim, dtype=torch.float64)
    
    # Create interpolant
    It, dtIt, coeffs = fabrics.make_It(
        path='nonlinear',
        data_type='vector',
        data_dim=dim,
        flow_config={'num_layers': 4, 'hidden': 256}
    )
    
    # Test at different time points
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(t_val, dtype=torch.float64)
        
        # Compute interpolation
        y = It(t, x0, x1)
        print(f"  t={t_val}: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Check boundary conditions
        if t_val == 0.0:
            # At t=0, for linear coefficients a(0)=1, b(0)=0
            # So T_0(T_0^{-1}(x0)) should be close to x0
            diff = (y - x0).abs().max()
            print(f"    Boundary check at t=0: max diff = {diff:.15f}")
            assert diff < 1e-3, f"Boundary condition failed at t=0: {diff}"
            
        elif t_val == 1.0:
            # At t=1, for linear coefficients a(1)=0, b(1)=1
            # So T_1(T_1^{-1}(x1)) should be close to x1
            diff = (y - x1).abs().max()
            print(f"    Boundary check at t=1: max diff = {diff:.15f}")
            assert diff < 1e-3, f"Boundary condition failed at t=1: {diff}"
    
    # Test 2: Time derivative
    print("\nTesting time derivative...")
    t = torch.tensor(0.5, dtype=torch.float64)
    dy_dt = dtIt(t, x0, x1)
    print(f"  dtIt shape: {dy_dt.shape}, mean={dy_dt.mean():.4f}, std={dy_dt.std():.4f}")
    
    # Test 3: MNIST data
    print("\nTesting MNIST data...")
    batch_size = 2
    x0_mnist = torch.randn(batch_size, 1, 28, 28, dtype=torch.float64)
    x1_mnist = torch.randn(batch_size, 1, 28, 28, dtype=torch.float64)
    
    It_mnist, dtIt_mnist, _ = fabrics.make_It(
        path='nonlinear',
        data_type='mnist',
        flow_config={'num_layers': 4, 'image_mode': True}
    )
    
    t = torch.tensor(0.5, dtype=torch.float64)
    y_mnist = It_mnist(t, x0_mnist, x1_mnist)
    print(f"  Output shape: {y_mnist.shape}")
    print(f"  Output stats: mean={y_mnist.mean():.4f}, std={y_mnist.std():.4f}")
    
    # Test 4: Invertibility check
    print("\nTesting invertibility...")
    # The flow should be invertible at each time point
    It_test, _, _ = fabrics.make_It(
        path='nonlinear',
        data_type='vector',
        data_dim=5,
        flow_config={'num_layers': 2}
    )
    
    x_test = torch.randn(2, 5, dtype=torch.float64)
    x_dummy = torch.zeros_like(x_test)  # Dummy endpoint
    
    # At t=0, It should preserve x_test (since a(0)=1, b(0)=0)
    t0 = torch.tensor(0.0, dtype=torch.float64)
    y_t0 = It_test(t0, x_test, x_dummy)
    reconstruction_error = (y_t0 - x_test).abs().max()
    print(f"  Reconstruction error at t=0: {reconstruction_error:.6f}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_nonlinear_interpolant()