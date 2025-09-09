# test_nonlinear_interpolant.py
import torch
import numpy as np
from interflow import fabrics

def test_nonlinear_interpolant():
    """Test the nonlinear interpolant implementation"""

    # Test 1: Vector data
    print("="*60)
    print("Testing VECTOR DATA (10-dim)...")
    print("="*60)
    dim = 10
    batch_size = 4

    # Create test data
    x0 = torch.randn(batch_size, dim, dtype=torch.float32)
    x1 = torch.randn(batch_size, dim, dtype=torch.float32)

    # Create interpolant
    It, dtIt, coeffs = fabrics.make_It(
        path='nonlinear',
        data_type='vector',
        data_dim=dim,
        flow_config={'num_layers': 4, 'hidden': 256}
    )

    # Test at different time points
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(t_val, dtype=torch.float32)

        # Compute interpolation
        y = It(t, x0, x1)
        print(f"  t={t_val}: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")

        # Check boundary conditions
        if t_val == 0.0:
            diff = (y - x0).abs().max()
            print(f"    Boundary check at t=0: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=0: {diff}"

        elif t_val == 1.0:
            diff = (y - x1).abs().max()
            print(f"    Boundary check at t=1: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=1: {diff}"

    # Test time derivative
    print("\n  Testing time derivative...")
    t = torch.tensor(0.5, dtype=torch.float32)
    dy_dt = dtIt(t, x0, x1)
    print(f"  dtIt shape: {dy_dt.shape}, mean={dy_dt.mean():.4f}, std={dy_dt.std():.4f}")

    # Test 2: MNIST data
    print("\n" + "="*60)
    print("Testing MNIST DATA (1x28x28)...")
    print("="*60)
    batch_size = 2
    x0_mnist = torch.randn(batch_size, 1, 28, 28, dtype=torch.float32)
    x1_mnist = torch.randn(batch_size, 1, 28, 28, dtype=torch.float32)

    It_mnist, dtIt_mnist, _ = fabrics.make_It(
        path='nonlinear',
        data_type='mnist',
        flow_config={'num_layers': 4, 'image_mode': True}
    )

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(t_val, dtype=torch.float32)
        y_mnist = It_mnist(t, x0_mnist, x1_mnist)
        print(f"  t={t_val}: shape={y_mnist.shape}, mean={y_mnist.mean():.4f}, std={y_mnist.std():.4f}")
        
        # Check boundary conditions
        if t_val == 0.0:
            diff = (y_mnist - x0_mnist).abs().max()
            print(f"    Boundary check at t=0: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=0: {diff}"
        elif t_val == 1.0:
            diff = (y_mnist - x1_mnist).abs().max()
            print(f"    Boundary check at t=1: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=1: {diff}"

    print("\n  Testing time derivative...")
    t = torch.tensor(0.5, dtype=torch.float32)
    dy_dt_mnist = dtIt_mnist(t, x0_mnist, x1_mnist)
    print(f"  dtIt shape: {dy_dt_mnist.shape}, mean={dy_dt_mnist.mean():.4f}, std={dy_dt_mnist.std():.4f}")

    # Test 3: CIFAR-10 data
    print("\n" + "="*60)
    print("Testing CIFAR-10 DATA (3x32x32)...")
    print("="*60)
    batch_size = 2
    x0_cifar = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)
    x1_cifar = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)

    It_cifar, dtIt_cifar, _ = fabrics.make_It(
        path='nonlinear',
        data_type='cifar10',
        flow_config={'num_layers': 4}
    )

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(t_val, dtype=torch.float32)
        y_cifar = It_cifar(t, x0_cifar, x1_cifar)
        print(f"  t={t_val}: shape={y_cifar.shape}, mean={y_cifar.mean():.4f}, std={y_cifar.std():.4f}")
        
        # Check boundary conditions
        if t_val == 0.0:
            diff = (y_cifar - x0_cifar).abs().max()
            print(f"    Boundary check at t=0: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=0: {diff}"
        elif t_val == 1.0:
            diff = (y_cifar - x1_cifar).abs().max()
            print(f"    Boundary check at t=1: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=1: {diff}"

    print("\n  Testing time derivative...")
    t = torch.tensor(0.5, dtype=torch.float32)
    dy_dt_cifar = dtIt_cifar(t, x0_cifar, x1_cifar)
    print(f"  dtIt shape: {dy_dt_cifar.shape}, mean={dy_dt_cifar.mean():.4f}, std={dy_dt_cifar.std():.4f}")

    # Test 4: ImageNet 128x128
    print("\n" + "="*60)
    print("Testing IMAGENET 128x128 DATA...")
    print("="*60)
    batch_size = 2
    x0_img128 = torch.randn(batch_size, 3, 128, 128, dtype=torch.float32)
    x1_img128 = torch.randn(batch_size, 3, 128, 128, dtype=torch.float32)

    It_img128, dtIt_img128, _ = fabrics.make_It(
        path='nonlinear',
        data_type='image',
        data_dim=(3, 128, 128),
        flow_config={'num_layers': 4, 'img_base_channels': 64, 'img_num_blocks': 3}  # Full config for powerful machines
    )

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(t_val, dtype=torch.float32)
        y_img128 = It_img128(t, x0_img128, x1_img128)
        print(f"  t={t_val}: shape={y_img128.shape}, mean={y_img128.mean():.4f}, std={y_img128.std():.4f}")
        
        # Check boundary conditions
        if t_val == 0.0:
            diff = (y_img128 - x0_img128).abs().max()
            print(f"    Boundary check at t=0: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=0: {diff}"
        elif t_val == 1.0:
            diff = (y_img128 - x1_img128).abs().max()
            print(f"    Boundary check at t=1: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=1: {diff}"

    print("\n  Testing time derivative...")
    t = torch.tensor(0.5, dtype=torch.float32)
    dy_dt_img128 = dtIt_img128(t, x0_img128, x1_img128)
    print(f"  dtIt shape: {dy_dt_img128.shape}, mean={dy_dt_img128.mean():.4f}, std={dy_dt_img128.std():.4f}")

    # Test 5: ImageNet 224x224 (using stable version)
    print("\n" + "="*60)
    print("Testing IMAGENET 224x224 DATA (stable version)...")
    print("="*60)
    batch_size = 2  # Full batch size for powerful machines
    x0_img224 = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    x1_img224 = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

    It_img224, dtIt_img224, _ = fabrics.make_It(
        path='nonlinear',
        data_type='imagenet_stable',
        flow_config={'resolution': 224, 'num_layers': 3}  # Increased layers for better quality
    )

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor(t_val, dtype=torch.float32)
        y_img224 = It_img224(t, x0_img224, x1_img224)
        print(f"  t={t_val}: shape={y_img224.shape}, mean={y_img224.mean():.4f}, std={y_img224.std():.4f}")
        
        # Check boundary conditions
        if t_val == 0.0:
            diff = (y_img224 - x0_img224).abs().max()
            print(f"    Boundary check at t=0: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=0: {diff}"
        elif t_val == 1.0:
            diff = (y_img224 - x1_img224).abs().max()
            print(f"    Boundary check at t=1: max diff = {diff:.6f}")
            assert diff < 1e-3, f"Boundary condition failed at t=1: {diff}"

    print("\n  Testing time derivative...")
    t = torch.tensor(0.5, dtype=torch.float32)
    dy_dt_img224 = dtIt_img224(t, x0_img224, x1_img224)
    print(f"  dtIt shape: {dy_dt_img224.shape}, mean={dy_dt_img224.mean():.4f}, std={dy_dt_img224.std():.4f}")

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    test_nonlinear_interpolant()