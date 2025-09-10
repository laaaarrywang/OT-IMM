#!/usr/bin/env python
"""Simple test for nonlinear interpolant with all data types"""

import torch
from interflow import fabrics

def test_data_type(name, data_type, x0, x1, flow_config=None, data_dim=None):
    """Test a specific data type"""
    print(f"\nTesting {name}...")
    
    try:
        # Create interpolant
        It, dtIt, _ = fabrics.make_It(
            path='nonlinear',
            data_type=data_type,
            data_dim=data_dim,
            flow_config=flow_config or {}
        )
        
        # Test boundary conditions
        t0 = torch.tensor(0.0, dtype=torch.float32)
        y0 = It(t0, x0, x1)
        diff0 = (y0 - x0).abs().max().item()
        print(f"  t=0.0: max_diff={diff0:.6f} (should be ~0)")
        assert diff0 < 1e-3, f"Failed at t=0: {diff0}"
        
        t1 = torch.tensor(1.0, dtype=torch.float32)
        y1 = It(t1, x0, x1)
        diff1 = (y1 - x1).abs().max().item()
        print(f"  t=1.0: max_diff={diff1:.6f} (should be ~0)")
        assert diff1 < 1e-3, f"Failed at t=1: {diff1}"
        
        # Test interpolation at t=0.5
        t_mid = torch.tensor(0.5, dtype=torch.float32)
        y_mid = It(t_mid, x0, x1)
        print(f"  t=0.5: shape={y_mid.shape}, mean={y_mid.mean():.4f}, std={y_mid.std():.4f}")
        
        # Test time derivative
        dy_dt = dtIt(t_mid, x0, x1)
        print(f"  dtIt:  shape={dy_dt.shape}, finite={dy_dt.isfinite().all()}")
        
        print(f"  ✓ {name} passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("NONLINEAR INTERPOLANT TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Vector data
    x0 = torch.randn(2, 10, dtype=torch.float32)
    x1 = torch.randn(2, 10, dtype=torch.float32)
    results.append(test_data_type(
        "Vector (10-dim)", "vector", x0, x1,
        flow_config={'num_layers': 2},
        data_dim=10
    ))
    
    # Test 2: MNIST
    x0 = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    x1 = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    results.append(test_data_type(
        "MNIST (1x28x28)", "mnist", x0, x1,
        flow_config={'num_layers': 2, 'image_mode': True}
    ))
    
    # Test 3: CIFAR-10
    x0 = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    x1 = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    results.append(test_data_type(
        "CIFAR-10 (3x32x32)", "cifar10", x0, x1,
        flow_config={'num_layers': 2}
    ))
    
    # Test 4: ImageNet 128x128
    x0 = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    x1 = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    results.append(test_data_type(
        "ImageNet 128x128", "image", x0, x1,
        flow_config={'num_layers': 2, 'img_base_channels': 32, 'img_blocks': 2},
        data_dim=(3, 128, 128)
    ))
    
    # Test 5: ImageNet 224x224 (stable)
    x0 = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    x1 = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    results.append(test_data_type(
        "ImageNet 224x224 (stable)", "imagenet_stable", x0, x1,
        flow_config={'resolution': 224, 'num_layers': 2}
    ))
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"✓ ALL {total} TESTS PASSED!")
    else:
        print(f"⚠ {passed}/{total} tests passed")
    print("="*60)