"""
Updated quick_run functions for test_realnvp.ipynb that include 
both time and input differentiability tests.

Copy these into your notebook to replace the existing quick_run functions.
"""

import torch
from test_differentiability import check_differentiability_t, check_differentiability_x

def count_params(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = n * 8 / 1e6  # float64 = 8 bytes per parameter
    return n, mb

@torch.no_grad()
def check_invertibility(model, x, t, name, atol=1e-5, rtol=1e-5):
    model.eval()
    y, ld1 = model(x, t)
    x_rec, ld2 = model.inverse(y, t)

    diff = (x_rec - x).reshape(x.shape[0], -1)
    l2 = diff.norm(dim=1)
    linf = diff.abs().max(dim=1).values
    rel = l2 / (x.reshape(x.shape[0], -1).norm(dim=1) + 1e-12)

    # logdet from inverse should be the negative of forward
    logdet_consistency = (ld1 + ld2).abs()

    print(f"\n[{name}] Invertibility check")
    print(f"  ||x_rec - x||_2    : mean {l2.mean():.3e} | max {l2.max():.3e}")
    print(f"  ||x_rec - x||_inf  : mean {linf.mean():.3e} | max {linf.max():.3e}")
    print(f"  rel L2 error       : mean {rel.mean():.3e} | max {rel.max():.3e}")
    print(f"  |logdet + logdet^-1|: mean {logdet_consistency.mean():.3e} | max {logdet_consistency.max():.3e}")

    ok = (l2.max() < atol + rtol * x.abs().max()) and (logdet_consistency.max() < 1e-4)  # Relaxed threshold
    print(f"  PASS: {ok}")
    return ok

def quick_run_vector(device="cpu"):
    print("\n=== VECTOR FLOW ===")
    from realnvp import create_vector_flow
    
    model = create_vector_flow(
        dim=32, num_layers=6, time_embed_dim=64,
        hidden=512, mlp_blocks=3, activation="gelu",
        use_layernorm=False, use_permutation=True
    ).to(device)
    
    nparams, mb = count_params(model)
    print(f"Params: {nparams:,} ({mb:.2f} MB)")
    
    # Use batch size 8 for comprehensive testing
    x = torch.randn(8, 32, device=device, dtype=torch.float64)
    t = torch.rand(8, device=device, dtype=torch.float64)
    
    # Test invertibility
    check_invertibility(model, x, t, "vector(32)")
    
    # Test differentiability w.r.t. time
    check_differentiability_t(model, x, "vector(32)")
    
    # Test differentiability w.r.t. input
    check_differentiability_x(model, x, "vector(32)")

def quick_run_mnist_like(device="cpu"):
    print("\n=== MNIST-LIKE IMAGE FLOW ===")
    from realnvp import create_mnist_flow
    
    model = create_mnist_flow(
        image_mode=True, num_layers=6, time_embed_dim=128,
        img_base_channels=96, img_blocks=3, img_groups=32,
        img_log_scale_clamp=10.0, use_permutation=True
    ).to(device)
    
    nparams, mb = count_params(model)
    print(f"Params: {nparams:,} ({mb:.2f} MB)")
    
    # Use batch size 8 for MNIST
    x = torch.randn(8, 1, 28, 28, device=device, dtype=torch.float64)
    t = torch.rand(8, device=device, dtype=torch.float64)
    
    # Test invertibility
    check_invertibility(model, x, t, "MNIST (1x28x28)")
    
    # Test differentiability w.r.t. time
    check_differentiability_t(model, x, "MNIST (1x28x28)")
    
    # Test differentiability w.r.t. input
    check_differentiability_x(model, x, "MNIST (1x28x28)")

def quick_run_cifar(device="cpu"):
    print("\n=== CIFAR-10 IMAGE FLOW ===")
    from realnvp import create_cifar10_flow
    
    model = create_cifar10_flow(
        num_layers=8, time_embed_dim=128,
        img_base_channels=128, img_blocks=4, img_groups=32,
        img_log_scale_clamp=10.0, use_permutation=True
    ).to(device)
    
    nparams, mb = count_params(model)
    print(f"Params: {nparams:,} ({mb:.2f} MB)")
    
    # Use smaller batch size for CIFAR due to memory
    batch_size = 4 if device == "cuda" else 8
    x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=torch.float64)
    t = torch.rand(batch_size, device=device, dtype=torch.float64)
    
    # Test invertibility
    check_invertibility(model, x, t, "CIFAR (3x32x32)")
    
    # Test differentiability w.r.t. time
    check_differentiability_t(model, x, "CIFAR (3x32x32)")
    
    # Test differentiability w.r.t. input
    check_differentiability_x(model, x, "CIFAR (3x32x32)")

def quick_run_imagenet(device="cpu"):
    print("\n=== IMAGENET-LIKE IMAGE FLOW ===")
    from realnvp import create_imagenet_flow_stable
    
    model = create_imagenet_flow_stable(
        resolution=128,  # Smaller than default 224 for memory efficiency
        num_layers=4, time_embed_dim=256,
        img_base_channels=128, img_blocks=3, img_groups=32,
        img_log_scale_clamp=10.0, use_permutation=True
    ).to(device)
    
    nparams, mb = count_params(model)
    print(f"Params: {nparams:,} ({mb:.2f} MB)")
    
    # Use very small batch for ImageNet
    batch_size = 2 if device == "cuda" else 4
    x = torch.randn(batch_size, 3, 128, 128, device=device, dtype=torch.float64)
    t = torch.rand(batch_size, device=device, dtype=torch.float64)
    
    # Test invertibility
    check_invertibility(model, x, t, "ImageNet (3x128x128)")
    
    # Test differentiability w.r.t. time
    check_differentiability_t(model, x, "ImageNet (3x128x128)")
    
    # Test differentiability w.r.t. input
    check_differentiability_x(model, x, "ImageNet (3x128x128)")

def quick_run_imagenet_full(device="cpu"):
    print("\n=== FULL IMAGENET FLOW ===")
    from realnvp import create_imagenet_flow_stable
    
    model = create_imagenet_flow_stable(
        resolution=224,
        num_layers=4,  # Reduced for memory
        time_embed_dim=256,
        img_base_channels=96,  # Reduced for memory
        img_blocks=3, img_groups=32,
        img_log_scale_clamp=5.0,  # Stable variant
        use_permutation=True
    ).to(device)
    
    nparams, mb = count_params(model)
    print(f"Params: {nparams:,} ({mb:.2f} MB)")
    
    # Use batch size 1 for full ImageNet
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float64)
    t = torch.rand(1, device=device, dtype=torch.float64)
    
    # Test invertibility
    check_invertibility(model, x, t, "ImageNet (3x224x224)")
    
    # Test differentiability w.r.t. time
    check_differentiability_t(model, x, "ImageNet (3x224x224)")
    
    # Test differentiability w.r.t. input
    check_differentiability_x(model, x, "ImageNet (3x224x224)")

# Memory-safe test runner
def run_tests_with_cleanup(device):
    """Run all tests with proper memory cleanup"""
    
    tests = [
        ("Vector", quick_run_vector),
        ("MNIST", quick_run_mnist_like),
        ("CIFAR", quick_run_cifar),
        ("ImageNet 128x128", quick_run_imagenet),
        ("ImageNet 224x224", quick_run_imagenet_full),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*80}")
        
        # Show memory before test
        if device == "cuda":
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory before: {memory_before:.2f} GB")
        
        try:
            # Run the test
            test_func(device)
            
            # Show memory after test
            if device == "cuda":
                memory_after = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory after: {memory_after:.2f} GB")
            
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            
        finally:
            # CRITICAL: Clean up memory
            if device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                memory_cleaned = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory after cleanup: {memory_cleaned:.2f} GB")
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETED!")
    print(f"{'='*80}")

# Quick test for specific batch sizes
def test_batch_size_scaling(device="cpu"):
    """Test how different batch sizes affect performance and memory"""
    from realnvp import create_vector_flow
    
    print("\n=== BATCH SIZE SCALING TEST ===")
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size {batch_size} ---")
        
        if device == "cuda":
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1e9
        
        model = create_vector_flow(dim=32).to(device)
        x = torch.randn(batch_size, 32, device=device, dtype=torch.float64)
        t = torch.rand(batch_size, device=device, dtype=torch.float64)
        
        # Quick forward pass
        with torch.no_grad():
            y, logdet = model(x, t)
        
        if device == "cuda":
            memory_after = torch.cuda.memory_allocated() / 1e9
            print(f"Memory usage: {memory_after - memory_before:.3f} GB")
        
        print(f"Output shapes: y={y.shape}, logdet={logdet.shape}")
        
        # Cleanup
        del model, x, t, y, logdet
        if device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Run individual tests
    quick_run_vector(device)
    
    # Or run all tests with cleanup
    # run_tests_with_cleanup(device) 