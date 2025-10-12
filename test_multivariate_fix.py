#!/usr/bin/env python
"""
Test script to verify the multivariate interpolant broadcasting fix.
"""
import torch
import sys
sys.path.append('/home/wang6559/Projects/stochastic-interpolants')

import interflow.stochastic_interpolant as stochastic_interpolant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Test configuration
matrix_config = {
    'matrix_type': 'diagonal',
    'exponent_p': [1.0, 0.5],
    'exponent_q': [1.0, 0.5],
}

# Create multivariate interpolant
interpolant = stochastic_interpolant.Interpolant(
    path='multivariate',
    gamma_type=None,
    data_type='vector',
    data_dim=2,
    matrix_config=matrix_config
)

print("="*70)
print("Testing Multivariate Interpolant Broadcasting Fix")
print("="*70)

# Test 1: Single time value [1] with batch
print("\n1. Testing with t shape [1] and batch size 16:")
batch_size = 16
x0 = torch.randn(batch_size, 2, device=device)
x1 = torch.randn(batch_size, 2, device=device)
t = torch.tensor([0.5], device=device)

xt = interpolant.calc_xt(t, x0, x1)
if isinstance(xt, tuple):
    xt = xt[0]

print(f"   x0 shape: {x0.shape}")
print(f"   x1 shape: {x1.shape}")
print(f"   t shape: {t.shape}")
print(f"   xt shape: {xt.shape}")
print(f"   Expected: torch.Size([{batch_size}, 2])")
print(f"   ✓ PASS" if xt.shape == torch.Size([batch_size, 2]) else f"   ✗ FAIL")

# Test 2: Batch time values [bs, 1]
print("\n2. Testing with t shape [bs, 1] and batch size 16:")
t_batch = torch.rand(batch_size, 1, device=device)

xt_batch = interpolant.calc_xt(t_batch, x0, x1)
if isinstance(xt_batch, tuple):
    xt_batch = xt_batch[0]

print(f"   x0 shape: {x0.shape}")
print(f"   x1 shape: {x1.shape}")
print(f"   t shape: {t_batch.shape}")
print(f"   xt shape: {xt_batch.shape}")
print(f"   Expected: torch.Size([{batch_size}, 2])")
print(f"   ✓ PASS" if xt_batch.shape == torch.Size([batch_size, 2]) else f"   ✗ FAIL")

# Test 3: Boundary conditions
print("\n3. Testing boundary conditions:")
t0 = torch.tensor([0.0], device=device)
t1 = torch.tensor([1.0], device=device)

xt0 = interpolant.calc_xt(t0, x0, x1)
xt1 = interpolant.calc_xt(t1, x0, x1)

if isinstance(xt0, tuple):
    xt0 = xt0[0]
if isinstance(xt1, tuple):
    xt1 = xt1[0]

err0 = torch.max(torch.abs(xt0 - x0)).item()
err1 = torch.max(torch.abs(xt1 - x1)).item()

print(f"   t=0: max|x_t - x_0| = {err0:.6f} (should be ~0)")
print(f"   t=1: max|x_t - x_1| = {err1:.6f} (should be ~0)")
print(f"   ✓ PASS" if err0 < 1e-5 and err1 < 1e-5 else f"   ✗ FAIL")

# Test 4: Time derivative shape
print("\n4. Testing time derivative dtIt:")
dt_xt = interpolant.dtIt(t, x0, x1)
if isinstance(dt_xt, tuple):
    dt_xt = dt_xt[0]

print(f"   dtIt shape: {dt_xt.shape}")
print(f"   Expected: torch.Size([{batch_size}, 2])")
print(f"   ✓ PASS" if dt_xt.shape == torch.Size([batch_size, 2]) else f"   ✗ FAIL")

# Test 5: Batch time derivative
print("\n5. Testing dtIt with batch times:")
dt_xt_batch = interpolant.dtIt(t_batch, x0, x1)
if isinstance(dt_xt_batch, tuple):
    dt_xt_batch = dt_xt_batch[0]

print(f"   dtIt shape: {dt_xt_batch.shape}")
print(f"   Expected: torch.Size([{batch_size}, 2])")
print(f"   ✓ PASS" if dt_xt_batch.shape == torch.Size([batch_size, 2]) else f"   ✗ FAIL")

print("\n" + "="*70)
print("All tests completed!")
print("="*70)
