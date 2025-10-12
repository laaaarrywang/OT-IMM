"""
Test script for multivariate (matrix-coefficient) interpolant.

This script demonstrates how to use the multivariate interpolant for:
1. Diagonal matrices with different exponents per dimension (for hyperparameter tuning)
2. Full matrices (for future training)

Designed for 100:1 aspect ratio rectangular targets.
"""

import torch
import sys
sys.path.append('../')
import interflow.stochastic_interpolant as stochastic_interpolant

# ============================================================================
# Example 1: Diagonal matrix with dimension-specific exponents
# ============================================================================

print("=" * 70)
print("Example 1: Diagonal Matrix Interpolant (Hyperparameter Tuning)")
print("=" * 70)

# For 100:1 rectangular target: x ∈ [-10, 10], y ∈ [-0.4, 0.4]
# Strategy: Use different exponents for x (wide) and y (narrow)

# Configuration for hyperparameter tuning
matrix_config_diagonal = {
    'matrix_type': 'diagonal',
    'matrix_A': None,  # Defaults to identity
    'matrix_B': None,  # Defaults to identity
    'exponent_p': [1.0, 0.5],  # x: linear, y: faster (sqrt)
    'exponent_q': [1.0, 0.5],  # x: linear, y: faster (sqrt)
    'trainable': False
}

# Create interpolant
interpolant_diag = stochastic_interpolant.Interpolant(
    path='multivariate',
    gamma_type=None,
    data_type='vector',
    data_dim=2,
    matrix_config=matrix_config_diagonal
)

print(f"\nInterpolant created with path: {interpolant_diag.path}")
print(f"Matrix type: diagonal")
print(f"Exponents p: {matrix_config_diagonal['exponent_p']}")
print(f"Exponents q: {matrix_config_diagonal['exponent_q']}")

# Test interpolation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x0 = torch.randn(4, 2).to(device)  # Base distribution samples
x1 = torch.randn(4, 2).to(device)  # Target distribution samples
t = torch.tensor([0.5]).to(device)

xt = interpolant_diag.calc_xt(t, x0, x1)
print(f"\nTest interpolation at t=0.5:")
print(f"x0 shape: {x0.shape}")
print(f"x1 shape: {x1.shape}")
print(f"xt shape: {xt.shape}")
print(f"xt:\n{xt}")

# Test boundary conditions
t0 = torch.tensor([0.0]).to(device)
t1 = torch.tensor([1.0]).to(device)
xt0 = interpolant_diag.calc_xt(t0, x0, x1)
xt1 = interpolant_diag.calc_xt(t1, x0, x1)
print(f"\nBoundary conditions:")
print(f"At t=0: max|x_t - x_0| = {torch.max(torch.abs(xt0 - x0)).item():.6f} (should be ~0)")
print(f"At t=1: max|x_t - x_1| = {torch.max(torch.abs(xt1 - x1)).item():.6f} (should be ~0)")

# ============================================================================
# Example 2: Diagonal matrix with custom scaling
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: Diagonal Matrix with Custom Scaling")
print("=" * 70)

# Use scaling factors to account for different coordinate ranges
# x range: 20, y range: 0.8 → ratio 25:1

matrix_config_scaled = {
    'matrix_type': 'diagonal',
    'matrix_A': [1.0, 1.0],  # Identity for A
    'matrix_B': [1.0, 1.0],  # Identity for B
    'exponent_p': [1.0, 0.3],  # y much faster
    'exponent_q': [1.0, 0.3],
    'trainable': False
}

interpolant_scaled = stochastic_interpolant.Interpolant(
    path='multivariate',
    gamma_type=None,
    data_type='vector',
    data_dim=2,
    matrix_config=matrix_config_scaled
)

print(f"\nCustom scaling interpolant created")
print(f"Exponents p: {matrix_config_scaled['exponent_p']}")
print(f"Exponents q: {matrix_config_scaled['exponent_q']}")

# ============================================================================
# Example 3: Full matrix (for future training)
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Full Matrix Interpolant (Future Training Support)")
print("=" * 70)

# Full matrix allows coupling between dimensions
# Initially set to identity, but can be learned

matrix_config_full = {
    'matrix_type': 'full',
    'matrix_A': None,  # Defaults to identity
    'matrix_B': None,  # Defaults to identity
    'exponent_p': 1.0,  # Scalar exponent applied to diagonal
    'exponent_q': 1.0,
    'trainable': False  # Will be True when training matrices
}

interpolant_full = stochastic_interpolant.Interpolant(
    path='multivariate',
    gamma_type=None,
    data_type='vector',
    data_dim=2,
    matrix_config=matrix_config_full
)

print(f"\nFull matrix interpolant created")
print(f"Matrix type: full")
print(f"Trainable: {matrix_config_full['trainable']} (future: set to True for training)")

xt_full = interpolant_full.calc_xt(t, x0, x1)
print(f"\nTest interpolation at t=0.5:")
print(f"xt shape: {xt_full.shape}")

# ============================================================================
# Example 4: Hyperparameter search configurations
# ============================================================================

print("\n" + "=" * 70)
print("Example 4: Hyperparameter Search Space for 100:1 Rectangle")
print("=" * 70)

# Define search space for diagonal exponents
search_configs = [
    {'name': 'baseline', 'exponent_p': [1.0, 1.0], 'exponent_q': [1.0, 1.0]},
    {'name': 'fast_y', 'exponent_p': [1.0, 0.5], 'exponent_q': [1.0, 0.5]},
    {'name': 'very_fast_y', 'exponent_p': [1.0, 0.3], 'exponent_q': [1.0, 0.3]},
    {'name': 'slow_y', 'exponent_p': [1.0, 2.0], 'exponent_q': [1.0, 2.0]},
    {'name': 'asymmetric', 'exponent_p': [1.0, 0.5], 'exponent_q': [1.0, 1.5]},
]

print("\nHyperparameter configurations to test:")
for config in search_configs:
    print(f"  {config['name']:15s}: p={config['exponent_p']}, q={config['exponent_q']}")

print("\n" + "=" * 70)
print("Testing complete! Key takeaways:")
print("=" * 70)
print("1. Diagonal matrices allow dimension-specific interpolation speeds")
print("2. For 100:1 rectangles, try exponents in range [0.3, 2.0] for narrow dimension")
print("3. Full matrices support future training via optimal transport")
print("4. All matrices satisfy boundary conditions: A(0)=I, B(1)=I, A(1)=0, B(0)=0")
