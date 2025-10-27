# Kink Analysis for Multivariate Polynomial Interpolants

## Understanding the Kinks

For polynomial coefficients:
- **A(t) = M_A ⊙ (1 - t^p)**
- **B(t) = M_B ⊙ t^q**

The interpolant is: **I_t = A(t)x₀ + B(t)x₁**

The velocity field is: **v(t) = dI/dt = dA/dt·x₀ + dB/dt·x₁**

### Where Do Kinks Occur?

Kinks (rapid velocity changes) occur where the **second derivatives** (curvature) peak:

**For A(t) = (1-t^p):**
- First derivative: dA/dt = -p·t^(p-1)
- Second derivative: d²A/dt² = -p(p-1)·t^(p-2)
- **Kink location**: Maximum curvature at **t ≈ (p-2)/(p-1)** when **p > 2**
- When 1 < p ≤ 2: Rapid change near **t → 1**

**For B(t) = t^q:**
- First derivative: dB/dt = q·t^(q-1)
- Second derivative: d²B/dt² = q(q-1)·t^(q-2)
- **Kink location**: Maximum curvature at **t ≈ (q-2)/(q-1)** when **q > 2**
- When 1 < q ≤ 2: Rapid change near **t → 0** and **t → 1**

## Specific Configurations

### Configuration 1: p=[1.0, 1.0], q=[1.0, 1.0] (Baseline)
- **Linear interpolation** - NO kinks
- Uniform sampling is optimal

### Configuration 2: p=[2.0, 1.0], q=[2.0, 1.0] (Quadratic)
- **Critical regions**:
  - Near **t ≈ 0.95** (A(t) rapid change)
  - Near **t ≈ 0.05** (B(t) rapid change)
- **Current sampling**: Beta(2, 1) gives 19% coverage at t≈0.95, only 1% at t≈0.05
- **Problem**: Missing the t≈0.05 region!

### Configuration 3: p=[5.0, 1.0], q=[5.0, 1.0] (High Exponent)
- **Primary kink at t ≈ 0.75**
  - From formula: (5-2)/(5-1) = 3/4 = 0.75
- **Current sampling**: Beta(5, 1) gives only 16% coverage
- **Problem**: Most samples pushed to t→1, missing the kink at t=0.75!

### Configuration 4: p=[10.0, 1.0], q=[10.0, 1.0] (Very High Exponent)
- **Primary kink at t ≈ 0.89**
  - From formula: (10-2)/(10-1) = 8/9 ≈ 0.889
- Velocity field extremely non-smooth near t=0.89

## Why Current Importance Sampling Fails

The current Beta(α, 1) approach with α = max(p, q):
- ✓ **Good**: Concentrates samples near t=1
- ✗ **Bad**: Misses intermediate kinks (e.g., t=0.75 for p=5)
- ✗ **Bad**: Completely ignores early kinks from B(t) with q>1

## Recommended Improvements

### Option 1: Beta Mixture Distribution
Use a mixture of Beta distributions:
```python
# Component 1: Beta(p, 1) → samples near t=1 (for A(t) kinks)
# Component 2: Beta(1, q) → samples near t=0 (for B(t) kinks)
# Component 3: Uniform   → baseline coverage
mix_weights = [0.4, 0.4, 0.2]
```

### Option 2: Adaptive Gaussian Mixture (Best for High Exponents)
Place Gaussians centered at analytical kink locations:
```python
# For p=5, q=5: Place Gaussian at t=0.75 with std≈0.03
# For p=2, q=2: Place Gaussians at t=0.05 and t=0.95
# + Uniform baseline component
```

### Option 3: Two-Stage Sampling
1. **Stage 1 (first 50 epochs)**: Use Beta mixture to explore kinks
2. **Stage 2 (remaining epochs)**: Adaptively concentrate on worst regions

## Implementation Guide

Update `_sample_times_with_importance()` in your training script:

```python
from improved_importance_sampling import sample_times_adaptive

def _sample_times_with_importance_v2(bs, device, dtype, matrix_cfg, keepdim=False):
    """Enhanced importance sampling targeting kink regions."""
    exponents_p = matrix_cfg.get('exponent_p', [1.0, 1.0])
    exponents_q = matrix_cfg.get('exponent_q', [1.0, 1.0])

    # Use beta_mixture strategy for best coverage
    samples, weights = sample_times_adaptive(
        bs, device, dtype,
        exponents_p, exponents_q,
        strategy="beta_mixture",  # or "adaptive_mixture" for high p,q
        keepdim=keepdim
    )

    return samples, weights
```

## Expected Improvements

With better kink sampling:
- **Smoother learned trajectories** (fewer visible kinks in visualizations)
- **Lower final loss** (better fitting near difficult regions)
- **Faster convergence** (velocity field learns correct dynamics earlier)
- **More stable training** (gradients more consistent across time)

## Quick Reference Table

| Exponents | Kink Location(s) | Beta(α,1) Coverage | Recommended Strategy |
|-----------|------------------|-------------------|---------------------|
| p=q=1.0   | None            | N/A               | Uniform             |
| p=q=2.0   | t≈0.05, t≈0.95  | 19%, 1%           | Beta mixture        |
| p=q=5.0   | t≈0.75          | 16%               | Gaussian @ t=0.75   |
| p=q=10.0  | t≈0.89          | ~20%              | Gaussian @ t=0.89   |

## Visualization

See `kink_analysis.png` for plots showing:
- Top row: Velocity magnitude |dI/dt| vs time
- Bottom row: Curvature |d²I/dt²| vs time (log scale)
- Red/blue lines: Critical time points

Run `analyze_kinks.py` with your specific exponents to see where YOUR kinks are!
