#!/usr/bin/env python3
"""Improved importance sampling for multivariate polynomial interpolants.

Addresses kinks by concentrating samples near critical t values where
velocity field changes rapidly.
"""

import numpy as np
import torch


def compute_critical_times(exponents_p, exponents_q):
    """
    Compute critical time points where kinks occur.

    For A(t) = (1-t^p): d²A/dt² = -p(p-1)t^(p-2)
    - Maximum curvature at t_crit ≈ (p-2)/(p-1) when p > 2
    - Near t → 1 when 1 < p ≤ 2

    For B(t) = t^q: d²B/dt² = q(q-1)t^(q-2)
    - Maximum curvature at t_crit ≈ (q-2)/(q-1) when q > 2
    - Near t → 0, 1 when 1 < q ≤ 2

    Returns:
        critical_regions: list of (t_center, width) tuples
    """
    critical_regions = []

    # Analyze each dimension's exponents
    max_p = max(exponents_p) if exponents_p else 1.0
    max_q = max(exponents_q) if exponents_q else 1.0

    # For A(t) = (1-t^p)
    if max_p > 2.0:
        t_crit_A = (max_p - 2.0) / (max_p - 1.0)
        width_A = 0.1  # Focus region width
        critical_regions.append((t_crit_A, width_A))
    elif max_p > 1.0:
        # Rapid change near t=1
        critical_regions.append((0.95, 0.1))

    # For B(t) = t^q
    if max_q > 2.0:
        t_crit_B = (max_q - 2.0) / (max_q - 1.0)
        width_B = 0.1
        # Only add if far from A's critical region
        if not critical_regions or abs(t_crit_B - critical_regions[0][0]) > 0.15:
            critical_regions.append((t_crit_B, width_B))
    elif max_q > 1.0:
        # Rapid change near t=0 and t=1
        critical_regions.append((0.05, 0.1))

    return critical_regions


def sample_times_adaptive(
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
    exponents_p: list,
    exponents_q: list,
    strategy: str = "adaptive_mixture",
    keepdim: bool = False,
):
    """
    Advanced importance sampling for polynomial interpolants.

    Strategies:
    - "uniform": Baseline uniform sampling
    - "beta_alpha": Current approach - Beta(max(p,q), 1) focuses on t→1
    - "adaptive_mixture": NEW - Mixture of Gaussians centered at critical t values
    - "beta_mixture": Mixture of Beta distributions

    Returns:
        samples: time samples [bs] or [bs, 1]
        weights: importance weights [bs] or [bs, 1]
    """
    max_p = max(exponents_p) if exponents_p else 1.0
    max_q = max(exponents_q) if exponents_q else 1.0

    if strategy == "uniform":
        samples = torch.rand(bs, device=device, dtype=dtype)
        weights = torch.ones_like(samples)

    elif strategy == "beta_alpha":
        # Current approach: Beta(α, 1) with α = max(p, q)
        alpha = max(1.0, max(max_p, max_q))
        concentration1 = torch.tensor(alpha, device=device, dtype=dtype)
        concentration0 = torch.tensor(1.0, device=device, dtype=dtype)
        beta_dist = torch.distributions.Beta(concentration1, concentration0)
        samples = beta_dist.sample((bs,))

        eps = torch.finfo(dtype).eps
        clamped = samples.clamp(min=eps, max=1 - eps)
        log_pdf = beta_dist.log_prob(clamped)
        weights = torch.exp(-log_pdf).detach()

    elif strategy == "adaptive_mixture":
        # NEW: Gaussian mixture centered at critical times
        critical_regions = compute_critical_times(exponents_p, exponents_q)

        if not critical_regions:
            # No critical regions, use uniform
            samples = torch.rand(bs, device=device, dtype=dtype)
            weights = torch.ones_like(samples)
        else:
            n_components = len(critical_regions) + 1  # +1 for uniform component
            component_weights = [1.0 / n_components] * n_components
            component_weights[-1] = 0.2  # Less weight on uniform
            # Renormalize
            total = sum(component_weights)
            component_weights = [w / total for w in component_weights]

            # Sample from mixture
            samples_list = []
            weights_list = []

            for i in range(bs):
                # Choose component
                component = np.random.choice(n_components, p=component_weights)

                if component < len(critical_regions):
                    # Gaussian centered at critical time
                    t_center, width = critical_regions[component]
                    std = width / 3.0  # 99.7% within width
                    sample = torch.normal(
                        mean=torch.tensor(t_center, dtype=dtype),
                        std=torch.tensor(std, dtype=dtype),
                    )
                    # Clamp to [0, 1]
                    sample = sample.clamp(0.0, 1.0)

                    # Compute mixture pdf
                    pdf = 0.0
                    for j, (tc, w) in enumerate(critical_regions):
                        s = w / 3.0
                        # Gaussian pdf (unnormalized, clamped domain)
                        pdf += component_weights[j] * torch.exp(
                            -0.5 * ((sample - tc) / s) ** 2
                        )
                    # Add uniform component
                    pdf += component_weights[-1] * 1.0

                else:
                    # Uniform component
                    sample = torch.rand(1, dtype=dtype)
                    pdf = component_weights[-1] * 1.0

                samples_list.append(sample)
                # Importance weight = 1 / pdf
                weight = 1.0 / (pdf + 1e-8)
                weights_list.append(weight)

            samples = torch.stack(samples_list).to(device=device)
            weights = torch.stack(weights_list).to(device=device)

            # Normalize weights to mean 1
            weights = weights / weights.mean()

    elif strategy == "beta_mixture":
        # Mixture of Beta distributions
        # Component 1: Beta(α, 1) for t→1
        # Component 2: Beta(1, β) for t→0
        # Component 3: Uniform

        alpha = max(2.0, max_p)
        beta = max(2.0, max_q)

        mix_probs = [0.4, 0.4, 0.2]  # Weights for each component

        # Sample component assignments
        component_choices = torch.multinomial(
            torch.tensor(mix_probs, dtype=dtype),
            bs,
            replacement=True,
        )

        samples = torch.zeros(bs, dtype=dtype, device=device)
        weights = torch.zeros(bs, dtype=dtype, device=device)

        # Component 0: Beta(α, 1)
        mask0 = component_choices == 0
        n0 = mask0.sum().item()
        if n0 > 0:
            beta_dist0 = torch.distributions.Beta(
                torch.tensor(alpha, dtype=dtype),
                torch.tensor(1.0, dtype=dtype),
            )
            samples[mask0] = beta_dist0.sample((n0,))

        # Component 1: Beta(1, β)
        mask1 = component_choices == 1
        n1 = mask1.sum().item()
        if n1 > 0:
            beta_dist1 = torch.distributions.Beta(
                torch.tensor(1.0, dtype=dtype),
                torch.tensor(beta, dtype=dtype),
            )
            samples[mask1] = beta_dist1.sample((n1,))

        # Component 2: Uniform
        mask2 = component_choices == 2
        n2 = mask2.sum().item()
        if n2 > 0:
            samples[mask2] = torch.rand(n2, dtype=dtype)

        # Compute weights (inverse of mixture pdf)
        eps = torch.finfo(dtype).eps
        clamped = samples.clamp(min=eps, max=1 - eps)

        beta_dist0 = torch.distributions.Beta(
            torch.tensor(alpha, dtype=dtype),
            torch.tensor(1.0, dtype=dtype),
        )
        beta_dist1 = torch.distributions.Beta(
            torch.tensor(1.0, dtype=dtype),
            torch.tensor(beta, dtype=dtype),
        )

        pdf = (
            mix_probs[0] * torch.exp(beta_dist0.log_prob(clamped))
            + mix_probs[1] * torch.exp(beta_dist1.log_prob(clamped))
            + mix_probs[2] * 1.0
        )
        weights = (1.0 / (pdf + eps)).detach()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if keepdim:
        samples = samples.unsqueeze(-1)
        weights = weights.unsqueeze(-1)

    return samples, weights


if __name__ == "__main__":
    # Test different strategies
    print("="*70)
    print("IMPORTANCE SAMPLING COMPARISON")
    print("="*70)

    configs = [
        ([1.0, 1.0], [1.0, 1.0], "Baseline"),
        ([2.0, 1.0], [2.0, 1.0], "Quadratic"),
        ([5.0, 1.0], [5.0, 1.0], "High exponent"),
    ]

    for exponents_p, exponents_q, name in configs:
        print(f"\n{name}: p={exponents_p}, q={exponents_q}")
        print("-"*70)

        critical_regions = compute_critical_times(exponents_p, exponents_q)
        print(f"Critical regions: {critical_regions}")

        # Sample with different strategies
        bs = 10000
        device = torch.device("cpu")
        dtype = torch.float32

        for strategy in ["uniform", "beta_alpha", "beta_mixture"]:
            samples, weights = sample_times_adaptive(
                bs, device, dtype, exponents_p, exponents_q, strategy=strategy
            )

            print(f"\n  {strategy}:")
            print(f"    Sample mean: {samples.mean():.3f}")
            print(f"    Sample std:  {samples.std():.3f}")
            print(f"    Weight mean: {weights.mean():.3f}")
            print(f"    Weight std:  {weights.std():.3f}")

            # Check coverage of critical regions
            if critical_regions:
                for t_center, width in critical_regions:
                    in_region = ((samples >= t_center - width/2) &
                                 (samples <= t_center + width/2)).float().mean()
                    print(f"    Coverage of t≈{t_center:.2f}: {in_region*100:.1f}%")
