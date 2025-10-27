#!/usr/bin/env python3
"""Analyze kink locations in multivariate polynomial interpolants.

For polynomial coefficients A(t) = (1-t^p) and B(t) = t^q, the velocity field
has rapid changes where derivatives peak. This script identifies critical t values.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def analyze_polynomial_derivatives(p_vals, q_vals):
    """
    Analyze where polynomial derivatives have maximum magnitude.

    For A(t) = (1-t^p): dA/dt = -p·t^(p-1)
    For B(t) = t^q:     dB/dt = q·t^(q-1)

    The second derivatives tell us where curvature (kink) is maximum:
    d²A/dt² = -p(p-1)·t^(p-2)
    d²B/dt² = q(q-1)·t^(q-2)
    """
    t = np.linspace(0.001, 0.999, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (p, q) in enumerate(zip(p_vals, q_vals)):
        # First derivatives (velocity contributions)
        dA_dt = -p * np.power(t, p - 1)  # Derivative of (1-t^p)
        dB_dt = q * np.power(t, q - 1)    # Derivative of t^q

        # Second derivatives (curvature/acceleration)
        d2A_dt2 = -p * (p - 1) * np.power(t, p - 2)
        d2B_dt2 = q * (q - 1) * np.power(t, q - 2)

        # Total velocity magnitude (assuming equal coefficients)
        velocity_mag = np.abs(dA_dt) + np.abs(dB_dt)

        # Find critical points
        # For A(t): max curvature at t = (p-2)/(p-1) if p > 2
        if p > 2:
            t_crit_A = (p - 2) / (p - 1)
        else:
            t_crit_A = 0.0  # Near t=0 for p ∈ (1,2]

        # For B(t): max curvature at t = (q-2)/(q-1) if q > 2
        if q > 2:
            t_crit_B = (q - 2) / (q - 1)
        else:
            t_crit_B = 1.0  # Near t=1 for q ∈ (1,2]

        # Row 0: Velocity contributions
        axes[0, i].plot(t, np.abs(dA_dt), label=f'|dA/dt| (p={p:.1f})', linewidth=2)
        axes[0, i].plot(t, np.abs(dB_dt), label=f'|dB/dt| (q={q:.1f})', linewidth=2)
        axes[0, i].plot(t, velocity_mag, label='Total velocity', linewidth=2, linestyle='--', color='black')
        if p > 2:
            axes[0, i].axvline(t_crit_A, color='red', linestyle=':', alpha=0.7, label=f't_crit_A={t_crit_A:.3f}')
        if q > 2:
            axes[0, i].axvline(t_crit_B, color='blue', linestyle=':', alpha=0.7, label=f't_crit_B={t_crit_B:.3f}')
        axes[0, i].set_xlabel('t')
        axes[0, i].set_ylabel('Magnitude')
        axes[0, i].set_title(f'Velocity: p={p:.1f}, q={q:.1f}')
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)

        # Row 1: Curvature (second derivatives)
        axes[1, i].plot(t, np.abs(d2A_dt2), label=f'|d²A/dt²| (p={p:.1f})', linewidth=2)
        axes[1, i].plot(t, np.abs(d2B_dt2), label=f'|d²B/dt²| (q={q:.1f})', linewidth=2)
        if p > 2:
            axes[1, i].axvline(t_crit_A, color='red', linestyle=':', alpha=0.7)
        if q > 2:
            axes[1, i].axvline(t_crit_B, color='blue', linestyle=':', alpha=0.7)
        axes[1, i].set_xlabel('t')
        axes[1, i].set_ylabel('Curvature')
        axes[1, i].set_title(f'Curvature: p={p:.1f}, q={q:.1f}')
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_yscale('log')

    plt.tight_layout()
    plt.savefig('kink_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved kink analysis to kink_analysis.png")
    plt.show()

    # Print critical t values
    print("\n" + "="*70)
    print("CRITICAL TIME POINTS (where kinks occur)")
    print("="*70)
    for p, q in zip(p_vals, q_vals):
        print(f"\np={p:.1f}, q={q:.1f}:")

        if p > 2:
            t_crit_A = (p - 2) / (p - 1)
            print(f"  A(t) kink at t ≈ {t_crit_A:.3f}")
        elif p > 1:
            print(f"  A(t) rapid change near t → 1")

        if q > 2:
            t_crit_B = (q - 2) / (q - 1)
            print(f"  B(t) kink at t ≈ {t_crit_B:.3f}")
        elif q > 1:
            print(f"  B(t) rapid change near t → 0 and t → 1")


def compute_optimal_beta_params(p, q):
    """
    Compute optimal Beta distribution parameters for importance sampling.

    Strategy:
    - If p > 1: Need more samples near t=1 → use Beta(α, 1) with α > 1
    - If q > 1: Need more samples near t=0 → use Beta(1, β) with β > 1
    - If both: Use Beta(α, β) with both > 1
    """
    # For polynomial (1-t^p), derivatives peak near t → 1 when p > 1
    # Want to sample more near t=1, so use larger alpha
    alpha = max(1.0, max(p, q))

    # Could also use a mixture or dual-mode distribution
    beta = 1.0  # Keep simple for now

    return alpha, beta


if __name__ == "__main__":
    # Example configurations from experiments
    configs = [
        ([1.0, 1.0], [1.0, 1.0]),  # Baseline (linear)
        ([2.0, 1.0], [2.0, 1.0]),  # Quadratic in first dimension
        ([5.0, 1.0], [5.0, 1.0]),  # High exponent in first dimension
    ]

    p_vals = [cfg[0][0] for cfg in configs]  # First dimension only
    q_vals = [cfg[1][0] for cfg in configs]

    analyze_polynomial_derivatives(p_vals, q_vals)

    print("\n" + "="*70)
    print("RECOMMENDED BETA DISTRIBUTION PARAMETERS")
    print("="*70)
    for p, q in zip(p_vals, q_vals):
        alpha, beta = compute_optimal_beta_params(p, q)
        print(f"p={p:.1f}, q={q:.1f}: Beta(α={alpha:.1f}, β={beta:.1f})")
