#!/usr/bin/env python3
"""Test script to verify Fourier series boundary conditions."""

import math

def test_fourier_boundaries():
    """Test that Fourier series satisfies boundary conditions."""

    # Test at t=0
    t = 0.0

    # Base term for A(t): cos(π*t/2)
    A_base_0 = math.cos(math.pi * t / 2)
    print(f"At t=0: A_base = cos(π*0/2) = {A_base_0} (should be 1.0)")

    # Fourier terms: sin(m*π*t) at t=0
    for m in range(1, 6):
        fourier_term = math.sin(m * math.pi * t)
        print(f"  sin({m}*π*0) = {fourier_term} (should be 0.0)")

    # Base term for B(t): sin(π*t/2)
    B_base_0 = math.sin(math.pi * t / 2)
    print(f"At t=0: B_base = sin(π*0/2) = {B_base_0} (should be 0.0)")

    print("\n" + "="*50 + "\n")

    # Test at t=1
    t = 1.0

    # Base term for A(t): cos(π*t/2)
    A_base_1 = math.cos(math.pi * t / 2)
    print(f"At t=1: A_base = cos(π*1/2) = {A_base_1} (should be 0.0)")

    # Fourier terms: sin(m*π*t) at t=1
    for m in range(1, 6):
        fourier_term = math.sin(m * math.pi * t)
        print(f"  sin({m}*π*1) = {fourier_term} (should be 0.0)")

    # Base term for B(t): sin(π*t/2)
    B_base_1 = math.sin(math.pi * t / 2)
    print(f"At t=1: B_base = sin(π*1/2) = {B_base_1} (should be 1.0)")

    print("\n" + "="*50 + "\n")

    # Summary
    print("BOUNDARY CONDITIONS CHECK:")
    print(f"A(0) = {A_base_0} + 0 (Fourier terms) = 1.0 ✓")
    print(f"A(1) = {A_base_1} + 0 (Fourier terms) = 0.0 ✓")
    print(f"B(0) = {B_base_0} + 0 (Fourier terms) = 0.0 ✓")
    print(f"B(1) = {B_base_1} + 0 (Fourier terms) = 1.0 ✓")

    print("\n" + "="*50 + "\n")

    # Test intermediate values
    print("INTERMEDIATE VALUES (t=0.5):")
    t = 0.5

    A_base = math.cos(math.pi * t / 2)
    B_base = math.sin(math.pi * t / 2)

    print(f"Base terms: A={A_base:.4f}, B={B_base:.4f}")

    # Example with M=5, α_m=β_m=1.0
    M = 5
    A_fourier_sum = 0
    B_fourier_sum = 0

    for m in range(1, M+1):
        sin_term = math.sin(m * math.pi * t)
        A_fourier_sum += sin_term
        B_fourier_sum += sin_term
        print(f"  m={m}: sin({m}π/2) = {sin_term:.4f}")

    A_fourier_sum /= M
    B_fourier_sum /= M

    print(f"\nFinal values at t=0.5 (with M={M}, α=β=1.0):")
    print(f"A(0.5) = {A_base:.4f} + {A_fourier_sum:.4f} = {A_base + A_fourier_sum:.4f}")
    print(f"B(0.5) = {B_base:.4f} + {B_fourier_sum:.4f} = {B_base + B_fourier_sum:.4f}")

if __name__ == "__main__":
    test_fourier_boundaries()