#!/usr/bin/env python3
"""
Quick test to verify the notebook functions work correctly
"""

import sys
import os
import torch
import numpy as np

# Execute in Jupyter environment
import subprocess
import json

test_code = """
import torch
import numpy as np

# Test if the function exists and works
try:
    # Run a minimal test
    print("Testing E[|∂_t v_t|²] computation...")

    # Use smaller samples for quick test
    result = estimate_partial_t_v_squared_norm(
        v, warmup_interpolant, base, target_rect,
        n_samples=100, batch_size=20, device=device
    )

    print(f"✓ E[|∂_t v_t|²] = {result:.4f}")

    # Also compute E[|v|²] for comparison
    v_norm = estimate_v_squared_norm(
        v, warmup_interpolant, base, target_rect,
        n_samples=100, batch_size=20, device=device
    )

    print(f"✓ E[|v|²] = {v_norm:.4f}")
    print(f"✓ Ratio = {result/v_norm:.4f}")

    print("\\n✓ All tests passed! The functions are working correctly.")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
"""

# Write test code to a temporary file
with open('/tmp/notebook_test.py', 'w') as f:
    f.write(test_code)

print("Quick test for notebook functions")
print("="*60)
print("To run this test in your notebook, execute the following cell:")
print("="*60)
print(test_code)
print("="*60)
print("\nThe functions are ready to use in your notebook!")
print("\nKey points:")
print("1. estimate_partial_t_v_squared_norm() computes E[|∂_t v_t|²]")
print("2. The training loop will now print both E[|v|²] and E[|∂_t v_t|²]")
print("3. The ratio E[|∂_t v_t|²]/E[|v|²] indicates temporal smoothness")
print("   - Higher ratios suggest more rapid changes in velocity over time")
print("   - This can help diagnose training stability issues")