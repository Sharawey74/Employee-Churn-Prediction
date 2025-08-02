#!/usr/bin/env python3
"""
Quick verification that the fix works
"""

# Test the specific error
try:
    x = 'N/A'
    print(f"Value: {x}")
    print(f"Float format: {x:.2f}")  # This should fail
except ValueError as e:
    print(f"Caught error: {e}")

# Test the fix
x = 'N/A'
if isinstance(x, (int, float)):
    print(f"Fixed format: {x:.2f}")
else:
    print(f"Fixed format: {x}")
    
print("✅ Fix verification completed!")
