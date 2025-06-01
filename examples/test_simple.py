"""Simple test of visualization imports."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing visualization imports...")

# Test 1: Direct module import
try:
    import repeatradar.visualization as viz
    print("✓ Module imported successfully")
    print(f"Module functions: {[name for name in dir(viz) if not name.startswith('_')]}")
except Exception as e:
    print(f"✗ Module import failed: {e}")

# Test 2: Direct function import
try:
    from repeatradar.visualization import plot_cohort_heatmap
    print("✓ Function imported successfully")
    print(f"Function: {plot_cohort_heatmap}")
except Exception as e:
    print(f"✗ Function import failed: {e}")

# Test 3: Check if function exists
try:
    import repeatradar.visualization as viz
    if hasattr(viz, 'plot_cohort_heatmap'):
        print("✓ Function exists in module")
    else:
        print("✗ Function does not exist in module")
        print(f"Available: {dir(viz)}")
except Exception as e:
    print(f"✗ Module check failed: {e}")
