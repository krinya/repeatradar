#!/usr/bin/env python3
"""Debug script to test visualization imports."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    print("1. Testing pandas import...")
    import pandas as pd
    print("   ✓ pandas imported successfully")
except ImportError as e:
    print(f"   ✗ pandas import failed: {e}")

try:
    print("2. Testing plotly imports...")
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("   ✓ plotly imported successfully")
except ImportError as e:
    print(f"   ✗ plotly import failed: {e}")

try:
    print("3. Testing visualization module structure...")
    
    # Import the module file directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "visualization", 
        "src/repeatradar/visualization.py"
    )
    viz_module = importlib.util.module_from_spec(spec)
    
    print("   Module spec created successfully")
    
    spec.loader.exec_module(viz_module)
    print("   Module executed successfully")
    
    # Check what's in the module
    all_attrs = dir(viz_module)
    functions = [attr for attr in all_attrs if not attr.startswith('_')]
    print(f"   Module attributes: {functions}")
    
    # Check specifically for our functions
    target_functions = [
        'plot_cohort_heatmap',
        'plot_retention_curves', 
        'plot_cohort_comparison',
        'plot_period_comparison',
        'plot_cohort_summary_stats',
        'create_cohort_dashboard'
    ]
    
    for func_name in target_functions:
        if hasattr(viz_module, func_name):
            print(f"   ✓ {func_name} found")
        else:
            print(f"   ✗ {func_name} not found")
            
except Exception as e:
    print(f"   ✗ Module loading failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Testing import through package...")
    from repeatradar import visualization
    print("   ✓ Package import successful")
    
    # Test specific function import
    from repeatradar.visualization import plot_cohort_heatmap
    print("   ✓ Function import successful")
    
except ImportError as e:
    print(f"   ✗ Package import failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug complete.")
