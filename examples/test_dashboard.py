#!/usr/bin/env python3
"""
Test the dashboard functionality of the visualization module.
"""

import sys
sys.path.insert(0, 'src')

import pickle
from repeatradar import (
    generate_cohort_data, 
    plot_cohort_heatmap, 
    plot_retention_curves,
    plot_cohort_comparison,
    plot_period_comparison,
    create_cohort_dashboard
)

def test_dashboard():
    """Test the comprehensive dashboard functionality."""
    print("Loading sample data...")
    
    # Load sample data
    with open('examples/data/ecommerce_data_1.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Filter to recent data for faster processing
    df = df.head(10000)
    print(f"Sample data shape: {df.shape}")
    
    # Generate different types of cohort data
    print("Generating user cohort data...")
    user_data = generate_cohort_data(
        df,
        date_column='InvoiceDateTime',
        user_column='CustomerID',
        cohort_period='M',
        period_duration=1
    )
    
    print("Generating retention data...")
    retention_data = generate_cohort_data(
        df,
        date_column='InvoiceDateTime',
        user_column='CustomerID',
        cohort_period='M',
        period_duration=1,
        calculate_retention_rate=True
    )
    
    print("Generating revenue data...")
    revenue_data = generate_cohort_data(
        df,
        date_column='InvoiceDateTime',
        user_column='CustomerID',
        value_column='TotalPrice',
        cohort_period='M',
        period_duration=1,
        aggregation_function='sum'
    )
    
    # Test individual visualizations
    print("\nTesting individual visualizations...")
    
    # Test heatmap
    fig1 = plot_cohort_heatmap(user_data, title="User Cohort Heatmap")
    print(f"✓ Heatmap: {type(fig1)}")
    
    # Test retention curves
    fig2 = plot_retention_curves(retention_data, title="Retention Curves")
    print(f"✓ Retention curves: {type(fig2)}")
    
    # Test comparison
    fig3 = plot_cohort_comparison(
        {'Users': user_data, 'Revenue': revenue_data},
        title="User vs Revenue Comparison"
    )
    print(f"✓ Cohort comparison: {type(fig3)}")
    
    # Test period comparison
    fig4 = plot_period_comparison(
        user_data, 
        periods_to_compare=[0, 1, 2], 
        chart_type='bar',
        title="Period Comparison"
    )
    print(f"✓ Period comparison: {type(fig4)}")
    
    # Test comprehensive dashboard
    print("\nTesting comprehensive dashboard...")
    dashboard = create_cohort_dashboard(
        cohort_data=user_data,
        retention_data=retention_data,
        revenue_data=revenue_data,
        title="Complete Cohort Analysis Dashboard"
    )
    print(f"✓ Dashboard: {type(dashboard)}")
    
    print("\n🎉 All dashboard tests passed!")

if __name__ == "__main__":
    test_dashboard()
