#!/usr/bin/env python3
"""
Demo script showing how to use all visualization functions in the repeatradar package.

This script demonstrates the complete workflow from data loading to visualization
using the enhanced repeatradar package with Plotly visualizations.
"""

import sys
sys.path.insert(0, 'src')

import pickle
import pandas as pd
from repeatradar import (
    generate_cohort_data,
    plot_cohort_heatmap,
    plot_retention_curves,
    plot_cohort_comparison,
    plot_period_comparison,
    plot_cohort_summary_stats,
    create_cohort_dashboard
)

def demo_visualization_functions():
    """Demonstrate all visualization functions with real data."""
    
    print("🎯 RepeatRadar Visualization Demo")
    print("=" * 50)
    
    # Load sample data
    print("📊 Loading sample ecommerce data...")
    with open('examples/data/ecommerce_data_1.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Use a subset for faster processing
    df = df.head(5000)
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['InvoiceDateTime'].min()} to {df['InvoiceDateTime'].max()}")
    print(f"   Unique customers: {df['CustomerID'].nunique()}")
    
    print("\n🔄 Generating cohort data...")
    
    # 1. Generate user cohort data
    user_cohorts = generate_cohort_data(
        df,
        date_column='InvoiceDateTime',
        user_column='CustomerID',
        cohort_period='M',
        period_duration=1
    )
    print(f"   User cohorts shape: {user_cohorts.shape}")
    
    # 2. Generate retention data
    retention_data = generate_cohort_data(
        df,
        date_column='InvoiceDateTime',
        user_column='CustomerID',
        cohort_period='M',
        period_duration=1,
        calculate_retention_rate=True
    )
    print(f"   Retention data shape: {retention_data.shape}")
    
    # 3. Generate revenue cohort data
    revenue_cohorts = generate_cohort_data(
        df,
        date_column='InvoiceDateTime',
        user_column='CustomerID',
        value_column='TotalPrice',
        cohort_period='M',
        period_duration=1,
        aggregation_function='sum'
    )
    print(f"   Revenue cohorts shape: {revenue_cohorts.shape}")
    
    print("\n🎨 Creating visualizations...")
    
    # 1. Cohort Heatmap
    print("   1. Creating cohort heatmap...")
    heatmap_fig = plot_cohort_heatmap(
        user_cohorts,
        title="User Cohort Analysis - Customer Count by Period",
        color_scale="Blues",
        show_values=True
    )
    
    # 2. Retention Curves
    print("   2. Creating retention curves...")
    retention_fig = plot_retention_curves(
        retention_data,
        title="Customer Retention Curves by Cohort",
        max_cohorts=5  # Limit for cleaner visualization
    )
    
    # 3. Cohort Comparison
    print("   3. Creating cohort comparison...")
    comparison_fig = plot_cohort_comparison(
        {
            'Customer Count': user_cohorts,
            'Total Revenue': revenue_cohorts
        },
        metric_names={
            'Customer Count': 'Active Customers',
            'Total Revenue': 'Revenue ($)'
        },
        title="Customer Count vs Revenue Comparison"
    )
    
    # 4. Period Comparison (Bar Chart)
    print("   4. Creating period comparison (bar chart)...")
    period_bar_fig = plot_period_comparison(
        user_cohorts,
        periods_to_compare=[0, 1, 2, 3],
        chart_type='bar',
        title="Customer Count by Period (Bar Chart)"
    )
    
    # 5. Period Comparison (Line Chart)
    print("   5. Creating period comparison (line chart)...")
    period_line_fig = plot_period_comparison(
        retention_data,
        periods_to_compare=[0, 1, 2, 3, 4],
        chart_type='line',
        title="Retention Rate by Period (Line Chart)"
    )
    
    # 6. Summary Statistics
    print("   6. Creating summary statistics...")
    stats_fig = plot_cohort_summary_stats(
        user_cohorts,
        title="Customer Count Summary Statistics by Period"
    )
    
    # 7. Comprehensive Dashboard
    print("   7. Creating comprehensive dashboard...")
    dashboard_fig = create_cohort_dashboard(
        cohort_data=user_cohorts,
        retention_data=retention_data,
        revenue_data=revenue_cohorts,
        title="Complete Cohort Analysis Dashboard"
    )
    
    print("\n✅ All visualizations created successfully!")
    print("\n📈 Visualization Summary:")
    print(f"   • Heatmap: {type(heatmap_fig).__name__}")
    print(f"   • Retention Curves: {type(retention_fig).__name__}")
    print(f"   • Cohort Comparison: {type(comparison_fig).__name__}")
    print(f"   • Period Bar Chart: {type(period_bar_fig).__name__}")
    print(f"   • Period Line Chart: {type(period_line_fig).__name__}")
    print(f"   • Summary Statistics: {type(stats_fig).__name__}")
    print(f"   • Dashboard: {type(dashboard_fig).__name__}")
    
    print("\n💡 Usage Tips:")
    print("   • Call .show() on any figure to display it in a browser")
    print("   • Call .write_html('filename.html') to save as HTML")
    print("   • Call .write_image('filename.png') to save as image (requires kaleido)")
    print("   • All figures are interactive Plotly objects")
    
    print("\n🎉 Demo completed successfully!")
    
    return {
        'heatmap': heatmap_fig,
        'retention_curves': retention_fig,
        'comparison': comparison_fig,
        'period_bar': period_bar_fig,
        'period_line': period_line_fig,
        'summary_stats': stats_fig,
        'dashboard': dashboard_fig
    }

if __name__ == "__main__":
    figures = demo_visualization_functions()
    
    # Example of displaying a figure (uncomment to show in browser)
    # figures['heatmap'].show()
