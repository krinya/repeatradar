"""
Tests for visualization functions in the repeatradar package.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from repeatradar.visualization_generator import (
    plot_cohort_heatmap,
    plot_retention_curves,
    plot_cohort_comparison,
    plot_period_comparison,
    plot_cohort_summary_stats,
    create_cohort_dashboard
)
from repeatradar.cohort_generator import generate_cohort_data


@pytest.fixture
def sample_cohort_data():
    """Create sample cohort data for testing."""
    # Create a simple cohort data DataFrame
    dates = pd.date_range('2023-01-01', periods=6, freq='MS')
    data = {
        0: [100, 90, 80, 70, 60, 50],
        1: [0, 50, 45, 40, 35, 30],
        2: [0, 0, 35, 30, 25, 20],
        3: [0, 0, 0, 20, 15, 12],
        4: [0, 0, 0, 0, 10, 8],
        5: [0, 0, 0, 0, 0, 5]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_retention_data():
    """Create sample retention data for testing."""
    dates = pd.date_range('2023-01-01', periods=6, freq='MS')
    data = {
        0: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        1: [0.0, 55.6, 56.3, 57.1, 58.3, 60.0],
        2: [0.0, 0.0, 43.8, 42.9, 41.7, 40.0],
        3: [0.0, 0.0, 0.0, 28.6, 25.0, 24.0],
        4: [0.0, 0.0, 0.0, 0.0, 16.7, 16.0],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_revenue_data():
    """Create sample revenue cohort data for testing."""
    dates = pd.date_range('2023-01-01', periods=6, freq='MS')
    data = {
        0: [10000, 9000, 8000, 7000, 6000, 5000],
        1: [0, 4500, 4000, 3500, 3000, 2500],
        2: [0, 0, 3000, 2500, 2000, 1500],
        3: [0, 0, 0, 1500, 1200, 1000],
        4: [0, 0, 0, 0, 800, 600],
        5: [0, 0, 0, 0, 0, 400]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing end-to-end functionality."""
    np.random.seed(42)
    
    # Generate users and dates
    users = [f"user_{i}" for i in range(1, 501)]
    start_date = datetime(2023, 1, 1)
    
    data = []
    for user in users:
        # Each user first transaction (acquisition)
        first_date = start_date + timedelta(days=np.random.randint(0, 180))
        data.append({
            'user_id': user,
            'transaction_date': first_date,
            'revenue': np.random.uniform(10, 100)
        })
        
        # Additional transactions (some users are more active)
        if np.random.random() > 0.3:  # 70% chance of repeat transaction
            for _ in range(np.random.randint(1, 5)):
                next_date = first_date + timedelta(days=np.random.randint(1, 365))
                if next_date <= start_date + timedelta(days=365):
                    data.append({
                        'user_id': user,
                        'transaction_date': next_date,
                        'revenue': np.random.uniform(5, 150)
                    })
    
    return pd.DataFrame(data)


class TestCohortHeatmap:
    """Test cases for plot_cohort_heatmap function."""
    
    def test_basic_heatmap_creation(self, sample_cohort_data):
        """Test basic heatmap creation with default parameters."""
        fig = plot_cohort_heatmap(sample_cohort_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Cohort Analysis Heatmap"
    
    def test_heatmap_with_custom_title(self, sample_cohort_data):
        """Test heatmap with custom title."""
        custom_title = "Custom Cohort Heatmap"
        fig = plot_cohort_heatmap(sample_cohort_data, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_heatmap_color_scales(self, sample_cohort_data):
        """Test different color scales."""
        color_scales = ["Blues", "Viridis", "RdYlBu", "Plasma"]
        
        for color_scale in color_scales:
            fig = plot_cohort_heatmap(sample_cohort_data, color_scale=color_scale)
            assert isinstance(fig, go.Figure)
    
    def test_heatmap_without_values(self, sample_cohort_data):
        """Test heatmap without showing values."""
        fig = plot_cohort_heatmap(sample_cohort_data, show_values=False)
        
        assert isinstance(fig, go.Figure)
        # When show_values=False, there should be no annotations
        assert len(fig.layout.annotations) == 0
    
    def test_heatmap_custom_dimensions(self, sample_cohort_data):
        """Test heatmap with custom width and height."""
        fig = plot_cohort_heatmap(sample_cohort_data, width=1000, height=700)
        
        assert fig.layout.width == 1000
        assert fig.layout.height == 700
    
    def test_heatmap_without_colorscale(self, sample_cohort_data):
        """Test heatmap without showing colorscale."""
        fig = plot_cohort_heatmap(sample_cohort_data, show_colorscale=False)
        
        assert isinstance(fig, go.Figure)
        # Check that colorscale is hidden
        assert fig.data[0].showscale == False
    
    def test_heatmap_reversed_y_axis(self, sample_cohort_data):
        """Test heatmap with reversed y-axis."""
        fig = plot_cohort_heatmap(sample_cohort_data, reverse_y_axis=True)
        
        assert isinstance(fig, go.Figure)
        # Check that the y-axis autorange is set to 'reversed'
        assert fig.layout.yaxis.autorange == 'reversed'
    
    def test_heatmap_show_values_with_formatting(self, sample_cohort_data):
        """Test heatmap with values shown and custom formatting."""
        fig = plot_cohort_heatmap(sample_cohort_data, show_values=True, value_format=".1f")
        
        assert isinstance(fig, go.Figure)
        # Check that text template is set correctly
        assert fig.data[0].texttemplate == "%{text:.1f}"


class TestRetentionCurves:
    """Test cases for plot_retention_curves function."""
    
    def test_basic_retention_curves(self, sample_retention_data):
        """Test basic retention curves creation."""
        fig = plot_retention_curves(sample_retention_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(sample_retention_data)  # One line per cohort
        assert fig.layout.title.text == "Cohort Retention Curves"
    
    def test_retention_curves_limited_cohorts(self, sample_retention_data):
        """Test retention curves with limited number of cohorts."""
        max_cohorts = 3
        fig = plot_retention_curves(sample_retention_data, max_cohorts=max_cohorts)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == max_cohorts
    
    def test_retention_curves_no_legend(self, sample_retention_data):
        """Test retention curves without legend."""
        fig = plot_retention_curves(sample_retention_data, show_legend=False)
        
        assert fig.layout.showlegend is False
    
    def test_retention_curves_custom_title(self, sample_retention_data):
        """Test retention curves with custom title."""
        custom_title = "Custom Retention Analysis"
        fig = plot_retention_curves(sample_retention_data, title=custom_title)
        
        assert fig.layout.title.text == custom_title


class TestCohortComparison:
    """Test cases for plot_cohort_comparison function."""
    
    def test_basic_comparison(self, sample_cohort_data, sample_revenue_data):
        """Test basic cohort comparison."""
        cohort_dict = {
            'users': sample_cohort_data,
            'revenue': sample_revenue_data
        }
        
        fig = plot_cohort_comparison(cohort_dict)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # One heatmap per metric
    
    def test_comparison_with_custom_names(self, sample_cohort_data, sample_revenue_data):
        """Test comparison with custom metric names."""
        cohort_dict = {
            'users': sample_cohort_data,
            'revenue': sample_revenue_data
        }
        metric_names = {
            'users': 'User Count',
            'revenue': 'Revenue ($)'
        }
        
        fig = plot_cohort_comparison(cohort_dict, metric_names=metric_names)
        
        assert isinstance(fig, go.Figure)
    
    def test_single_metric_comparison(self, sample_cohort_data):
        """Test comparison with single metric."""
        cohort_dict = {'users': sample_cohort_data}
        
        fig = plot_cohort_comparison(cohort_dict)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
    
    def test_comparison_without_colorscale(self, sample_cohort_data, sample_revenue_data):
        """Test comparison without showing colorscale."""
        cohort_dict = {
            'users': sample_cohort_data,
            'revenue': sample_revenue_data
        }
        
        fig = plot_cohort_comparison(cohort_dict, show_colorscale=False)
        
        assert isinstance(fig, go.Figure)
        # Check that colorscale is hidden for both heatmaps
        assert fig.data[0].showscale == False
        assert fig.data[1].showscale == False
    
    def test_comparison_reversed_y_axis(self, sample_cohort_data, sample_revenue_data):
        """Test comparison with reversed y-axis."""
        cohort_dict = {
            'users': sample_cohort_data,
            'revenue': sample_revenue_data
        }
        
        fig = plot_cohort_comparison(cohort_dict, reverse_y_axis=True)
        
        assert isinstance(fig, go.Figure)
        # Check that y-axis autorange is set to 'reversed' for all subplots
        # Since we have 2 metrics, they should be in a 1x2 layout
        for i in range(1, 3):  # subplot positions 1 and 2
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1
            yaxis_name = f'yaxis{i}' if i > 1 else 'yaxis'
            if hasattr(fig.layout, yaxis_name):
                yaxis = getattr(fig.layout, yaxis_name)
                assert yaxis.autorange == 'reversed'


class TestPeriodComparison:
    """Test cases for plot_period_comparison function."""
    
    def test_basic_period_comparison_bar(self, sample_cohort_data):
        """Test basic period comparison with bar chart."""
        fig = plot_period_comparison(sample_cohort_data)
        
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == 'bar'
    
    def test_period_comparison_line(self, sample_cohort_data):
        """Test period comparison with line chart."""
        fig = plot_period_comparison(sample_cohort_data, chart_type="line")
        
        assert isinstance(fig, go.Figure)
        assert all(trace.type == 'scatter' for trace in fig.data)
    
    def test_period_comparison_custom_periods(self, sample_cohort_data):
        """Test period comparison with custom periods."""
        periods = [0, 2, 4]
        fig = plot_period_comparison(sample_cohort_data, periods_to_compare=periods)
        
        assert isinstance(fig, go.Figure)
    
    def test_period_comparison_invalid_periods(self, sample_cohort_data):
        """Test period comparison with invalid periods."""
        periods = [10, 20, 30]  # These periods don't exist in the data
        
        with pytest.raises(ValueError, match="None of the specified periods are available"):
            plot_period_comparison(sample_cohort_data, periods_to_compare=periods)


class TestCohortSummaryStats:
    """Test cases for plot_cohort_summary_stats function."""
    
    def test_basic_summary_stats(self, sample_cohort_data):
        """Test basic summary statistics visualization."""
        fig = plot_cohort_summary_stats(sample_cohort_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 5  # Mean, Median, Min, Max, Std
    
    def test_summary_stats_custom_title(self, sample_cohort_data):
        """Test summary stats with custom title."""
        custom_title = "Custom Summary Statistics"
        fig = plot_cohort_summary_stats(sample_cohort_data, title=custom_title)
        
        assert fig.layout.title.text == custom_title


class TestCohortDashboard:
    """Test cases for create_cohort_dashboard function."""
    
    def test_dashboard_single_metric(self, sample_cohort_data):
        """Test dashboard with only user data."""
        fig = create_cohort_dashboard(sample_cohort_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    
    def test_dashboard_with_retention(self, sample_cohort_data, sample_retention_data):
        """Test dashboard with user and retention data."""
        fig = create_cohort_dashboard(sample_cohort_data, retention_data=sample_retention_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2
    
    def test_dashboard_all_metrics(self, sample_cohort_data, sample_retention_data, sample_revenue_data):
        """Test dashboard with all three metrics."""
        fig = create_cohort_dashboard(
            sample_cohort_data, 
            retention_data=sample_retention_data,
            revenue_data=sample_revenue_data
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3
    
    def test_dashboard_custom_title(self, sample_cohort_data):
        """Test dashboard with custom title."""
        custom_title = "Custom Dashboard"
        fig = create_cohort_dashboard(sample_cohort_data, title=custom_title)
        
        assert fig.layout.title.text == custom_title
    
    def test_dashboard_reversed_y_axis(self, sample_cohort_data, sample_retention_data, sample_revenue_data):
        """Test dashboard with reversed y-axis."""
        fig = create_cohort_dashboard(
            sample_cohort_data, 
            retention_data=sample_retention_data,
            revenue_data=sample_revenue_data,
            reverse_y_axis=True
        )
        
        assert isinstance(fig, go.Figure)
        # With 3 metrics, the layout should be 2x2, so we have 3 subplots with data
        # Check that y-axis autorange is set to 'reversed' for all subplots
        for i in range(1, 4):  # We have 3 subplots with data
            yaxis_name = f'yaxis{i}' if i > 1 else 'yaxis'
            if hasattr(fig.layout, yaxis_name):
                yaxis = getattr(fig.layout, yaxis_name)
                assert yaxis.autorange == 'reversed'
    
    def test_dashboard_without_colorscale(self, sample_cohort_data, sample_retention_data):
        """Test dashboard without showing colorscale."""
        fig = create_cohort_dashboard(
            sample_cohort_data, 
            retention_data=sample_retention_data,
            show_colorscale=False
        )
        
        assert isinstance(fig, go.Figure)
        # Check that colorscale is hidden for heatmaps
        assert fig.data[0].showscale == False
        assert fig.data[1].showscale == False


class TestEndToEndVisualization:
    """End-to-end tests using generated cohort data."""
    
    def test_end_to_end_user_cohorts(self, sample_transaction_data):
        """Test end-to-end workflow: generate data -> visualize."""
        # Generate cohort data
        cohort_data = generate_cohort_data(
            data=sample_transaction_data,
            date_column='transaction_date',
            user_column='user_id',
            cohort_period='M',
            period_duration=30
        )
        
        # Test that visualization functions work with generated data
        fig1 = plot_cohort_heatmap(cohort_data)
        assert isinstance(fig1, go.Figure)
        
        fig2 = plot_period_comparison(cohort_data)
        assert isinstance(fig2, go.Figure)
        
        fig3 = plot_cohort_summary_stats(cohort_data)
        assert isinstance(fig3, go.Figure)
    
    def test_end_to_end_retention_analysis(self, sample_transaction_data):
        """Test end-to-end retention analysis."""
        # Generate retention data
        retention_data = generate_cohort_data(
            data=sample_transaction_data,
            date_column='transaction_date',
            user_column='user_id',
            cohort_period='M',
            period_duration=30,
            calculate_retention_rate=True
        )
        
        # Test retention-specific visualizations
        fig = plot_retention_curves(retention_data)
        assert isinstance(fig, go.Figure)
    
    def test_end_to_end_revenue_analysis(self, sample_transaction_data):
        """Test end-to-end revenue analysis."""
        # Generate revenue cohort data
        revenue_data = generate_cohort_data(
            data=sample_transaction_data,
            date_column='transaction_date',
            user_column='user_id',
            value_column='revenue',
            aggregation_function='sum',
            cohort_period='M',
            period_duration=30
        )
        
        # Test that visualizations work with revenue data
        fig = plot_cohort_heatmap(revenue_data, title="Revenue Cohorts")
        assert isinstance(fig, go.Figure)
    
    def test_end_to_end_dashboard(self, sample_transaction_data):
        """Test complete dashboard with all metrics."""
        # Generate all types of cohort data
        user_data = generate_cohort_data(
            data=sample_transaction_data,
            date_column='transaction_date',
            user_column='user_id',
            cohort_period='M',
            period_duration=30
        )
        
        retention_data = generate_cohort_data(
            data=sample_transaction_data,
            date_column='transaction_date',
            user_column='user_id',
            cohort_period='M',
            period_duration=30,
            calculate_retention_rate=True
        )
        
        revenue_data = generate_cohort_data(
            data=sample_transaction_data,
            date_column='transaction_date',
            user_column='user_id',
            value_column='revenue',
            aggregation_function='sum',
            cohort_period='M',
            period_duration=30
        )
        
        # Create comprehensive dashboard
        fig = create_cohort_dashboard(
            user_data,
            retention_data=retention_data,
            revenue_data=revenue_data,
            title="Complete Cohort Analysis"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3
    
    def test_dashboard_without_colorscale(self, sample_cohort_data, sample_retention_data):
        """Test dashboard without showing colorscales."""
        fig = create_cohort_dashboard(
            cohort_data=sample_cohort_data,
            retention_data=sample_retention_data,
            show_colorscale=False
        )
        
        assert isinstance(fig, go.Figure)
        # Check that colorscales are hidden
        assert fig.data[0].showscale == False
        assert fig.data[1].showscale == False


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""
    
    def test_empty_dataframe_error(self):
        """Test that empty DataFrames raise appropriate errors."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            plot_cohort_heatmap(empty_df)
    
    def test_invalid_color_scale(self, sample_cohort_data):
        """Test that invalid color scales raise appropriate errors."""
        # This should raise a ValueError for invalid color scale
        with pytest.raises(ValueError, match="Invalid color scale"):
            plot_cohort_heatmap(sample_cohort_data, color_scale="InvalidColorScale")
