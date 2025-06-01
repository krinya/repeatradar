# RepeatRadar 📡

[![PyPI version](https://badge.fury.io/py/repeatradar.svg)](https://badge.fury.io/py/repeatradar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![Tests](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml/badge.svg)](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml)

A comprehensive Python package for calculating and visualizing cohort retention and other cohort-based metrics. RepeatRadar helps you understand user behavior and trends over time by grouping users into cohorts based on their acquisition period and tracking their activity or value in subsequent periods.

## ✨ Features

- **Cohort Data Generation**: Calculate user retention, revenue cohorts, and custom metrics
- **Rich Visualizations**: Interactive heatmaps, retention curves, and comprehensive dashboards
- **Flexible Analysis**: Support for different time periods, aggregation functions, and cohort types
- **Easy Integration**: Simple API that works seamlessly with pandas DataFrames
- **Plotly-Powered**: Beautiful, interactive visualizations using Plotly

## 🚀 Installation

```bash
pip install repeatradar
```

## 📊 Quick Start

### Basic Cohort Analysis

```python
from repeatradar import generate_cohort_data
import pandas as pd

# Load your data
ecommerce_data = pd.read_pickle("https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl")

# Generate user cohort data
user_cohorts = generate_cohort_data(
    data=ecommerce_data,
    datetime_column_name='InvoiceDateTime',
    user_column_name='CustomerID',
    base_period='M',  # Monthly cohorts
    period_duration=30,  # 30-day periods
    output_format='pivot'
)

print(user_cohorts)
```

### Cohort Visualizations

```python
from repeatradar.visualization_generator import (
    plot_cohort_heatmap, 
    plot_retention_curves, 
    create_cohort_dashboard
)

# Create an interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    user_cohorts, 
    title="User Retention Cohorts",
    color_scale="Blues"
)
heatmap_fig.show()

# Generate retention rate data
retention_data = generate_cohort_data(
    data=ecommerce_data,
    datetime_column_name='InvoiceDateTime',
    user_column_name='CustomerID',
    calculate_retention_rate=True,
    output_format='pivot'
)

# Plot retention curves
retention_fig = plot_retention_curves(
    retention_data,
    title="Cohort Retention Curves"
)
retention_fig.show()

# Create a comprehensive dashboard
dashboard_fig = create_cohort_dashboard(
    cohort_data=user_cohorts,
    retention_data=retention_data,
    title="Cohort Analysis Dashboard"
)
dashboard_fig.show()
```

### Revenue Cohort Analysis

```python
# Analyze revenue cohorts
revenue_cohorts = generate_cohort_data(
    data=ecommerce_data,
    datetime_column_name='InvoiceDateTime',
    user_column_name='CustomerID',
    value_column='Revenue',  # Specify revenue column
    aggregation_function='sum',  # Sum revenue per cohort/period
    output_format='pivot'
)

# Visualize revenue heatmap
revenue_fig = plot_cohort_heatmap(
    revenue_cohorts,
    title="Revenue Cohorts",
    color_scale="Viridis",
    value_format=".0f"
)
revenue_fig.show()
```
![Example Output](img/example_1.png)

## 🎨 Available Visualizations

RepeatRadar offers several types of visualizations:

- **`plot_cohort_heatmap`**: Interactive heatmaps showing cohort performance over time
- **`plot_retention_curves`**: Line charts comparing retention rates across cohorts
- **`plot_cohort_comparison`**: Side-by-side comparison of multiple metrics
- **`plot_period_comparison`**: Compare specific periods across all cohorts
- **`plot_cohort_summary_stats`**: Statistical summaries of cohort performance
- **`create_cohort_dashboard`**: Comprehensive dashboard with multiple visualizations

## 📚 API Reference

### Core Functions

#### `generate_cohort_data()`

Generate cohort analysis data from transactional data.

**Parameters:**
- `data` (pd.DataFrame): Input DataFrame with user transactions
- `datetime_column_name` (str): Name of the datetime column
- `user_column_name` (str): Name of the user identifier column
- `value_column` (str, optional): Column to aggregate (default: count users)
- `aggregation_function` (str): How to aggregate values ('count', 'sum', 'mean', etc.)
- `base_period` (str): Cohort grouping period ('D', 'W', 'M', 'Q', 'Y')
- `period_duration` (int): Duration of analysis periods in days
- `calculate_retention_rate` (bool): Whether to calculate retention percentages
- `output_format` (str): Output format ('pivot' or 'long')

**Returns:**
- `pd.DataFrame`: Cohort analysis results

### Visualization Functions

All visualization functions return Plotly Figure objects that can be displayed with `.show()` or saved with `.write_html()` or `.write_image()`.

## 🛠️ Advanced Usage

### Custom Aggregation Functions

```python
# Calculate average order value per cohort
aov_cohorts = generate_cohort_data(
    data=ecommerce_data,
    datetime_column_name='InvoiceDateTime',
    user_column_name='CustomerID',
    value_column='Revenue',
    aggregation_function='mean',  # Average revenue per user
    output_format='pivot'
)
```

### Different Time Periods

```python
# Weekly cohorts with 7-day periods
weekly_cohorts = generate_cohort_data(
    data=ecommerce_data,
    datetime_column_name='InvoiceDateTime',
    user_column_name='CustomerID',
    base_period='W',  # Weekly cohorts
    period_duration=7,  # 7-day periods
    output_format='pivot'
)
```

### Comparing Multiple Metrics

```python
from repeatradar.visualization_generator import plot_cohort_comparison

# Compare user counts and revenue
metrics_comparison = plot_cohort_comparison(
    cohort_data_dict={
        'users': user_cohorts,
        'revenue': revenue_cohorts
    },
    metric_names={
        'users': 'Active Users',
        'revenue': 'Total Revenue ($)'
    },
    title="Users vs Revenue Comparison"
)
metrics_comparison.show()
```

## 🔗 Useful Links

- **GitHub Repository**: https://github.com/krinya/repeatradar
- **PyPI Package**: https://pypi.org/project/repeatradar/
- **Documentation**: Coming soon!
- **Examples**: Check the `examples/` directory in the repository

## 🗺️ Roadmap

- [ ] Additional visualization types (funnel charts, cohort tables)
- [ ] Export functionality for reports
- [ ] Integration with popular BI tools
- [ ] Advanced statistical analysis features
- [ ] Jupyter notebook widgets for interactive analysis
- [ ] Performance optimizations for large datasets

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes and commit them (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please make sure to update tests as appropriate and follow the coding standards outlined in the project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pandas](https://pandas.pydata.org/) for data manipulation
- Powered by [Plotly](https://plotly.com/) for interactive visualizations
- Package management with [Poetry](https://python-poetry.org/)

---

**RepeatRadar** - Making cohort analysis simple and beautiful! 📡✨