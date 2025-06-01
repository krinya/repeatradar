# RepeatRadar 📡

[![PyPI version](https://badge.fury.io/py/repeatradar.svg)](https://badge.fury.io/py/repeatradar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![Tests](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml/badge.svg)](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml)

A comprehensive Python package for calculating and visualizing cohort retention and other cohort-based metrics. RepeatRadar helps you understand user behavior and trends over time by grouping users into cohorts based on their acquisition period and tracking their activity or value in subsequent periods.

## 🔗 Useful Links

- **GitHub Repository**: https://github.com/krinya/repeatradar
- **PyPI Package**: https://pypi.org/project/repeatradar/
- **Documentation**: Coming soon!
- **Examples**: 
  - [📊 Cohort Calculation Demo](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb)
  - [🎨 Cohort Visualization Demo](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb)
  - [📁 Example Data](https://github.com/krinya/repeatradar/tree/main/examples/data)

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

# Load your data (example dataset available in examples/data/)
ecommerce_data = pd.read_pickle("https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl")

# Generate user cohort data
basic_cohorts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M',       # Monthly cohorts (users grouped by acquisition month)
    period_duration=30       # Track in 30-day periods
)

print(basic_cohorts.head())
```

### Cohort Visualizations

```python
from repeatradar import plot_cohort_heatmap

# Create an interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    basic_cohorts, 
    title="User Retention Cohorts",
    color_scale="Blues"
)
heatmap_fig.show()
```

For detailed examples and advanced usage, see our Jupyter notebooks:
- **[📊 Cohort Calculation Examples](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb)** - Complete guide to generating different types of cohort data
- **[🎨 Visualization Examples](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb)** - Interactive visualizations and dashboard creation

## 🎨 Available Visualizations

RepeatRadar offers several types of visualizations:

- **`plot_cohort_heatmap`**: Interactive heatmaps showing cohort performance over time
- **`plot_retention_curves`**: Line charts comparing retention rates across cohorts
- **`plot_cohort_comparison`**: Side-by-side comparison of multiple metrics
- **`plot_period_comparison`**: Compare specific periods across all cohorts
- **`plot_cohort_summary_stats`**: Statistical summaries of cohort performance
- **`create_cohort_dashboard`**: Comprehensive dashboard with multiple visualizations

> 💡 **Tip:** Explore the [visualization examples notebook](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb) for detailed usage of all visualization functions.


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