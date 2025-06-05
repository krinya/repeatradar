# RepeatRadar 📡

[![PyPI version](https://badge.fury.io/py/repeatradar.svg)](https://badge.fury.io/py/repeatradar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![Tests](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml/badge.svg)](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml)

A streamlined Python package for calculating and visualizing cohort retention and other cohort-based metrics. RepeatRadar helps you understand user behavior and trends over time by grouping users into cohorts based on their acquisition period and tracking their activity or value in subsequent periods.

## 🔗 Links

- **GitHub Repository**: https://github.com/krinya/repeatradar
- **PyPI Package**: https://pypi.org/project/repeatradar/
- **Documentation**: https://krinya.github.io/repeatradar/
- **Examples**: 
  - [📊 Cohort Calculation Demo](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb)
  - [🎨 Cohort Visualization Demo](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb)
  - [📁 Example Data](https://github.com/krinya/repeatradar/tree/main/examples/data)

## ✨ Features

- **📊 Cohort Data Generation**: Calculate user retention, revenue cohorts, and custom metrics
- **🔥 Interactive Heatmaps**: Visualize cohort performance patterns with customizable styling
- **📈 Retention Curves**: Track customer journey and retention trends over time
- **📁 Export Options**: Save charts as HTML, PNG, or PDF for reports and presentations

## 🚀 Installation

```bash
pip install repeatradar
```

## 📊 Quick Start

### Basic Cohort Generation
This is how you can quickly calculate the user cohort data from your dataset:

```python
from repeatradar import generate_cohort_data

# Load your data (example dataset available in examples/data/)
ecommerce_data = pd.read_pickle("https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl")

# Generate user cohort data
basic_cohorts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M', # Set cohort_period to monthly grouping
    period_duration=30 # Set period duration (this is the granularity of retention tracking = the subsequent columns in the output)
)

basic_cohorts
```

![User Cohort Table ](https://raw.githubusercontent.com/krinya/repeatradar/refs/heads/main/img/example_user_1.png)

### Cohort Visualizations

```python
from repeatradar import plot_cohort_heatmap

# Create an interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    cohort_data=basic_cohorts, # The cohort data generated above
    title="📊 User Retention",
    color_scale="viridis",
    show_values=True,
    show_colorscale=False,
    width=900,
    height=600
)
heatmap_fig.show()
```

![heatmap of user cohort table](https://raw.githubusercontent.com/krinya/repeatradar/refs/heads/main/img/example_user_heatmap_1.png)

For detailed examples and advanced usage, see the examples Jupyter notebooks:
- **[📊 Cohort Calculation Examples](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb)** - Guide to generating different types of cohort data
- **[🎨 Visualization Examples](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb)** - Interactive visualizations creation

## 🎨 Core Visualization Functions

RepeatRadar provides two essential visualization functions for cohort analysis:

- **`plot_cohort_heatmap()`**: Interactive heatmaps showing cohort performance patterns over time
- **`plot_retention_curves()`**: Line charts comparing retention rates and trends across cohorts

These core functions provide comprehensive insights into customer behavior and other cohort-based metrics, allowing you to visualize and analyze user retention, revenue trends, and more.

> 💡 **Tip:** For detailed examples and advanced usage, see the [📊 Cohort Calculation Examples](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb) and [🎨 Visualization Examples](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb) notebooks.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.