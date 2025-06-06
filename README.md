# RepeatRadar 📡

[![PyPI version](https://badge.fury.io/py/repeatradar.svg)](https://badge.fury.io/py/repeatradar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![Tests](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml/badge.svg)](https://github.com/krinya/repeatradar/actions/workflows/publish_to_pypi.yml) ![PyPI - Downloads](https://img.shields.io/pypi/dm/repeatradar)

A lightweight Python package for calculating and visualizing cohort retention and other cohort-based metrics. With just a few simple commands, you can analyze user behavior and trends over time by grouping users into cohorts based on their acquisition period and tracking their activity or value in subsequent periods.

## 🔗 Links

**GitHub**: https://github.com/krinya/repeatradar
**PyPI**: https://pypi.org/project/repeatradar/
**Documentation**: https://krinya.github.io/repeatradar/
**Example notebooks**:
* [Cohort Calculation Demo](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb)
* [Cohort Visualization Demo](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb)
* [Example Data](https://github.com/krinya/repeatradar/tree/main/examples/data)

## 🌟 Features

**Cohort Analysis Made Easy**: Quickly calculate user retention, revenue trends, and other cohort-based metrics.
**Insightful Visualizations**: Create heatmaps and retention curves to uncover patterns in your data.
**Highly Customizable**: Flexible options for cohort periods, tracking granularity, and visualization styles.
**Seamless Pandas & Plotly Integration**: Works effortlessly with popular Python libraries for a smooth analysis experience.

## 🚀 Installation
To install RepeatRadar, simply use pip:

```bash
pip install repeatradar
```

## 📊 Quick Start

### Cohort Generation
Here's how you can quickly calculate user cohort data from your dataset:

```python
from repeatradar import generate_cohort_data

# Load your data (example dataset available in examples/data/)
ecommerce_data = pd.read_pickle("https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl")

# Basic Monthly User Cohort Analysis
# This creates a cohort table showing unique user counts for each period

basic_cohorts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M',       # Monthly cohorts (users grouped by acquisition month)
    period_duration=30       # Track in 30-day periods
)

# This shows how many unique users return in each 30-day period
basic_cohorts
```

![User Cohort Table](https://raw.githubusercontent.com/krinya/repeatradar/refs/heads/main/img/example_user_1.png)

### Cohort Visualizations

You can visualize the cohort data using heatmaps to better understand user retention trends:

```python
from repeatradar import plot_cohort_heatmap

# Create an interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    cohort_data=basic_cohorts,  # The cohort data generated above
    title="📊 User Retention",
    color_scale="viridis",
    show_values=True,
    show_colorscale=False,
    width=900,
    height=600
)
heatmap_fig.show()
```

![Heatmap of User Cohort Table](https://raw.githubusercontent.com/krinya/repeatradar/refs/heads/main/img/example_user_heatmap_1.png)

> 💡 **Tip:** For detailed examples and advanced usage, check out the [Cohort Calculation Examples](https://github.com/krinya/repeatradar/blob/main/examples/01_cohort_calculation_demo.ipynb) and [Visualization Examples](https://github.com/krinya/repeatradar/blob/main/examples/02_cohort_visualization_demo.ipynb) notebooks.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.