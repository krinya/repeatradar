# Cohort Retention Analysis Made Easy with RepeatRadar

As a Python enthusiast, I wanted to create something **genuinely useful** for the data science community - that's how RepeatRadar was born! Cohort retention analysis is one of the **most powerful techniques** for understanding user behavior, but existing solutions were either too complex or lacked flexibility.

**This article shows you:**
* What cohort retention analysis is and why it matters
* How to use RepeatRadar to get insights in just a few lines of code

# What is cohort retention analysis?

Cohort analysis groups users into *"cohorts"* based on when they first engaged with your product, then tracks how many come back over time.

Think Netflix subscribers: track different monthly signup groups to see how many are **still actively watching** 3, 6, or 12 months later.

**Why it's powerful:**
- **User Loyalty**: How loyal are your users over time?
- **Marketing Effectiveness**: Which channels bring the **most loyal users**?
- **Revenue Prediction**: Understand **lifetime value** of user segments

Instead of just *"1000 active users last month,"* you get insights like *"January cohort has 40% retention after 3 months"* or *"Q4 cohort performs 20% better than Q3!"*

# Quick Start with RepeatRadar

## Installation
```bash
pip install repeatradar
```

## Your Data
You need a pandas DataFrame with:
* **Date column** (transaction/signup date)
* **User ID column** 
* **Value column** (optional, for revenue analysis, watch time, etc.)

```python
# Example data structure
   user_id purchase_date  purchase_amount
0  user_1    2024-01-15             45.99
1  user_2    2024-01-15             23.50
2  user_1    2024-02-10             67.25
```

# Example: E-commerce Analysis

```python
from repeatradar import generate_cohort_data, plot_cohort_heatmap
import pandas as pd

# Load sample e-commerce data
data_url = "https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl"
ecommerce_data = pd.read_pickle(data_url)

# Generate retention counts
retention_counts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M',     # Monthly cohorts
    period_duration=30     # Track in 30-day periods
)

print("📊 User Retention Counts:")
print(retention_counts)
```

```python
# Generate retention rates (percentages)
retention_rates = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    calculate_retention_rate=True,  # Convert to percentages
    cohort_period='M',              # Monthly cohorts
    period_duration=30              # Track in 30-day periods
)

print("📈 User Retention Rates (%):")
print(retention_rates.round(1))
```

## Revenue Analysis
```python
# Track revenue over time instead of user counts
revenue_cohorts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    value_column='TotalPrice',    # Track revenue
    aggregation_function='sum',   # Sum up revenue
    cohort_period='M',            # Monthly cohorts
    period_duration=30            # Track in 30-day periods
)

print("💰 Revenue by Cohort:")
print(revenue_cohorts.round(0))
```

## Visualizations
```python
# Create interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    cohort_data=retention_rates,
    title="User Retention Rates Over Time"
)
heatmap_fig.show()
```

**Key insights you'll discover:**
- Which cohorts generate the **most revenue**?
- How does retention change over time?
- Which acquisition periods produce **loyal customers**?

# Why RepeatRadar?

✅ **Simple**: A few lines of code for complex analysis  
✅ **Flexible**: User counts, revenue, custom metrics  
✅ **Visual**: Beautiful interactive charts  
✅ **Fast**: Built on pandas for performance  

# Get Started Today

RepeatRadar makes cohort analysis **accessible and powerful**. Every business has retention patterns waiting to be uncovered - RepeatRadar just makes finding them **a lot easier!**

🔗 **GitHub**: [github.com/krinya/repeatradar](https://github.com/krinya/repeatradar)  
📦 **Install**: `pip install repeatradar`

*Happy analyzing! 🚀*
