# Beyond Transactions: Unlock Customer Purchasing Patterns with RepeatRadar 📊

*Turn complex cohort analysis into simple Python commands*

After working with customer data for years, I noticed that cohort retention analysis—one of the most powerful techniques for understanding user behavior—was often **time-consuming** to implement. I also wanted to learn how to build a Python package, so I created **RepeatRadar**: a Python package that simplifies cohort analysis calculations and visualizations into just a few lines of code.

This guide will show you:
1. **What cohort analysis is** and why it's essential for your business
2. **How to use RepeatRadar** to quickly extract actionable insights from your transaction data

## What is Cohort Retention Analysis? 🔍

**Cohort analysis** is a method to understand user behavior over time by grouping users into *"cohorts"* based on when they first engaged with your product or service.

Think of tracking different groups of **Netflix subscribers** who joined during different months—you want to see how many people from each signup group are still actively watching shows 3, 6, or 12 months later.

### Other examples where this comes in handy:
- 📅 All users who made their first purchase in **January 2024**
- 🛍️ All customers who signed up during a **Black Friday campaign**
- 📱 All users who downloaded your app from a **specific ad campaign** at a given time

The *"retention"* part tracks how many of these users **come back and engage again** in subsequent periods.

## How This Gives You Additional Value

Instead of just knowing *"we had 1,000 active users last month,"* cohort analysis tells you **more useful stories** like:

> *"Users who joined in January have **40% retention** after 3 months"*
> 
> *"Our Q4 cohort is performing **20% better** than Q3—the holiday campaign worked!"*

This deeper understanding helps you **predict revenue** by understanding lifetime value of different user segments, **optimize marketing** by investing more in channels that bring loyal users, and guide **product development** by seeing which features drive long-term engagement.

---

## Getting Started with RepeatRadar 🚀

### Installation

I built a package called **RepeatRadar** that makes cohort retention analysis in Python much simpler. It's available on PyPI and can be installed with a single command:

```bash
pip install repeatradar
```

That's it! RepeatRadar will automatically install its dependencies (pandas, numpy, and plotly for visualizations). You can find more details and examples in the [GitHub repository](https://github.com/krinya/repeatradar).

### Preparing Your Data 📊

You need your transaction data in a **pandas DataFrame**. The data should have at least two columns, but if you have more (like price, profit, etc.), you can perform more advanced analysis.

#### **Minimum Required Columns:**
- **📅 Date column**: Contains the date of the transaction, shipment, or any other date you want to use as a reference point
- **👤 User ID column**: Contains the user ID of the user who made the transaction *(essential for cohort analysis)*

#### **Extra Columns for Deeper Insights:**
- **💰 Value column** (like `purchase_amount`, `revenue`, etc.) - unlocks revenue cohort analysis
- **🏷️ Segmentation columns** (`product_id`, `category`, `channel`) - for more detailed analysis

Your data might look something like this:

```
user_id purchase_date  purchase_amount
0  user_1    2024-01-15             45.99
1  user_2    2024-01-15             23.50
2  user_1    2024-02-10             67.25
3  user_3    2024-01-16             15.00
```


## Hands-On Tutorial: Real E-commerce Analysis 🛍️

Now let's dive into some **practical examples**! I'll walk you through the most common use cases you'll encounter.

### Step 1: Install and Load RepeatRadar

```python
# Install if you haven't already
# !pip install repeatradar

# Import the main function
from repeatradar import generate_cohort_data
import pandas as pd
import numpy as np

# Check that everything works
import repeatradar
print(f"RepeatRadar version: {repeatradar.__version__}")
```

*Running this will display your installed version. For optimal compatibility, please confirm you are using the latest version (RepeatRadar 1.x.x, the version used in this article).*

### Step 2: Load Your Data

For this example, I'll use a **real e-commerce dataset**. You can follow along with your own data by replacing the data loading part.

```python
# Load sample e-commerce data
# This dataset contains customer transactions with CustomerID, InvoiceDateTime, and TotalPrice
data_url = "https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl"
ecommerce_data = pd.read_pickle(data_url)

# Let's see what we're working with
print(f"📊 Dataset Overview:")
print(f"{ecommerce_data.shape[0]:,} transactions from {ecommerce_data['CustomerID'].nunique():,} customers")
print(f"📅 Date range: {ecommerce_data['InvoiceDateTime'].min().strftime('%Y-%m-%d')} to {ecommerce_data['InvoiceDateTime'].max().strftime('%Y-%m-%d')}")
print(f"💰 Total revenue: ${ecommerce_data['TotalPrice'].sum():,.2f}")

# Take a peek at the data structure
ecommerce_data.head()
```

**Output:**
```
📊 Dataset Overview:
401,604 transactions from 4,372 customers
📅 Date range: 2010–12–01 to 2011–12–09
💰 Total revenue: $8,278,519.42
```

*What you should notice is that certain columns are essential for our cohort analysis. These include `InvoiceDateTime` and `CustomerID`. `TotalPrice` will be useful for revenue analysis later on.*

---

## Step 3: Run Your First Cohort Analysis 🎯

### Basic User Retention Analysis

Let's start with the most fundamental question: **"How many users come back over time?"**

```python
# Generate basic user retention cohorts
user_cohorts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M',        # Monthly cohorts
    period_duration=30        # 30-day analysis periods
)

print("🎯 Basic User Retention Cohorts:")
user_cohorts
```

*Output: Cohort table showing months as rows (2010–12–01, 2011–01–01, etc.) and periods 0–12 as columns with user counts*

#### **How to Read This Table:**
- **Rows**: Each row represents a cohort (users acquired in that month)
- **Columns**: Time periods after acquisition (0, 1, 2, 3, etc.)
- **Values**: Number of users who were active in each period

For example, if you look at the second row (for the 2011–01–01 cohort), you'll see "421" in Period 0 and "107" in Period 1. This means **421 users** made their first transaction in January 2011, and **107 of those same users** returned in the subsequent 30-day period.

### Understanding Retention Rates

Raw numbers are good, but **percentages tell a different story**:

```python
# Calculate retention rates as percentages
retention_rates = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    calculate_retention_rate=True,
    period_duration=30
)

print("📈 User Retention Rates (%):")
retention_rates.round(1)
```

*Output: Same structure but showing percentages like 100.0%, 45.0%, 32.1%, etc.*

**This is where it gets interesting!** You can now see patterns like:
- *"Our December 2010 cohort had **37% retention** after 1 month"*
- *"Retention rates are **relatively stable** after the first few months"*
- *"Some cohorts perform **consistently better** than others"*

### Revenue Cohort Analysis 💰

Now let's analyze **revenue patterns** instead of just user counts, and experiment with different time periods:

```python
# Analyze revenue patterns over time
revenue_cohorts = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    value_column='TotalPrice',    # Track revenue instead of user counts
    aggregation_function='sum',   # Sum up the revenue
    cohort_period='M',
    period_duration='W'           # Weekly revenue tracking for more detail
)

print("💰 Revenue Cohorts (Weekly Analysis):")
revenue_cohorts
```

*Output: Revenue cohort table showing dollar amounts by cohort and week*

#### **Business Insights You Can Extract:**
- Which cohorts generate the **most revenue** over time?
- Do newer cohorts **spend more or less** than historical ones?
- How quickly does **revenue decline** after acquisition?

---

## Step 4: Create Visualizations 📈

RepeatRadar makes it **easy** to create nice, interactive visualizations:

```python
from repeatradar import plot_cohort_heatmap, plot_retention_curves

# Create an interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    cohort_data=retention_rates,
    title="User Retention Rates Over Time",
    color_scale="RdYlBu_r"  # Red for low retention, blue for high
)
heatmap_fig
```

*Output: Plotly Interactive heatmap showing retention rates with color coding*

```python
# Create retention curves to compare cohorts
curves_fig = plot_retention_curves(
    retention_data=user_cohorts
)

curves_fig.show()
```

*Output: Line chart showing retention curves for different cohorts*

### Why Visualizations Help 🎨

- **Pattern Recognition**: Heatmaps make it easier to spot patterns and outliers across all cohorts
- **Interactive Exploration**: You can explore your data in real-time
- **Team Communication**: Clear charts make it easier to share insights with non-technical team members

---

## Tips for Other Analysis 🚀

### 🎯 **Segment Your Analysis**

You can experiment with creating different cohorts for different user segments:

```python
# Filter for high-value customers
high_value_customers = ecommerce_data[ecommerce_data['TotalPrice'] > 50]

high_value_retention = generate_cohort_data(
    data=high_value_customers,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    calculate_retention_rate=True
)

high_value_retention
```

*Maybe you want to focus on transactions where the price is high to understand your premium customer behavior.*

### 📅 **Experiment with Time Periods**

It's worth experimenting with different time periods for both cohorts and period durations that make sense for your business:

```python
# Monthly cohorts with quarterly analysis periods
monthly_cohorts_with_quarter_durations = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M',       
    period_duration=90  # 90-day periods might be more meaningful for your business
)

monthly_cohorts_with_quarter_durations
```

*Maybe you want to track quarterly patterns because that's more meaningful for your business cycles.*

**That's what makes RepeatRadar useful**—a few lines of code give you **good insights** into your user behavior patterns! 🎯

---

## Let's Connect! 🤝

If you found this article helpful or have questions about RepeatRadar, **I'd love to hear from you!**

### Want to Dive Deeper? 🔗

- **📚 GitHub Repository**: Visit the [full source code](https://github.com/krinya/repeatradar) for more examples and documentation
- **📧 Email**: Reach out directly at [menyhert.kristof@gmail.com](mailto:menyhert.kristof@gmail.com)

### Connect with Me:

- **🐦 Twitter**: Follow me for more data science content and RepeatRadar updates
- **💼 LinkedIn**: Let's connect and discuss data science, Python packages, or cohort analysis
- **🚀 Available for Projects**: I'm open to consulting work to help improve your business with data insights

### Try It Yourself! 

**Learning by doing works best!** Download your own transaction data and give RepeatRadar a try. I'd love to see what insights you discover.

> 💡 **Remember**: Every business has retention patterns waiting to be found. RepeatRadar just makes finding them easier! 📊

**Happy analyzing!** 🚀

---

*Did this article help you understand cohort analysis better? Give it a clap 👏 and share it with your data team!*
