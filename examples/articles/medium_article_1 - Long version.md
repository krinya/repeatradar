Beyond Transactions: Unlock Customer Purchasing Patterns with RepeatRadar package in Python
As a daily Python user, I've always been fascinated by the idea of creating my own package and making it available on PyPI. But instead of building just another "hello world" package, I wanted to create something genuinely useful for the data community. That's how RepeatRadar was born!
Having worked with customer data for years, I noticed that cohort retention analysis - one of the most powerful techniques for understanding user behavior - was often complicated and time-consuming to implement. Existing solutions were either too complex for quick analysis or lacked the flexibility needed for different business scenarios. 
So I thought: "Why not build a Python package that makes cohort analysis as simple as a few lines of code?"
This article has two main goals:
Give you a solid introduction to cohort retention analysis - what it is, why it matters, and how it can transform your business insights
Show you how to use RepeatRadar - the Python package I developed to make cohort analysis accessible, powerful, and fun to use

Whether you're a data analyst looking to add cohort analysis to your toolkit, or a business stakeholder wanting to understand customer behavior patterns, this guide will get you up and running with actionable insights in no time!
What is cohort retention analysis?
Definition
Cohort analysis (or cohort retention analysis) is a method to understand user behavior over time by grouping users into "cohorts" based on when they first engaged with your product or service.
Think of it like tracking different groups of Netflix subscribers who joined during different months - you want to see how many people from each monthly signup group are still actively watching shows 3, 6, or 12 months later. Or picture tracking different groups of first-time shoppers on your online store - those who discovered you during Black Friday sales, through a Google ad campaign, or via a social media influencer - and seeing which acquisition channels bring customers who become loyal repeat buyers.
In business terms, a cohort is a group of users who first engaged with your product during the same time period - like all customers who made their first purchase in January 2024. For example:
All users who made their first purchase in January 2024
All customers who signed up for your service in Q1 2024
All users who downloaded your app during a specific marketing campaign

The "retention" part tracks how many of these users come back and engage again in subsequent periods.
How and why is it useful for your business?
Cohort retention analysis answers critical business questions that simple metrics can't:
🔍 What it reveals:
User Loyalty: How loyal are your users over time?
Product-Market Fit: Are users finding long-term value in what you offer?
Seasonal Patterns: Do certain times of year produce more loyal customers?
Feature Impact: How do product changes affect user retention?
Marketing Effectiveness: Which acquisition channels bring the most loyal users?

💡 Why it's better than basic metrics:
Instead of just knowing "we had 1000 active users last month," cohort analysis tells you stories like:
"Users who joined in January have 40% retention after 3 months"
"Our Q4 cohort is performing 20% better than Q3 - the holiday campaign worked!"
"Revenue from our March cohort is actually increasing over time - great sign!"

🎯 Business impact:
Predict Revenue: Understand the lifetime value of different user segments
Optimize Marketing: Invest more in channels that bring sticky users
Product Development: Identify which features drive long-term engagement
Resource Planning: Forecast future user base and revenue streams

How to do cohort retention analysis with RepeatRadar?
Installation
I developed a package called RepeatRadar that allows you to do cohort retention analysis in Python. It is available on PyPI and can be installed using pip.
pip install repeatradar
That's it! RepeatRadar will automatically install its dependencies (pandas, numpy, and plotly for visualizations).
Get your transaction data
You need to have your transaction data in a pandas DataFrame. The data should have at least two columns, but if you have more like price, profit, etc., you can do more advanced analysis.
The minimum required columns are:
a date column that contains the date of the transaction, shipment, or any other date that you want to use as a reference point
a user_id column that contains the user ID of the user who made the transaction. This is needed; otherwise, you will not be able to do cohort retention analysis.

Extra columns for deeper insights:
value column (like purchase_amount, revenue, etc.) - this unlocks revenue cohort analysis
product_id, category, channel - for more detailed segmentation

Your data might look something like this:
user_id purchase_date  purchase_amount
0  user_1    2024-01-15             45.99
1  user_2    2024-01-15             23.50
2  user_1    2024-02-10             67.25
3  user_3    2024-01-16             15.00
Example usage with code and output
Now let's dive into some practical examples! I'll walk you through the most common use cases you'll encounter.
Install and load RepeatRadar
# Install if you haven't already
# !pip install repeatradar
# Import the main function
from repeatradar import generate_cohort_data
import pandas as pd
import numpy as np
# Check that everything works
import repeatradar
print(f"RepeatRadar version: {repeatradar.__version__}")
Running this will display your installed version. For optimal compatibility, please confirm you are using the latest version (or RepeatRadar 1.x.x, the version used in this article).
Load data
For this example, I'll use a real e-commerce dataset. You can follow along with your own data by replacing the data loading part.
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
📊 Dataset Overview:
401,604 transactions from 4,372 customers
📅 Date range: 2010–12–01 to 2011–12–09
💰 Total revenue: $8,278,519.42
sample of the ecommerce_dataWhat you should notice is that certain columns are essential for our cohort analysis. These include, for example, InvoiceDateTime and CustomerID. TotalPrice might also be useful later on.
Run cohort retention analysis
Basic User Retention Analysis
Let's start with the most fundamental question: "How many users come back over time?"
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
Cohort table showing months as rows (2010–12–01, 2011–01–01, etc.) and periods 0–12 as columns with user countsHow to read this table:
Rows: Each row represents a cohort (users acquired in that month)
Columns: Time periods after acquisition (0, 1, 2, 3, etc.)
Values: Number of users who were active in each period

For example, if you look at the second row (for the 2011–01–01 cohort), you'll see "421" in Period 0 and "107" in Period 1. This means 421 users made their first transaction in January 2011, and 107 of those same users returned in the subsequent 30-day period.
Retention Rates
Raw numbers are good, but percentages tells a different aspect of the story:
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
Same structure but showing percentages like 100.0%, 45.0%, 32.1%, etc.This is where it gets interesting! You can now see patterns like:
"Our December 2010 cohort had 37% retention after 1 month"
"Retention are relatevely stable after few months"
"Some cohorts perform consistently better than others"

Revenue Cohort Analysis
Now let's do the same but not with users but with the revenue. And maybe use a differnet period duration. We can do that too.
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
Revenue cohort table showing dollar amounts by cohort and weekBusiness insights you can get:
Which cohorts generate the most revenue over time?
Do newer cohorts spend more or less than historical ones?
How quickly does revenue decline after acquisition?

Run visualization
RepeatRadar makes it super easy to create beautiful, interactive visualizations:
from repeatradar import plot_cohort_heatmap, plot_retention_curves

# Create an interactive heatmap
heatmap_fig = plot_cohort_heatmap(
    cohort_data=retention_rates,
    title="User Retention Rates Over Time",
    color_scale="RdYlBu_r"  # Red for low retention, blue for high
)
heatmap_fig
Plotly Interactive heatmap showing retention raw numbers with color codingfrom repeatradar import plot_cohort_heatmap, plot_retention_curves

# Create an interactive heatmap
heatmap_fig_retention = plot_cohort_heatmap(
    cohort_data=retention_rates,
    title="User Retention Rates Over Time",
    color_scale="RdYlBu_r"  # Red for low retention, blue for high
)
heatmap_fig_retention
Plotly Interactive heatmap showing retention rates with color coding# Create retention curves to compare cohorts
curves_fig = plot_retention_curves(
    retention_data=user_cohorts
)

curves_fig.show()
Line chart showing retention curves for different cohortsWhy visualizations matter:
Heatmaps and Retention curves make it easy to spot patterns and outliers across all cohorts
Interactive features let you explore your data dynamically

Quick Tips for Better Analysis
🎯 You can just exeriment with creating different cohorts for different users (that have different features):
Maybe you just want to keep transcation where the price is high.
# Filter for high-value customers
high_value_customers = ecommerce_data[ecommerce_data['TotalPrice'] > 50]

high_value_retention = generate_cohort_data(
    data=high_value_customers,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    calculate_retention_rate=True
)

high_value_retention
📅 Try different time periods:
It is worth to experiment with different time periods for both the cohorts and the period durations that is applicable for a diven business. Maybe you want to track quarters becase that is more meaningful for the business. You can do that too.
# Weekly cohorts for more granular analysis
monthly_cohorts_with_quarter_durations = generate_cohort_data(
    data=ecommerce_data,
    date_column='InvoiceDateTime',
    user_column='CustomerID',
    cohort_period='M',       
    period_duration=90 
)

monthly_cohorts_with_quarter_durations
That's the power of RepeatRadar - a few lines of code give you different insights into your user behavior patterns!
Contact me
If you found this article helpful or have questions about RepeatRadar, I'd love to hear from you!
Want to dive deeper?
🔗 Visit the GitHub repository to see the full source code. Here you can find more examples and documentation.
Or just send me an email: menyhert.kristof@gmail.com

Connect with me:
🐦 Follow me for more data science content and updates about RepeatRadar
💼 Let's connect and discuss data science, Python packages, or cohort analysis
I am open for projects so you can hire me to elevate your business

Try it yourself:
The best way to learn is by doing! Download your own transaction data and give RepeatRadar a spin. I'd love to see what insights you discover.
Remember: every business has retention patterns waiting to be uncovered. RepeatRadar just makes finding them a lot easier! 📊
Happy analyzing! 🚀