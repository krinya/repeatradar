"""
Demonstration of the improved generate_cohort_data function.

This script shows the new features:
1. Better parameter names (date_column, user_column, value_column)
2. Period duration support for both int and string ('D', 'W', 'M', 'Q', 'Y')
3. Missing period handling (all periods included with 0 values)
4. Retention rate calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from repeatradar.cohort_generator import generate_cohort_data

def create_sample_data():
    """Create sample e-commerce data for demonstration."""
    rng = np.random.default_rng(42)
    
    # Generate users
    user_ids = [f"user_{i}" for i in range(20)]
    
    # Generate transactions over 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    transactions = []
    for user_id in user_ids:
        # Each user has first transaction at a random time
        first_transaction = start_date + timedelta(
            days=rng.integers(0, 90)  # Users join in first 3 months
        )
        
        # Generate 1-5 transactions per user
        num_transactions = rng.integers(1, 6)
        for i in range(num_transactions):
            transaction_date = first_transaction + timedelta(
                days=rng.integers(0, 90)  # Spread transactions over 3 months
            )
            
            transactions.append({
                'customer_id': user_id,
                'purchase_date': transaction_date,
                'purchase_amount': round(rng.uniform(10, 200), 2),
                'product_id': f"product_{rng.integers(1, 11)}"
            })
    
    df = pd.DataFrame(transactions)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    return df.sort_values('purchase_date').reset_index(drop=True)

def demonstrate_improvements():
    """Demonstrate all the improvements made to the function."""
    print("=== Improved generate_cohort_data Function Demo ===\n")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} transactions from {df['customer_id'].nunique()} users")
    print(f"Date range: {df['purchase_date'].min().date()} to {df['purchase_date'].max().date()}\n")
    
    # 1. Basic user cohort analysis with new parameter names
    print("1. Basic user cohort analysis (improved parameter names):")
    basic_cohorts = generate_cohort_data(
        data=df,
        date_column='purchase_date',        # Previously: datetime_column_name
        user_column='customer_id',          # Previously: user_column_name
        cohort_period='M'                   # Previously: base_period
    )
    print(f"Shape: {basic_cohorts.shape}")
    print(basic_cohorts.head())
    print()
    
    # 2. Period duration as string (new feature)
    print("2. Weekly analysis periods (period_duration='W'):")
    weekly_cohorts = generate_cohort_data(
        data=df,
        date_column='purchase_date',
        user_column='customer_id',
        period_duration='W'  # New: can use 'D', 'W', 'M', 'Q', 'Y'
    )
    print(f"Shape: {weekly_cohorts.shape}")
    print(weekly_cohorts.head())
    print()
    
    # 3. Retention rate calculation (new feature)
    print("3. Retention rate analysis (new feature):")
    retention_rates = generate_cohort_data(
        data=df,
        date_column='purchase_date',
        user_column='customer_id',
        calculate_retention_rate=True  # New feature!
    )
    print(f"Shape: {retention_rates.shape}")
    print("Retention rates (% of users returning compared to period 0):")
    print(retention_rates.head())
    print()
    
    # 4. Missing periods handling (improved)
    print("4. Long format showing missing periods filled with 0:")
    long_format = generate_cohort_data(
        data=df,
        date_column='purchase_date',
        user_column='customer_id',
        output_format='long'
    )
    print(f"Total rows: {len(long_format)}")
    print("Sample showing periods with 0 values:")
    print(long_format[long_format['metric_value'] == 0].head())
    print()
    
    # 5. Revenue cohort analysis with weekly periods
    print("5. Revenue analysis with weekly periods:")
    revenue_cohorts = generate_cohort_data(
        data=df,
        date_column='purchase_date',
        user_column='customer_id',
        value_column='purchase_amount',     # Previously: value_column_name
        aggregation_function='sum',
        period_duration='W'
    )
    print(f"Shape: {revenue_cohorts.shape}")
    print(revenue_cohorts.head())
    print()
    
    # 6. Product variety analysis
    print("6. Unique products per cohort period:")
    product_variety = generate_cohort_data(
        data=df,
        date_column='purchase_date',
        user_column='customer_id',
        value_column='product_id',
        aggregation_function='nunique',
        period_duration='M'
    )
    print(f"Shape: {product_variety.shape}")
    print(product_variety.head())

if __name__ == "__main__":
    demonstrate_improvements()
