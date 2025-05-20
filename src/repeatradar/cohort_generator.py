from typing import Tuple, Optional, Union, Literal
import pandas as pd
import numpy as np
from datetime import datetime

def generate_cohort_data(
    data: pd.DataFrame, 
    datetime_column_name: str,
    user_column_name: str, 
    value_column_name: Optional[str] = None,
    aggregation_function: Literal['sum', 'mean', 'count', 'median', 'min', 'max', 'nunique'] = 'sum',
    base_period: Literal['D', 'W', 'M', 'Q', 'Y'] = 'M', 
    period_duration: int = 30,
    output_format: Literal['long', 'pivot'] = 'pivot'
) -> pd.DataFrame:
    """
    Creates cohort analysis data in a specified format with optimized performance.
    Supports both user retention analysis and transaction value analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input data containing transaction information
    datetime_column_name : str
        Column name containing the datetime information
    user_column_name : str
        Column name containing the user/customer ID
    value_column_name : Optional[str], default None
        Column name containing values to aggregate (e.g., transaction amount)
        If None, the function counts unique users (traditional cohort analysis)
    aggregation_function : {'sum', 'mean', 'count', 'median', 'min', 'max', 'nunique'}, default 'sum'
        Function to apply when aggregating values
        Only used when value_column_name is provided
        'nunique' counts the number of unique values in each group
    base_period : {'D', 'W', 'M', 'Q', 'Y'}, default 'M'
        Period to group dates by
    period_duration : int, default 30
        Number of days to consider as one period
    output_format : {'long', 'pivot'}, default 'pivot'
        Format of the output data - long format or pivot table
        
    Returns:
    --------
    pd.DataFrame
        Either a long-format DataFrame with columns [first_period, period_number, metric_value]
        or a pivoted DataFrame in triangle format with cohorts as rows and periods as columns
        
    Examples:
    ---------
    # User retention analysis
    >>> user_cohorts = generate_cohort_data(
    ...     data=df, 
    ...     datetime_column_name='purchase_date',
    ...     user_column_name='customer_id'
    ... )
    
    # Transaction value analysis
    >>> revenue_cohorts = generate_cohort_data(
    ...     data=df, 
    ...     datetime_column_name='purchase_date',
    ...     user_column_name='customer_id',
    ...     value_column_name='purchase_amount',
    ...     aggregation_function='sum'
    ... )
    
    # Count unique products per cohort period
    >>> unique_products = generate_cohort_data(
    ...     data=df,
    ...     datetime_column_name='purchase_date',
    ...     user_column_name='customer_id',
    ...     value_column_name='product_id',
    ...     aggregation_function='nunique'
    ... )
    """
    # Input validation
    if datetime_column_name not in data.columns:
        raise ValueError(f"Column '{datetime_column_name}' not found in data")
    if user_column_name not in data.columns:
        raise ValueError(f"Column '{user_column_name}' not found in data")
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_column_name]):
        raise TypeError(f"Column '{datetime_column_name}' must be of datetime type")
    if output_format not in ['long', 'pivot']:
        raise ValueError("output_format must be either 'long' or 'pivot'")
    
    # Validate value_column if provided
    if value_column_name is not None and value_column_name not in data.columns:
        raise ValueError(f"Column '{value_column_name}' not found in data")
    
    # Create an explicit copy of only the required columns to avoid SettingWithCopyWarning
    required_cols = [datetime_column_name, user_column_name]
    if value_column_name is not None:
        required_cols.append(value_column_name)
    base_data = data[required_cols].copy()
    
    # Set pandas option to suppress the warning (optional)
    pd.options.mode.chained_assignment = None
    
    try:
        # Convert datetime to period and back to timestamp in one efficient operation
        period_col = 'period_date'
        base_data[period_col] = pd.to_datetime(
            base_data[datetime_column_name].dt.to_period(base_period).dt.to_timestamp()
        )
        
        # Get first period for each user using transform (avoids merge operation)
        first_period_col = 'first_period'
        base_data[first_period_col] = base_data.groupby(user_column_name)[period_col].transform('min')
        
        # Calculate days since first purchase and period number in vectorized operations
        first_purchase_dates = base_data.groupby(user_column_name)[datetime_column_name].transform('min')
        base_data['days_since_first_purchase'] = (base_data[datetime_column_name] - first_purchase_dates).dt.days
        base_data['period_number'] = base_data['days_since_first_purchase'] // period_duration
        
        # Calculate the metric based on presence of value_column_name
        if value_column_name is None:
            # Count unique users per cohort and period
            cohort_data_long = base_data.groupby([first_period_col, 'period_number'])[user_column_name].nunique().reset_index(
                name='metric_value'
            )
        else:
            # Apply the specified aggregation function to the value column
            cohort_data_long = base_data.groupby([first_period_col, 'period_number'])[value_column_name].agg(aggregation_function).reset_index(
                name='metric_value'
            )
        
        if output_format == 'pivot':
            # Create the pivot table for the triangle view
            cohort_data_pivot = cohort_data_long.pivot(
                index=first_period_col,
                columns='period_number',
                values='metric_value'
            ).fillna(0)
            
            # Convert to int for user counts, leave as float for value metrics
            if value_column_name is None or (aggregation_function in ['count', 'nunique'] and not pd.api.types.is_float_dtype(data[value_column_name])):
                cohort_data_pivot = cohort_data_pivot.astype(np.int32)
            
            return cohort_data_pivot
        else:
            return cohort_data_long
        
    finally:
        # Reset pandas option to default
        pd.options.mode.chained_assignment = 'warn'