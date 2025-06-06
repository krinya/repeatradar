from typing import Tuple, Optional, Union, Literal
import pandas as pd
import numpy as np
from datetime import datetime

def _get_period_offset_days(period: str) -> int:
    """Convert period strings to approximate days for cohort analysis calculations.

    :param period: Period abbreviation string
    :type period: str
    :return: Number of days corresponding to the period
    :rtype: int
    """
    period_mapping = {
        'D': 1,
        'W': 7,
        'M': 30,
        'Q': 90,
        'Y': 365
    }
    return period_mapping.get(period, 30)

def _ensure_complete_periods(cohort_data_long: pd.DataFrame, first_period_col: str) -> pd.DataFrame:
    """Ensure all cohorts have all possible periods with 0 values where missing.

    :param cohort_data_long: Long-format cohort data DataFrame
    :type cohort_data_long: pd.DataFrame
    :param first_period_col: Column name containing the first period (cohort) information
    :type first_period_col: str
    :return: Complete cohort data with all periods filled
    :rtype: pd.DataFrame
    """
    # Get all unique cohorts and all possible period numbers
    all_cohorts = cohort_data_long[first_period_col].unique()
    max_period = cohort_data_long['period_number'].max()
    all_periods = range(max_period + 1)
    
    # Create a complete index with all combinations
    complete_index = pd.MultiIndex.from_product(
        [all_cohorts, all_periods],
        names=[first_period_col, 'period_number']
    )
    
    # Reindex to include all combinations, fill missing with 0
    cohort_data_complete = cohort_data_long.set_index([first_period_col, 'period_number']).reindex(
        complete_index, fill_value=0
    ).reset_index()
    
    return cohort_data_complete

def generate_cohort_data(
    data: pd.DataFrame, 
    date_column: str,
    user_column: str, 
    value_column: Optional[str] = None,
    aggregation_function: Literal['sum', 'mean', 'count', 'median', 'min', 'max', 'nunique'] = 'sum',
    cohort_period: Literal['D', 'W', 'M', 'Q', 'Y'] = 'M', 
    period_duration: Union[int, Literal['D', 'W', 'M', 'Q', 'Y']] = 30,
    output_format: Literal['long', 'pivot'] = 'pivot',
    calculate_retention_rate: bool = False
) -> pd.DataFrame:
    """
    Create cohort analysis data in a specified format with optimized performance.
    
    Supports both user retention analysis and transaction value analysis with retention rates.
    This function groups users into cohorts based on their acquisition period and tracks 
    their activity or value in subsequent periods.

    :param data: The input data containing transaction information
    :type data: pd.DataFrame
    :param date_column: Column name containing the datetime information
    :type date_column: str
    :param user_column: Column name containing the user/customer ID
    :type user_column: str
    :param value_column: Column name containing values to aggregate (e.g., transaction amount). If None, the function counts unique users (traditional cohort analysis), defaults to None
    :type value_column: Optional[str], optional
    :param aggregation_function: Function to apply when aggregating values. Only used when value_column is provided. 'nunique' counts the number of unique values in each group, defaults to 'sum'
    :type aggregation_function: Literal['sum', 'mean', 'count', 'median', 'min', 'max', 'nunique'], optional
    :param cohort_period: Period to group cohorts by (how to define cohort acquisition periods), defaults to 'M'
    :type cohort_period: Literal['D', 'W', 'M', 'Q', 'Y'], optional
    :param period_duration: Duration of analysis periods. Can be number of days (int) or period string. If string: 'D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly, 'Y'=yearly, defaults to 30
    :type period_duration: Union[int, Literal['D', 'W', 'M', 'Q', 'Y']], optional
    :param output_format: Format of the output data - long format or pivot table, defaults to 'pivot'
    :type output_format: Literal['long', 'pivot'], optional
    :param calculate_retention_rate: If True, calculates retention rate as percentage compared to period 0. Only applicable when value_column is None (user count analysis), defaults to False
    :type calculate_retention_rate: bool, optional
    :raises ValueError: If required columns are not found in data or invalid parameters are provided
    :raises TypeError: If date_column is not of datetime type
    :return: Either a long-format DataFrame with columns [cohort_period, period_number, metric_value] or a pivoted DataFrame in triangle format with cohorts as rows and periods as columns. If calculate_retention_rate=True, values represent percentage retention rates
    :rtype: pd.DataFrame

    Examples::

        # Basic user retention analysis
        >>> user_cohorts = generate_cohort_data(
        ...     data=df, 
        ...     date_column='purchase_date',
        ...     user_column='customer_id'
        ... )
        
        # User retention with retention rates
        >>> retention_rates = generate_cohort_data(
        ...     data=df,
        ...     date_column='purchase_date', 
        ...     user_column='customer_id',
        ...     calculate_retention_rate=True
        ... )
        
        # Revenue cohort analysis with weekly periods
        >>> revenue_cohorts = generate_cohort_data(
        ...     data=df, 
        ...     date_column='purchase_date',
        ...     user_column='customer_id',
        ...     value_column='purchase_amount',
        ...     period_duration='W',
        ...     aggregation_function='sum'
        ... )
        
        # Count unique products per cohort period
        >>> unique_products = generate_cohort_data(
        ...     data=df,
        ...     date_column='purchase_date',
        ...     user_column='customer_id',
        ...     value_column='product_id',
        ...     aggregation_function='nunique'
        ... )
    """
    # Input validation
    if date_column not in data.columns:
        raise ValueError(f"Column '{date_column}' not found in data")
    if user_column not in data.columns:
        raise ValueError(f"Column '{user_column}' not found in data")
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        raise TypeError(f"Column '{date_column}' must be of datetime type")
    if output_format not in ['long', 'pivot']:
        raise ValueError("output_format must be either 'long' or 'pivot'")
    
    # Validate value_column if provided
    if value_column is not None and value_column not in data.columns:
        raise ValueError(f"Column '{value_column}' not found in data")
    
    # Validate retention rate calculation
    if calculate_retention_rate and value_column is not None:
        raise ValueError("Retention rate calculation is only available for user count analysis (value_column=None)")
    
    # Convert period_duration to days if it's a string
    if isinstance(period_duration, str):
        period_duration_days = _get_period_offset_days(period_duration)
    else:
        period_duration_days = period_duration
    
    # Create an explicit copy of only the required columns to avoid SettingWithCopyWarning
    required_cols = [date_column, user_column]
    if value_column is not None:
        required_cols.append(value_column)
    base_data = data[required_cols].copy()
    
    # Set pandas option to suppress the warning (optional)
    pd.options.mode.chained_assignment = None
    
    try:
        # Convert datetime to period and back to timestamp in one efficient operation
        period_col = 'period_date'
        base_data[period_col] = pd.to_datetime(
            base_data[date_column].dt.to_period(cohort_period).dt.to_timestamp()
        )
        
        # Get first period for each user using transform (avoids merge operation)
        first_period_col = 'cohort_period'
        base_data[first_period_col] = base_data.groupby(user_column)[period_col].transform('min')
        
        # Calculate days since first purchase and period number in vectorized operations
        first_purchase_dates = base_data.groupby(user_column)[date_column].transform('min')
        base_data['days_since_first_purchase'] = (base_data[date_column] - first_purchase_dates).dt.days
        base_data['period_number'] = base_data['days_since_first_purchase'] // period_duration_days
        
        # Calculate the metric based on presence of value_column
        if value_column is None:
            # Count unique users per cohort and period
            cohort_data_long = base_data.groupby([first_period_col, 'period_number'])[user_column].nunique().reset_index(
                name='metric_value'
            )
        else:
            # Apply the specified aggregation function to the value column
            cohort_data_long = base_data.groupby([first_period_col, 'period_number'])[value_column].agg(aggregation_function).reset_index(
                name='metric_value'
            )
        
        # Ensure all periods are present (fill missing periods with 0)
        cohort_data_long = _ensure_complete_periods(cohort_data_long, first_period_col)
        
        # Calculate retention rates if requested
        if calculate_retention_rate:
            # Get period 0 values for each cohort
            period_0_values = cohort_data_long[cohort_data_long['period_number'] == 0].set_index(first_period_col)['metric_value']
            
            # Calculate retention rate as percentage for each row
            def calculate_retention_rate(row):
                cohort_date = row[first_period_col]
                period_0_value = period_0_values.get(cohort_date, 1)  # Avoid division by zero
                if period_0_value == 0:
                    return 0.0  # Return 0% if no users in period 0
                return round((row['metric_value'] / period_0_value * 100), 2)
            
            cohort_data_long['metric_value'] = cohort_data_long.apply(calculate_retention_rate, axis=1)
        
        if output_format == 'pivot':
            # Create the pivot table for the triangle view
            cohort_data_pivot = cohort_data_long.pivot(
                index=first_period_col,
                columns='period_number',
                values='metric_value'
            ).fillna(0)
            
            # Convert to appropriate data type
            if calculate_retention_rate:
                # Keep as float for retention rates
                cohort_data_pivot = cohort_data_pivot.astype(np.float32)
            elif value_column is None or (aggregation_function in ['count', 'nunique'] and not pd.api.types.is_float_dtype(data[value_column])):
                # Convert to int for user counts
                cohort_data_pivot = cohort_data_pivot.astype(np.int32)
            
            return cohort_data_pivot
        else:
            return cohort_data_long
        
    finally:
        # Reset pandas option to default
        pd.options.mode.chained_assignment = 'warn'