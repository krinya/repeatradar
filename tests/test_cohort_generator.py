import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from repeatradar.cohort_generator import generate_cohort_data

@pytest.fixture
def sample_ecommerce_data() -> pd.DataFrame:
    """Generates a sample DataFrame emulating e-commerce transaction data."""
    rng = np.random.default_rng(42)  # For reproducibility
    num_users = 50
    num_transactions = 500

    user_ids = [f"user_{i}" for i in range(num_users)]
    
    # Generate somewhat realistic transaction dates over the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    transaction_dates = [start_date + timedelta(seconds=int(rng.integers((end_date - start_date).total_seconds()))) for _ in range(num_transactions)]
    transaction_dates.sort()

    data = pd.DataFrame({
        'transaction_id': range(num_transactions),
        'customer_id': rng.choice(user_ids, size=num_transactions, replace=True),
        'purchase_date': pd.to_datetime(transaction_dates),
        'purchase_amount': rng.uniform(5.0, 200.0, size=num_transactions).round(2),
        'product_id': [f"product_{rng.integers(1, 21)}" for _ in range(num_transactions)] # 20 unique products
    })
    return data

def test_generate_cohort_data_runs_without_error(sample_ecommerce_data: pd.DataFrame):
    """
    Tests that generate_cohort_data runs without raising an error with basic parameters
    and returns a pandas DataFrame.
    """
    try:
        result_df = generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='customer_id'
        )
        assert isinstance(result_df, pd.DataFrame), "Function should return a pandas DataFrame."
        assert not result_df.empty, "Resulting DataFrame should not be empty."

        # Test with value column and different aggregation
        result_value_df = generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='customer_id',
            value_column_name='purchase_amount',
            aggregation_function='sum'
        )
        assert isinstance(result_value_df, pd.DataFrame), "Function should return a pandas DataFrame for value aggregation."
        assert not result_value_df.empty, "Resulting DataFrame for value aggregation should not be empty."

        # Test with 'nunique' aggregation
        result_nunique_df = generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='customer_id',
            value_column_name='product_id',
            aggregation_function='nunique'
        )
        assert isinstance(result_nunique_df, pd.DataFrame), "Function should return a pandas DataFrame for nunique aggregation."
        assert not result_nunique_df.empty, "Resulting DataFrame for nunique aggregation should not be empty."

        # Test long format
        result_long_df = generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='customer_id',
            output_format='long'
        )
        assert isinstance(result_long_df, pd.DataFrame), "Function should return a pandas DataFrame for long format."
        assert not result_long_df.empty, "Resulting DataFrame for long format should not be empty."
        assert 'first_period' in result_long_df.columns
        assert 'period_number' in result_long_df.columns
        assert 'metric_value' in result_long_df.columns


    except Exception as e:
        pytest.fail(f"generate_cohort_data raised an exception: {e}")

def test_generate_cohort_data_invalid_inputs(sample_ecommerce_data: pd.DataFrame):
    """Tests that generate_cohort_data raises appropriate errors for invalid inputs."""
    with pytest.raises(ValueError, match="Column 'non_existent_date_col' not found in data"):
        generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='non_existent_date_col',
            user_column_name='customer_id'
        )

    with pytest.raises(ValueError, match="Column 'non_existent_user_col' not found in data"):
        generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='non_existent_user_col'
        )
    
    data_copy = sample_ecommerce_data.copy()
    data_copy['purchase_date_str'] = data_copy['purchase_date'].astype(str)
    with pytest.raises(TypeError, match="Column 'purchase_date_str' must be of datetime type"):
        generate_cohort_data(
            data=data_copy,
            datetime_column_name='purchase_date_str',
            user_column_name='customer_id'
        )

    with pytest.raises(ValueError, match="output_format must be either 'long' or 'pivot'"):
        generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='customer_id',
            output_format='invalid_format' # type: ignore
        )

    with pytest.raises(ValueError, match="Column 'non_existent_value_col' not found in data"):
        generate_cohort_data(
            data=sample_ecommerce_data,
            datetime_column_name='purchase_date',
            user_column_name='customer_id',
            value_column_name='non_existent_value_col'
        )
