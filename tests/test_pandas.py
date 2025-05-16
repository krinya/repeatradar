import pytest
import pandas as pd
from repeatradar import create_sample_data, filter_for_name

def test_create_sample_data():
    """Test that create_sample_data returns a DataFrame with expected structure and values."""
    df = create_sample_data()
    
    # Check DataFrame structure and values
    expected_columns = ['Name', 'Age', 'City']
    expected_data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }
    
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns
    assert len(df) == 3
    for column, values in expected_data.items():
        assert df[column].tolist() == values

def test_filter_for_name():
    """Test that filter_for_name correctly filters a DataFrame by name."""
    df = create_sample_data()
    test_cases = [
        ("Alice", 1, {'Name': 'Alice', 'Age': 25, 'City': 'New York'}),
        ("Bob", 1, {'Name': 'Bob'}),
        ("David", 0, None)
    ]
    
    for name, expected_len, expected_values in test_cases:
        filtered = filter_for_name(df, name)
        assert len(filtered) == expected_len
        if expected_values:
            for key, value in expected_values.items():
                assert filtered.iloc[0][key] == value

def test_filter_for_name_type_error():
    """Test that filter_for_name raises TypeError with non-string input."""
    df = create_sample_data()
    invalid_inputs = [123, None, ["Alice"]]
    
    for invalid_input in invalid_inputs:
        with pytest.raises(TypeError):
            filter_for_name(df, invalid_input)  # type: ignore