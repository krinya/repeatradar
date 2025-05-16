import pytest
import pandas as pd
from repeatradar import create_sample_data, filter_for_name

class TestDataFunctions:
    def test_create_sample_data(self):
        """Test that create_sample_data returns a DataFrame with expected structure and values."""
        df = create_sample_data()
        
        # Check if result is a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Check if DataFrame has the expected columns
        expected_columns = ['Name', 'Age', 'City']
        assert list(df.columns) == expected_columns
        
        # Check if DataFrame has the expected number of rows
        assert len(df) == 3
        
        # Check specific values
        assert df['Name'].tolist() == ['Alice', 'Bob', 'Charlie']
        assert df['Age'].tolist() == [25, 30, 35]
        assert df['City'].tolist() == ['New York', 'Los Angeles', 'Chicago']

    def test_filter_for_name(self):
        """Test that filter_for_name correctly filters a DataFrame by name."""
        # Create a test DataFrame
        df = create_sample_data()
        
        # Test filtering for Alice
        filtered = filter_for_name(df, "Alice")
        assert len(filtered) == 1
        assert filtered.iloc[0]['Name'] == 'Alice'
        assert filtered.iloc[0]['Age'] == 25
        assert filtered.iloc[0]['City'] == 'New York'
        
        # Test filtering for Bob
        filtered = filter_for_name(df, "Bob")
        assert len(filtered) == 1
        assert filtered.iloc[0]['Name'] == 'Bob'
        
        # Test filtering for a name that doesn't exist
        filtered = filter_for_name(df, "David")
        assert len(filtered) == 0
        
    def test_filter_for_name_type_error(self):
        """Test that filter_for_name raises TypeError with non-string input."""
        df = create_sample_data()
        
        # Test with non-string inputs
        with pytest.raises(TypeError):
            filter_for_name(df, 123)  # type: ignore
        
        with pytest.raises(TypeError):
            filter_for_name(df, None)  # type: ignore
        
        with pytest.raises(TypeError):
            filter_for_name(df, ["Alice"])  # type: ignore