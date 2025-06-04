import pandas as pd

def greet(name: str) -> str:
    """
    Generates a friendly greeting.
    
    :param name: The name of the person to greet
    :type name: str
    :raises TypeError: If the input name is not a string
    :return: A greeting string
    :rtype: str
    """
    if not isinstance(name, str):
        raise TypeError("Input 'name' must be a string")
    if not name:
        # Handle empty string case if desired
        return "Hello there!"
    return f"Hello, {name}! Nice to see you."

def create_sample_data() -> pd.DataFrame:
    """
    Generates a sample DataFrame.
    
    :return: A DataFrame with sample data
    :rtype: pd.DataFrame
    """
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }
    return pd.DataFrame(data)

def filter_for_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Filters the DataFrame for a specific name.
    
    :param df: The DataFrame to filter
    :type df: pd.DataFrame
    :param name: The name to filter by
    :type name: str
    :raises TypeError: If the input name is not a string
    :return: A filtered DataFrame
    :rtype: pd.DataFrame
    """
    if not isinstance(name, str):
        raise TypeError("Input 'name' must be a string")
    return df[df['Name'] == name]