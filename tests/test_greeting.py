import pytest
from repeatradar import greet # Import the function to test

def test_greet_with_name():
    """Test greeting with a valid name."""
    assert greet("Alice") == "Hello, Alice! Nice to see you."
    assert greet("Bob") == "Hello, Bob! Nice to see you."

def test_greet_empty_name():
    """Test greeting with an empty string."""
    assert greet("") == "Hello there!"

def test_greet_type_error():
    """Test that non-string input raises TypeError."""
    with pytest.raises(TypeError):
        greet(123) # type: ignore

    with pytest.raises(TypeError):
        greet(None) # type: ignore

    with pytest.raises(TypeError):
        greet(["List", "Not", "Allowed"]) # type: ignore