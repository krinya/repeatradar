def greet(name: str) -> str:
    """
    Generates a friendly greeting.
    Args:
        name: The name of the person to greet.
    Returns:
        A greeting string.
    Raises:
        TypeError: If the input name is not a string.
    """
    if not isinstance(name, str):
        raise TypeError("Input 'name' must be a string")
    if not name:
        # Handle empty string case if desired
        return "Hello there!"
    return f"Hello, {name}! Nice to see you."