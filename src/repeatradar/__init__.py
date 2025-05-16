# my_simple_package/my_simple_package/__init__.py
# Make the version easily accessible (matches pyproject.toml)
__version__ = "0.1.0"

# To keep __init__.py clean for larger projects, you could put 'greet'
# in a separate file like 'my_simple_package/aux.py' and import it here:
from .aux import greet

__all__ = ['greet'] # Explicitly define the public API