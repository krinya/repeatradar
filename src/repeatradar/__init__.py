# repeatradar package initialization
# Make the version easily accessible (matches pyproject.toml)

__version__ = "0.6.1"

# Import core functionality
from .cohort_generator import generate_cohort_data

# Import visualization functions
from .visualization_generator import (
    plot_cohort_heatmap,
    plot_retention_curves
)

# Explicitly define the public API
__all__ = [
    'generate_cohort_data',
    'plot_cohort_heatmap', 
    'plot_retention_curves'
] 