# repeatradar package initialization
# Make the version easily accessible (matches pyproject.toml)

__version__ = "0.4.0"

# Import core functionality
from .sample import greet, create_sample_data, filter_for_name
from .cohort_generator import generate_cohort_data

# Import visualization functions
from .visualization_generator import (
    plot_cohort_heatmap,
    plot_retention_curves,
    plot_cohort_comparison,
    plot_period_comparison,
    plot_cohort_summary_stats,
    create_cohort_dashboard
)

# Explicitly define the public API
__all__ = [
    'greet', 'create_sample_data', 'filter_for_name', 'generate_cohort_data',
    'plot_cohort_heatmap', 'plot_retention_curves', 'plot_cohort_comparison',
    'plot_period_comparison', 'plot_cohort_summary_stats', 'create_cohort_dashboard'
] 