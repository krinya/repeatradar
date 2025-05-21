# RepeatRadar

[![PyPI version](https://badge.fury.io/py/repeatradar.svg)](https://badge.fury.io/py/repeatradar) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A Python package for calculating and visualizing cohort retention and other cohort-based metrics. RepeatRadar helps you understand user behavior and trends over time by grouping users into cohorts based on their acquisition period and tracking their activity or value in subsequent periods.

## Useful links
- GitHub: https://github.com/krinya/repeatradar
- PyPI: https://pypi.org/project/repeatradar/
# Installation
```bash
pip install repeatradar
```

# Example of Use

The primary function in this package is `generate_cohort_data`.

```python
from repeatradar import generate_cohort_data
import pandas as pd

ecommerce_data = pd.read_pickle("https://github.com/krinya/repeatradar/raw/refs/heads/main/examples/data/ecommerce_data_1.pkl")

generated_data_pivot_user = generate_cohort_data(
    data=ecommerce_data,  # Your pandas DataFrame
    datetime_column_name='InvoiceDateTime',  # Column with transaction/event dates
    user_column_name='CustomerID',  # Column with user identifiers
    base_period='M',  # Cohorts grouped by Month
    period_duration=30,  # Each period is 30 days long
    output_format='pivot'  # Output as a pivot table
)

generated_data_pivot_user
```
![example_1](img/example_1.png)
# Roadmap

More features are coming soon!

# Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

Please make sure to update tests as appropriate.