"""Test Configuration."""

import pyarrow


def pytest_report_header(config, start_path):
    """Add the PyArrow version to the pytest report header."""
    return [
        f"PyArrow: {pyarrow.__version__}",
    ]
