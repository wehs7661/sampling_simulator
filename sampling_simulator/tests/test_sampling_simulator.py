"""
Unit and regression test for the sampling_simulator package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import sampling_simulator


def test_sampling_simulator_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "sampling_simulator" in sys.modules
