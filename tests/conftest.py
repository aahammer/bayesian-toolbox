# conftest.py
import pytest
import numpy as np

@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
