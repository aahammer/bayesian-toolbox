import pytest

from product_analytics.power_analysis import _binary_search

import numpy as np
import pytest
from product_analytics.power_analysis import _estimate_success_prob,_estimate_minimum_sample_size


def test_detectable_effect_detected():
    # Small detectable effect, expect a non-None result
    result = _estimate_minimum_sample_size(
        baseline_rate=0.05,
        minimum_detectable_effect=0.01,
    )
    assert result == 3906

def test_impossible_effect_returns_none():
    # Require unrealistic confidence for tiny effect, should return None
    result = _estimate_minimum_sample_size(
        baseline_rate=0.05,
        minimum_detectable_effect=0.001,
        required_probability=0.9999,
        max_sample_size=1000
    )
    assert result is None

def test_high_uplift_needs_smaller_sample():
    small = _estimate_minimum_sample_size(
        baseline_rate=0.05,
        minimum_detectable_effect=0.05,
    )
    large = _estimate_minimum_sample_size(
        baseline_rate=0.05,
        minimum_detectable_effect=0.01,
    )
    assert small < large

def test_sample_size_respects_max_limit():
    result = _estimate_minimum_sample_size(
        baseline_rate=0.05,
        minimum_detectable_effect=0.01,
        max_sample_size=500
    )
    
    assert result is None


def test_binary_search_simple_true():
    # condition becomes True from x >= 5
    cond = lambda x: x >= 5
    assert _binary_search(1, 10, cond) == 5

def test_binary_search_immediate_true():
    # always True
    assert _binary_search(10, 20, lambda x: True) == 10

def test_binary_search_all_false():
    # never True
    assert _binary_search(1, 5, lambda x: False) is None

def test_binary_search_exact_boundary():
    # cond(x) is True only at upper bound
    cond = lambda x: x == 100
    assert _binary_search(1, 100, cond, min_step=0) == 100

def test_binary_search_min_step_stops_early():
    calls = []

    def cond(x):
        calls.append(x)
        return x >= 50

    result = _binary_search(1, 100, cond, min_step=10)

    # Expect exactly these midpoints checked
    assert calls == [50, 25, 37, 43]

    # The best satisfying value found
    assert result == 50



@pytest.mark.parametrize("lo, hi, true_at, expected", [
    (1, 100, 1, 1),
    (1, 100, 100, 100),
    (10, 10, 10, 10),
    (0, 0, 1, None),  # impossible
])
def test_binary_search_various(lo, hi, true_at, expected):
    cond = lambda x: x >= true_at
    assert _binary_search(lo, hi, cond, min_step=0) == expected
    
    
def test_estimate_success_prob_with_clear_uplift():
    prob = _estimate_success_prob(
        baseline_rate=0.1,
        uplift=0.1,
        sample_size=1000,
        num_samples=5000
    )
    print(prob)
    assert prob == 1.0

def test_estimate_success_prob_no_uplift():
    prob = _estimate_success_prob(
        baseline_rate=0.1,
        uplift=0.0,
        sample_size=1000,
        num_samples=5000
    )
    assert prob == pytest.approx(0.4898, abs=0.005)

def test_estimate_success_prob_negative_uplift():
    prob = _estimate_success_prob(
        baseline_rate=0.1,
        uplift=-0.05,
        sample_size=1000,
        num_samples=5000
    )
    assert prob == 0.0

