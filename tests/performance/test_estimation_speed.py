import pytest
from product_analytics.power_analysis import power_analysis

def test_estimate_min_sample_size_perf(benchmark):
    
    result = benchmark(
        power_analysis,
        baseline_rate=0.05,
        minimum_detectable_effect=0.01
    )

