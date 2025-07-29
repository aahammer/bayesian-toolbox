import pytest
import numpy as np


from product_analytics.variant_evaluation import (
    evaluate_product_variants,
    _evaluate_single_variant,
    _sample_posterior,
    _compute_posterior_stats,
   _make_comparative_stats_fn,
    _get_best_variant_index,
    _format_variant_results,
    VariantEvidence,
    VariantEvaluation,
    VariantStats,
    ComparativeVariantStats, 
)

@pytest.fixture
def example_variants():
    return [
        VariantEvidence(name="A", observations=1000, conversions=100),
        VariantEvidence(name="B", observations=1000, conversions=120),
    ]

def test_evaluate_product_variants(example_variants):
    
    sample_size = 1000
    
    results = evaluate_product_variants(example_variants, sample_size=sample_size)

    assert isinstance(results, list)
    assert len(results) == len(example_variants)
    
    for result in results:
        assert isinstance(result, VariantEvaluation)
        assert isinstance(result.samples, (list, tuple))
        assert isinstance(result.stats, VariantStats)
        assert isinstance(result.comparative_stats, ComparativeVariantStats)
        assert isinstance(result.is_reference, bool)
        
def test_evaluate_single_variant_copies_essential_values():
    np.random.seed(42)
    reference_samples = np.random.beta(10, 20, size=1000)
    rope = 0.05
    compute_comparative_stats = _make_comparative_stats_fn(reference_samples, rope)

    result = _evaluate_single_variant(
        name="VariantX",
        conversions=25,
        observations=100,
        alpha_prior=1,
        beta_prior=1,
        sample_size=1000,
        is_reference=False,
        compute_comparative_stats=compute_comparative_stats
    )

    assert isinstance(result, VariantEvaluation)

    # Key values preserved or transformed correctly
    assert result.name == "VariantX"
    assert len(result.samples) == 1000
    assert result.is_reference is False

    # Check that stats and comp stats are present (trust unit tests for details)
    assert isinstance(result.stats, VariantStats)
    assert isinstance(result.comparative_stats, ComparativeVariantStats)


def test_sample_posterior_output_properties():
    
    np.random.seed(42)
     
    samples = _sample_posterior(
        conversions=50,
        observations=100,
        alpha_prior=1,
        beta_prior=1,
        sample_size=5000
    )
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 5000
    assert np.all((0 <= samples) & (samples <= 1)), "Samples must be in [0, 1]"


def test_compute_posterior_stats_seeded_stable_output():
    
    np.random.seed(42)
    
    samples = np.random.beta(a=100, b=100, size=10000)
    stats = _compute_posterior_stats(samples)

    assert isinstance(stats, VariantStats)

    assert stats.mean == pytest.approx(0.49986, rel=1e-3)
    assert stats.std == pytest.approx(0.03503, rel=1e-2)
    assert stats.ci_95_lower == pytest.approx(0.4418, abs=0.005)
    assert stats.ci_95_upper == pytest.approx(0.5568, abs=0.005)


def test_compute_comparative_stats_expected_outputs():
    np.random.seed(42)

    reference = np.random.beta(10, 20, size=10000)
    variant = np.random.beta(10, 30, size=10000)

    compute_stats = _make_comparative_stats_fn(reference, rope_percentage=0.05)
    result = compute_stats(variant)

    assert isinstance(result, ComparativeVariantStats)
    assert result.prob_better == pytest.approx(0.2216, abs=0.01)
    assert result.avg_uplift == pytest.approx(-0.1947, rel=1e-2)
    assert result.avg_regret == pytest.approx(0.0960, rel=1e-2)



def test_get_best_variant_index_correct_selection():
    
    a = np.random.beta(10, 20, size=10000)
    b = np.random.beta(10, 30, size=10000)
    c = np.random.beta(10, 25, size=10000)

    index = _get_best_variant_index([a, b, c])
    assert index == 0 
    

@pytest.fixture
def dummy_evaluations():
    return [
        VariantEvaluation(
            name="A",
            samples=[0.1] * 10,
            stats=VariantStats(0.1, 0.01, 0.08, 0.12),
            comparative_stats=ComparativeVariantStats(0.5, 0.0, 0.01, 0.95),
            is_reference=True
        ),
        VariantEvaluation(
            name="B",
            samples=[0.12] * 10,
            stats=VariantStats(0.12, 0.01, 0.10, 0.14),
            comparative_stats=ComparativeVariantStats(0.75, 0.20, 0.05, 0.90),
            is_reference=False
        )
    ]

def test_format_variant_results_output(dummy_evaluations):
    table_str = _format_variant_results(dummy_evaluations)
    

    print(table_str)
    assert "name" in table_str
    assert "| A" in table_str
    assert "| B" in table_str
    assert "0.12" in table_str
    assert "20.00%" in table_str  # uplift
    assert "0.05" in table_str  # regret
