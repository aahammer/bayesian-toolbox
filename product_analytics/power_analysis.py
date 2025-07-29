import numpy as np
from typing import Optional, Callable
from .variant_evaluation import _sample_posterior  # assumed correct


def _estimate_success_prob(
    baseline_rate: float,
    uplift: float,
    sample_size: int,
    num_samples: int,
    alpha_priors: tuple[int, int] = (1, 1),
    beta_priors: tuple[int, int] = (1, 1)
) -> float:
    """
    Simulates posterior draws for control and variant groups and returns the
    probability that variant B outperforms variant A.
    """

    successes_a = int(sample_size * baseline_rate)
    successes_b = int(sample_size * (baseline_rate + uplift))

    samples_a = _sample_posterior(successes_a, sample_size, alpha_priors[0], beta_priors[0], num_samples)
    samples_b = _sample_posterior(successes_b, sample_size, alpha_priors[1], beta_priors[1], num_samples)

    return (samples_b > samples_a).mean()


def binary_search(
    lo: int,
    hi: int,
    condition: Callable[[int], bool],
    min_step: int = 1
) -> Optional[int]:
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if condition(mid):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
        if hi - lo < min_step:
            return best
    return best


def estimate_minimum_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    required_probability: float = 0.975,
    max_sample_size: int = 100_000,
    num_samples: int = 10_000,
    min_step: int = 100,
    alpha_priors: tuple[int, int] = (1, 1),
    beta_priors: tuple[int, int] = (1, 1)
) -> Optional[int]:
    """
    Estimate the minimum sample size needed to detect a given minimum detectable effect (MDE)
    with a specified probability, using Bayesian simulation and binary search.
    """

    def simulate(sample_size: int) -> float:
        return _estimate_success_prob(
            baseline_rate,
            minimum_detectable_effect,
            sample_size,
            num_samples,
            alpha_priors,
            beta_priors
        )

    return binary_search(
        lo=1,
        hi=max_sample_size,
        condition=lambda n: simulate(n) >= required_probability,
        min_step=min_step
    )
