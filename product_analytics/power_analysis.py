import numpy as np
from typing import Optional, Callable
from .variant_evaluation import _sample_posterior 


def _estimate_success_prob(
    baseline_rate: float,
    uplift: float,
    sample_size: int,
    num_samples: int,
    alpha_priors: tuple[int, int] = (1, 1),
    beta_priors: tuple[int, int] = (1, 1)
) -> float:

    successes_a = int(sample_size * baseline_rate)
    successes_b = int(sample_size * (baseline_rate + uplift))

    samples_a = _sample_posterior(successes_a, sample_size, alpha_priors[0], beta_priors[0], num_samples)
    samples_b = _sample_posterior(successes_b, sample_size, alpha_priors[1], beta_priors[1], num_samples)

    return (samples_b > samples_a).mean()


def _binary_search(
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


def _estimate_minimum_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    required_probability: float = 0.975,
    max_sample_size: int = 100_000,
    num_samples: int = 10_000,
    min_step: int = 100,
    alpha_priors: tuple[int, int] = (1, 1),
    beta_priors: tuple[int, int] = (1, 1)
) -> Optional[int]:

    def simulate_n_samples(sample_size: int) -> float:
        return _estimate_success_prob(
            baseline_rate,
            minimum_detectable_effect,
            sample_size,
            num_samples,
            alpha_priors,
            beta_priors
        )

    return _binary_search(
        lo=1,
        hi=max_sample_size,
        condition=lambda n: simulate_n_samples(n) >= required_probability,
        min_step=min_step
    )

def power_analysis(
    baseline_rate: float,
    minimum_detectable_effect: float,
    required_probability: float = 0.975,
    max_sample_size: int = 100_000,
    num_samples: int = 10_000,
    min_step: int = 100,
    alpha_priors: tuple[int, int] = (1, 1),
    beta_priors: tuple[int, int] = (1, 1),
    n_runs: int = 9
) -> Optional[float]:
    """
    Estimate the median minimum sample size required to detect a specified uplift
    with a given confidence level, using repeated Bayesian simulations.

    This function runs multiple binary Bayesian tests to stabilize variance caused
    by stochastic sampling. It reports the median of the minimum required sample sizes.

    Parameters:
        baseline_rate (float): Conversion rate of the control group (e.g. 0.05).
        minimum_detectable_effect (float): Minimum absolute uplift to detect over the baseline.
        required_probability (float): Confidence level to consider the uplift detectable (default: 0.975).
        max_sample_size (int): Upper bound of sample size search space (default: 100,000).
        num_samples (int): Number of posterior samples per simulation run (default: 10,000).
        min_step (int): Minimum step resolution for binary search (default: 100).
        alpha_priors (tuple[int, int]): Beta prior (alpha) parameters for control and variant (default: (1, 1)).
        beta_priors (tuple[int, int]): Beta prior (beta) parameters for control and variant (default: (1, 1)).
        n_runs (int): Number of independent simulation runs to average over (default: 9).

    Returns:
        Optional[float]: Median estimated minimum sample size, or None if no result was found.
    """

    results = [
        _estimate_minimum_sample_size(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=minimum_detectable_effect,
            required_probability=required_probability,
            max_sample_size=max_sample_size,
            num_samples=num_samples,
            min_step=min_step,
            alpha_priors=alpha_priors,
            beta_priors=beta_priors
        )
        for _ in range(n_runs)
    ]
    valid = [r for r in results if r is not None]
    return float(np.median(valid)) if valid else None
