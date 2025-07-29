from dataclasses import dataclass
from typing import List, Sequence, Optional, Callable, NamedTuple
import numpy as np
from tabulate import tabulate



@dataclass(frozen=True)
class VariantEvidence:
    name: str
    observations: int
    conversions: int
    alpha_prior: int = 1
    beta_prior: int = 1


class VariantStats(NamedTuple):
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float


class ComparativeVariantStats(NamedTuple):
    prob_better: float
    avg_uplift: float
    avg_regret: float
    p_in_rope: float


@dataclass(frozen=True)
class VariantEvaluation:
    name: str
    samples: Sequence[float]
    stats: VariantStats
    comparative_stats: ComparativeVariantStats
    is_reference: bool

    def __str__(self):
        prob_better_display = (
            f"{self.comparative_stats.prob_better:.2%}"
            if self.comparative_stats.prob_better is not None else "N/A"
        )
        is_equal = "yes" if self.comparative_stats.is_equivalent else "no"

        return (
            f"| name={self.name} \t"
            f"| is_reference={self.is_reference} \t"
            f"| prob_better={prob_better_display} \t"
            f"| uplift={self.comparative_stats.avg_uplift:.2%} \t"
            f"| regret={self.comparative_stats.avg_regret:.4f} \t"
            f"| p_in_rope={self.comparative_stats.p_in_rope:.2%} \t"
            f"| mean={self.stats.mean:.4f} \t"
            f"| CI=({self.stats.ci_95_lower:.4f}, {self.stats.ci_95_upper:.4f}) |"
        )





def _sample_posterior(conversions: int, observations: int, alpha_prior: int, beta_prior: int, sample_size: int):
    posterior_alpha = alpha_prior + conversions
    posterior_beta = beta_prior + observations - conversions
    return np.random.beta(posterior_alpha, posterior_beta, sample_size)


def _compute_posterior_stats(samples: Sequence[float]) -> VariantStats:
    mean = samples.mean()
    std = samples.std()
    ci_95_lower, ci_95_upper = np.percentile(samples, [5, 95])
    return VariantStats(mean, std, ci_95_lower, ci_95_upper)


def _make_comparative_stats_fn(reference_samples: np.ndarray, rope_percentage: float
                               ) -> Callable[[np.ndarray], ComparativeVariantStats]:
    def compute(samples: np.ndarray) -> ComparativeVariantStats:
        prob_better = (samples > reference_samples).mean()
        avg_uplift = ((samples / reference_samples) - 1).mean()
        regret = (reference_samples > samples) * (reference_samples - samples)
        avg_regret = regret.mean()

        lower_bound = reference_samples * (1 - rope_percentage)
        upper_bound = reference_samples * (1 + rope_percentage)
        p_in_rope = np.mean((samples >= lower_bound) & (samples <= upper_bound))

        return ComparativeVariantStats(
            prob_better, avg_uplift, avg_regret, p_in_rope
        )
    return compute




def _get_best_variant_index(all_samples: List[np.ndarray]) -> int:
    stacked_samples = np.stack(all_samples)
    winners = np.argmax(stacked_samples, axis=0)
    counts = np.bincount(winners, minlength=stacked_samples.shape[0])
    return int(np.argmax(counts))


def _evaluate_single_variant(
    name: str,
    conversions: int,
    observations: int,
    alpha_prior: int,
    beta_prior: int,
    sample_size: int,
    is_reference: bool,
    compute_comparative_stats: Callable[[np.ndarray], ComparativeVariantStats]
) -> VariantEvaluation:
    samples = _sample_posterior(conversions, observations, alpha_prior, beta_prior, sample_size)
    stats = _compute_posterior_stats(samples)

    comp_stats = (
        ComparativeVariantStats(0.0, 0.0, 0.0, 1.0)
        if is_reference
        else compute_comparative_stats(samples)
    )

    return VariantEvaluation(
        name=name,
        samples=samples.tolist(),
        stats=stats,
        comparative_stats=comp_stats,
        is_reference=is_reference
    )



def evaluate_product_variants(
    variant_evidence: List[VariantEvidence],
    sample_size: int = 100_000,
    reference_index: Optional[int] = None,
    rope: float = 0.01
) -> List[VariantEvaluation]:
    
    all_samples = [
        _sample_posterior(v.conversions, v.observations, v.alpha_prior, v.beta_prior, sample_size)
        for v in variant_evidence
    ]

    ref_idx = reference_index if reference_index is not None else _get_best_variant_index(all_samples)
    reference_samples = all_samples[ref_idx]

    compute_comparative_stats = _make_comparative_stats_fn(reference_samples, rope)

    return [
        _evaluate_single_variant(
            name=v.name,
            conversions=v.conversions,
            observations=v.observations,
            alpha_prior=v.alpha_prior,
            beta_prior=v.beta_prior,
            sample_size=sample_size,
            is_reference=(i == ref_idx),
            compute_comparative_stats=compute_comparative_stats
        )
        for i, v in enumerate(variant_evidence)
    ]



def _format_variant_results(evaluations: List[VariantEvaluation]) -> str:
    rows = [
        [
            ev.name,
            ev.is_reference,
            f"{ev.comparative_stats.prob_better:.2%}",
            f"{ev.comparative_stats.avg_uplift:.2%}",
            f"{ev.comparative_stats.avg_regret:.4f}",
            f"{ev.comparative_stats.p_in_rope:.2%}",
            f"{ev.stats.mean:.4f}",
            f"{ev.stats.ci_95_lower:.4f}",
            f"{ev.stats.ci_95_upper:.4f}"
        ]
        for ev in evaluations
    ]
    headers = ["name", "ref", "p_better", "uplift", "regret", "p_in_rope", "mean", "CI_low", "CI_up"]
    return tabulate(rows, headers=headers, tablefmt="github")


def display_variant_results(evaluations):
    print(_format_variant_results(evaluations))


