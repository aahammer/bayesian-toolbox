# Product Analytics Toolbox

Simple functionality to solve common product analytics problems — the **Bayesian way**.

# ✅ Features

- A/B testing 
- Multivariate testing (3+ variants)  
- Tabular summary output  

---

## A/B & Multivariate test

## 📊 A/B Test (with reference)

```python

from product_analytics import VariantEvidence, evaluate_product_variants, display_variant_results


variants = [
    VariantEvidence(name="A", observations=1000, conversions=100),
    VariantEvidence(name="B", observations=1000, conversions=120),
]

results = evaluate_product_variants(variants, reference_index=0)
display_variant_results(results)
```

```shell
| name   | ref   | p_better   | uplift   |   regret | p_in_rope   |   mean |   CI_low |   CI_up |
|--------|-------|------------|----------|----------|-------------|--------|----------|---------|
| A      | True  | 0.00%      | 0.00%    |   0      | 100.00%     | 0.1007 |   0.0856 |  0.1168 |
| B      | False | 92.49%     | 20.91%   |   0.0005 | 11.47%      | 0.1208 |   0.1043 |  0.1382 |

```

## Multivariate test — best variant auto-selected as reference

```python

from product_analytics import VariantEvidence, evaluate_product_variants, display_variant_results

variants = [
    VariantEvidence(name="A", observations=800, conversions=88),
    VariantEvidence(name="B", observations=1000, conversions=110),
    VariantEvidence(name="C", observations=950, conversions=130),
]

results = evaluate_product_variants(variants)
display_variant_results(results)
```

```shell
| name   | ref   | p_better   | uplift   |   regret | p_in_rope   |   mean |   CI_low |   CI_up |
|--------|-------|------------|----------|----------|-------------|--------|----------|---------|
| A      | False | 4.59%      | -18.84%  |   0.027  | 7.89%       | 0.111  |   0.0933 |  0.1298 |
| B      | False | 3.60%      | -19.00%  |   0.0271 | 7.04%       | 0.1108 |   0.0949 |  0.1275 |
| C      | True  | 0.00%      | 0.00%    |   0      | 100.00%     | 0.1376 |   0.1197 |  0.1564 |

```

## 📄 Output Explanation

Each row in the result table corresponds to one variant. The columns report Bayesian estimates:

| Column       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `name`       | Name of the variant                                                         |
| `ref`        | Whether this variant is used as the reference baseline                      |
| `p_better`   | Probability this variant is better than the reference                       |
| `uplift`     | Average relative uplift vs. the reference                                   |
| `regret`     | Expected loss if this variant is chosen but it's not the best               |
| `p_in_rope`  | Probability the difference is within the *Region of Practical Equivalence* (ROPE) |
| `mean`       | Mean of the posterior distribution for the conversion rate                 |
| `CI_low`     | 5th percentile of the posterior (lower credible interval bound)             |
| `CI_up`      | 95th percentile of the posterior (upper credible interval bound)            |

### ℹ️ Notes

- ROPE (Region of Practical Equivalence) defines a margin within which variants are considered practically equal (deafult is 5%)
- You can configure the ROPE width via the `rope` parameter:
  
```python
evaluate_product_variants(variants, rope=0.02)  # 2% equivalence threshold
```

## 🧪 Power Analysis

Estimate how many observations you need to confidently detect a desired uplift using Bayesian simulation and binary search.

### 🔍 Estimate Required Sample Size

```python
from product_analytics import power_analysis

sample_size = power_analysis(
    baseline_rate=0.05,
    minimum_detectable_effect=0.01
)

print(f"Estimated minimum sample size per variant: {sample_size}")
```


```shell
Estimated minimum sample size per variant: 4003.0
```