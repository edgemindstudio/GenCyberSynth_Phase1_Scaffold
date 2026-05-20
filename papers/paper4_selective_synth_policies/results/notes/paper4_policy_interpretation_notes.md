# Paper 4 Result Interpretation Notes

## Current evidence state

The current paper-facing result table contains the baseline, keep_all identity-policy, strict confidence-acceptance, and confidence-ranked top-k policy results.

## Main policy findings

1. The keep_all identity policy validates that policy-specific manifests can be routed through the same downstream evaluation pipeline.

2. The strict confidence_accept_t080 policy is diagnostic rather than practical. It exposes that strict same-label confidence filtering can collapse the accepted synthetic set.

3. The confidence_ranked_topk500 policy preserves class balance but is too restrictive on average.

4. The confidence_ranked_topk1000 policy preserves class balance and produces positive macro-F1 and balanced-accuracy gains across seeds while using half the full b2000 synthetic volume.

## Mean ± SD summary

- Baseline b500 Δ Macro-F1: 0.002028 ± 0.002913
- Baseline b2000 Δ Macro-F1: 0.004356 ± 0.005115
- keep_all b2000 Δ Macro-F1: 0.005696 ± 0.003787
- confidence_ranked topk500 Δ Macro-F1: -0.000362 ± 0.003330
- confidence_ranked topk1000 Δ Macro-F1: 0.003583 ± 0.002292

## Safe paper claim

Selective synthetic-data policies are not automatically superior to keeping all generated samples. Strict confidence filtering can collapse accepted samples. Balanced confidence ranking avoids collapse, but its utility depends on the retained budget. In the current results, topk1000 preserves positive utility with half the full b2000 synthetic volume, while the full keep_all b2000 setting remains strongest on average.

## Claims to avoid

- Do not claim confidence ranking universally improves augmentation.
- Do not claim topk1000 beats full b2000 keep_all.
- Do not claim confidence_accept_t080 is a good augmentation policy.
- Do not claim policy quality can be judged from confidence alone; downstream utility remains necessary.
