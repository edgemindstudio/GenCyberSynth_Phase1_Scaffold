# Paper 4 Result Interpretation Notes

## Current evidence state

The current paper-facing result table contains fixed-budget baselines, keep-all identity policies, strict confidence acceptance, balanced confidence-ranked top-k policies, and adaptive class-repair allocation.

## Short policy labels used in paper

- Keep-all b2000 = `paper4_policy_keep_all_b2000`
- Strict-conf. = `paper4_policy_confidence_accept_t080_b2000`
- Top-k500 = `paper4_policy_confidence_ranked_topk500_b2000`
- Top-k1000 = `paper4_policy_confidence_ranked_topk1000_b2000`
- Class-repair = `paper4_policy_class_repair_topk9000_b2000`

## Main policy findings

1. The keep-all identity policy validates that policy-specific manifests can be routed through the same downstream evaluation pipeline.

2. Strict-conf. is diagnostic rather than practical. It exposes that strict same-label confidence filtering can collapse the accepted synthetic set.

3. Top-k500 preserves class balance but is too restrictive on average.

4. Top-k1000 preserves class balance and produces positive Macro-F1 and balanced-accuracy gains across seeds while using half the full b2000 synthetic volume.

5. Class-repair adds adaptive allocation. It uses the same 9000-sample total budget as Top-k1000 but reallocates samples using an audit-based class difficulty signal.

## Mean ± SD summary

- Baseline b500 Δ Macro-F1: 0.002028 ± 0.002913
- Baseline b2000 Δ Macro-F1: 0.004356 ± 0.005115
- Keep-all b2000 Δ Macro-F1: 0.005696 ± 0.003787
- Top-k500 Δ Macro-F1: -0.000362 ± 0.003330
- Top-k1000 Δ Macro-F1: 0.003583 ± 0.002292
- Class-repair Δ Macro-F1: 0.003723 ± 0.002966
- Class-repair Δ Macro-AUPRC: 0.003000 ± 0.001768

## Safe paper claim

Selective synthetic-data policies are not automatically superior to keeping all generated samples. Strict confidence filtering can collapse accepted samples. Balanced confidence ranking avoids collapse, but its utility depends on the retained budget. Class-repair shows that adaptive allocation can improve the reduced-budget setting, especially for Macro-AUPRC, while the full Keep-all b2000 setting remains strongest on mean Macro-F1 and balanced accuracy.

## Claims to avoid

- Do not claim confidence ranking universally improves augmentation.
- Do not claim Top-k1000 beats full Keep-all b2000.
- Do not claim Strict-conf. is a good augmentation policy.
- Do not claim Class-repair universally dominates all policies.
- Do not claim policy quality can be judged from confidence alone; downstream utility remains necessary.
