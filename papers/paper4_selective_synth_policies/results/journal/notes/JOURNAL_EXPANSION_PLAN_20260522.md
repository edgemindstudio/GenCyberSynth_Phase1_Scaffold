# Paper 4 Journal Expansion Plan

Paper 4 title:
Selective Synthetic Data Policies for Malware Classification: Allocation, Filtering, and Acceptance Rules Under Utility Constraints

## Current conference-ready contribution

Paper 4 currently evaluates selective synthetic-data policies on USTC-TFC2016 malware images using a fixed GAN-generated synthetic source pool. It compares fixed-budget baselines, keep-all policies, strict confidence acceptance, balanced confidence-ranked Top-k selection, and adaptive Class-repair allocation.

The central claim is that synthetic malware augmentation should be treated as a policy-controlled intervention. The question is not only whether synthetic samples can be generated, but which samples should be accepted, filtered, ranked, or allocated before downstream training.

## Current policy families

- Baseline b25, b500, b2000
- Keep-all b25, b500, b2000
- Strict-conf.
- Top-k500
- Top-k1000
- Class-repair

## Current main results

Keep-all b2000 gives the strongest mean Macro-F1 and balanced accuracy but uses 18,000 synthetic samples.

Top-k1000 preserves positive Macro-F1 and balanced accuracy using 9,000 samples.

Top-k500 is too restrictive.

Strict-conf. collapses the accepted set and is diagnostic rather than practical.

Class-repair uses the same 9,000-sample budget as Top-k1000 but adaptively reallocates samples across classes. It slightly improves Macro-F1 and balanced accuracy over Top-k1000 and gives stronger Macro-AUPRC.

## Relationship to Papers 2 and 3

Paper 2 provides the external audit logic: requested-label fidelity should be externally audited using real-only classifiers, and downstream utility is distinct from class fidelity.

Paper 3 provides the regime logic: synthetic-data utility depends on augmentation regime, generator family, budget, class structure, and downstream metric.

Paper 4 provides the policy logic: given a generated synthetic pool, sample acceptance, filtering, ranking, and allocation policies materially affect downstream utility.

## Journal expansion priorities

1. Deepen current USTC evidence with per-class policy analysis.
2. Add Class-repair allocation diagnostics.
3. Add Strict-conf. collapse diagnostics.
4. Analyze KID/MS-SSIM versus downstream utility for selected policies.
5. Add calibration metrics only if summaries contain ECE/Brier; current paper-facing CSV does not include them.
6. Add CICMalDroid2020 as second dataset after current USTC deepening.
7. Add at least one additional generator family, preferably VAE first, after second-dataset planning.
8. Add audit-classifier robustness later using Paper 2 audit classifiers if feasible.

## Safe claims

- Selective policies are not automatically better than Keep-all.
- Strict confidence acceptance can collapse the selected set.
- Balanced Top-k selection avoids collapse but is budget-sensitive.
- Class-repair is the strongest reduced-budget policy in the current USTC/GAN setting.
- Policy choice materially affects downstream utility.
- Downstream utility remains the final judge of policy value.

## Claims to avoid

- Do not claim selective policies universally beat Keep-all.
- Do not claim Class-repair is universally best.
- Do not claim confidence alone measures utility.
- Do not claim results generalize to all datasets or generator families until we run those experiments.