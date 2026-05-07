# Paper 2 Results Synthesis

## Core Question

Paper 2 asks whether class-conditional synthetic cybersecurity image generators actually respect requested class labels, and whether internal conditioning objectives are sufficient evidence of class-faithful generation.

## Evidence Summary

| Method / Intervention | Config | Seed | Epochs | Audited samples | Conditioning accuracy | Leakage rate | Predicted-label collapse | Downstream utility | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Baseline cDCGAN smoke | paper2_smoke | 42 | smoke | 225 | 0.1111 | 0.8889 | collapsed | not final | Label injection alone failed in smoke audit |
| ACGAN smoke | paper2_acgan_smoke | 42 | 1 | 90 | ~0.12 | ~0.88 | collapsed | not final | ACGAN route worked, but smoke still collapsed |
| ACGAN auxloss clean | paper2_acgan_auxloss | 42 | 50 | 90 | 0.1111 | 0.8889 | class 7 collapse | not primary | Internal class loss did not guarantee external class fidelity |
| Fake-class ACGAN smoke | paper2_acgan_fakeclass_smoke | 42 | 1 | 90 | 0.1000 | 0.9000 | mostly class 4 | not final | Fake-class-head route worked, but smoke still collapsed |
| Fake-class ACGAN full | paper2_acgan_fakeclass | 42 | 50 | 225 | 0.1111 | 0.8889 | class 7 collapse | small positive deltas | Fake-class-head supervision improved utility slightly but did not repair external conditioning |

## Main Finding

Across the tested GAN conditioning interventions, internal conditioning mechanisms did not guarantee externally class-faithful generation. Even when the auxiliary classifier losses became very small, an independent real-only classifier still judged generated samples as belonging almost entirely to a single class.

## Important Distinction

The fake-class-head ACGAN result suggests that synthetic data can slightly improve downstream classification utility while still failing class-conditional semantic fidelity. Therefore, downstream utility and conditioning correctness should be evaluated separately.

## Paper 2 Claim

The evidence supports the following central claim:

> Synthetic cybersecurity image generators require external conditioning audits because internal conditioning objectives can appear successful while generated samples remain class-collapsed under an independent classifier.

## Seed-Stability Result for Fake-Class ACGAN

The fake-class-head ACGAN intervention was evaluated across seeds 42, 43, and 44. Across all three seeds, the external real-only CNN audit found severe class-conditioning failure.

| Seed | Config | Audited samples | Conditioning accuracy | Leakage rate | Predicted-label pattern |
|---:|---|---:|---:|---:|---|
| 42 | paper2_acgan_fakeclass | 225 | 0.1111 | 0.8889 | full collapse to class 7 |
| 43 | paper2_acgan_fakeclass_seed43 | 225 | 0.0356 | 0.9644 | collapse into classes 2, 5, and 8 |
| 44 | paper2_acgan_fakeclass_seed44 | 225 | 0.1111 | 0.8889 | full collapse to class 8 |

This result suggests that external class-conditioning failure is not an isolated seed-42 artifact. The exact collapse pattern varies by seed, but all tested seeds fail to produce class-faithful synthetic samples under the independent real-only classifier audit.

Importantly, the evaluation summaries for seeds 43 and 44 still showed positive downstream utility deltas, indicating that downstream utility improvement and class-conditional semantic fidelity are not equivalent.

## Next Experimental Options

1. Run seeds 43 and 44 for the strongest current intervention to test whether collapse is seed-stable. DONE
2. Add a frozen external classifier guidance loss to directly align generator outputs with an independent classifier.
3. Extend the audit to class-wise cKID/cFID and per-class utility deltas.