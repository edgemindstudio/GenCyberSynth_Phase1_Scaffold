# Paper 4 Strict-conf. Collapse Diagnostics

Strict-conf. was evaluated on the seed42 b2000 source pool with 18000 generated samples.

It accepted 1004 samples, for an acceptance rate of 0.055778.

The source pool contains 9 requested classes before filtering, but the accepted set contains 1 requested class after filtering.

Main interpretation:

- Strict confidence acceptance is diagnostic rather than practical.
- The policy applies a plausible confidence rule but destroys class coverage.
- This supports the claim that sample-level confidence alone is insufficient for synthetic-data policy selection.
- Confidence filtering must be combined with class coverage, budget, diversity, or downstream utility constraints.
