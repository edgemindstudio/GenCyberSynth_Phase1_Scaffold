# M7.3 — Makefile Safety and Authority Review

**Status:** PASS

## Summary

- Checks: 27/27 passed
- Targets reviewed: 21
- Canonical-safe: 3
- Historical-preserved: 4
- Mixed-authority: 12
- Destructive: 2

## Target classification

| Target | Classification | Migration action |
|---|---|---|
| `clean-summaries` | destructive | operator_only_no_migration |
| `clean-synth` | destructive | operator_only_no_migration |
| `figs-all` | mixed_authority | do_not_relabel_as_canonical |
| `figs-core` | mixed_authority | do_not_relabel_as_canonical |
| `figs-diversity` | mixed_authority | do_not_relabel_as_canonical |
| `figs-hparams` | mixed_authority | do_not_relabel_as_canonical |
| `figs-imbalance` | mixed_authority | do_not_relabel_as_canonical |
| `figs-qual` | mixed_authority | do_not_relabel_as_canonical |
| `normalize-summaries` | historical_preserved | preserve_historical |
| `paper1` | mixed_authority | do_not_relabel_as_canonical |
| `paper1-jsonl` | mixed_authority | do_not_relabel_as_canonical |
| `paper1_build` | mixed_authority | do_not_relabel_as_canonical |
| `paper1_prepare` | historical_preserved | preserve_historical |
| `phase1_backfill` | historical_preserved | preserve_historical |
| `phase1_check` | canonical_safe | preserve |
| `phase1_freeze` | historical_preserved | preserve_historical |
| `phase1_gate` | canonical_safe | preserve |
| `phase1_scores` | canonical_safe | preserve |
| `report` | mixed_authority | do_not_relabel_as_canonical |
| `scores-csv` | mixed_authority | do_not_relabel_as_canonical |
| `table` | mixed_authority | do_not_relabel_as_canonical |

## Decision

- Makefile modification authorized: **NO**
- Recommendation: `NO_CHANGE_DURING_M7_3_REVIEW`

The Makefile safely exposes canonical P0 gates separately, while historical and destructive targets remain visibly distinct. However paper1-jsonl and its dependent report/figure/build chain remain mixed-authority by default. M7.3 records this boundary; it does not authorize editing historical orchestration. Any future canonical Make entrypoint should be additive and separately named rather than silently changing historical targets.

## Authority statement

The Makefile is an orchestration surface, not a source of scientific authority. Canonical Paper 1 authority remains in TrustForge linkage/export interfaces. Historical and destructive Make targets retain their existing semantics and must not be silently reinterpreted as canonical.
