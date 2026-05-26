# Paper 4 CICMalDroid2020 Keep-all Validation

This diagnostic collects Paper 4 CICMalDroid2020 real-only and keep-all b2000 GAN evaluations.

The collector deduplicates by config_id and seed, keeping the latest summary path when duplicate seed runs exist.

Main interpretation:

- CICMalDroid2020 provides the second-dataset validation layer for Paper 4.
- Keep-all b2000 uses 10000 synthetic GAN samples, corresponding to 2000 samples per class across five classes.
- Across seeds, keep-all tends to reduce Macro-F1 and balanced accuracy.
- Macro-AUPRC and calibration metrics may improve even when Macro-F1 and balanced accuracy decline.
- This supports a journal-safe claim that synthetic-data policy effects are dataset-sensitive and metric-sensitive.
