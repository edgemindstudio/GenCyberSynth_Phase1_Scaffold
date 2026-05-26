# Paper 4 CICMalDroid2020 Policy Validation

This diagnostic compares CICMalDroid2020 Keep-all b2000, Top-k1000, and Class-repair policies.

The collector deduplicates by config_id and seed, keeping the latest summary when duplicate seed runs exist.

Main interpretation:

- CICMalDroid2020 acts as the second-dataset validation layer for Paper 4.
- Keep-all b2000 uses 10000 synthetic samples, while Top-k1000 and Class-repair use 5000 selected samples.
- Reduced-budget policies do not consistently reverse negative balanced-accuracy effects on CICMalDroid2020.
- Top-k1000 is generally more favorable than Class-repair on CICMalDroid in the current runs, especially for Macro-AUPRC.
- This contrasts with USTC-TFC2016 and supports the journal claim that selective synthetic-data policies are dataset-sensitive.
- The paper should avoid claiming that Class-repair is universally best.
