# Paper 4 Journal Quality-Utility Diagnostics

This diagnostic compares downstream policy utility against KID and MS-SSIM values already present in the paper-facing result table.

Key interpretation to verify from the generated tables:

- KID and MS-SSIM should be treated as diagnostics, not final policy-selection criteria.
- Similar KID/MS-SSIM values can correspond to different downstream utility values.
- Class-repair and Top-k1000 are especially important because both use 9000 selected samples but differ in allocation.
- Calibration metrics are not included because current Paper 4 summaries do not contain usable ECE/Brier deltas.
