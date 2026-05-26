# Paper 4 Top-k Budget Sensitivity Diagnostics

This diagnostic compares Top-k500, Top-k1000, and Class-repair.

Main interpretation:

- Top-k500 is the most restrictive balanced confidence-ranked policy and is near-zero or negative on Macro-F1 and balanced accuracy.
- Top-k1000 keeps twice as many samples per class as Top-k500 and restores positive Macro-F1 and balanced-accuracy utility.
- Class-repair uses the same total 9000-sample budget as Top-k1000, but reallocates across classes and improves Macro-AUPRC.
- This supports the journal claim that reduced-budget policy utility depends on both retained sample count and allocation structure.
