# Paper 4 Class-repair Allocation Diagnostics

This diagnostic summarizes how Class-repair reallocates the fixed 9000-sample budget across classes and seeds.

Important naming note: the source allocation CSV column `real_only_f1` is used here as `audit_confidence_score`. It reflects the audit-derived class confidence score used by the allocation rule, not true downstream real-only F1.

Main interpretation:

- Class-repair is not equivalent to balanced Top-k1000. Top-k1000 assigns 1000 samples per class, while Class-repair reallocates the same 9000 total samples adaptively.
- Classes with higher audit confidence generally receive the base allocation or near-base allocation.
- Classes with lower audit confidence receive larger allocations.
- The allocation pattern is seed-dependent, which should be reported as a policy behavior rather than hidden.
- This supports the journal claim that allocation structure can affect downstream utility under a fixed selected-sample budget.
