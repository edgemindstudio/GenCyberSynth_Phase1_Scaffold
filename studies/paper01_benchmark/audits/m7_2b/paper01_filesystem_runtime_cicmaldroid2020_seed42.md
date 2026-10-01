# M7.2B — Filesystem-coupled Runtime Compatibility Audit

**Status:** PASS

- Dataset: `cicmaldroid2020`
- Seed: `42`
- Checks: 58/58 passed

## Per-model runtime compatibility

| Model | Manifest exists | Manifest records | Simple-stats images | Classes | Class-triptych counts |
|---|---:|---:|---:|---|---|
| `autoregressive` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |
| `diffusion` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |
| `gan` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |
| `gaussianmixture` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |
| `maskedautoflow` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |
| `restrictedboltzmann` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |
| `vae` | yes | 10000 | 64 | 0, 1, 2, 3, 4 | 0:32, 1:32, 2:32, 3:32, 4:32 |

## Authority statement

Runtime filesystem compatibility was validated only through manifest_path carried by canonical-derived rows. Fallback synthetic discovery was not invoked and is not scientific authority.

> A PASS means the exact canonical manifest paths resolve and both consumers can parse historical synthetic images from them directly. It does not authorize fallback globs to select scientific evidence.
