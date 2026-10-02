# TrustForge HPC

**Status:** `SCAFFOLD_ONLY`

This directory is the native TrustForge destination for future execution-backend support.

Creating this directory does **not** migrate, replace, supersede, or reinterpret any historical Slurm script.

Current historical/global HPC compatibility surfaces remain authoritative in their existing locations, including:

- `slurm/`
- `run_gcs.slurm`
- `run_suite.slurm`
- `run_tuning.slurm`

Authority boundary:

- `SCAFFOLD_CREATION != MIGRATION`
- `DESTINATION_EXISTENCE != AUTHORITY_TRANSFER`
- Historical execution semantics remain unchanged.
- No historical Slurm file has been moved or copied here.
- No import, execution, or scheduling behavior is changed by this scaffold.

Authorized by M8.4.2 acceptance commit `954ad5f`.
