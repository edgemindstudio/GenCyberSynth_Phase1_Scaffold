#!/usr/bin/env bash
#SBATCH -J gcs-test
#SBATCH -o gcs.%j.out
#SBATCH -e gcs.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH -t 00:10:00

# Clean, predictable env on compute node
module purge
module load python3
module load slurm/slurm/24.05.7

# Activate your venv created on login node
source "$SCRATCH/gencys/runs/.venv-cpu/bin/activate"

python -V
python - <<'PY'
print("Hello from Slurm Python!")
PY
