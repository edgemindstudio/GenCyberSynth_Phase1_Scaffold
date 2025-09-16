
#!/usr/bin/env bash
#SBATCH --job-name=gcs
#SBATCH --partition=talon-short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=0-2

# Usage:
# sbatch scripts/slurm_array_example.sh MODEL=TEMPLATE SEEDS=42,43,44

set -euo pipefail

MODEL=${MODEL:-TEMPLATE}
SEEDS_CSV=${SEEDS:-42,43,44}
IFS=',' read -ra SEEDS_ARR <<< "$SEEDS_CSV"
SEED=${SEEDS_ARR[$SLURM_ARRAY_TASK_ID]}

echo "Running $MODEL (simulate) seed=$SEED"
python app/main.py train --simulate --model-name "$MODEL" --seed "$SEED"
python app/main.py synth --simulate --model-name "$MODEL" --seed "$SEED" --per-class 1000 --out artifacts/synthetic/seed_${SEED}
python app/main.py eval  --simulate --model-name "$MODEL" --seed "$SEED" --fid-cap 200 --json runs/summary_seed_${SEED}.jsonl --console runs/console_seed_${SEED}.txt
