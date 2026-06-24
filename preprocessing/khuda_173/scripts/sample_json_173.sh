#!/usr/bin/bash

#SBATCH -J sample-json
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/sample-json-173-%A.out

ANN_DIR="/local_datasets/$USER/KHUDA_173/raw/extracted/173/annotations"

echo "=== SAMPLE JSON START ==="
date
hostname

SAMPLE=$(find "$ANN_DIR" -type f -iname "*.json" | head -n 1)

echo "Sample JSON:"
echo "$SAMPLE"

echo "=== JSON CONTENT HEAD ==="
python -m json.tool "$SAMPLE" | head -120

echo "=== SAMPLE JSON END ==="
date
