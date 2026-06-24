#!/usr/bin/bash

#SBATCH -J sample-manifest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/sample-manifest-173-%A.out

ROOT="/local_datasets/$USER/KHUDA_173"
MANIFEST_DIR="$ROOT/processed_173_manifest/manifests"
REPORT_DIR="$ROOT/processed_173_manifest/reports"
META_DIR="$ROOT/processed_173_manifest/metadata"

echo "=== CLIPS_ALL SAMPLE ==="
python -m json.tool "$MANIFEST_DIR/clips_all.json" | head -120

echo ""
echo "=== EVENTS_ALL SAMPLE ==="
python -m json.tool "$MANIFEST_DIR/events_all.json" | head -120

echo ""
echo "=== PREPROCESSING CONFIG ==="
python -m json.tool "$META_DIR/preprocessing_config.json"

echo ""
echo "=== VALIDATION REPORT ==="
python -m json.tool "$REPORT_DIR/manifest_validation_report.json" | head -200

echo ""
echo "=== SAMPLE MANIFEST END ==="
