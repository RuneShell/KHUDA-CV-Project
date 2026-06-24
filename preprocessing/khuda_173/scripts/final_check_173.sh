#!/usr/bin/bash

#SBATCH -J check-173
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/final-check-173-%A.out

ROOT="/local_datasets/$USER/KHUDA_173/raw/extracted/173"
VIDEO_DIR="$ROOT/videos"
ANN_DIR="$ROOT/annotations"

echo "=== FINAL CHECK START ==="
date
hostname

echo "=== COUNTS ==="
echo "All files:"
find "$ROOT" -type f | wc -l

echo "Videos:"
find "$VIDEO_DIR" -type f | wc -l

echo "Annotations:"
find "$ANN_DIR" -type f | wc -l

echo "Zip files:"
find "$ROOT" -type f -iname "*.zip" | wc -l

echo "=== SAMPLE VIDEOS ==="
find "$VIDEO_DIR" -type f | head -20

echo "=== SAMPLE ANNOTATIONS ==="
find "$ANN_DIR" -type f | head -20

echo "=== DISK USAGE ==="
du -sh /local_datasets/$USER/KHUDA_173

echo "=== FINAL CHECK END ==="
date
