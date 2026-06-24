#!/usr/bin/bash

#SBATCH -J inspect-173
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/inspect-173-%A.out

set -e

ROOT="/local_datasets/$USER/KHUDA_173/raw/extracted/173"

echo "=== INSPECT START ==="
date
hostname

echo "=== ROOT ==="
ls -lh "$ROOT"

echo "=== DIRECTORY STRUCTURE ==="
find "$ROOT" -maxdepth 5 -type d | sort

echo "=== FILE LIST SAMPLE ==="
find "$ROOT" -maxdepth 6 -type f | head -100

echo "=== ZIP FILES ==="
find "$ROOT" -type f -iname "*.zip" -print

echo "=== VIDEO FILES SAMPLE ==="
find "$ROOT" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | head -50

echo "=== JSON FILES SAMPLE ==="
find "$ROOT" -type f -iname "*.json" | head -50

echo "=== COUNTS ==="
echo "All files:"
find "$ROOT" -type f | wc -l

echo "Zip files:"
find "$ROOT" -type f -iname "*.zip" | wc -l

echo "Videos:"
find "$ROOT" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | wc -l

echo "JSON:"
find "$ROOT" -type f -iname "*.json" | wc -l

echo "=== DISK USAGE ==="
du -sh /local_datasets/$USER/KHUDA_173

echo "=== INSPECT END ==="
date
