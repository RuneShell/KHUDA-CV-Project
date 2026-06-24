#!/usr/bin/bash

#SBATCH -J tree-173
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/tree-173-%A.out

ROOT="/local_datasets/$USER/KHUDA_173"

echo "=== TREE CHECK START ==="
date
hostname

echo ""
echo "=== ROOT ==="
ls -lh "$ROOT"

echo ""
echo "=== DIRECTORY TREE DEPTH 5 ==="
find "$ROOT" -maxdepth 5 -type d | sort

echo ""
echo "=== RAW EXTRACTED 173 ==="
ls -lh "$ROOT/raw/extracted/173"

echo ""
echo "=== VIDEOS DIR INFO ==="
ls -lh "$ROOT/raw/extracted/173/videos" | head -50

echo ""
echo "=== ANNOTATIONS DIR INFO ==="
ls -lh "$ROOT/raw/extracted/173/annotations" | head -50

echo ""
echo "=== FILE COUNTS ==="
echo "All files:"
find "$ROOT" -type f | wc -l

echo "Videos:"
find "$ROOT/raw/extracted/173/videos" -type f | wc -l

echo "Annotations:"
find "$ROOT/raw/extracted/173/annotations" -type f | wc -l

echo "Zip files:"
find "$ROOT" -type f -iname "*.zip" | wc -l

echo ""
echo "=== SAMPLE VIDEOS ==="
find "$ROOT/raw/extracted/173/videos" -type f | head -30

echo ""
echo "=== SAMPLE ANNOTATIONS ==="
find "$ROOT/raw/extracted/173/annotations" -type f | head -30

echo ""
echo "=== DISK USAGE BY TOP FOLDERS ==="
du -h --max-depth=2 "$ROOT" | sort -hr | head -50

echo ""
echo "=== TREE CHECK END ==="
date
