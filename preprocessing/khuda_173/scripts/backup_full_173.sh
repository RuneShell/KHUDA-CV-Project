#!/usr/bin/bash

#SBATCH -J backup-full-173
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 1-0
#SBATCH -o /data/%u/seraph_jobs/logs/backup-full-173-%A.out

set -e

SRC="/local_datasets/$USER/KHUDA_173"
DST="/data/$USER/KHUDA_173"

echo "=== BACKUP FULL START ==="
date
hostname

mkdir -p "$DST"

rsync -avP "$SRC/" "$DST/"

echo ""
echo "=== BACKUP FULL RESULT ==="
du -sh "$DST"

echo ""
echo "=== FILE COUNTS ==="
echo "Videos:"
find "$DST/raw/extracted/173/videos" -type f -iname "*.mp4" | wc -l

echo "Annotations:"
find "$DST/raw/extracted/173/annotations" -type f -iname "*.json" | wc -l

echo "Manifests:"
find "$DST/processed_173_manifest/manifests" -type f | sort

echo ""
echo "=== BACKUP FULL END ==="
date
