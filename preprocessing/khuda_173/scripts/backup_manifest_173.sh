#!/usr/bin/bash

#SBATCH -J backup-manifest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/backup-manifest-173-%A.out

set -e

SRC="/local_datasets/$USER/KHUDA_173/processed_173_manifest"
DST="/data/$USER/KHUDA_173/processed_173_manifest"

echo "=== BACKUP MANIFEST START ==="
date
hostname

mkdir -p "/data/$USER/KHUDA_173"

rsync -avP "$SRC/" "$DST/"

echo ""
echo "=== BACKUP RESULT ==="
find "$DST" -maxdepth 3 -type f | sort
du -sh "$DST"

echo ""
echo "=== BACKUP MANIFEST END ==="
date
