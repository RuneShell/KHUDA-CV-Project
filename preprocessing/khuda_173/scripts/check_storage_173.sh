#!/usr/bin/bash

#SBATCH -J check-storage
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/check-storage-173-%A.out

echo "=== STORAGE CHECK START ==="
date
hostname

echo ""
echo "=== LOCAL DATASETS ==="
du -sh /local_datasets/$USER/KHUDA_173

echo ""
echo "=== /data/$USER ==="
ls -ld /data/$USER
df -h /data/$USER

echo ""
echo "=== /data2 CHECK ==="
ls -ld /data2 2>/dev/null || echo "/data2 not found"
ls -ld /data2/$USER 2>/dev/null || echo "/data2/$USER not found"

echo ""
echo "=== STORAGE CHECK END ==="
date
