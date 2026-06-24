#!/usr/bin/bash

#SBATCH -J match-173
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/match-173-%A.out

ROOT="/local_datasets/$USER/KHUDA_173/raw/extracted/173"
VIDEO_DIR="$ROOT/videos"
ANN_DIR="$ROOT/annotations"

echo "=== MATCH CHECK START ==="
date
hostname

echo "Videos:"
find "$VIDEO_DIR" -type f -iname "*.mp4" | wc -l

echo "Annotations:"
find "$ANN_DIR" -type f -iname "*.json" | wc -l

echo "=== VIDEOS WITHOUT JSON ==="
comm -23 \
  <(find "$VIDEO_DIR" -type f -iname "*.mp4" -exec basename {} .mp4 \; | sort) \
  <(find "$ANN_DIR" -type f -iname "*.json" -exec basename {} .json \; | sort) \
  | head -50

echo "=== JSON WITHOUT VIDEO ==="
comm -13 \
  <(find "$VIDEO_DIR" -type f -iname "*.mp4" -exec basename {} .mp4 \; | sort) \
  <(find "$ANN_DIR" -type f -iname "*.json" -exec basename {} .json \; | sort) \
  | head -50

echo "=== MATCH CHECK END ==="
date
