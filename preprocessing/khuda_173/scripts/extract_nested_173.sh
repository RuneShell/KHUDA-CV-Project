#!/usr/bin/bash

#SBATCH -J nested-173
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 1-0
#SBATCH -o /data/%u/seraph_jobs/logs/nested-173-%A.out

set -e

ROOT="/local_datasets/$USER/KHUDA_173/raw/extracted/173"
VIDEO_DIR="$ROOT/videos"
ANN_DIR="$ROOT/annotations"

echo "=== NESTED EXTRACT START ==="
date
echo "HOSTNAME: $(hostname)"
echo "ROOT: $ROOT"

mkdir -p "$VIDEO_DIR"
mkdir -p "$ANN_DIR"

echo "=== ZIP FILES BEFORE EXTRACTION ==="
find "$ROOT" -type f -iname "*.zip" -print

echo "=== UNZIP NESTED ZIP FILES ==="
find "$ROOT" -type f -iname "*.zip" -print0 | while IFS= read -r -d '' z; do
  outdir="${z%.zip}"
  echo "----------------------------------------"
  echo "Unzipping: $z"
  echo "Output dir: $outdir"
  mkdir -p "$outdir"
  unzip -q "$z" -d "$outdir"
done

echo "=== ORGANIZE VIDEOS ==="
find "$ROOT" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) \
  ! -path "$VIDEO_DIR/*" \
  -exec mv -n {} "$VIDEO_DIR/" \;

echo "=== ORGANIZE JSONS ==="
find "$ROOT" -type f -iname "*.json" \
  ! -path "$ANN_DIR/*" \
  -exec mv -n {} "$ANN_DIR/" \;

echo "=== RESULT COUNT ==="
echo "All files:"
find "$ROOT" -type f | wc -l

echo "Zip files:"
find "$ROOT" -type f -iname "*.zip" | wc -l

echo "Videos:"
find "$VIDEO_DIR" -type f | wc -l

echo "Annotations:"
find "$ANN_DIR" -type f | wc -l

echo "=== SAMPLE VIDEOS ==="
find "$VIDEO_DIR" -type f | head -20

echo "=== SAMPLE ANNOTATIONS ==="
find "$ANN_DIR" -type f | head -20

echo "=== FINAL STRUCTURE ==="
find "$ROOT" -maxdepth 4 -type d | sort

echo "=== DISK USAGE ==="
du -sh /local_datasets/$USER/KHUDA_173

echo "=== NESTED EXTRACT END ==="
date
