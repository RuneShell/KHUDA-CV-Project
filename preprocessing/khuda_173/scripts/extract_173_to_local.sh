#!/usr/bin/bash

#SBATCH -J extract-173
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/%u/seraph_jobs/logs/extract-173-%A.out

set -e

echo "=== JOB START ==="
date
echo "HOSTNAME: $(hostname)"
echo "USER: $USER"

SRC_DIR="/data/dlgkrwls09/KHUDA/datasets"
SRC_ZIP=$(find "$SRC_DIR" -maxdepth 1 -type f -name "173*공원*불법행위*zip" | head -n 1)

WORK_DIR="/local_datasets/$USER/KHUDA_173"
ZIP_DST="$WORK_DIR/173_dataset.zip"
EXTRACT_DIR="$WORK_DIR/raw/extracted/173"

echo "SRC_ZIP: $SRC_ZIP"
echo "WORK_DIR: $WORK_DIR"
echo "EXTRACT_DIR: $EXTRACT_DIR"

if [ -z "$SRC_ZIP" ]; then
  echo "ERROR: zip file not found in $SRC_DIR"
  exit 1
fi

if [ ! -r "$SRC_ZIP" ]; then
  echo "ERROR: source zip is not readable"
  ls -lh "$SRC_ZIP"
  exit 1
fi

echo "=== MAKE DIRECTORIES ==="
mkdir -p "$WORK_DIR"
mkdir -p "$EXTRACT_DIR/videos"
mkdir -p "$EXTRACT_DIR/annotations"
mkdir -p "$WORK_DIR/processed_173_manifest/manifests"
mkdir -p "$WORK_DIR/processed_173_manifest/metadata"
mkdir -p "$WORK_DIR/processed_173_manifest/reports"
mkdir -p "$WORK_DIR/processed_173_manifest/logs"

echo "=== SOURCE ZIP INFO ==="
ls -lh "$SRC_ZIP"

echo "=== COPY ZIP TO LOCAL DATASETS ==="
date
cp -av "$SRC_ZIP" "$ZIP_DST"

echo "=== LOCAL ZIP INFO ==="
ls -lh "$ZIP_DST"

echo "=== UNZIP START ==="
date
time unzip -q "$ZIP_DST" -d "$EXTRACT_DIR"

echo "=== UNZIP DONE ==="
date

echo "=== ORGANIZE VIDEOS ==="
find "$EXTRACT_DIR" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) \
  ! -path "$EXTRACT_DIR/videos/*" \
  -exec mv -n {} "$EXTRACT_DIR/videos/" \;

echo "=== ORGANIZE JSONS ==="
find "$EXTRACT_DIR" -type f -iname "*.json" \
  ! -path "$EXTRACT_DIR/annotations/*" \
  -exec mv -n {} "$EXTRACT_DIR/annotations/" \;

echo "=== RESULT COUNT ==="
echo "All files:"
find "$EXTRACT_DIR" -type f | wc -l

echo "Videos:"
find "$EXTRACT_DIR/videos" -type f | wc -l

echo "Annotations:"
find "$EXTRACT_DIR/annotations" -type f | wc -l

echo "=== SAMPLE VIDEOS ==="
find "$EXTRACT_DIR/videos" -type f | head -20

echo "=== SAMPLE ANNOTATIONS ==="
find "$EXTRACT_DIR/annotations" -type f | head -20

echo "=== FINAL STRUCTURE ==="
find "$WORK_DIR" -maxdepth 4 -type d | sort

echo "=== DISK USAGE ==="
du -sh "$WORK_DIR"

echo "=== JOB END ==="
date
