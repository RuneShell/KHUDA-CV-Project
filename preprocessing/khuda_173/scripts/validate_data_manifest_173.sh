#!/usr/bin/bash

#SBATCH -J validate-data-173
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/validate-data-173-%A.out

set -e

ROOT="/data/$USER/KHUDA_173"
MANIFEST_DIR="$ROOT/processed_173_manifest/manifests"
META_DIR="$ROOT/processed_173_manifest/metadata"

echo "=== VALIDATE DATA MANIFEST START ==="
date
hostname

python - "$ROOT" "$MANIFEST_DIR" "$META_DIR" <<'PY'
import json
import sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
meta_dir = Path(sys.argv[3])

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

clips_all = load(manifest_dir / "clips_all.json")
clips_train = load(manifest_dir / "clips_train.json")
clips_val = load(manifest_dir / "clips_val.json")
clips_test = load(manifest_dir / "clips_test.json")
videos_json = load(meta_dir / "videos.json")

errors = []

if len(clips_train) + len(clips_val) + len(clips_test) != len(clips_all):
    errors.append("split count mismatch")

clip_ids = [c["clip_id"] for c in clips_all]
dups = [k for k, v in Counter(clip_ids).items() if v > 1]
if dups:
    errors.append(f"duplicate clip_id count: {len(dups)}")

train_videos = set(c["video_id"] for c in clips_train)
val_videos = set(c["video_id"] for c in clips_val)
test_videos = set(c["video_id"] for c in clips_test)

if train_videos & val_videos:
    errors.append(f"train-val leakage: {len(train_videos & val_videos)}")
if train_videos & test_videos:
    errors.append(f"train-test leakage: {len(train_videos & test_videos)}")
if val_videos & test_videos:
    errors.append(f"val-test leakage: {len(val_videos & test_videos)}")

missing = []
local_paths = []

for c in clips_all:
    p = c["video_path"]

    if p.startswith("/local_datasets"):
        local_paths.append(p)

    if not Path(p).exists():
        missing.append(p)

if local_paths:
    errors.append(f"still has /local_datasets paths: {len(local_paths)}")

if missing:
    errors.append(f"missing video paths: {len(missing)}")

def label_count(clips):
    return dict(Counter(c["label_name"] for c in clips))

print("=== FINAL DATA VALIDATION SUMMARY ===")
print("Status:", "ok" if not errors else "error")
print("Errors:", len(errors))
for e in errors:
    print("ERROR:", e)

print()
print("clips_all:", len(clips_all))
print("clips_train:", len(clips_train))
print("clips_val:", len(clips_val))
print("clips_test:", len(clips_test))

print()
print("label_all:", label_count(clips_all))
print("label_train:", label_count(clips_train))
print("label_val:", label_count(clips_val))
print("label_test:", label_count(clips_test))

print()
print("train_videos:", len(train_videos))
print("val_videos:", len(val_videos))
print("test_videos:", len(test_videos))

print()
print("Sample video paths:")
for c in clips_all[:5]:
    print(c["video_path"])
PY

echo ""
echo "=== VALIDATE DATA MANIFEST END ==="
date
