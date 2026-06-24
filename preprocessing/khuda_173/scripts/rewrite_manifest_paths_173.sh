#!/usr/bin/bash

#SBATCH -J rewrite-paths-173
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/rewrite-paths-173-%A.out

set -e

ROOT="/data/$USER/KHUDA_173"
MANIFEST_DIR="$ROOT/processed_173_manifest/manifests"
META_DIR="$ROOT/processed_173_manifest/metadata"
REPORT_DIR="$ROOT/processed_173_manifest/reports"

OLD_PREFIX="/local_datasets/$USER/KHUDA_173"
NEW_PREFIX="/data/$USER/KHUDA_173"

echo "=== REWRITE PATHS START ==="
date
hostname

python - "$MANIFEST_DIR" "$META_DIR" "$REPORT_DIR" "$OLD_PREFIX" "$NEW_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

manifest_dir = Path(sys.argv[1])
meta_dir = Path(sys.argv[2])
report_dir = Path(sys.argv[3])
old_prefix = sys.argv[4]
new_prefix = sys.argv[5]

target_files = [
    manifest_dir / "events_all.json",
    manifest_dir / "clips_all.json",
    manifest_dir / "clips_train.json",
    manifest_dir / "clips_val.json",
    manifest_dir / "clips_test.json",
    meta_dir / "videos.json",
    meta_dir / "preprocessing_config.json",
    report_dir / "validation_report.json",
    report_dir / "clip_statistics.json",
    report_dir / "manifest_validation_report.json",
]

def replace_obj(obj):
    if isinstance(obj, dict):
        return {k: replace_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_obj(x) for x in obj]
    if isinstance(obj, str):
        return obj.replace(old_prefix, new_prefix)
    return obj

for p in target_files:
    if not p.exists():
        print("SKIP missing:", p)
        continue

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    data = replace_obj(data)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("rewritten:", p)

print("OLD_PREFIX:", old_prefix)
print("NEW_PREFIX:", new_prefix)
PY

echo ""
echo "=== REWRITE PATHS END ==="
date
