#!/usr/bin/bash

#SBATCH -J summary-json
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/summary-json-173-%A.out

set -e

ANN_DIR="/local_datasets/$USER/KHUDA_173/raw/extracted/173/annotations"

echo "=== JSON SUMMARY START ==="
date
hostname

python - "$ANN_DIR" <<'PY'
import json
import sys
from pathlib import Path
from collections import Counter

ann_dir = Path(sys.argv[1])
json_files = sorted(ann_dir.glob("*.json"))

class_counter = Counter()
frame_counter = Counter()
json_error = []
total_objects = 0
files_with_cur_frame = 0
files_with_bbox = 0

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x)

for p in json_files:
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        json_error.append((str(p), str(e)))
        continue

    has_cur_frame = False
    has_bbox = False

    for d in walk(data):
        if "class_name" in d:
            class_counter[str(d.get("class_name"))] += 1
            total_objects += 1

        if "cur_frame" in d:
            try:
                frame_counter[int(d.get("cur_frame"))] += 1
                has_cur_frame = True
            except Exception:
                pass

        if "bbox" in d:
            has_bbox = True

    if has_cur_frame:
        files_with_cur_frame += 1

    if has_bbox:
        files_with_bbox += 1

print("JSON files:", len(json_files))
print("JSON errors:", len(json_error))
print("Total objects with class_name:", total_objects)
print("Files with cur_frame:", files_with_cur_frame)
print("Files with bbox:", files_with_bbox)

print()
print("=== CLASS NAME COUNTS ===")
for name, cnt in class_counter.most_common():
    print(f"{name}: {cnt}")

print()
print("=== CUR_FRAME RANGE ===")
if frame_counter:
    frames = sorted(frame_counter)
    print("min cur_frame:", frames[0])
    print("max cur_frame:", frames[-1])
    print("unique cur_frame count:", len(frames))
else:
    print("No cur_frame found")

print()
print("=== JSON ERRORS SAMPLE ===")
for path, err in json_error[:20]:
    print(path, err)
PY

echo ""
echo "=== JSON SUMMARY END ==="
date
