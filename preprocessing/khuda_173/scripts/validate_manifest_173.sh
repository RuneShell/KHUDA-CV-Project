#!/usr/bin/bash

#SBATCH -J validate-manifest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/validate-manifest-173-%A.out

set -e

ROOT="/local_datasets/$USER/KHUDA_173"
MANIFEST_DIR="$ROOT/processed_173_manifest/manifests"
META_DIR="$ROOT/processed_173_manifest/metadata"
REPORT_DIR="$ROOT/processed_173_manifest/reports"

echo "=== VALIDATE MANIFEST START ==="
date
hostname

python - "$ROOT" "$MANIFEST_DIR" "$META_DIR" "$REPORT_DIR" <<'PY'
import json
import sys
from pathlib import Path
from collections import Counter

root = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
meta_dir = Path(sys.argv[3])
report_dir = Path(sys.argv[4])

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

events = load(manifest_dir / "events_all.json")
clips_all = load(manifest_dir / "clips_all.json")
clips_train = load(manifest_dir / "clips_train.json")
clips_val = load(manifest_dir / "clips_val.json")
clips_test = load(manifest_dir / "clips_test.json")
videos_json = load(meta_dir / "videos.json")

video_meta = {v["stem"]: v for v in videos_json["videos"] if v.get("stem")}

errors = []
warnings = []

# split 개수 검증
if len(clips_train) + len(clips_val) + len(clips_test) != len(clips_all):
    errors.append("train/val/test split count does not match clips_all")

# clip_id 중복 검증
clip_ids = [c["clip_id"] for c in clips_all]
dups = [k for k, v in Counter(clip_ids).items() if v > 1]
if dups:
    errors.append(f"duplicate clip_id count: {len(dups)}")

# video leakage 검증
train_videos = set(c["video_id"] for c in clips_train)
val_videos = set(c["video_id"] for c in clips_val)
test_videos = set(c["video_id"] for c in clips_test)

if train_videos & val_videos:
    errors.append(f"train-val video leakage: {len(train_videos & val_videos)}")
if train_videos & test_videos:
    errors.append(f"train-test video leakage: {len(train_videos & test_videos)}")
if val_videos & test_videos:
    errors.append(f"val-test video leakage: {len(val_videos & test_videos)}")

# frame range / video path 검증
bad_frame = []
missing_video_path = []

for c in clips_all:
    vid = c["video_id"]
    meta = video_meta.get(vid)
    if meta is None:
        bad_frame.append({"clip_id": c["clip_id"], "error": "missing video metadata"})
        continue

    num_frames = meta.get("nb_frames") or meta.get("estimated_frames") or 370
    num_frames = int(num_frames)

    start = int(c["start_frame"])
    end = int(c["end_frame"])

    if start < 1 or end < start or end > num_frames:
        bad_frame.append({
            "clip_id": c["clip_id"],
            "video_id": vid,
            "start_frame": start,
            "end_frame": end,
            "num_frames": num_frames,
        })

    if not Path(c["video_path"]).exists():
        missing_video_path.append(c["video_path"])

if bad_frame:
    errors.append(f"bad frame range clips: {len(bad_frame)}")
if missing_video_path:
    errors.append(f"missing video path clips: {len(missing_video_path)}")

# positive clip source event 검증
event_ids = set(e["event_id"] for e in events)
missing_events = []

for c in clips_all:
    if c["label"] == 1:
        for eid in c.get("source_events", []):
            if eid not in event_ids:
                missing_events.append({"clip_id": c["clip_id"], "missing_event_id": eid})

if missing_events:
    errors.append(f"missing source events: {len(missing_events)}")

def label_count(clips):
    return dict(Counter(c["label_name"] for c in clips))

summary = {
    "status": "ok" if not errors else "error",
    "errors": errors,
    "warnings": warnings,
    "counts": {
        "events": len(events),
        "clips_all": len(clips_all),
        "clips_train": len(clips_train),
        "clips_val": len(clips_val),
        "clips_test": len(clips_test),
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "test_videos": len(test_videos),
    },
    "label_counts": {
        "all": label_count(clips_all),
        "train": label_count(clips_train),
        "val": label_count(clips_val),
        "test": label_count(clips_test),
    },
    "bad_frame_samples": bad_frame[:20],
    "missing_video_path_samples": missing_video_path[:20],
    "missing_source_event_samples": missing_events[:20],
}

out_path = report_dir / "manifest_validation_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("=== VALIDATION SUMMARY ===")
print("Status:", summary["status"])
print("Errors:", len(errors))
for e in errors:
    print("ERROR:", e)

print()
print("Counts:", summary["counts"])
print("Label counts:", summary["label_counts"])
print("Saved:", out_path)
PY

echo "=== VALIDATE MANIFEST END ==="
date
