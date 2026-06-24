#!/usr/bin/bash

#SBATCH -J manifest-173
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 02:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/manifest-173-%A.out

set -e

ROOT="/local_datasets/$USER/KHUDA_173"
VIDEO_DIR="$ROOT/raw/extracted/173/videos"
ANN_DIR="$ROOT/raw/extracted/173/annotations"
OUT_DIR="$ROOT/processed_173_manifest"

mkdir -p "$OUT_DIR/manifests"
mkdir -p "$OUT_DIR/metadata"
mkdir -p "$OUT_DIR/reports"
mkdir -p "$OUT_DIR/logs"

echo "=== MAKE CLIP MANIFEST START ==="
date
hostname

python - "$ROOT" "$VIDEO_DIR" "$ANN_DIR" "$OUT_DIR" <<'PY'
import json
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(sys.argv[1])
VIDEO_DIR = Path(sys.argv[2])
ANN_DIR = Path(sys.argv[3])
OUT_DIR = Path(sys.argv[4])

MANIFEST_DIR = OUT_DIR / "manifests"
META_DIR = OUT_DIR / "metadata"
REPORT_DIR = OUT_DIR / "reports"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 설정값
# =========================
RANDOM_SEED = 42

ABNORMAL_CLASSES = {
    "trash_dump",
    "fliers_action",
    "smoking",
}

NORMAL_CLASSES = {
    "moving",
    "stand",
    "sit_down_floor",
    "sit_down_bench",
}

CLIP_LEN = 48
EVENT_GAP_TOLERANCE = 3
NORMAL_EXCLUSION_BUFFER = 10
NORMAL_TO_ABNORMAL_RATIO = 2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FPS = 3.0

random.seed(RANDOM_SEED)

VIDEO_META_PATH = META_DIR / "videos.json"

if not VIDEO_META_PATH.exists():
    raise FileNotFoundError(f"videos.json not found: {VIDEO_META_PATH}")

with open(VIDEO_META_PATH, encoding="utf-8") as f:
    video_meta_json = json.load(f)

videos_meta = video_meta_json["videos"]
video_meta_by_stem = {v["stem"]: v for v in videos_meta if v.get("stem")}

video_files = sorted(VIDEO_DIR.glob("*.mp4"))
json_files = sorted(ANN_DIR.glob("*.json"))

video_by_stem = {p.stem: p for p in video_files}
json_by_stem = {p.stem: p for p in json_files}

common_stems = sorted(set(video_by_stem) & set(json_by_stem))

print("Video files:", len(video_files))
print("JSON files:", len(json_files))
print("Matched pairs:", len(common_stems))
print("Abnormal classes:", sorted(ABNORMAL_CLASSES))
print("Normal classes:", sorted(NORMAL_CLASSES))
print("Clip length:", CLIP_LEN)
print("Clip seconds:", CLIP_LEN / FPS)
print("Event gap tolerance:", EVENT_GAP_TOLERANCE)
print("Normal exclusion buffer:", NORMAL_EXCLUSION_BUFFER)
print("Normal ratio:", NORMAL_TO_ABNORMAL_RATIO)

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x)

def get_num_frames(stem):
    meta = video_meta_by_stem.get(stem, {})
    nb = meta.get("nb_frames")
    est = meta.get("estimated_frames")

    if nb is not None:
        return int(nb)
    if est is not None:
        return int(est)

    return 370

def load_annotations(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    anns = []

    for d in walk(data):
        if not isinstance(d, dict):
            continue

        cls = d.get("class_name")
        cur_frame = d.get("cur_frame")

        if cls is None or cur_frame is None:
            continue

        try:
            cur_frame = int(cur_frame)
        except Exception:
            continue

        item = {
            "cur_frame": cur_frame,
            "class_name": str(cls),
        }

        if "object_id" in d:
            item["object_id"] = d.get("object_id")

        if "bbox" in d:
            item["bbox"] = d.get("bbox")

        anns.append(item)

    anns.sort(key=lambda x: x["cur_frame"])
    return anns

def make_event(stem, idx, cls, anns, num_frames):
    frames = [max(1, min(int(a["cur_frame"]), num_frames)) for a in anns]

    bbox_frames = []
    for a in anns:
        item = {
            "cur_frame": max(1, min(int(a["cur_frame"]), num_frames)),
        }
        if "bbox" in a:
            item["bbox"] = a["bbox"]
        if "object_id" in a:
            item["object_id"] = a["object_id"]
        bbox_frames.append(item)

    return {
        "event_id": f"{stem}_event_{idx:04d}",
        "video_id": stem,
        "class_name": cls,
        "start_frame": min(frames),
        "end_frame": max(frames),
        "duration_frames": max(frames) - min(frames) + 1,
        "duration_seconds": (max(frames) - min(frames) + 1) / FPS,
        "num_annotated_frames": len(frames),
        "bbox_frames": bbox_frames,
    }

def merge_events(stem, anns, num_frames):
    abnormal_anns = [a for a in anns if a["class_name"] in ABNORMAL_CLASSES]

    if not abnormal_anns:
        return []

    by_class = defaultdict(list)
    for a in abnormal_anns:
        by_class[a["class_name"]].append(a)

    events = []
    event_idx = 0

    for cls, cls_anns in by_class.items():
        cls_anns = sorted(cls_anns, key=lambda x: x["cur_frame"])

        current = []
        last_frame = None

        for a in cls_anns:
            f = max(1, min(int(a["cur_frame"]), num_frames))

            if last_frame is None:
                current = [a]
                last_frame = f
                continue

            if f - last_frame <= EVENT_GAP_TOLERANCE:
                current.append(a)
            else:
                event_idx += 1
                events.append(make_event(stem, event_idx, cls, current, num_frames))
                current = [a]

            last_frame = f

        if current:
            event_idx += 1
            events.append(make_event(stem, event_idx, cls, current, num_frames))

    events.sort(key=lambda e: (e["start_frame"], e["end_frame"], e["class_name"]))
    return events

def make_positive_clip(stem, video_path, event, num_frames):
    ev_start = event["start_frame"]
    ev_end = event["end_frame"]
    ev_center = (ev_start + ev_end) // 2

    start = ev_center - CLIP_LEN // 2
    start = max(1, start)
    start = min(start, max(1, num_frames - CLIP_LEN + 1))

    end = min(num_frames, start + CLIP_LEN - 1)

    return {
        "clip_id": f"{stem}_pos_{event['event_id'].split('_event_')[-1]}",
        "video_id": stem,
        "video_path": str(video_path),
        "label": 1,
        "label_name": "abnormal",
        "abnormal_classes": [event["class_name"]],
        "start_frame": start,
        "end_frame": end,
        "clip_len": end - start + 1,
        "start_time_sec": (start - 1) / FPS,
        "end_time_sec": end / FPS,
        "fps": FPS,
        "source_events": [event["event_id"]],
    }

def overlaps_event_or_buffer(start, end, events):
    for e in events:
        s = e["start_frame"] - NORMAL_EXCLUSION_BUFFER
        t = e["end_frame"] + NORMAL_EXCLUSION_BUFFER
        if not (end < s or start > t):
            return True
    return False

def make_normal_candidates(stem, video_path, events, num_frames):
    candidates = []

    if num_frames < CLIP_LEN:
        return candidates

    stride = CLIP_LEN // 2
    idx = 0

    for start in range(1, num_frames - CLIP_LEN + 2, stride):
        end = start + CLIP_LEN - 1

        if overlaps_event_or_buffer(start, end, events):
            continue

        idx += 1
        candidates.append({
            "clip_id": f"{stem}_neg_candidate_{idx:04d}",
            "video_id": stem,
            "video_path": str(video_path),
            "label": 0,
            "label_name": "normal",
            "abnormal_classes": [],
            "start_frame": start,
            "end_frame": end,
            "clip_len": CLIP_LEN,
            "start_time_sec": (start - 1) / FPS,
            "end_time_sec": end / FPS,
            "fps": FPS,
            "source_events": [],
        })

    return candidates

events_all = []
positive_clips = []
normal_candidates = []

video_event_count = Counter()
class_event_count = Counter()
annotation_class_count = Counter()
videos_without_abnormal = 0
videos_with_abnormal = 0

for i, stem in enumerate(common_stems, 1):
    if i % 200 == 0:
        print(f"manifest progress: {i}/{len(common_stems)}")

    video_path = video_by_stem[stem]
    json_path = json_by_stem[stem]
    num_frames = get_num_frames(stem)

    anns = load_annotations(json_path)

    for a in anns:
        annotation_class_count[a["class_name"]] += 1

    events = merge_events(stem, anns, num_frames)

    if events:
        videos_with_abnormal += 1
    else:
        videos_without_abnormal += 1

    for e in events:
        e["video_path"] = str(video_path)
        e["annotation_path"] = str(json_path)
        e["num_frames"] = num_frames
        events_all.append(e)
        class_event_count[e["class_name"]] += 1

    video_event_count[stem] = len(events)

    for e in events:
        positive_clips.append(make_positive_clip(stem, video_path, e, num_frames))

    cands = make_normal_candidates(stem, video_path, events, num_frames)
    normal_candidates.extend(cands)

target_normal = min(len(normal_candidates), len(positive_clips) * NORMAL_TO_ABNORMAL_RATIO)
normal_clips = random.sample(normal_candidates, target_normal) if target_normal > 0 else []

for i, c in enumerate(normal_clips, 1):
    c["clip_id"] = f"{c['video_id']}_neg_{i:06d}"

clips_all = positive_clips + normal_clips
random.shuffle(clips_all)

# video-level split
video_ids = sorted(set(c["video_id"] for c in clips_all))
random.shuffle(video_ids)

n = len(video_ids)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_videos = set(video_ids[:n_train])
val_videos = set(video_ids[n_train:n_train + n_val])
test_videos = set(video_ids[n_train + n_val:])

def assign_split(video_id):
    if video_id in train_videos:
        return "train"
    if video_id in val_videos:
        return "val"
    return "test"

for c in clips_all:
    c["split"] = assign_split(c["video_id"])

clips_train = [c for c in clips_all if c["split"] == "train"]
clips_val = [c for c in clips_all if c["split"] == "val"]
clips_test = [c for c in clips_all if c["split"] == "test"]

def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

events_path = MANIFEST_DIR / "events_all.json"
clips_all_path = MANIFEST_DIR / "clips_all.json"
clips_train_path = MANIFEST_DIR / "clips_train.json"
clips_val_path = MANIFEST_DIR / "clips_val.json"
clips_test_path = MANIFEST_DIR / "clips_test.json"

dump_json(events_path, events_all)
dump_json(clips_all_path, clips_all)
dump_json(clips_train_path, clips_train)
dump_json(clips_val_path, clips_val)
dump_json(clips_test_path, clips_test)

config = {
    "root": str(ROOT),
    "video_dir": str(VIDEO_DIR),
    "annotation_dir": str(ANN_DIR),
    "abnormal_classes": sorted(ABNORMAL_CLASSES),
    "normal_classes": sorted(NORMAL_CLASSES),
    "clip_len_frames": CLIP_LEN,
    "clip_len_seconds": CLIP_LEN / FPS,
    "fps": FPS,
    "event_gap_tolerance_frames": EVENT_GAP_TOLERANCE,
    "normal_exclusion_buffer_frames": NORMAL_EXCLUSION_BUFFER,
    "normal_to_abnormal_ratio": NORMAL_TO_ABNORMAL_RATIO,
    "split_policy": "video_level_random_split",
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "random_seed": RANDOM_SEED,
    "frame_indexing": "1-based inclusive",
    "actual_video_cutting": False,
}

dump_json(META_DIR / "preprocessing_config.json", config)

def count_labels(clips):
    return dict(Counter(c["label_name"] for c in clips))

def count_video_ids(clips):
    return len(set(c["video_id"] for c in clips))

stats = {
    "video_pairs": len(common_stems),
    "annotation_class_count": dict(annotation_class_count),
    "abnormal_classes": sorted(ABNORMAL_CLASSES),
    "normal_classes": sorted(NORMAL_CLASSES),
    "event_count": len(events_all),
    "event_count_by_class": dict(class_event_count),
    "positive_clip_count": len(positive_clips),
    "normal_candidate_count": len(normal_candidates),
    "normal_clip_count": len(normal_clips),
    "clips_all_count": len(clips_all),
    "clips_train_count": len(clips_train),
    "clips_val_count": len(clips_val),
    "clips_test_count": len(clips_test),
    "train_label_counts": count_labels(clips_train),
    "val_label_counts": count_labels(clips_val),
    "test_label_counts": count_labels(clips_test),
    "train_video_count": count_video_ids(clips_train),
    "val_video_count": count_video_ids(clips_val),
    "test_video_count": count_video_ids(clips_test),
    "videos_with_abnormal_events": videos_with_abnormal,
    "videos_without_abnormal_events": videos_without_abnormal,
    "sample_events": events_all[:5],
    "sample_clips": clips_all[:5],
}

dump_json(REPORT_DIR / "clip_statistics.json", stats)

validation = {
    "video_count": len(video_files),
    "annotation_count": len(json_files),
    "matched_pairs": len(common_stems),
    "videos_without_json": sorted(set(video_by_stem) - set(json_by_stem))[:100],
    "json_without_video": sorted(set(json_by_stem) - set(video_by_stem))[:100],
    "abnormal_classes": sorted(ABNORMAL_CLASSES),
    "clip_len_frames": CLIP_LEN,
    "clip_len_seconds": CLIP_LEN / FPS,
    "event_count": len(events_all),
    "positive_clip_count": len(positive_clips),
    "normal_clip_count": len(normal_clips),
    "notes": [
        "trash_dump, fliers_action, smoking are treated as abnormal.",
        "moving, stand, sit_down_floor, sit_down_bench are treated as normal.",
        "Frame indices are 1-based inclusive.",
        "Actual mp4 files are not cut; clips are represented by frame ranges.",
        "Split is video-level, not clip-level.",
    ],
}

dump_json(REPORT_DIR / "validation_report.json", validation)

print()
print("=== MANIFEST RESULT ===")
print("Events:", len(events_all))
print("Event count by class:", dict(class_event_count))
print("Positive clips:", len(positive_clips))
print("Normal candidates:", len(normal_candidates))
print("Normal clips:", len(normal_clips))
print("All clips:", len(clips_all))
print("Train clips:", len(clips_train))
print("Val clips:", len(clips_val))
print("Test clips:", len(clips_test))

print()
print("=== LABEL COUNTS ===")
print("Train:", count_labels(clips_train))
print("Val:", count_labels(clips_val))
print("Test:", count_labels(clips_test))

print()
print("=== VIDEO COUNTS ===")
print("Train videos:", count_video_ids(clips_train))
print("Val videos:", count_video_ids(clips_val))
print("Test videos:", count_video_ids(clips_test))

print()
print("=== SAVED FILES ===")
for p in [
    events_path,
    clips_all_path,
    clips_train_path,
    clips_val_path,
    clips_test_path,
    META_DIR / "preprocessing_config.json",
    REPORT_DIR / "clip_statistics.json",
    REPORT_DIR / "validation_report.json",
]:
    print(p)

PY

echo ""
echo "=== OUTPUT FILES ==="
find "$OUT_DIR" -maxdepth 3 -type f | sort

echo ""
echo "=== MAKE CLIP MANIFEST END ==="
date
