#!/usr/bin/bash

#SBATCH -J analyze-173
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 02:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/analyze-173-%A.out

set -e

ROOT="/local_datasets/$USER/KHUDA_173"
VIDEO_DIR="$ROOT/raw/extracted/173/videos"
ANN_DIR="$ROOT/raw/extracted/173/annotations"
OUT_DIR="$ROOT/processed_173_manifest"

mkdir -p "$OUT_DIR/manifests"
mkdir -p "$OUT_DIR/metadata"
mkdir -p "$OUT_DIR/reports"
mkdir -p "$OUT_DIR/logs"

echo "=== ANALYZE 173 START ==="
date
hostname

echo "ROOT: $ROOT"
echo "VIDEO_DIR: $VIDEO_DIR"
echo "ANN_DIR: $ANN_DIR"
echo "OUT_DIR: $OUT_DIR"

python - "$VIDEO_DIR" "$ANN_DIR" "$OUT_DIR" <<'PY'
import json
import sys
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

video_dir = Path(sys.argv[1])
ann_dir = Path(sys.argv[2])
out_dir = Path(sys.argv[3])

metadata_dir = out_dir / "metadata"
reports_dir = out_dir / "reports"
metadata_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

video_files = sorted(video_dir.glob("*.mp4"))
json_files = sorted(ann_dir.glob("*.json"))

print("=== BASIC COUNTS ===")
print("Videos:", len(video_files))
print("Annotations:", len(json_files))

video_stems = {p.stem for p in video_files}
json_stems = {p.stem for p in json_files}

videos_without_json = sorted(video_stems - json_stems)
json_without_video = sorted(json_stems - video_stems)

print("Videos without JSON:", len(videos_without_json))
print("JSON without Video:", len(json_without_video))

def parse_rate(rate):
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        a, b = rate.split("/")
        try:
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(rate)
    except Exception:
        return None

def ffprobe_video(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "json",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return {
                "ok": False,
                "error": "no video stream",
                "path": str(path),
            }

        s = streams[0]
        fps = parse_rate(s.get("avg_frame_rate")) or parse_rate(s.get("r_frame_rate"))

        duration = s.get("duration")
        try:
            duration = float(duration) if duration is not None else None
        except Exception:
            duration = None

        nb_frames = s.get("nb_frames")
        try:
            nb_frames = int(nb_frames) if nb_frames is not None and nb_frames != "N/A" else None
        except Exception:
            nb_frames = None

        estimated_frames = None
        if nb_frames is None and duration is not None and fps is not None:
            estimated_frames = round(duration * fps)

        return {
            "ok": True,
            "path": str(path),
            "file_name": path.name,
            "stem": path.stem,
            "width": int(s["width"]) if s.get("width") else None,
            "height": int(s["height"]) if s.get("height") else None,
            "r_frame_rate": s.get("r_frame_rate"),
            "avg_frame_rate": s.get("avg_frame_rate"),
            "fps": fps,
            "duration": duration,
            "nb_frames": nb_frames,
            "estimated_frames": estimated_frames,
        }

    except Exception as e:
        return {
            "ok": False,
            "path": str(path),
            "file_name": path.name,
            "stem": path.stem,
            "error": str(e),
        }

print()
print("=== FFPROBE VIDEOS ===")

videos_meta = []
for i, vp in enumerate(video_files, 1):
    if i % 100 == 0:
        print(f"ffprobe progress: {i}/{len(video_files)}")
    videos_meta.append(ffprobe_video(vp))

ok_videos = [v for v in videos_meta if v.get("ok")]
bad_videos = [v for v in videos_meta if not v.get("ok")]

fps_counter = Counter(str(v.get("fps")) for v in ok_videos)
resolution_counter = Counter(f"{v.get('width')}x{v.get('height')}" for v in ok_videos)

frame_values = []
for v in ok_videos:
    nf = v.get("nb_frames") or v.get("estimated_frames")
    if nf is not None:
        frame_values.append(nf)

print("ffprobe ok:", len(ok_videos))
print("ffprobe failed:", len(bad_videos))

print()
print("=== FPS COUNTS ===")
for k, v in fps_counter.most_common():
    print(k, v)

print()
print("=== RESOLUTION COUNTS ===")
for k, v in resolution_counter.most_common():
    print(k, v)

print()
print("=== VIDEO FRAME SUMMARY ===")
if frame_values:
    print("min frames:", min(frame_values))
    print("max frames:", max(frame_values))
    print("avg frames:", sum(frame_values) / len(frame_values))
else:
    print("No frame count available")

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x)

print()
print("=== PARSE JSON ANNOTATIONS ===")

class_counter = Counter()
file_class_counter = defaultdict(Counter)
cur_frame_counter = Counter()
json_errors = []
bbox_errors = []
events_by_file = {}

for i, jp in enumerate(json_files, 1):
    if i % 200 == 0:
        print(f"json progress: {i}/{len(json_files)}")

    try:
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        json_errors.append({"path": str(jp), "error": str(e)})
        continue

    anns = []

    for d in walk(data):
        if not isinstance(d, dict):
            continue

        cls = d.get("class_name")
        cur_frame = d.get("cur_frame")
        bbox = d.get("bbox")

        if cls is not None:
            cls = str(cls)
            class_counter[cls] += 1
            file_class_counter[jp.stem][cls] += 1

        if cur_frame is not None:
            try:
                cur_frame = int(cur_frame)
                cur_frame_counter[cur_frame] += 1
            except Exception:
                continue

            item = {
                "cur_frame": cur_frame,
                "class_name": cls,
            }

            if bbox is not None:
                item["bbox"] = bbox
                if not isinstance(bbox, list):
                    bbox_errors.append({
                        "json": str(jp),
                        "cur_frame": cur_frame,
                        "class_name": cls,
                        "bbox": bbox,
                        "error": "bbox is not list",
                    })

            anns.append(item)

    events_by_file[jp.stem] = anns

print("JSON parse errors:", len(json_errors))
print("Total class_name annotations:", sum(class_counter.values()))

print()
print("=== CLASS NAME COUNTS ===")
for name, cnt in class_counter.most_common():
    print(f"{name}: {cnt}")

print()
print("=== CUR_FRAME SUMMARY ===")
if cur_frame_counter:
    frames = sorted(cur_frame_counter)
    print("min cur_frame:", frames[0])
    print("max cur_frame:", frames[-1])
    print("unique cur_frame count:", len(frames))
else:
    print("No cur_frame found")

print()
print("=== SAMPLE VIDEO META ===")
for v in videos_meta[:5]:
    print(json.dumps(v, ensure_ascii=False))

print()
print("=== SAMPLE ANNOTATION FRAMES ===")
sample_items = list(events_by_file.items())[:3]
for stem, anns in sample_items:
    print(stem, "ann_count:", len(anns), "sample:", anns[:5])

class_map = {
    "normal": 0,
}

abnormal_classes = sorted([c for c in class_counter.keys() if c not in {"normal", "None", ""}])
for idx, c in enumerate(abnormal_classes, start=1):
    class_map[c] = idx

videos_json = {
    "root": str(video_dir),
    "count": len(videos_meta),
    "videos": videos_meta,
}

class_map_json = {
    "class_map": class_map,
    "class_counts": dict(class_counter),
}

summary_json = {
    "video_count": len(video_files),
    "annotation_count": len(json_files),
    "videos_without_json_count": len(videos_without_json),
    "json_without_video_count": len(json_without_video),
    "videos_without_json": videos_without_json[:100],
    "json_without_video": json_without_video[:100],
    "ffprobe_ok": len(ok_videos),
    "ffprobe_failed": len(bad_videos),
    "ffprobe_failed_samples": bad_videos[:20],
    "fps_counts": dict(fps_counter),
    "resolution_counts": dict(resolution_counter),
    "frame_summary": {
        "min": min(frame_values) if frame_values else None,
        "max": max(frame_values) if frame_values else None,
        "avg": sum(frame_values) / len(frame_values) if frame_values else None,
    },
    "json_parse_errors": json_errors[:50],
    "class_counts": dict(class_counter),
    "cur_frame_summary": {
        "min": min(cur_frame_counter) if cur_frame_counter else None,
        "max": max(cur_frame_counter) if cur_frame_counter else None,
        "unique_count": len(cur_frame_counter),
    },
    "bbox_error_samples": bbox_errors[:50],
}

with open(metadata_dir / "videos.json", "w", encoding="utf-8") as f:
    json.dump(videos_json, f, ensure_ascii=False, indent=2)

with open(metadata_dir / "class_map.json", "w", encoding="utf-8") as f:
    json.dump(class_map_json, f, ensure_ascii=False, indent=2)

with open(reports_dir / "annotation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, ensure_ascii=False, indent=2)

print()
print("=== SAVED FILES ===")
print(metadata_dir / "videos.json")
print(metadata_dir / "class_map.json")
print(reports_dir / "annotation_summary.json")

PY

echo ""
echo "=== OUTPUT FILES ==="
find "$OUT_DIR" -maxdepth 3 -type f | sort

echo ""
echo "=== ANALYZE 173 END ==="
date
