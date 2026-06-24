#!/usr/bin/bash

#SBATCH -J test-open-clip
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p debug_ugrad
#SBATCH -t 00:30:00
#SBATCH -o /data/%u/seraph_jobs/logs/test-open-clip-173-%A.out

set -e

MANIFEST="/data/$USER/KHUDA_173/processed_173_manifest/manifests/clips_train.json"

echo "=== TEST OPEN CLIP START ==="
date
hostname

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])

with open(manifest_path, encoding="utf-8") as f:
    clips = json.load(f)

print("manifest:", manifest_path)
print("num clips:", len(clips))

clip = clips[0]
print("sample clip:")
print(json.dumps(clip, ensure_ascii=False, indent=2))

video_path = Path(clip["video_path"])
print("video exists:", video_path.exists())
print("video path:", video_path)

try:
    import cv2
except Exception as e:
    print("ERROR: cv2 import failed")
    print(e)
    raise SystemExit(1)

cap = cv2.VideoCapture(str(video_path))

print("cap opened:", cap.isOpened())

if not cap.isOpened():
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("fps:", fps)
print("frame_count:", frame_count)
print("width:", width)
print("height:", height)

# manifest는 1-based frame index이므로 OpenCV 위치는 start_frame - 1
start_frame = int(clip["start_frame"])
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

ret, frame = cap.read()
print("read start_frame:", start_frame)
print("read success:", ret)

if ret:
    print("frame shape:", frame.shape)

cap.release()

print("=== TEST PASSED ===")
PY

echo ""
echo "=== TEST OPEN CLIP END ==="
date
