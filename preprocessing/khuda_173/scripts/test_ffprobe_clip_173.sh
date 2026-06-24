#!/usr/bin/bash

#SBATCH -J test-ffprobe
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p debug_ugrad
#SBATCH -t 00:30:00
#SBATCH -o /data/%u/seraph_jobs/logs/test-ffprobe-clip-173-%A.out

set -e

MANIFEST="/data/$USER/KHUDA_173/processed_173_manifest/manifests/clips_train.json"

echo "=== TEST FFPROBE CLIP START ==="
date
hostname

python - "$MANIFEST" <<'PY'
import json
import sys
import subprocess
from pathlib import Path

manifest = Path(sys.argv[1])

with open(manifest, encoding="utf-8") as f:
    clips = json.load(f)

clip = clips[0]
video_path = Path(clip["video_path"])

print("manifest:", manifest)
print("num clips:", len(clips))
print("video_path:", video_path)
print("video exists:", video_path.exists())

print("start_frame:", clip["start_frame"])
print("end_frame:", clip["end_frame"])
print("label:", clip["label_name"])

cmd = [
    "ffprobe",
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
    "-of", "json",
    str(video_path),
]

result = subprocess.run(cmd, capture_output=True, text=True)

print("ffprobe returncode:", result.returncode)
print("ffprobe stdout:")
print(result.stdout)

if result.returncode != 0:
    print("ffprobe stderr:")
    print(result.stderr)
    raise SystemExit(1)

print("=== TEST PASSED ===")
PY

echo ""
echo "=== TEST FFPROBE CLIP END ==="
date
