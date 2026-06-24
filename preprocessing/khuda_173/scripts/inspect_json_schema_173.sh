#!/usr/bin/bash

#SBATCH -J inspect-json
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 01:00:00
#SBATCH -o /data/%u/seraph_jobs/logs/inspect-json-173-%A.out

set -e

ANN_DIR="/local_datasets/$USER/KHUDA_173/raw/extracted/173/annotations"

SAMPLE=$(find "$ANN_DIR" -type f -iname "*.json" | head -n 1)

echo "=== SAMPLE JSON FILE ==="
echo "$SAMPLE"

echo ""
echo "=== TOP LEVEL KEYS / STRUCTURE ==="
python - "$SAMPLE" <<'PY'
import json
import sys

p = sys.argv[1]

with open(p, encoding="utf-8") as f:
    data = json.load(f)

def brief(x, depth=0):
    indent = "  " * depth

    if isinstance(x, dict):
        print(f"{indent}dict keys:", list(x.keys())[:50])
        for k, v in list(x.items())[:15]:
            print(f"{indent}- {k}: {type(v).__name__}")
            if depth < 3:
                brief(v, depth + 1)

    elif isinstance(x, list):
        print(f"{indent}list len:", len(x))
        if x:
            print(f"{indent}first item type:", type(x[0]).__name__)
            if depth < 3:
                brief(x[0], depth + 1)

    else:
        print(f"{indent}{repr(x)[:150]}")

brief(data)
PY

echo ""
echo "=== FIRST 200 PRETTY JSON LINES ==="
python -m json.tool "$SAMPLE" | head -200

echo ""
echo "=== INSPECT JSON END ==="
date
