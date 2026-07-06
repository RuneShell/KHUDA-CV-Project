import argparse, json, os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from dataset import GarbageDumpingClipDataset, collate_fn
from model import GarbageDumpingVideoMAE

MANIFEST_DIR = "/data/leecg1219/KHUDA_173/processed_173_manifest/manifests"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--manifest_dir", type=str, default=MANIFEST_DIR)
    parser.add_argument("--manifest_version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--video_root", type=str, default=None,
                        help="Aurora: /data/philipn337/KHUDA_173/raw/extracted/173/videos")
    return parser.parse_args()

def load_model(checkpoint_path, device):
    model = GarbageDumpingVideoMAE().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, loader, device):
    all_clip_probs, all_clip_preds, all_clip_labels = [], [], []
    all_frame_probs, all_frame_labels = [], []
    clip_ids = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        clip_labels = batch["clip_label"].to(device)
        frame_labels = batch["frame_labels"]
        outputs = model(pixel_values)
        clip_probs = torch.softmax(outputs["clip_logits"], dim=1)[:, 1]
        frame_probs = torch.sigmoid(outputs["frame_logits"])
        all_clip_probs.extend(clip_probs.cpu().numpy().tolist())
        all_clip_preds.extend((clip_probs > 0.5).long().cpu().numpy().tolist())
        all_clip_labels.extend(clip_labels.cpu().numpy().tolist())
        all_frame_probs.extend(frame_probs.cpu().numpy().tolist())
        all_frame_labels.extend(frame_labels.numpy().tolist())
        clip_ids.extend(batch["clip_id"])
    return {
        "clip_probs": np.array(all_clip_probs),
        "clip_preds": np.array(all_clip_preds),
        "clip_labels": np.array(all_clip_labels),
        "frame_probs": np.array(all_frame_probs),
        "frame_labels": np.array(all_frame_labels),
        "clip_ids": clip_ids,
    }

def compute_video_classification_metrics(results):
    acc = accuracy_score(results["clip_labels"], results["clip_preds"])
    f1 = f1_score(results["clip_labels"], results["clip_preds"])
    try:
        auroc = roc_auc_score(results["clip_labels"], results["clip_probs"])
    except ValueError:
        auroc = float("nan")
    return {"accuracy": acc, "f1": f1, "auroc": auroc}

def compute_frame_level_metrics(results, threshold=0.5):
    preds = (results["frame_probs"] > threshold).astype(int).flatten()
    labels = results["frame_labels"].astype(int).flatten()
    return {"frame_f1": f1_score(labels, preds, zero_division=0)}

def frames_to_intervals(frame_array):
    intervals, start = [], None
    for i, v in enumerate(frame_array):
        if v == 1 and start is None: start = i
        elif v == 0 and start is not None:
            intervals.append((start, i - 1)); start = None
    if start is not None: intervals.append((start, len(frame_array) - 1))
    return intervals

def interval_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1]-a[0]+1) + (b[1]-b[0]+1) - inter
    return inter / union if union > 0 else 0.0

def compute_temporal_iou(results, threshold=0.5, iou_threshold=0.5):
    matched, total_gt = 0, 0
    for pred_row, label_row in zip(results["frame_probs"], results["frame_labels"]):
        pred_binary = (pred_row > threshold).astype(int)
        gt_intervals = frames_to_intervals(label_row.astype(int))
        pred_intervals = frames_to_intervals(pred_binary)
        total_gt += len(gt_intervals)
        for gt in gt_intervals:
            if max((interval_iou(gt, p) for p in pred_intervals), default=0.0) >= iou_threshold:
                matched += 1
    return {"temporal_iou_recall": matched / total_gt if total_gt > 0 else float("nan"),
            "iou_threshold": iou_threshold}

def main():
    args = parse_args()
    suffix = "_v2" if args.manifest_version == "v2" else ""
    test_ds = GarbageDumpingClipDataset(
        os.path.join(args.manifest_dir, f"clips_{args.split}{suffix}.json"),
        split=args.split,
        video_root=args.video_root
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    model = load_model(args.checkpoint, args.device)
    results = run_inference(model, test_loader, args.device)
    video_cls = compute_video_classification_metrics(results)
    frame_m = compute_frame_level_metrics(results, threshold=args.frame_threshold)
    iou_m = compute_temporal_iou(results, threshold=args.frame_threshold, iou_threshold=args.iou_threshold)
    print("=== Video Classification ===")
    print(json.dumps(video_cls, indent=2))
    print("=== Frame-level Event Detection ===")
    print(json.dumps(frame_m, indent=2))
    print("=== Temporal IoU ===")
    print(json.dumps(iou_m, indent=2))
    report = {"split": args.split, "video_classification": video_cls,
              "frame_level": frame_m, "temporal_iou": iou_m}
    report_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_report_{args.split}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"저장: {report_path}")

if __name__ == "__main__":
    main()
