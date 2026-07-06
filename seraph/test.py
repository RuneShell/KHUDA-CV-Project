import argparse
import os

import cv2
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader

from dataset import GarbageDumpingClipDataset, collate_fn
# from models.model import VideoMAEMultiHead
from models.model import GarbageDumpingVideoMAE


MANIFEST_DIR = "/data/leecg1219/KHUDA_173/processed_173_manifest/manifests"
VIDEO_ROOT = "/data/philipn337/KHUDA_173/raw/extracted/173/videos"
CHECKPOINT_PATH = "/data/leecg1219/KHUDA_173/checkpoints/best_model.pt"
SAVE_PATH = "result_gradcam.mp4"

CLIP_LEN = 48
MODEL_INPUT_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--manifest_dir", type=str, default=MANIFEST_DIR)
    parser.add_argument("--video_root", type=str, default=VIDEO_ROOT)

    # VIDEO PATH HERE
    parser.add_argument("--video_path", type=str, default=
                        "/data/philipn337/KHUDA_173/raw/extracted/173/videos/C_3_18_smp_cl_09-10_17-21-00_c_aft_DF1.mp4")
    
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    parser.add_argument("--visualize_index", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    # model = VideoMAEMultiHead(pretrained="MCG-NJU/videomae-base", num_frames=CLIP_LEN).to(device)
    model = GarbageDumpingVideoMAE().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    model.eval()
    return model


def build_dataloader(args):
    manifest_path = os.path.join(args.manifest_dir, f"clips_{args.split}_v2.json")
    dataset = GarbageDumpingClipDataset(
        manifest_path,
        split=args.split,
        num_frames=CLIP_LEN,
        image_size=IMAGE_SIZE,
        video_root=args.video_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return dataset, loader


def _sample_indices(start_frame, end_frame, num_samples, total_frames):
    end_frame = max(start_frame + 1, end_frame)
    ids = np.linspace(start_frame, end_frame - 1, num=num_samples)
    ids = np.clip(ids, 0, total_frames - 1)
    return np.round(ids).astype(np.int64)


def _normalize_frames(frames):
    frames = frames.float() / 255.0
    mean = IMAGENET_MEAN.to(frames.device)
    std = IMAGENET_STD.to(frames.device)
    return (frames - mean) / std


def load_video_clips(video_path, clip_len=CLIP_LEN, model_input_frames=MODEL_INPUT_FRAMES,
                     image_size=IMAGE_SIZE, window_stride=CLIP_LEN):
    cap = cv2.VideoCapture(video_path)
    clips = []

    original_frames = []
    resized_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (image_size, image_size))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        original_frames.append(original_frame)
        resized_frames.append(resized_frame)
    cap.release()

    total_frames = len(original_frames)
    if total_frames <= 0:
        return [], 0

    for start in range(0, total_frames, window_stride):
        end = min(start + clip_len, total_frames)
        sampled_ids = _sample_indices(start, end, model_input_frames, total_frames)
        sampled_frames = torch.from_numpy(np.stack([resized_frames[idx] for idx in sampled_ids], axis=0)).permute(0, 3, 1, 2)
        sampled_frames = _normalize_frames(sampled_frames)
        clips.append({
            "pixel_values": sampled_frames,
            "full_frames": original_frames[start:end],
            "frame_indices": sampled_ids,
            "start": start,
            "end": end,
            "video_path": video_path,
        })

    return clips, total_frames


@torch.no_grad()
def run_inference(model, loader, device):
    clip_probs, clip_preds, clip_labels = [], [], []
    clip_ids = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["clip_label"].to(device)

        outputs = model(pixel_values)
        clip_logits = outputs["clip_logits"]
        frame_logits = outputs["frame_logits"]
        probs = torch.softmax(clip_logits, dim=1)[:, 1]
        preds = (probs > 0.5).long()

        clip_probs.extend(probs.detach().cpu().numpy().tolist())
        clip_preds.extend(preds.detach().cpu().numpy().tolist())
        clip_labels.extend(labels.detach().cpu().numpy().tolist())
        clip_ids.extend(batch["clip_id"])

        print(
            "batch:",
            tuple(pixel_values.shape),
            "clip_logits:",
            tuple(clip_logits.shape),
            "frame_logits:",
            tuple(frame_logits.shape),
        )

    return {
        "clip_probs": np.array(clip_probs),
        "clip_preds": np.array(clip_preds),
        "clip_labels": np.array(clip_labels),
        "clip_ids": clip_ids,
    }


def run_video_inference(model, clips, device):
    results = []
    for idx, clip in enumerate(clips):
        pixel_values = clip["pixel_values"].unsqueeze(0).to(device)
        outputs = model(pixel_values)
        clip_logits = outputs["clip_logits"]
        frame_logits = outputs["frame_logits"]
        prob = torch.softmax(clip_logits, dim=1)[0, 1].item()
        pred = int(prob > 0.5)
        results.append({
            "index": idx,
            "prob": prob,
            "pred": pred,
            "frame_logits": frame_logits.squeeze(0).detach().cpu().numpy(),
            "pixel_values": clip["pixel_values"],
            "start": clip["start"],
            "end": clip["end"],
            "frame_indices": clip["frame_indices"],
        })
        print(
            "clip:", idx,
            "range:", (clip["start"], clip["end"]),
            "prob:", f"{prob:.4f}",
            "pred:", pred,
            "clip_logits:", tuple(clip_logits.shape),
            "frame_logits:", tuple(frame_logits.shape),
        )
    return results


def _select_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _denormalize_frames(frames):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=frames.dtype).view(1, 3, 1, 1)
    frames = frames * std + mean
    frames = frames.clamp(0.0, 1.0)
    frames = (frames * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    return frames


def extract_gradcam(model, pixel_values):
    for param in model.backbone.parameters():
        param.requires_grad = True

    target_layer = model.backbone.encoder.layer[-1]
    features, grads = [], []

    def forward_hook(module, input, output):
        features.append(_select_tensor(output).detach())

    def backward_hook(module, grad_input, grad_output):
        grads.append(_select_tensor(grad_output).detach())

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    outputs = model(pixel_values)
    clip_logit = outputs["clip_logits"]
    frame_logits = outputs["frame_logits"]
    model.zero_grad(set_to_none=True)
    clip_score = torch.softmax(clip_logit, dim=1)[0, 1].item()
    frame_scores = torch.sigmoid(frame_logits[0]).detach().cpu().numpy()
    clip_logit[0, 1].backward()

    handle_fw.remove()
    handle_bw.remove()

    activations = features[0].squeeze(0)
    gradients = grads[0].squeeze(0)

    weights = gradients.mean(dim=0)
    cam = torch.relu((activations * weights).sum(dim=-1))

    spatial_tokens = (IMAGE_SIZE // 16) ** 2
    if cam.numel() % spatial_tokens != 0:
        raise RuntimeError(
            f"Unexpected token count for Grad-CAM: {cam.numel()} tokens cannot be split into {spatial_tokens} spatial patches."
        )

    t_tokens = cam.numel() // spatial_tokens
    spatial_side = int(np.sqrt(spatial_tokens))
    cam = cam.view(t_tokens, spatial_side, spatial_side).cpu().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    zoom_t = pixel_values.shape[1] / t_tokens
    grayscale_cam = scipy.ndimage.zoom(cam, (zoom_t, 1, 1), order=1)
    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    return clip_score, frame_scores, grayscale_cam


def extract_gradcam_for_clip(model, pixel_values, segment_len):
    clip_score, frame_scores, grayscale_cam = extract_gradcam(model, pixel_values)
    grayscale_cam = scipy.ndimage.zoom(grayscale_cam, (segment_len / grayscale_cam.shape[0], 1, 1), order=1)
    grayscale_cam = np.clip(grayscale_cam, 0, 1)
    return clip_score, frame_scores, grayscale_cam


def save_gradcam_video(frames, grayscale_cam, save_path, fps, clip_score, frame_scores):
    frames = _denormalize_frames(frames)
    height, width = frames.shape[1], frames.shape[2]

    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_idx, frame in enumerate(frames):
        cam = np.uint8(255 * grayscale_cam[frame_idx])
        cam = cv2.resize(cam, (width, height))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.4, 0)

        score_idx = min(
            int(frame_idx * len(frame_scores) / max(1, len(frames))),
            len(frame_scores) - 1,
        )
        # cv2.putText(
        #     overlay,
        #     f"clip={clip_score:.3f} frame={frame_scores[score_idx]:.3f}",
        #     (20, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (255, 255, 255),
        #     2,
        # )
        writer.write(overlay)

    writer.release()


def main():
    args = parse_args()
    model = load_model(args.checkpoint, args.device)
    if args.video_path is not None:
        clips, total_frames = load_video_clips(args.video_path)
        if not clips:
            raise ValueError(f"No frames could be read from {args.video_path}")

        first_original_frame = clips[0]["full_frames"][0]
        height, width = first_original_frame.shape[0], first_original_frame.shape[1]
        cap = cv2.VideoCapture(args.video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 3.0)
        cap.release()

        writer = cv2.VideoWriter(
            args.save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for clip_idx, clip in enumerate(clips):
            sampled_pixels = clip["pixel_values"].unsqueeze(0).to(args.device)
            clip_score, frame_scores, grayscale_cam = extract_gradcam_for_clip(
                model,
                sampled_pixels,
                len(clip["full_frames"]),
            )
            full_frames = np.array(clip["full_frames"], dtype=np.uint8)
            for frame_idx, frame in enumerate(full_frames):
                cam = np.uint8(255 * grayscale_cam[frame_idx])
                cam = cv2.resize(cam, (width, height))
                heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

                score_idx = min(int(frame_idx * len(frame_scores) / max(1, len(full_frames))), len(frame_scores) - 1)
                cv2.putText(
                    overlay,
                    f"clip={clip_score:.3f} frame={frame_scores[score_idx]:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                writer.write(overlay)

            print("clip:", clip_idx, "range:", (clip["start"], clip["end"]), "score:", f"{clip_score:.4f}")

        writer.release()
        print("video_path:", args.video_path)
        print("num_clips:", len(clips))
        print("total_frames:", total_frames)
    else:
        dataset, loader = build_dataloader(args)

        results = run_inference(model, loader, args.device)
        print("clip_probs:", results["clip_probs"].shape)
        print("clip_preds:", results["clip_preds"].shape)
        print("clip_labels:", results["clip_labels"].shape)

        vis_index = min(max(args.visualize_index, 0), len(dataset) - 1)
        sample = dataset[vis_index]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(args.device)
        clip_score, frame_scores, grayscale_cam = extract_gradcam(model, pixel_values)

        fps = float(dataset.items[vis_index].get("fps", 3.0))
        save_gradcam_video(
            sample["pixel_values"],
            grayscale_cam,
            args.save_path,
            fps,
            clip_score,
            frame_scores,
        )
    print("Saved:", args.save_path)


if __name__ == "__main__":
    main()