# transforms.py

import json
import os
import random

import decord
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as TT


VIDEO_EXTENSIONS = {".mp4"}


def _resolve_label_json_path(video_path, clip_label):
    base_dir = os.path.dirname(os.path.dirname(video_path))
    label_root = os.path.join(base_dir, "label")
    label_class = "illegal" if int(clip_label) == 1 else "legal"
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    candidates = [
        os.path.join(label_root, label_class, f"{video_name}.json"),
        os.path.join(label_root, f"{video_name}.json"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_frame_annotations(label_json_path, total_frames):
    if label_json_path is None:
        return None

    with open(label_json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    frame_labels = np.zeros(total_frames, dtype=np.float32)

    if "events" in annotation:
        for event in annotation["events"]:
            start_frame = event.get("ev_start_frame")
            end_frame = event.get("ev_end_frame")
            if start_frame is None or end_frame is None:
                continue
            start_frame = max(0, int(start_frame))
            end_frame = min(total_frames - 1, int(end_frame))
            frame_labels[start_frame : end_frame + 1] = 1.0
        return frame_labels

    if "annotations" in annotation:
        for frame_item in annotation["annotations"]:
            cur_frame = frame_item.get("cur_frame")
            if cur_frame is None:
                continue
            cur_frame = int(cur_frame)
            if 0 <= cur_frame < total_frames:
                frame_labels[cur_frame] = 1.0
        return frame_labels

    return None


class _ClipDataset(Dataset):
    def __init__(self, VIDEO_PATH, transform, num_frames):
        self.VIDEO_PATH = VIDEO_PATH
        self.transform = transform
        self.num_frames = num_frames
        self.clips = []

        for class_name, label in [("legal", 0), ("illegal", 1)]:
            class_dir = os.path.join(VIDEO_PATH, class_name)
            for name in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, name)
                ext = os.path.splitext(name)[1].lower()
                if os.path.isfile(path) and ext in VIDEO_EXTENSIONS:
                    self.clips.append({
                        "video_path": path,
                        "label": label,
                    })

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_item = self.clips[idx]

        video_path = clip_item["video_path"]
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        ids = np.linspace(0, total_frames - 1, num=self.num_frames)
        ids = np.round(ids).astype(np.int64)

        frames = torch.from_numpy(vr.get_batch(ids).asnumpy()).permute(0, 3, 1, 2)
        frames = self.transform(frames)

        clip_label = torch.tensor(float(clip_item["label"]), dtype=torch.float32)

        label_json_path = _resolve_label_json_path(video_path, clip_item["label"])
        sampled_frame_labels = _load_frame_annotations(label_json_path, total_frames)

        if sampled_frame_labels is None:
            sampled_frame_labels = np.full(total_frames, clip_label.item(), dtype=np.float32)

        sampled_frame_labels = sampled_frame_labels[ids]
        frame_labels = torch.from_numpy(sampled_frame_labels).float()

        # VideoMAE uses tubelet_size=2, so align frame supervision to the temporal tubelets.
        frame_labels = frame_labels.view(-1, 2).amax(dim=1)

        return frames, clip_label, frame_labels


class MyDataset:
    def __init__(self, mode, BASE_PATH, BATCH_SIZE, image_size, num_frames, frame_stride=3):

        self.mode = mode
        self.BATCH_SIZE = BATCH_SIZE
        self.image_size = image_size
        self.num_frames = num_frames
        self.frame_stride = frame_stride

        self.TRAIN_VIDEO = os.path.join(BASE_PATH, "train", "video")
        self.VAL_VIDEO = os.path.join(BASE_PATH, "val", "video")

        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        seed = 42
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.train_transform = TT.Compose([
            TT.ToDtype(torch.float32, scale=True),
            TT.RandomResizedCrop(self.image_size, scale=(0.7, 1.0), antialias=True),
            TT.RandomHorizontalFlip(p=0.5),
            TT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            TT.Normalize(mean=self.MEAN, std=self.STD),
        ])

        self.val_transform = TT.Compose([
            TT.ToDtype(torch.float32, scale=True),
            TT.Resize(self.image_size, antialias=True),
            TT.CenterCrop(self.image_size),
            TT.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def get_train_dataset(self, train_batch_size=None, val_batch_size=None):
        train_batch_size = self.BATCH_SIZE if train_batch_size is None else train_batch_size
        val_batch_size = self.BATCH_SIZE if val_batch_size is None else val_batch_size

        train_set = _ClipDataset(
            self.TRAIN_VIDEO,
            self.train_transform,
            self.num_frames,
        )

        val_set = _ClipDataset(
            self.VAL_VIDEO,
            self.val_transform,
            self.num_frames,
        )
        train_loader = DataLoader(train_set, batch_size=train_batch_size,
                                    shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=val_batch_size,
                                shuffle=False, pin_memory=True)
        return train_loader, val_loader
