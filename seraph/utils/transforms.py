# transforms.py
# version 2

import json
import os
import random

import decord
import numpy as np
import torch
import torchvision.transforms.v2 as TT
from torch.utils.data import DataLoader, Dataset

from dataset import GarbageDumpingClipDataset


VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_MANIFEST_DIR = "/data/leecg1219/KHUDA_173/processed_173_manifest/manifests"
DEFAULT_TRAIN_MANIFEST = "clips_train_v2.json"
DEFAULT_VAL_MANIFEST = "clips_val_v2.json"


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


def _video_batch_to_tensor(batch):
    if hasattr(batch, "asnumpy"):
        batch = batch.asnumpy()
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    return torch.from_numpy(batch)


class _FolderClipDataset(Dataset):
    def __init__(self, split_root, transform, num_frames):
        self.split_root = split_root
        self.transform = transform
        self.num_frames = num_frames
        self.clips = []

        for class_name, label in [("legal", 0), ("illegal", 1)]:
            class_dir = os.path.join(split_root, "video", class_name)
            if not os.path.isdir(class_dir):
                continue
            for name in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, name)
                ext = os.path.splitext(name)[1].lower()
                if os.path.isfile(path) and ext in VIDEO_EXTENSIONS:
                    try:
                        vr = decord.VideoReader(path)
                        if len(vr) == 0:
                            continue
                    except Exception:
                        continue
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

        frames = _video_batch_to_tensor(vr.get_batch(ids)).permute(0, 3, 1, 2)
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
    def __init__(self, mode, BASE_PATH, BATCH_SIZE, image_size, num_frames, frame_stride=3,
                 manifest_dir=DEFAULT_MANIFEST_DIR, video_root=None):
        self.mode = mode
        self.BATCH_SIZE = BATCH_SIZE
        self.image_size = image_size
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.BASE_PATH = BASE_PATH
        self.manifest_dir = manifest_dir
        self.video_root = video_root

        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        seed = 42
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.train_transform = TT.Compose([
            TT.RandomResizedCrop(self.image_size, scale=(0.7, 1.0), antialias=True),
            TT.RandomHorizontalFlip(p=0.5),
            TT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

        self.val_transform = TT.Compose([
            TT.Resize(self.image_size, antialias=True),
            TT.CenterCrop(self.image_size),
        ])

    def _manifest_path(self, filename):
        return os.path.join(self.manifest_dir, filename)

    def _tuple_collate_fn(self, batch):
        frames = torch.stack([item[0] for item in batch])
        clip_labels = torch.stack([item[1] for item in batch])
        frame_labels = torch.stack([item[2] for item in batch])
        return frames, clip_labels, frame_labels

    def _build_manifest_dataset(self, manifest_path, split, train_augment=None):
        return GarbageDumpingClipDataset(
            manifest_path,
            split=split,
            num_frames=self.num_frames,
            image_size=self.image_size,
            train_augment=train_augment,
            video_root=self.video_root,
        )

    def get_train_dataset(self, train_batch_size=None, val_batch_size=None):
        train_batch_size = self.BATCH_SIZE if train_batch_size is None else train_batch_size
        val_batch_size = self.BATCH_SIZE if val_batch_size is None else val_batch_size

        train_manifest = self._manifest_path(DEFAULT_TRAIN_MANIFEST)
        val_manifest = self._manifest_path(DEFAULT_VAL_MANIFEST)

        if os.path.exists(train_manifest) and os.path.exists(val_manifest):
            train_set = self._build_manifest_dataset(
                train_manifest,
                split="train",
                train_augment=self.train_transform,
            )
            val_set = self._build_manifest_dataset(
                val_manifest,
                split="val",
                train_augment=None,
            )
        else:
            train_set = _FolderClipDataset(
                os.path.join(self.BASE_PATH, "train"),
                TT.Compose([
                    TT.ToDtype(torch.float32, scale=True),
                    self.train_transform,
                    TT.Normalize(mean=self.MEAN, std=self.STD),
                ]),
                self.num_frames,
            )
            val_set = _FolderClipDataset(
                os.path.join(self.BASE_PATH, "val"),
                TT.Compose([
                    TT.ToDtype(torch.float32, scale=True),
                    self.val_transform,
                    TT.Normalize(mean=self.MEAN, std=self.STD),
                ]),
                self.num_frames,
            )

        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self._tuple_collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self._tuple_collate_fn,
        )
        return train_loader, val_loader
