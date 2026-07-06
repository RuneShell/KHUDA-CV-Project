"""
dataset.py
KHUDA_173 manifest(clips_train/val/test.json) 기반 VideoMAE 클립 데이터셋.

manifest 1개 항목 예시:
{
  "clip_id": "...",
  "video_id": "...",
  "video_path": "/data/leecg1219/KHUDA_173/raw/extracted/173/videos/xxx.mp4",
  "label": 0 or 1,                 # 0 = normal(legal), 1 = abnormal(illegal)
  "label_name": "normal"/"abnormal",
  "abnormal_classes": [...],
  "start_frame": int,
  "end_frame": int,
  "clip_len": 48,                  # 원본 클립 프레임 수 (3fps 기준 약 16초)
  "start_time_sec": float,
  "end_time_sec": float,
  "fps": 3.0,
  "source_events": [...],          # frame-level event 구간 (illegal일 때만 존재)
  "split": "train"/"val"/"test"
}

VideoMAE-base는 16프레임 고정 입력이라, 48프레임 클립에서 균등 간격으로
16프레임을 샘플링한다(stride=3). 모델은 dual head:
  - clip head: binary classification (illegal/legal)
  - frame head: 16프레임 각각에 대한 event 여부(0/1) - source_events를 변환해서 생성
"""

import json
import os
import random

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")

VIDEOMAE_NUM_FRAMES = 16          # HuggingFace VideoMAE-base 고정 입력 프레임 수
VIDEOMAE_IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class GarbageDumpingClipDataset(Dataset):
    """manifest json 1개(clips_train.json 등)를 읽어 클립 단위 샘플을 생성."""

    def __init__(self, manifest_path, split="train", num_frames=VIDEOMAE_NUM_FRAMES,
                 image_size=VIDEOMAE_IMAGE_SIZE, train_augment=None, events_path=None,
                 video_root=None):
        """
        manifest_path : clips_train_v2.json 등 경로
        split         : "train" | "val" | "test"
        events_path   : events_all.json 경로. None이면 manifest와 같은 폴더에서 찾음.
        video_root    : raw 영상 루트 경로. None이면 manifest의 video_path 그대로 사용.
                        서버마다 경로가 다를 때 사용:
                          Moana:  /data/leecg1219/KHUDA_173/raw/extracted/173/videos
                          Aurora: /data/philipn337/KHUDA_173/raw/extracted/173/videos
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        if events_path is None:
            events_path = os.path.join(os.path.dirname(manifest_path), "events_all.json")
        with open(events_path, "r", encoding="utf-8") as f:
            events_list = json.load(f)
        # event_id -> event dict 조회용 lookup 테이블
        self.events_by_id = {ev["event_id"]: ev for ev in events_list}

        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_train = split == "train"
        self.train_augment = train_augment
        self.video_root = video_root  # None이면 manifest의 video_path 그대로 사용

        self._validate_paths()

    def _get_video_path(self, item):
        """video_root가 지정된 경우 파일명만 추출해서 새 경로로 교체."""
        if self.video_root is None:
            return item["video_path"]
        filename = os.path.basename(item["video_path"])
        return os.path.join(self.video_root, filename)

    def _validate_paths(self):
        missing = [self._get_video_path(it) for it in self.items
                   if not os.path.exists(self._get_video_path(it))]
        if missing:
            raise FileNotFoundError(
                f"[{self.split}] {len(missing)}개 비디오 경로 누락. 예: {missing[:3]}"
            )

    def __len__(self):
        return len(self.items)

    def _sample_frame_indices(self, start_frame, end_frame, total_frames_in_video):
        """48프레임 클립 구간 안에서 16프레임을 균등 샘플링."""
        clip_indices = np.linspace(start_frame, end_frame - 1, num=self.num_frames)
        clip_indices = np.clip(clip_indices, 0, total_frames_in_video - 1)
        return clip_indices.astype(np.int64)

    def _build_frame_level_labels(self, item, sampled_indices):
        """
        source_events는 event_id(문자열) 리스트라서, events_by_id에서 실제
        start_frame/end_frame을 조회해 16프레임 샘플 각각에 대한 0/1 라벨로 변환.
        legal 클립이거나 이벤트가 없으면 전부 0.

        주의: events_all.json에는 abnormal_classes에 없는 다른 종류의 이벤트
        (예: smoking)도 섞여 있을 수 있으므로, frame label은 항상 abnormal_classes
        기준이 아니라 source_events에 실제로 연결된 이벤트만 사용한다.
        """
        frame_labels = np.zeros(self.num_frames, dtype=np.float32)
        if item["label"] == 0 or not item.get("source_events"):
            return frame_labels

        for event_id in item["source_events"]:
            event = self.events_by_id.get(event_id)
            if event is None:
                continue  # lookup 실패 시 조용히 skip (manifest와 events 파일 불일치 대비)
            ev_start = event["start_frame"]
            ev_end = event["end_frame"]
            for i, frame_idx in enumerate(sampled_indices):
                if ev_start <= frame_idx <= ev_end:
                    frame_labels[i] = 1.0
        return frame_labels

    def _normalize(self, frames):
        """frames: (T, H, W, C) uint8 torch tensor -> (T, C, H, W) float normalized."""
        frames = frames.float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 1, 1, 3)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 1, 1, 3)
        frames = (frames - mean) / std
        return frames.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

    def __getitem__(self, idx):
        item = self.items[idx]

        vr = VideoReader(self._get_video_path(item), ctx=cpu(0),
                          width=self.image_size, height=self.image_size)
        total_frames = len(vr)

        sampled_indices = self._sample_frame_indices(
            item["start_frame"], item["end_frame"], total_frames
        )

        # train일 때만 약한 temporal jitter (±1 frame) 적용해 과적합 완화
        if self.is_train:
            jitter = np.random.randint(-1, 2, size=sampled_indices.shape)
            sampled_indices = np.clip(sampled_indices + jitter, 0, total_frames - 1)

        frames = vr.get_batch(sampled_indices)  # (T, H, W, C) uint8
        frames = self._normalize(frames)

        if self.is_train and self.train_augment is not None:
            frames = self.train_augment(frames)

        frame_labels = self._build_frame_level_labels(item, sampled_indices)

        sample = {
            "pixel_values": frames,
            "clip_label": torch.tensor(item["label"], dtype=torch.long),
            "frame_labels": torch.from_numpy(frame_labels),
            "clip_id": item["clip_id"],
            "video_path": self._get_video_path(item),
        }
        return sample


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])      # (B, T, C, H, W)
    clip_labels = torch.stack([b["clip_label"] for b in batch])         # (B,)
    frame_labels = torch.stack([b["frame_labels"] for b in batch])      # (B, T)
    clip_ids = [b["clip_id"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "clip_label": clip_labels,
        "frame_labels": frame_labels,
        "clip_id": clip_ids,
    }


if __name__ == "__main__":
    # 빠른 동작 확인용
    manifest_dir = "/data/leecg1219/KHUDA_173/processed_173_manifest/manifests"
    ds = GarbageDumpingClipDataset(os.path.join(manifest_dir, "clips_train.json"), split="train")
    print(f"train clips: {len(ds)}")
    sample = ds[0]
    print("pixel_values:", sample["pixel_values"].shape)
    print("clip_label:", sample["clip_label"])
    print("frame_labels:", sample["frame_labels"])
