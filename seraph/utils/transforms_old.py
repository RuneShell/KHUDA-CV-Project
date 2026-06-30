# transforms.py

import os
import random
import numpy as np

import json
import decord # video를 빠르게 읽기 위한 라이브러리. pip install decord

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as TT


class _ClipDataset(Dataset):
    def __init__(self, MANIFEST_PATH, VIDEO_PATH, transform, num_frames):
        with open(MANIFEST_PATH) as f:
            self.clips = json.load(f)
        self.VIDEO_PATH = VIDEO_PATH
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_item = self.clips[idx]

        # get frames
        video_path = clip_item["video_path"]
        vr = decord.VideoReader(video_path)
        start = clip_item["start_frame"]
        end = clip_item["end_frame"]

        ids = range(start, end+1)
        frames = torch.from_numpy(vr.get_batch(ids).asnumpy()).permute(0, 3, 1, 2)  # (T,C,H,W)
        frames = self.transform(frames)

        # get clip label
        clip_label = clip_item["label"] == 1 and "trash_dump" in clip_item["abnormal_classes"]
        clip_label = torch.tensor(int(clip_label), dtype=torch.float32)

        # get frame labels
        frame_labels = torch.tensor(clip_item["frame_labels"], dtype=torch.float32)
        frame_labels = frame_labels.view(-1, 2).amax(dim=1)  # VideoMAE는 frame_labels를 (T, 2)로 받음. 각 frame별로 [0,1] or [1,0]으로 one-hot encoding 되어 있음. 따라서 max(dim=1)으로 0/1로 변환.

        return frames, clip_label, frame_labels


class MyDataset:
    def __init__(self, mode, VIDEO_PATH, JSON_PATH, BATCH_SIZE, image_size=224, num_frames=16, frame_stride=3):

        self.mode = mode
        self.BATCH_SIZE = BATCH_SIZE
        self.image_size = image_size
        self.num_frames = num_frames
        self.frame_stride = frame_stride

        # json
        self.TRAIN_JSON = os.path.join(JSON_PATH, "clips_train.json")
        self.VAL_JSON = os.path.join(JSON_PATH, "clips_val.json")
        self.TEST_JSON = os.path.join(JSON_PATH, "clips_test.json")
        # video 
        self.TRAIN_VIDEO = os.path.join(VIDEO_PATH, "train", "video")
        self.VAL_VIDEO = os.path.join(VIDEO_PATH, "val", "video")
        self.TEST_VIDEO = os.path.join(VIDEO_PATH, "test", "video")
        
        # VideoMAE (Kinetics) 정규화 통계
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        seed = 42
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


        # Transforms
        self.train_transform = TT.Compose([
            TT.ToDtype(torch.float32, scale=True),
            TT.RandomResizedCrop(self.image_size, scale=(0.7, 1.0), antialias=True),
            TT.RandomHorizontalFlip(p=0.5),
            TT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            TT.Normalize(mean=self.MEAN, std=self.STD),
        ])

        self.test_transform = TT.Compose([
            TT.ToDtype(torch.float32, scale=True),
            TT.Resize(self.image_size, antialias=True),
            TT.CenterCrop(self.image_size),
            TT.Normalize(mean=self.MEAN, std=self.STD),
        ])


    def get_train_dataset(self):
        train_set = _ClipDataset(
            self.TRAIN_JSON,
            self.TRAIN_VIDEO,
            self.train_transform,
            self.num_frames,
            self.frame_stride,
        )

        val_set = _ClipDataset(
            self.VAL_JSON,
            self.VAL_VIDEO,
            self.test_transform,
            self.num_frames,
            self.frame_stride,
        )
        train_loader = DataLoader(train_set, batch_size=self.BATCH_SIZE,
                                    shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.BATCH_SIZE,
                                shuffle=False, pin_memory=True)
        return train_loader, val_loader



    def get_test_dataset(self):
        test_set = _ClipDataset(
            self.TEST_JSON,
            self.TEST_VIDEO,
            self.test_transform,
            self.num_frames,
            self.frame_stride,
        )
        test_loader = DataLoader(test_set, batch_size=self.BATCH_SIZE,
                                    shuffle=False, pin_memory=True)
        return test_loader