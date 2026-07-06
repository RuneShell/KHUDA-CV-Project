# model.py

import torch
import torch.nn as nn
from transformers import VideoMAEModel


VIDEOMAE_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"



class VideoMAEMultiHead(nn.Module):
    def __init__(self, pretrained, num_frames):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(
            pretrained,
            num_frames=num_frames,
            ignore_mismatched_sizes=True,
        )
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable()
        hidden = self.backbone.config.hidden_size # 768
        self.num_frames = num_frames
        self.t_tokens = num_frames // self.backbone.config.tubelet_size # VideoMAE는 tubelet_size=2로 시간축을 2프레임씩 묶음 → T_token = num_frames // 2

        self.clip_head = nn.Linear(hidden, 1) # Clip이 무단투기이다/정상이다
        self.frame_head = nn.Linear(hidden, 1) # Frame별로 무단투기이다/정상이다 (frame 개수만큼)

    def forward(self, x):
        # x: (B, T, C, H, W)
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if backbone_trainable:
            out = self.backbone(x).last_hidden_state # (B, N, D), N = t_tokens * patches
        else:
            with torch.no_grad():
                out = self.backbone(x).last_hidden_state # (B, N, D), N = t_tokens * patches
        B, N, D = out.shape
        patches = N // self.t_tokens

        # clip head: 전체 토큰 평균 풀링
        clip_logit = self.clip_head(out.mean(dim=1)).squeeze(-1)  # (B, num_classes)

        # frame head: 시간 토큰별 공간 패치 평균 → 프레임별 logit
        tokens = out.view(B, self.t_tokens, patches, D).mean(dim=2)  # (B, t_tokens, D)
        frame_logits = self.frame_head(tokens).squeeze(-1)  # (B, t_tokens)

        return clip_logit, frame_logits
    
    # progressive unfreezing methods
    def unfreeze_heads(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.clip_head.parameters():
            p.requires_grad = True
        for p in self.frame_head.parameters():
            p.requires_grad = True

    def unfreeze_last_layers(self, num_layers=1):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.backbone.encoder.layer[-num_layers:].parameters():
            p.requires_grad = True
        for p in self.clip_head.parameters():
            p.requires_grad = True
        for p in self.frame_head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


