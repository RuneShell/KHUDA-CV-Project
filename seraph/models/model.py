# 일단 채경님 노션에 있던거 복붙함

from transformers import VideoMAEModel
import torch.nn as nn

class DumpingDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        # Output 1 - 클립 분류
        self.clip_head = nn.Linear(768, 2)
        # Output 2 - 프레임별 분류
        self.frame_head = nn.Linear(768, 2)

    def forward(self, x):
        output = self.encoder(x)
        cls_token = output.last_hidden_state[:, 0, :]
        frame_tokens = output.last_hidden_state[:, 1:17, :]

        clip_logits = self.clip_head(cls_token)
        frame_logits = self.frame_head(frame_tokens)

        return clip_logits, frame_logits