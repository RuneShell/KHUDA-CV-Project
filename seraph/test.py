import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import decord
import torchvision.transforms.v2 as TT
import scipy.ndimage

from models.model import VideoMAEMultiHead


#####################
# Configuration
#####################
BASE_DIR = ""
VIDEO_PATH = os.path.join(BASE_DIR, "C_30_3_smp_su_09-11_16-23-00_a_aft_DF2.mp4")
SAVE_PATH = os.path.join(BASE_DIR, "result.mp4")

CLIP_LEN = 48
IMAGE_SIZE = 224
DEVICE = "cuda"
# CHECKPOINT_PATH = ""  # ???: 사용할 가중치 파일 경로

from dataset import GarbageDumpingClipDataset
ds = GarbageDumpingClipDataset("/data/leecg1219/KHUDA_173/processed_173_manifest/manifests/clips_test_v2.json", 
                               split="test", 
                               video_root="/data/philipn337/KHUDA_173/raw/extracted/173/videos")


# 채경님 model 로드
from models.model import GarbageDumpingVideoMAE
model = GarbageDumpingVideoMAE()
ckpt = torch.load(
    "/data/leecg1219/KHUDA_173/checkpoints/best_model.pt",
    map_location="cuda"
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()


########################
# load video
cap = cv2.VideoCapture(VIDEO_PATH)
# video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


####################
## test code here
###################
vr = decord.VideoReader(VIDEO_PATH)
total_frames = len(vr)

ids = np.linspace(0, total_frames - 1, num=CLIP_LEN)
ids = np.round(ids).astype(np.int64)

frames = torch.from_numpy(vr.get_batch(ids).asnumpy()).permute(0, 3, 1, 2)

val_transform = TT.Compose([
    TT.ToDtype(torch.float32, scale=True),
    TT.Resize(IMAGE_SIZE, antialias=True),
    TT.CenterCrop(IMAGE_SIZE),
    TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = val_transform(frames).unsqueeze(0).to(DEVICE) # (1, T, C, H, W)


# 모델 및 Hook 설정
for p in model.backbone.parameters():
    p.requires_grad = True

target_layer = model.backbone.encoder.layer[-1]
features, grads = [], []

def forward_hook(module, input, output):
    features.append(output[0].detach())

def backward_hook(module, grad_input, grad_output):
    grads.append(grad_output[0].detach())

handle_fw = target_layer.register_forward_hook(forward_hook)
handle_bw = target_layer.register_full_backward_hook(backward_hook)


# Forward & Backward (Grad-CAM 추출)
clip_logit, frame_logits = model(input_tensor)
model.zero_grad()
# debugging
clip_score = torch.sigmoid(clip_logit[0]).item()
frame_scores = torch.sigmoid(frame_logits[0]).detach().cpu().numpy()
print("clip_logit shape:", tuple(clip_logit.shape))
print("frame_logits shape:", tuple(frame_logits.shape))
print("clip_score:", clip_score)
print("frame_scores shape:", frame_scores.shape)

score = clip_logit[0]
score.backward()

handle_fw.remove()
handle_bw.remove()

act = features[0].squeeze(0)  
grad = grads[0].squeeze(0)    

weights = grad.mean(dim=0)
cam = (act * weights).sum(dim=-1)
cam = torch.relu(cam) 

t_tokens = CLIP_LEN // 2            
h_tokens = IMAGE_SIZE // 16         
w_tokens = IMAGE_SIZE // 16         

cam = cam.view(t_tokens, h_tokens, w_tokens).cpu().numpy()
# 0-1 min-max normalization & resize to original video size
cam_min, cam_max = cam.min(), cam.max()
if cam_max > cam_min:
    cam = (cam - cam_min) / (cam_max - cam_min)

zoom_t = total_frames / t_tokens
grayscale_cam = scipy.ndimage.zoom(cam, (zoom_t, 1, 1), order=1)
grayscale_cam = np.clip(grayscale_cam, 0, 1)


#####################
## YOLO & BoT-SORT here (나중에)
#####################

# 추가할 예정


#####################
## visualization here
#####################

writer = cv2.VideoWriter(
    SAVE_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(grayscale_cam):
        break


    # get grad-cam
    cam = grayscale_cam[frame_idx]
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (width, height))

    # Heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(
        frame,
        0.6,
        heatmap,
        0.4,
        0
    )

    score_idx = min(
        int(frame_idx * len(frame_scores) / max(1, len(grayscale_cam))),
        len(frame_scores) - 1,
    )
    cv2.putText(
        overlay,
        f"clip={clip_score:.3f} frame={frame_scores[score_idx]:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    #= Yolo bbox =
    # cv2.rectangle(
    #     overlay,
    #     (x1, y1),
    #     (x2, y2),
    #     (0,255,0),
    #     2
    # )
    # = tracking ID =
    # cv2.putText(
    #     overlay,
    #     f"ID {track_id}",
    #     (x1, y1-10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6,
    #     (0,255,0),
    #     2
    # )


    writer.write(overlay)
    frame_idx += 1


cap.release()
writer.release()
print("Saved:", SAVE_PATH)



# 추후 task : Gradio | streamlit으로 시각화.