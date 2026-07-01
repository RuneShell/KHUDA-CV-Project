import os
import cv2
import numpy as np
import torch
import decord
import torchvision.transforms.v2 as TT
import scipy.ndimage

from models.model import VideoMAEMultiHead
BASE_DIR = ""
VIDEO_PATH = os.path.join(BASE_DIR, ".mp4")
SAVE_PATH = os.path.join(BASE_DIR, "result.mp4")

CLIP_LEN = 48
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = ""  # ???: 사용할 가중치 파일 경로

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(
    SAVE_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)



####################
## test code here
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

#모델 로드 및 Hook 설정

model = VideoMAEMultiHead(pretrained="MCG-NJU/videomae-base", num_frames=CLIP_LEN).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

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

cam_min, cam_max = cam.min(), cam.max()
if cam_max > cam_min:
    cam = (cam - cam_min) / (cam_max - cam_min)

zoom_t = total_frames_cv / t_tokens
grayscale_cam = scipy.ndimage.zoom(cam, (zoom_t, 1, 1), order=1)
grayscale_cam = np.clip(grayscale_cam, 0, 1)
####################
    
# TODO
# 영상이랑 model 불러오기
# 영상에 transform 적용해서 tensor로 만들기
# model에 넣어서 grad-cam 구하기







##  finally, grad-cam data


#####################
## YOLO & BoT-SORT here (나중에)
#####################




#####################
## visualization here
#####################


frame_idx = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx >= len(grayscale_cam): # ?
        break




    # get grad-cam
    cam = grayscale_cam[frame_idx]

    # 0~255
    cam = np.uint8(255 * cam)

    # 영상 크기로 resize
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

    # bbox
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

    # ID
    cv2.putText(
        overlay,
        f"ID {track_id}",
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )
    writer.write(overlay)

    frame_idx += 1

cap.release()
writer.release()

print("Saved:", SAVE_PATH)



# 
