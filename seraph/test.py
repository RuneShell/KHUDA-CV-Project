import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

BASE_DIR = ""
VIDEO_PATH = os.path.join(BASE_DIR, ".mp4")
SAVE_PATH = os.path.join(BASE_DIR, "result.mp4")


cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    SAVE_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)



####################
## test code here
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