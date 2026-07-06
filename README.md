# CCTV 기반 쓰레기 무단 투기 행동 탐지

고정형 CCTV 영상에서 쓰레기 무단 투기 상황을 탐지하는 비디오 분석 프로젝트입니다.  
본 프로젝트는 단순히 쓰레기 객체를 찾는 것이 아니라, 사람이 물체를 버리고 이탈하는 **시간적 행동 패턴**을 VideoMAE 기반 비디오 모델로 판단하고, YOLOv8을 이용해 실제 투기 순간의 사람 bbox를 시각화합니다.

## 1. 프로젝트 개요

쓰레기 무단 투기 감시 시스템은 CCTV와 객체 탐지 모델을 활용하는 경우가 많습니다. 하지만 기존 방식은 대체로 사람과 쓰레기 객체의 위치 관계나 거리 변화에 의존하기 때문에 실제 환경에서 오탐이 발생하기 쉽습니다.

예를 들어 다음과 같은 상황은 단순 거리 기반 규칙으로 판단하기 어렵습니다.

- 사람이 물건을 잠시 내려놓는 경우
- 쓰레기통 근처를 지나가는 경우
- 쓰레기 객체가 작거나 가려지는 경우
- CCTV 배경, 각도, 조명 변화가 큰 경우
- 사람과 물체가 멀어졌지만 실제 투기가 아닌 경우

따라서 본 프로젝트는 무단 투기를 **객체 탐지 문제**가 아니라 **비디오 행동 인식 문제**로 정의했습니다.

```text
입력: CCTV 영상
출력:
  1. 무단 투기 의심 구간
  2. 실제 투기 순간의 사람 bbox 시각화
  3. 결과 영상(mp4)
  4. 탐지 정보(json)
```

## 2. 핵심 아이디어

본 프로젝트의 핵심은 두 단계로 구성됩니다.

```text
CCTV 영상
    ↓
VideoMAE
    ↓
무단 투기 의심 구간 탐지
    ↓
YOLOv8
    ↓
의심 구간 내 투기 순간 탐지
    ↓
사람 bbox 강조 + 결과 영상/JSON 생성
```

### VideoMAE

VideoMAE는 여러 프레임으로 구성된 비디오 클립을 입력받아 시간적 변화를 학습하는 모델입니다.  
본 프로젝트에서는 VideoMAE를 이용해 영상 클립이 정상 상황인지, 무단 투기 상황인지 분류합니다.

### YOLOv8

YOLOv8은 VideoMAE가 찾은 의심 구간 안에서 실제 투기 순간의 사람 영역을 찾고 시각화하는 데 사용됩니다.  
정상 구간에서는 bbox를 파란색으로, 무단 투기 발생 구간에서는 bbox를 빨간색으로 표시합니다.

## 3. 전체 파이프라인

```text
원본 영상(mp4)
    ↓
[1] 영상 전처리
    - annotation JSON 매칭
    - 무단 투기 이벤트 구간 추출
    - 일정 길이의 clip 생성
    - train / val / test 분리

    ↓
[2] VideoMAE 학습
    - clip-level classification
    - 정상 / 무단 투기 분류
    - progressive unfreezing 적용

    ↓
[3] YOLOv8 학습 및 추론
    - 투기 순간의 사람 bbox 탐지
    - VideoMAE 의심 구간 내부에서 동작

    ↓
[4] 시각화 결과 생성
    - 결과 영상 mp4 저장
    - 탐지 정보 json 저장
```

## 4. 데이터셋

본 프로젝트는 AI Hub의 공원 주요시설 및 불법행위 감시 CCTV 영상 데이터를 기반으로 진행했습니다.

| 항목 | 내용 |
| --- | --- |
| 데이터 유형 | 고정형 CCTV 영상 |
| 영상 길이 | 약 2분 단위 원본 영상 |
| FPS | 3fps |
| 카메라 수 | 5개 고정 카메라 |
| 주요 클래스 | `trash_dump` |
| task | 정상 / 쓰레기 무단 투기 분류 |

최종적으로 원본 영상을 clip 단위로 분할해 학습 데이터셋을 구성했습니다.

| Split | Clip 수 |
| --- | ---: |
| Train | 8,122 |
| Validation | 1,954 |
| Test | 1,790 |

## 5. 데이터 전처리

전처리 과정에서는 원본 영상과 annotation JSON을 매칭한 뒤, 무단 투기 이벤트 구간을 기준으로 학습용 clip을 생성했습니다.

```text
원본 영상
    ↓
annotation JSON 파싱
    ↓
trash_dump 이벤트 추출
    ↓
clip 단위 분할
    ↓
normal / illegal 라벨 부여
    ↓
train / val / test 구성
```

### Shortcut bias 문제

초기 데이터 분석 과정에서 특정 카메라 배경이 무단 투기 라벨과 과도하게 함께 등장하는 문제가 있었습니다.

이 경우 모델은 실제 투기 행동이 아니라 아래와 같은 shortcut을 학습할 수 있습니다.

```text
"이 배경이 나오면 illegal"
"이 위치에서 사람이 움직이면 illegal"
"클립의 특정 시점에 변화가 생기면 illegal"
```

이를 줄이기 위해 다음과 같이 데이터셋을 재구성했습니다.

- 카메라별 정상 / 무단 투기 clip 비율 보정
- 특정 배경이 특정 라벨에만 치우치지 않도록 normal clip 추가
- 무단 투기 이벤트가 clip 내부의 다양한 위치에 오도록 offset 조정
- train / validation / test 분리 시 데이터 누수 방지

## 6. 모델 구조

본 프로젝트의 VideoMAE 모델은 clip-level head와 frame-level head를 함께 사용하는 구조입니다.

```text
Input Video Clip
    ↓
VideoMAE Backbone
    ↓
Temporal Tokens
    ↓
┌─────────────────────┬─────────────────────┐
│ Clip-level Head      │ Frame-level Head     │
│ normal / illegal     │ suspicious frames    │
└─────────────────────┴─────────────────────┘
```

| 구성 요소 | 역할 |
| --- | --- |
| VideoMAE backbone | 비디오 프레임 간 시간적 특징 추출 |
| Clip-level head | 전체 clip이 정상인지 무단 투기인지 분류 |
| Frame-level head | 무단 투기 의심 frame 구간 추정 |
| YOLOv8 | 의심 구간 내 사람 bbox 탐지 및 강조 |

## 7. 학습 방식

VideoMAE fine-tuning은 progressive unfreezing 방식으로 진행했습니다.

| Stage | 학습 대상 | 목적 |
| --- | --- | --- |
| Stage 1 | classification heads | 새 task에 맞는 분류기 우선 학습 |
| Stage 2 | backbone 마지막 layer + heads | 일부 비디오 표현 조정 |
| Stage 3 | 전체 모델 | 전체 네트워크 fine-tuning |

학습 시 clip-level loss와 frame-level loss를 함께 사용해 전체 영상 분류와 의심 구간 추정을 동시에 학습하도록 구성했습니다.

## 8. 실험 결과

최종 모델은 clip 단위 무단 투기 분류에서 높은 성능을 보였습니다.

| Metric | Score |
| --- | ---: |
| Accuracy | 97.7% |
| F1 Score | 0.977 |
| AUROC | 0.997 |
| Frame-level F1 | 0.754 |
| Temporal IoU | 0.764 |

Clip-level classification 성능은 높게 나타났지만, frame-level detection은 상대적으로 개선 여지가 있습니다.  
즉, “이 영상이 무단 투기인가?”는 잘 판단하지만, “정확히 어느 frame에서 발생했는가?”는 추가 개선이 필요합니다.

## 9. Repository 구조

학습 및 실험 코드는 다음 구조를 기준으로 구성됩니다.

```text
.
├── preprocessing/
│   └── khuda_173/
│       └── README_KHUDA_173_PREPROCESSING.md
│
├── seraph/
│   ├── models/
│   │   └── model.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── transforms.py
│   ├── train.py
│   └── test.py
│
├── README.md
└── README_GITHUB.md
```

### 주요 파일

| 경로 | 설명 |
| --- | --- |
| `preprocessing/khuda_173/` | AI Hub CCTV 데이터 전처리 관련 코드 및 정리 |
| `seraph/train.py` | VideoMAE 학습 코드 |
| `seraph/test.py` | 학습된 모델 테스트 코드 |
| `seraph/models/model.py` | VideoMAE multi-head 모델 정의 |
| `seraph/utils/transforms.py` | 데이터 로딩 및 transform 정의 |

## 10. 시각화 / 실행 패키지 구조

최종 데모 및 시각화 실행 패키지는 아래 구조를 기준으로 합니다.

```text
project_package.zip
│
├── pipeline.py
│     메인 실행 코드.
│     VideoMAE로 영상에서 투기 의심 구간을 찾고,
│     YOLOv8로 그 구간 안에서 실제 투기 순간의 사람 bbox를 빨간색으로 강조합니다.
│     최종 결과 영상(mp4)과 탐지 정보(json)를 생성합니다.
│
├── model.py
│     VideoMAE 모델 구조 정의 파일.
│
├── checkpoints/
│   └── best_model.pt
│         학습 완료된 VideoMAE 가중치입니다.
│         입력 영상 구간이 무단 투기인지 아닌지 분류하는 모델의 실제 학습 결과물입니다.
│
├── yolo_runs/
│   └── trash_dump_ep5/
│       └── weights/
│           └── best.pt
│                 투기 행위 탐지용으로 fine-tuning된 YOLOv8 가중치입니다.
│                 VideoMAE가 찾은 의심 구간 안에서 실제 투기 순간을 찾는 데 사용합니다.
│
├── sample_video.mp4
│     테스트용 예시 CCTV 영상입니다.
│
└── requirements.txt
      실행에 필요한 Python package 목록입니다.
      torch, transformers, decord, ultralytics, opencv-python, numpy, scipy 등을 포함합니다.
```

> `best_model.pt`는 약 1GB 크기의 모델 가중치이므로 GitHub에 직접 올리기 어렵습니다.  
> 필요한 경우 GitHub Releases, Google Drive, Hugging Face Hub 등의 외부 링크로 제공하는 것을 권장합니다.

## 11. 실행 방법

### 1. 패키지 압축 해제

```bash
unzip project_package.zip
cd project_package
```

### 2. Python 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음 패키지가 포함됩니다.

```text
torch
transformers
decord
ultralytics
opencv-python
numpy
scipy
```

### 3. 데모 실행

```bash
python pipeline.py --input sample_video.mp4
```

실행 후 다음 결과물이 생성됩니다.

```text
result.mp4
detection_result.json
```

예상 출력 흐름은 다음과 같습니다.

```text
1. sample_video.mp4 입력
2. VideoMAE가 무단 투기 의심 구간 탐지
3. YOLOv8이 의심 구간 내부에서 사람 bbox 탐지
4. 투기 전 bbox는 파란색으로 표시
5. 투기 발생 순간 bbox는 빨간색으로 표시
6. 결과 영상과 json 파일 저장
```

## 12. 결과 예시

데모 영상에서는 다음과 같은 방식으로 결과를 확인할 수 있습니다.

```text
불법 투기 감지중입니다
        ↓
불법 투기가 감지되었습니다
        ↓
해당 frame 근처 사람 bbox 빨간색 강조
        ↓
결과 영상 저장
```

탐지 결과 JSON에는 다음과 같은 정보가 포함될 수 있습니다.

```json
{
  "input_video": "sample_video.mp4",
  "is_illegal": true,
  "suspicious_start_frame": 120,
  "suspicious_end_frame": 168,
  "detections": [
    {
      "frame": 142,
      "bbox": [320, 180, 410, 360],
      "label": "trash_dump",
      "score": 0.91
    }
  ]
}
```

## 13. 한계점

본 프로젝트는 clip-level 무단 투기 분류에서는 높은 성능을 보였지만, 실제 서비스 적용을 위해서는 다음 한계점을 추가로 검증해야 합니다.

### 1. 새로운 배경에 대한 일반화

현재 데이터는 5개 고정 카메라 기반입니다.  
따라서 학습에 포함되지 않은 새로운 공원, 골목, 야간 환경, 다른 카메라 각도에서도 잘 동작하는지 검증이 필요합니다.

### 2. Background shortcut 가능성

데이터 재구성을 통해 배경 편향을 줄였지만, CCTV 데이터 특성상 특정 배경과 특정 라벨의 상관관계를 완전히 제거하기는 어렵습니다.

### 3. Frame-level detection 성능

Clip 단위 분류 성능은 높지만, 정확히 어느 frame에서 투기 행동이 발생했는지 찾는 성능은 상대적으로 부족합니다.

### 4. YOLO bbox 시각화 안정성

YOLOv8은 데모 시각화에 유용하지만, 사람 가림, 저해상도, 야간 환경, 멀리 있는 사람에 대해서는 bbox 품질이 낮아질 수 있습니다.

### 5. 모델 가중치 배포 문제

VideoMAE 가중치 파일은 약 1GB로 크기가 큽니다.  
GitHub 저장소에는 코드와 실행 방법을 올리고, 가중치는 별도 링크로 제공하는 방식이 적절합니다.

## 14. 기대 효과

본 프로젝트를 통해 다음과 같은 효과를 기대할 수 있습니다.

- 관제 인력의 CCTV 모니터링 부담 감소
- 무단 투기 의심 구간 자동 선별
- 단순 객체 탐지보다 행동 맥락을 반영한 판단 가능
- 결과 영상과 JSON을 통한 후처리 및 기록 관리 가능
- 향후 다른 불법행위 감지 task로 확장 가능

## 15. References

- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training  
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html
- YOLOv8 / Ultralytics  
  https://docs.ultralytics.com
- AI Hub  
  https://www.aihub.or.kr
