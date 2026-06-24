# KHUDA 173 CCTV 데이터 전처리 총정리

## 1. 개요

`173. 공원 주요시설 및 불법행위 감시 CCTV 영상 데이터`는 실제 영상 clip 파일로 자르지 않고, 원본 영상 경로와 frame 구간을 담은 JSON manifest 방식으로 전처리했다.

즉, 학습 시에는 manifest의 `video_path`, `start_frame`, `end_frame`, `label` 정보를 읽어 원본 영상에서 필요한 frame 구간을 동적으로 로딩한다.

최종 저장 위치:

```text
/data/philipn337/KHUDA_173
```

전체 크기:

```text
159G
```

## 2. 최종 폴더 구조

```text
/data/philipn337/KHUDA_173/
├── 173_dataset.zip
├── raw/
│   └── extracted/
│       └── 173/
│           ├── videos/
│           │   ├── C_33_1_smp_su_09-11_11-02-00_c_for_DF1.mp4
│           │   ├── C_3_2_smp_cl_09-08_11-42-00_b_for_DF1.mp4
│           │   └── ...
│           │
│           ├── annotations/
│           │   ├── C_33_1_smp_su_09-11_11-02-00_c_for_DF1.json
│           │   ├── C_3_2_smp_cl_09-08_11-42-00_b_for_DF1.json
│           │   └── ...
│           │
│           └── 원본 압축 해제 폴더들
│
└── processed_173_manifest/
    ├── manifests/
    │   ├── events_all.json
    │   ├── clips_all.json
    │   ├── clips_train.json
    │   ├── clips_val.json
    │   └── clips_test.json
    │
    ├── metadata/
    │   ├── videos.json
    │   ├── class_map.json
    │   └── preprocessing_config.json
    │
    └── reports/
        ├── annotation_summary.json
        ├── clip_statistics.json
        ├── validation_report.json
        └── manifest_validation_report.json
```

## 3. 원본 데이터 정리 결과

영상과 JSON은 파일명 기준으로 1:1 매칭된다.

예:

```text
videos/C_33_1_smp_su_09-11_11-02-00_c_for_DF1.mp4
annotations/C_33_1_smp_su_09-11_11-02-00_c_for_DF1.json
```

정리 결과:

| 항목 | 값 |
| --- | ---: |
| 영상 파일 수 | 1728 |
| annotation JSON 수 | 1728 |
| Videos without JSON | 0 |
| JSON without Video | 0 |
| JSON parse errors | 0 |

## 4. 영상 메타데이터

전체 영상은 동일한 기본 속성을 가진다.

| 항목 | 값 |
| --- | ---: |
| FPS | 3.0 |
| 해상도 | 1920x1080 |
| 최소 frame 수 | 182 |
| 최대 frame 수 | 419 |
| 평균 frame 수 | 335.87 |

Annotation의 `cur_frame` 범위:

```text
min cur_frame: 1
max cur_frame: 370
unique cur_frame count: 370
```

## 5. Class 분포

JSON에서 확인된 `class_name` 분포는 다음과 같다.

| class_name | annotation 수 |
| --- | ---: |
| moving | 45666 |
| trash_dump | 42826 |
| stand | 33721 |
| sit_down_floor | 19728 |
| sit_down_bench | 14787 |
| fliers_action | 69 |
| smoking | 22 |

## 6. 정상/비정상 라벨 기준

이진 분류 기준으로 라벨을 구성했다.

### label = 1, abnormal

```text
trash_dump
fliers_action
smoking
```

### label = 0, normal

```text
moving
stand
sit_down_floor
sit_down_bench
```

주의:

```text
비정상 이벤트 대부분은 trash_dump이다.
따라서 현재 데이터셋은 사실상 쓰레기 무단투기 탐지 중심으로 해석하는 것이 안전하다.
```

## 7. 전처리 방식

이번 전처리는 실제 `.mp4` clip 파일을 만들지 않았다.

진행 방식:

1. 각 JSON에서 `cur_frame`, `class_name`, `bbox`, `object_id` 정보를 읽는다.
2. 비정상 class를 가진 frame을 positive frame으로 표시한다.
3. 연속된 positive frame들을 하나의 abnormal event로 병합한다.
4. 각 event를 기준으로 positive clip manifest를 생성한다.
5. 비정상 event와 겹치지 않는 구간에서 normal clip 후보를 생성한다.
6. normal/abnormal clip을 거의 1:1 비율로 구성한다.
7. 같은 영상이 train/val/test에 동시에 들어가지 않도록 video 단위 split을 수행한다.
8. 최종 manifest 내부 `video_path`를 `/data/philipn337/KHUDA_173` 기준으로 수정한다.

## 8. Event 및 Clip 생성 결과

| 항목 | 값 |
| --- | ---: |
| abnormal events | 5943 |
| positive clips | 5943 |
| normal candidates | 5997 |
| normal clips | 5997 |
| all clips | 11940 |

Event count by abnormal class:

| class | event 수 |
| --- | ---: |
| trash_dump | 5933 |
| fliers_action | 8 |
| smoking | 2 |

## 9. Train / Val / Test Split

Split은 clip 단위가 아니라 video 단위로 수행했다.

이유:

```text
같은 영상에서 나온 clip이 train과 val/test에 동시에 들어가면 데이터 누수가 생길 수 있기 때문이다.
```

| split | clip 수 | normal | abnormal | video 수 |
| --- | ---: | ---: | ---: | ---: |
| train | 8441 | 4370 | 4071 | 1209 |
| val | 1696 | 719 | 977 | 259 |
| test | 1803 | 908 | 895 | 260 |

Leakage 검증:

```text
train-val leakage: 없음
train-test leakage: 없음
val-test leakage: 없음
```

## 10. 학습에 사용할 핵심 파일

학습 시 주로 사용할 manifest는 다음 세 파일이다.

```text
/data/philipn337/KHUDA_173/processed_173_manifest/manifests/clips_train.json
/data/philipn337/KHUDA_173/processed_173_manifest/manifests/clips_val.json
/data/philipn337/KHUDA_173/processed_173_manifest/manifests/clips_test.json
```

전체 clip 목록:

```text
/data/philipn337/KHUDA_173/processed_173_manifest/manifests/clips_all.json
```

Event 목록:

```text
/data/philipn337/KHUDA_173/processed_173_manifest/manifests/events_all.json
```

## 11. Clip manifest 예시

각 clip 항목은 대략 다음 구조를 가진다.

```json
{
  "clip_id": "...",
  "video_id": "...",
  "video_path": "/data/philipn337/KHUDA_173/raw/extracted/173/videos/....mp4",
  "label": 1,
  "label_name": "abnormal",
  "start_frame": 120,
  "end_frame": 167,
  "start_time_sec": 40.0,
  "end_time_sec": 56.0,
  "fps": 3.0,
  "source_events": ["..."],
  "split": "train"
}
```

정상 clip은 다음 특징을 가진다.

```text
label: 0
label_name: normal
source_events: []
```

비정상 clip은 다음 특징을 가진다.

```text
label: 1
label_name: abnormal
source_events: 해당 abnormal event id 목록
```

## 12. 최종 검증 결과

최종 manifest 검증 결과:

```text
Status: ok
Errors: 0
```

검증된 항목:

```text
split count 정상
clip_id 중복 없음
train/val/test video leakage 없음
frame range 오류 없음
video_path 존재함
/local_datasets 경로 잔존 없음
source_event 누락 없음
```

또한 `ffprobe`로 manifest의 `video_path`가 실제 영상 파일을 정상적으로 가리키는 것도 확인했다.

OpenCV 테스트는 현재 base 환경에 `cv2`가 없어 실패했지만, 이는 데이터 문제가 아니다.

```text
ERROR: cv2 import failed
No module named 'cv2'
```

## 13. 현재 완료 상태

완료된 작업:

```text
[완료] 원본 zip 확보
[완료] 압축 해제
[완료] videos / annotations 정리
[완료] 영상-JSON 1:1 매칭 검증
[완료] JSON 파싱 검증
[완료] 영상 메타데이터 분석
[완료] class 분포 분석
[완료] event 생성
[완료] clip manifest 생성
[완료] train/val/test split
[완료] manifest validation
[완료] 전체 데이터 /data/philipn337/KHUDA_173 로 백업
[완료] manifest video_path 영구 경로로 변경
[완료] ffprobe로 영상 접근 테스트
```

## 14. 다음 단계

전처리는 완료됐고, 다음 단계는 학습 파이프라인 구성이다.

해야 할 일:

1. 학습 conda 환경에 OpenCV, PyAV, decord 중 하나를 준비한다.
2. `clips_train.json`, `clips_val.json`, `clips_test.json`을 읽는 Dataset 클래스를 만든다.
3. 각 sample에서 `video_path`, `start_frame`, `end_frame`, `label`을 읽는다.
4. 원본 영상에서 해당 frame 구간을 로딩한다.
5. 첫 batch가 정상적으로 생성되는지 확인한다.
6. 모델 학습을 시작한다.

## 15. 한 줄 요약

`173. 공원 주요시설 및 불법행위 감시 CCTV 영상 데이터`는 `/data/philipn337/KHUDA_173`에 정리 및 백업됐고, 실제 영상을 자르지 않는 방식으로 정상/비정상 clip manifest까지 생성 및 검증 완료됐다.

