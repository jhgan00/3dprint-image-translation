# Pix2Pix

## Training

- `main.py` 스크립트 실행 (`scripts` 디렉토리 참고)
- 학습 로그는 `experiments/$DATASET/logs` 디렉토리에 기록 (`tensorboard --logdir .experiments/$DATASET/logs`)
- 체크포인트는 `experiments/$DATASET/checkpoints` 디렉토리에 기록
  - 1 에포크마다 성능 평기
  - 베스트 FID, 베스트 픽셀 로스 기준으로 모델 선택
  - 추가로 `ckpt_freq` 에포크마다 체크포인트 기록

```bash
# 학습 스크립트 예시
DATASET="g-fdm"
python main.py \
    --netG resnet \
    --batch_size 4 \
    --norm_type instance \
    --device cuda:0 \
    
tensorboard --logdir ./experiments/$DATASET/logs  # ssh -NfL -p $SSH_PORT localhost:6006:localhost:6006 $USER_NAME@$HOST
```

## Dataset

- `main.py` 스크립트 실행 시 어떤 데이터셋을 사용할지 명시해야 함 
- 데이터셋별로 관련 경로 명시 (`src_dir`, `dst_dir`, `csv_fpath`)
- 현재 다음 두 가지 데이터셋 지원하며, 데이터셋을 추가할 경우 클래스 구현하고 `utils.get_dataset` 함수에 등록

argument|class
---|---
`g-fdm` | `GFDMDataset` 
`sla` | `SLADataset`
