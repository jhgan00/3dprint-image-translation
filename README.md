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


## Docker

- 시작 전 호스트 머신에 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치
- `docker.sample.sh` 파일 참고
- 이미지 빌드 또는 로드 후 컨테이너 생성, 접속 과정
- 접속 후에는 동일하게 `main.py` 실행하여 학습 진행

```bash
# 변수 설정
DATA_DIR="path/to/data"
WORKDIR="/workspace/3dprint-image-translation" # 도커 이미지의 WORKDIR 경로
DATA_VOL=translation-data-vol  # 데이터 볼륨
EXPR_VOL=translation-expr-vol  # 실험 결과를 저장할 볼륨
SHM_SIZE="32G"

# 볼륨 생성
docker volume create \
    --name $DATA_VOL \
    --opt type=none \
    --opt device=$DATA_DIR \
    --opt o=bind
sudo docker volume create --name $EXPR_VOL

# 이미지 빌드 또는 로드
docker build -t translation .  # Dockerfile 에서 직접 이미지를 빌드하는 경우
# docker load --input translation.tar  # 도커 이미지 tar 파일을 사용하는 경우

# 컨테이너 시작
docker run \
    --name translation \
    --gpus all \
    -it \
    -d \
    --rm \
    -p 6006:6006 \
    --shm-size $SHM_SIZE \
    --mount 'src='"$DATA_VOL"',dst='"$WORKDIR"'/data' \
    --mount 'src='"$EXPR_VOL"',dst='"$WORKDIR"'/experiments' \
    translation:latest

# 컨테이너 진입
docker exec -it translation /bin/bash 
```