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
