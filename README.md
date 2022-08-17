# Pix2Pix


- python 3.9.5 에서 테스트됨

```
tensorboard==2.9.1
torch==1.11.0
torchvision==0.12.0
```

# Training

- `main.py` 스크립트 실행
- 학습 로그는 `logs` 디렉토리에 기록 (`tensorboard --logdir ./logs`)
- 체크포인트는 `checkpoints` 디렉토리에 기록
- 50 에포크마다 체크포인트, 테스트셋 출력물 텐서보드에 기록

```bash
# 학습 스크립트 예시
python main.py \
    --netG resnet \
    --batch_size 4 \
    --norm_type instance \
    --device cuda:0 \
    
tensorboard --logdir ./logs
```

## Details

### Initial commit

- `ResnetGenerator` 백본
- `InstanceNormalization`, 작은 배치 사이즈가 효과 좋은듯
- LSGAN 로스에 스무딩 적용 ( `main.py` 117번 줄): 효과 있는지는 잘 모르겠음
- 러닝 레이트 스케쥴은 200 에폭까지 `1e-4`, 이후 50 에폭마다 `1/10` 로 감소
- `metric.py` 는 아직 다 작성하지 못함 (CRI, pixel loss 함수)
- 아직 randomness 컨트롤이 완벽하지 않은듯 (돌릴때마다 뭔가 조금씩 다른 느낌)

### 2022/07/19

- `ResnetGenerator` 의 residual blocks 에서 드랍아웃 사용 -> 해수면 부분의 경계를 제대로 생성하지 못함
- `ResnetGenerator` 인코더 단에 드랍아웃 추가 -> 해수면 부분의 경계를 안정적으로 생성하는 듯 보임

### 2022/08/17

- 원사이클 점검 코드 작성
- 데이터셋 업데이트 (현대중 데이터셋 -> 3D 프린팅 데이터셋)
    - 이미지 해상도 업데이트됨 (256 -> 512)
    - -1 ~ 1 사이의 회귀 문제로 치환 (수축, 정상/그외, 팽창)
    - 이미지 해상도 업데이트에 따라 판별기 설정 변경: `ndf=64`, `n_layers=2`
