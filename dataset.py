import os
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_dataset(args) -> Tuple[Dataset]:

    src_transform = build_transform(args.input_size, args.input_nc)

    dataset_train = TDPDataset(args.dst_dir, args.csv_fpath, src_transform, split="train")
    dataset_valid = TDPDataset(args.dst_dir, args.csv_fpath, src_transform, split="valid")
    dataset_test  = TDPDataset(args.dst_dir, args.csv_fpath, src_transform, split="test")

    return dataset_train, dataset_valid, dataset_test


def padding(image, value=255):
    """ 높이, 너비를 4로 나누어 떨어지도록 패딩 """
    w, h = image.size[0], image.size[1]
    if w % 4 == (0 or 2):
        p_left, p_right = (w % 4) / 2, (w % 4) / 2
    else:
        p_left, p_right = (w % 4) // 2, (w % 4) // 2 - (w % 4) // 2
    if h % 4 == (0 or 2):
        p_top, p_bottom = (h % 4) / 2, (h % 4) / 2
    else:
        p_top, p_bottom = (h % 4) // 2, (h % 4) // 2 - (h % 4) // 2
    padding = (int(p_left), int(p_top), int(p_right), int(p_bottom))
    return transforms.functional.pad(image, padding, value, 'constant')


def build_transform(input_size, num_channels):
    t = []
    if input_size:
        t.append(transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize([0.5] * num_channels, [0.5] * num_channels))
    return transforms.Compose(t)


class TDPDataset(Dataset):

    l_lo = np.array([0, 0, 0])  # 도면 라인 추정
    l_hi = np.array([179, 255, 150])
    g_lo = np.array([40, 20, 100])  # 공차범위 내
    g_hi = np.array([89, 255, 255])
    b_lo = np.array([90, 20, 100])  # 수축
    b_hi = np.array([139, 255, 255])
    ry_lo = np.array([0, 20, 100])  # 팽창
    ry_hi = np.array([39, 255, 255])
    rp_lo = np.array([140, 20, 100])
    rp_hi = np.array([179, 255, 255])

    def __init__(self, dst_dir, csv_fpath, src_transform, split):

        """도면 이미지, 출력 이미지, 프린터 파라미터, 변형률 정보"""

        df = pd.read_csv(csv_fpath, encoding='utf-8-sig').query(f"split=='{split}'")

        self.dst_images = df['dst'].values
        self.conditions = df.iloc[:, 2:-8].values.astype(np.float32)
        self.real_error = df['Out.Tol.(%)'].values
        self.dst_dir = dst_dir

        self.src_transform = src_transform

        self.split = split

    def __getitem__(self, i):
        # 이미지 경로
        dst_fpath = os.path.join(self.dst_dir, self.dst_images[i])

        # 이미지 읽기 : 패딩 & 리사이즈를 먼저 하기: 처리를 다 하고 나중에 BICUBIC 리사이징 하는 경우 리사이징 알고리즘으로 인해 값이 틀어짐
        src = Image.open(dst_fpath)
        src = padding(src)

        # Augmentation 적용
        if self.split == "train":

            if random.random() > 0.5: src = transforms.functional.hflip(src)
            if random.random() > 0.5: src = transforms.functional.vflip(src)

        src = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2HSV)

        l = cv2.inRange(src, TDPDataset.l_lo,  TDPDataset.l_hi)
        g = cv2.inRange(src,  TDPDataset.g_lo,  TDPDataset.g_hi)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        g = cv2.erode(g, k)
        b = cv2.inRange(src,  TDPDataset.b_lo,  TDPDataset.b_hi)
        ry = cv2.inRange(src,  TDPDataset.ry_lo,  TDPDataset.ry_hi)
        rp = cv2.inRange(src,  TDPDataset.rp_lo,  TDPDataset.rp_hi)
        r = ry + rp

        # 0, 255 to 0, 1
        src = np.where(l + g > 1, 1., 0.)  # 인풋
        contra = np.where(b > 1, 2, 0)  # 수축
        expand = np.where(r > 1, 3, 0)  # 팽창
        dst = contra + expand
        dst = np.where(src == 1, 1, dst)
        src = self.src_transform(src)

        conditions = self.conditions[i]
        real_error = self.real_error[i]

        return src, dst, conditions, real_error

    def __len__(self):
        return len(self.dst_images)
