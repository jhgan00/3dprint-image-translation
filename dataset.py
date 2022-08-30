import random
from collections import defaultdict
import os

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


def build_transform(input_size):
    t = [transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC)]
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def np2ts(x):
    x = Image.fromarray(x)
    x = transforms.functional.to_tensor(x)
    y = transforms.functional.normalize(x, (0.5,), (0.5,))
    return y


class CustomDataset(Dataset):

    def __init__(self, dst_dir, csv_fpath, split=False):

        """도면 이미지, 출력 이미지, 프린터 파라미터, 변형률 정보"""

        df = pd.read_csv(csv_fpath, encoding='utf-8-sig')
        self.split = split
        self.dst_images = df['dst']
        self.conditions = df.iloc[:, 2:-8].values
        self.real_error = df['Out.Tol.(%)']
        self.dst_dir = dst_dir

    def __getitem__(self, i):
        # 이미지 경로
        dst_fpath = os.path.join(self.dst_dir, self.dst_images[i])

        # 이미지 읽기 : 패딩 & 리사이즈를 먼저 하기: 처리를 다 하고 나중에 BICUBIC 리사이징 하는 경우 리사이징 알고리즘으로 인해 값이 틀어짐
        src = Image.open(dst_fpath)
        src = padding(src)
        src = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2HSV)

        w_lo = np.array([0, 0, 240])  # 배경
        w_hi = np.array([179, 255, 255])
        w = cv2.inRange(src, w_lo, w_hi)

        l_lo = np.array([0, 0, 0])  # 도면 라인 추정
        l_hi = np.array([179, 255, 150])
        l = cv2.inRange(src, l_lo, l_hi)

        g_lo = np.array([40, 20, 100])  # 공차범위 내
        g_hi = np.array([89, 255, 255])
        g = cv2.inRange(src, g_lo, g_hi)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        g = cv2.erode(g, k)

        b_lo = np.array([90, 20, 100])  # 수축
        b_hi = np.array([139, 255, 255])
        b = cv2.inRange(src, b_lo, b_hi)

        ry_lo = np.array([0, 20, 100])  # 팽창
        ry_hi = np.array([39, 255, 255])
        rp_lo = np.array([140, 20, 100])
        rp_hi = np.array([179, 255, 255])

        ry = cv2.inRange(src, ry_lo, ry_hi)
        rp = cv2.inRange(src, rp_lo, rp_hi)
        r = ry + rp

        # 0, 255 to 0, 1
        src = np.where(l + g > 1, 1., 0.)  # 인풋
        target_1 = np.where(b > 1, 1., 0.)  # 수축
        target_2 = np.where(r > 1, 1., 0.)  # 팽창

        # 인풋&타겟 이미지 생성
        src = np2ts(src)
        dst = (-target_1 + target_2 + 1.) * 0.5
        dst = np2ts(dst)

        # Augmentation 적용
        if self.split == "train":

            if random.random() > 0.5:
                src = transforms.functional.hflip(src)
                dst = transforms.functional.hflip(dst)

            if random.random() > 0.5:
                src = transforms.functional.vflip(src)
                dst = transforms.functional.vflip(dst)

        conditions = self.conditions[i]
        real_error = self.real_error[i]

        return src, dst, conditions, real_error

    def __len__(self):

        return len(self.dst_images)
