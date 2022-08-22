import random
from collections import defaultdict
import os

import torch
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def parse_levels(hierarchy):
    """컨투어 트리의 hierarchy를 입력받아 각 컨투어의 레벨을 추출"""

    # 방문 여부를 기록
    visited = set()
    levels = [-1 for _ in range(len(hierarchy))]
    level_dict = defaultdict(set)

    # 부모를 갖지 않는 노드(루트)에서 시작
    root_nodes = []
    for i, node in enumerate(hierarchy):
        if node[-1] >= 0: continue
        root_nodes.append(i)
        levels[i] = 0
        level_dict[0].add(i)

    for root in root_nodes:

        # Depth First Search
        to_visit = list()
        to_visit.append(root)

        while to_visit:

            node = to_visit.pop()
            level = levels[node]

            if node not in visited:

                visited.add(node)

                next_sibling = hierarchy[node][1]
                if next_sibling >= 0:
                    levels[next_sibling] = level
                    level_dict[level].add(next_sibling)
                    to_visit.append(next_sibling)

                first_child = hierarchy[node][2]
                if first_child >= 0:
                    levels[first_child] = level + 1
                    level_dict[level + 1].add(first_child)
                    to_visit.append(first_child)

    return level_dict


def get_mask(src_image):
    """도면 이미지를 입력받아서 (수축 마스크, 팽창 마스크)의 튜플을 반환"""

    # 도면 이미지에서 컨투어 추출: 가우시안 필터 -> 오츠 이진화
    blur = cv2.GaussianBlur(255 - src_image, (5, 5), 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # get the actual inner list of hierarchy descriptions

    # 컨투어 레벨 파싱
    # 컨투어가 제대로 검출되었다면 컨투어의 레벨은 2개 또는 4개임
    # 컨투어의 레벨은 4개 보다 많을 수 없음. 왜? 한번 구멍을 뚫으면 빈 공간이기 때문에 그 안에 다시 구멍이 생길 수 없음.
    level_dict = parse_levels(hierarchy)
    assert len(level_dict) in {2, 4}, f"컨투어를 올바르게 검출하지 못했습니다: {len(level_dict)} 개의 컨투어 레벨이 검출됨"

    contra_mask = np.zeros_like(thresh, dtype=np.uint8)
    expand_mask = np.ones_like(thresh, dtype=np.uint8) * 255

    # 팽창 마스크 : outer 컨투어 밖은 하얗고 안은 까맣게
    pts = [contours[i] for i in level_dict[0]]  # 레벨 0에 해당하는 컨투어들의 인덱스
    cv2.fillPoly(expand_mask, pts=pts, color=(0, 0, 0))  # 안쪽을 검은색으로 채우기

    # 수축 마스크 : inner 컨투어 밖은 까맣고 안은 하얗게
    contour_idxs = level_dict[1]
    pts = [contours[i] for i in contour_idxs]
    cv2.fillPoly(contra_mask, pts=pts, color=(255, 255, 255))  # 안쪽을 흰색으로 채우기

    # 컨투어 레벨이 4개인 경우 추가 처리 필요
    if len(level_dict) == 4:
        # 수축 마스크
        contour_idxs = level_dict[2]
        pts = [contours[i] for i in contour_idxs]
        cv2.fillPoly(contra_mask, pts=pts, color=(0, 0, 0))  # 안쪽을 검은색으로 채우기

        # 팽창 마스크
        contour_idxs = level_dict[3]
        pts = [contours[i] for i in contour_idxs]
        cv2.fillPoly(expand_mask, pts=pts, color=(255, 255, 255))  # 안쪽을 흰색으로 채우기

    return contra_mask, expand_mask


def square_padding(image, value=0):
    """ 높이, 너비 중 긴쪽에 맞춰서 정사각형으로 패딩 """
    max_wh = max(image.size)
    p_left, p_top = [(max_wh - s) // 2 for s in image.size]
    p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
    padding = (p_left, p_top, p_right, p_bottom)
    return transforms.functional.pad(image, padding, value, 'constant')



class CustomDataset(Dataset):

    def __init__(self, src_dir, dst_dir, csv_fpath, input_size=512, split=False, threshold=0):

        """도면 이미지, 출력 이미지, 프린터 파라미터, 변형률 정보"""

        self.split = split

        df = pd.read_csv(csv_fpath, encoding='utf-8-sig')

        self.src_images = df['src']
        self.dst_images = df['dst']
        self.conditions = df.iloc[:, 2:-8].values
        self.real_error = df['Out.Tol.(%)']

        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self.threshold = threshold
        self.input_size = input_size

    def __getitem__(self, i):

        # 이미지 경로
        src_fpath = os.path.join(self.src_dir, self.src_images[i])
        dst_fpath = os.path.join(self.dst_dir, self.dst_images[i])

        # 이미지 읽기
        src = cv2.imread(src_fpath, cv2.IMREAD_GRAYSCALE)
        contra_mask, expand_mask = get_mask(src)
        blur = cv2.GaussianBlur(255 - src, (5, 5), 5)
        _, src = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 패딩 & NEAREST 리사이즈: BICUBIC 리사이징 하는 경우 리사이징 알고리즘으로 인해 값이 틀어짐
        src = square_padding(Image.fromarray(src), value=0)
        src = transforms.functional.resize(src, (self.input_size, self.input_size),
                                           interpolation=transforms.InterpolationMode.NEAREST)
        src = transforms.functional.to_tensor(src)
        src = transforms.functional.normalize(src, (0.5,), (0.5,))

        # 패딩 & 리사이즈를 먼저 하기: 처리를 다 하고 나중에 BICUBIC 리사이징 하는 경우 리사이징 알고리즘으로 인해 값이 틀어짐
        dst = Image.open(dst_fpath)
        dst = square_padding(dst, value=255)
        dst = transforms.functional.resize(dst, (self.input_size, self.input_size),
                                           interpolation=transforms.InterpolationMode.BICUBIC)
        dst = np.array(dst)

        contra_mask = square_padding(Image.fromarray(contra_mask), value=0)
        contra_mask = transforms.functional.resize(contra_mask, (self.input_size, self.input_size),
                                                   interpolation=transforms.InterpolationMode.NEAREST)
        contra_mask = np.array(contra_mask)

        expand_mask = square_padding(Image.fromarray(expand_mask), value=255)
        expand_mask = transforms.functional.resize(expand_mask, (self.input_size, self.input_size),
                                                   interpolation=transforms.InterpolationMode.NEAREST)
        expand_mask = np.array(expand_mask)

        # 타겟의 배경을 흰색에서 검은색으로 전환
        dst = np.where(np.repeat((dst >= 128).all(axis=-1)[:, :, np.newaxis], 3, -1), 0, dst)

        # 실제로 마스크를 씌우기
        contra_masked = cv2.bitwise_and(dst, dst, mask=contra_mask)
        expand_masked = cv2.bitwise_and(dst, dst, mask=expand_mask)

        # 정상: 0, 내부: 1, 수축: 2, 배경: 3, 팽창: 4
        inner = (contra_mask > 0).astype(int)  # 내부 마스크: 0, 1
        inner = np.where((contra_masked > 0).any(axis=-1), 2, inner)  # 0, 1, 2
        outer = np.where(expand_mask > 0, 3, 0).astype(int)  # 0, 3
        outer = np.where((expand_masked > 0).any(axis=-1), 4, outer)  # 0, 3, 4
        dst = inner + outer
        dst = torch.LongTensor(dst).unsqueeze(0)

        # Augmentation 적용
        if self.split == "train":

            if random.random() > 0.5:
                src = transforms.functional.hflip(src)
                dst = transforms.functional.hflip(dst)

            if random.random() > 0.5:
                src = transforms.functional.vflip(src)
                dst = transforms.functional.vflip(dst)

            angle = random.choice([0, 90, 180, 270])
            src = transforms.functional.rotate(src, angle)
            dst = transforms.functional.rotate(dst, angle)

        conditions = self.conditions[i]
        real_error = self.real_error[i]

        return src, dst.squeeze(), conditions, real_error

    def __len__(self):

        return len(self.src_images)