import random
from collections import defaultdict
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F_


sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
resize = transforms.Compose([transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)])

x_h, x_w = 480, 2064
y_h, y_w = 200, 2096


def build_transform(input_size):
    t = [transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC)]
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


class CustomDataset(Dataset):
    def __init__(self, src_dir, dst_dir, csv_fpath, split=False):
        """도면 이미지, 출력 이미지, 프린터 파라미터, 변형률 정보"""
        self.split = split
        self.df = pd.read_csv(csv_fpath, encoding='utf-8-sig').fillna(0.)
        self.src_images = self.df['src']
        self.dst_images = self.df['dst']
        self.conditions = self.df.iloc[:, 2:-8].values
        self.real_error = self.df['Avg']
        self.src_dir = src_dir
        self.dst_dir = dst_dir

    def __getitem__(self, i):
        src = cv2.imread(os.path.join(self.src_dir, self.src_images[i]), cv2.IMREAD_GRAYSCALE)[900:1450, 250:2404]
        axis = self.src_images[i].split('.')[0][-1]
        dst = cv2.imread(os.path.join(self.dst_dir, f'{axis}', self.dst_images[i]))[900:1450, 250:2404, :]
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        dst = np.where(np.repeat((dst >= 128).all(axis=-1)[:, :, np.newaxis], 3, -1), 0, dst)
        src_thresh = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)[1]
        src_contours, src_hierarchy = cv2.findContours(src_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        src_contours = list(src_contours)
        src_contours.sort(key=cv2.contourArea)
        contra_mask, expand_mask = get_mask(src)
        contour_index = -2
        if axis == 'X':
            input_size = x_h, x_w
            the_contour = src_contours[contour_index]
            x, y, w, h = cv2.boundingRect(the_contour)
            src_thresh = src_thresh[y:y + h, x:x + w]
            contra_masked = cv2.bitwise_and(dst, dst, mask=contra_mask)[y:y + h, x:x + w]
            expand_masked = cv2.bitwise_and(dst, dst, mask=expand_mask)[y:y + h, x:x + w]
            contra_masked = cv2.cvtColor(contra_masked, cv2.COLOR_RGB2GRAY)
            expand_masked = cv2.cvtColor(expand_masked, cv2.COLOR_RGB2GRAY)
            target_1 = cv2.threshold(contra_masked, 127, 255, cv2.THRESH_BINARY)[1]
            target_2 = cv2.threshold(expand_masked, 127, 255, cv2.THRESH_BINARY)[1]
            tf = build_transform(input_size)
            src = tanh(tf(Image.fromarray(src_thresh)))
            dst = tanh(tf(Image.fromarray(-1 * target_1 + target_2)))
            p2d = (int((y_w - x_w) / 2), int((y_w - x_w) / 2), int((y_w - x_h) / 2), int((y_w - x_h) / 2))
            src, dst = resize(F_.pad(src, p2d, 'constant', 0)), resize(F_.pad(dst, p2d, 'constant', 0))

        elif axis == 'Y':
            input_size = y_h, y_w
            l_the_contour = src_contours[contour_index]
            l_x, l_y, l_w, l_h = cv2.boundingRect(l_the_contour)
            r_the_contour = src_contours[contour_index - 1]
            r_x, r_y, r_w, r_h = cv2.boundingRect(r_the_contour)
            src_thresh = src_thresh[min(l_y, r_y):max(l_y + l_h, r_y + r_h), min(l_x, r_x):max(l_x + r_w, r_x + r_w)]
            contra_masked = cv2.bitwise_and(dst, dst, mask=contra_mask)
            expand_masked = cv2.bitwise_and(dst, dst, mask=expand_mask)
            contra_masked = contra_masked[min(l_y, r_y):max(l_y + l_h, r_y + r_h),
                            min(l_x, r_x):max(l_x + r_w, r_x + r_w)]
            expand_masked = expand_masked[min(l_y, r_y):max(l_y + l_h, r_y + r_h),
                            min(l_x, r_x):max(l_x + r_w, r_x + r_w)]
            contra_masked = cv2.cvtColor(contra_masked, cv2.COLOR_RGB2GRAY)
            expand_masked = cv2.cvtColor(expand_masked, cv2.COLOR_RGB2GRAY)
            target_1 = cv2.threshold(contra_masked, 127, 255, cv2.THRESH_BINARY)[1]
            target_2 = cv2.threshold(expand_masked, 127, 255, cv2.THRESH_BINARY)[1]
            tf = build_transform(input_size)
            src = tanh(tf(Image.fromarray(src_thresh)))
            dst = tanh(tf(Image.fromarray(-1 * target_1 + target_2)))
            p2d = (0, 0, int((y_w - y_h) / 2), int((y_w - y_h) / 2))
            src, dst = resize(F_.pad(src, p2d, 'constant', 0)), resize(F_.pad(dst, p2d, 'constant', 0))
        conditions = self.conditions[i]
        real_error = self.real_error[i]

        if self.split == "train":
            if random.random() > 0.5:
                src = transforms.functional.hflip(src)
                dst = transforms.functional.hflip(dst)
            angle = random.randint(-30, 30)
            src = transforms.functional.rotate(src, angle)
            dst = transforms.functional.rotate(dst, angle)
        return tanh(src), tanh(dst), conditions, real_error

    def __len__(self):
        return len(self.src_images)


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
    # 컨투어의 레벨은 4개 보다 많은 수 없음. 한번 구멍을 뚫으면 빈 공간이기 때문에 그 안에 다시 구멍이 생길 수 없음.
    level_dict = parse_levels(hierarchy)
    assert len(level_dict) in {2, 4}, f"컨투어를 올바르게 검출하지 못했습니다: {len(level_dict)} 개의 컨투어 레벨이 검출됨"

    contra_mask = np.zeros_like(thresh, dtype=np.uint8)
    expand_mask = np.ones_like(thresh, dtype=np.uint8) * 255

    # 팽창 마스크 : outer 컨투어 밖은 하얗고 안은 까맣게
    pts = [contours[i] for i in level_dict[0]]
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