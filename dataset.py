from torch.utils.data import Dataset
import glob
import os
from PIL import Image, ImageOps
import csv
from pathlib import Path
from torchvision import transforms
import random
import torch
import numpy as np


def build_transform(input_size=256):
    t = [transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC)]
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(t)


class CustomDataset(Dataset):

    def __init__(self, src_dir, dst_dir, csv_fpath, input_size, split="train"):
        """도면 이미지, 출력 이미지, 프린터 파라미터"""

        self.split = split
        self.src_images = []
        self.dst_images = []

        for dst in Path(dst_dir).rglob('*.png'):
            conditions = dst.stem.split("_")
            src = os.path.join(src_dir, f"base_{conditions[1]}_{conditions[3]}.jpg")
            assert os.path.isfile(src)
            dst = str(dst)
            self.src_images.append(src)
            self.dst_images.append(dst)

        self.src_images = np.array(self.src_images)
        self.dst_images = np.array(self.dst_images)
        idxsort = np.argsort(self.dst_images)
        self.dst_images = self.dst_images[idxsort]
        self.src_images = self.src_images[idxsort]

        self.transform = build_transform(input_size)

        with open(csv_fpath, "r", encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            # 환경변수는 앞에서부터 7개만 사용
            self.conditions = [[float(x) for x in line][:7] for line in reader]
            self.conditions = torch.tensor(self.conditions)

    def __getitem__(self, i):

        src = Image.open(self.src_images[i])
        src = ImageOps.grayscale(src)

        dst = Image.open(self.dst_images[i])
        dst = ImageOps.grayscale(dst)

        if self.split == "train":

            # random horizontal flip
            if random.random() > 0.5:
                src = transforms.functional.hflip(src)
                dst = transforms.functional.hflip(dst)

            # # random horizontal flip
            if random.random() > 0.5:
                src = transforms.functional.vflip(src)
                dst = transforms.functional.vflip(dst)

            # random rotation
            angle = random.randint(-30, 30)
            src = transforms.functional.rotate(src, angle)
            dst = transforms.functional.rotate(dst, angle)

        src = self.transform(src)
        dst = self.transform(dst)

        return src, dst, self.conditions[i]

    def __len__(self):
        return len(self.conditions)


class TestDataset(Dataset):

    def __init__(self, src_dir, csv_fpath, input_size):
        """도면 이미지, 출력 이미지, 프린터 파라미터"""

        self.src_images = glob.glob(os.path.join(src_dir, "*.jpg"))
        self.src_images.sort()
        self.transform = build_transform(input_size)

        with open(csv_fpath, "r") as f:
            reader = csv.reader(f)
            self.conditions = [[float(x) for x in line] for line in reader]
            self.conditions = torch.tensor(self.conditions)

    def __getitem__(self, i):
        src = Image.open(self.src_images[i])
        src = ImageOps.grayscale(src)
        src = self.transform(src)

        return src, self.conditions[i % len(self.conditions)]

    def __len__(self):
        return len(self.src_images)
