import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


def get_dataset(args) -> Tuple[Dataset]:

    if args.dataset == "g-fdm":
        dataset_train = GFDMDataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="train")
        dataset_valid = GFDMDataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="valid")
        dataset_test  = GFDMDataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="test")

    elif args.dataset == "sla":
        dataset_train = SLADataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="train")
        dataset_valid = SLADataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="valid")
        dataset_test = SLADataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="test")

    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented yet")

    return dataset_train, dataset_valid, dataset_test


def resize_and_pad(image, input_size, value=255):
    """ 높이, 너비를 4로 나누어 떨어지도록 패딩 """
    w, h = image.size[0], image.size[1]
    longer = max(w, h)

    if isinstance(input_size, list):
        input_size = max(input_size)

    w, h = int(w * (input_size / longer)), int(h * (input_size / longer))
    image = image.resize((w, h))
    if w % 2 == 0:
        p_left, p_right = (input_size - w) // 2, (input_size - w) // 2
    else:
        p_left, p_right = (input_size - w) // 2, (input_size - w) - ((input_size - w) // 2)
    if h % 2 == 0:
        p_top, p_bottom = (input_size - h) // 2, (input_size - h) // 2
    else:
        p_top, p_bottom = (input_size - h) // 2, (input_size - h) - ((input_size - h) // 2)

    padding = (int(p_left), int(p_top), int(p_right), int(p_bottom))
    return transforms.functional.pad(image, padding, value, 'constant')


def build_transform(input_size, num_channels):
    t = []
    if input_size:
        t.append(transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize([0.5] * num_channels, [0.5] * num_channels))
    return transforms.Compose(t)


class GFDMDataset(Dataset):

    b_lo = np.array([80, 20, 100])  # 수축
    b_hi = np.array([139, 255, 255])
    ry_lo = np.array([0, 20, 100])  # 팽창
    ry_hi = np.array([39, 255, 255])
    rp_lo = np.array([140, 20, 100])
    rp_hi = np.array([179, 255, 255])

    def __init__(self, src_dir, dst_dir, csv_fpath, input_size, split):
        """도면 이미지, 출력 이미지, 프린터 파라미터, 변형률 정보"""

        df = pd.read_csv(csv_fpath, encoding='utf-8-sig').query(f"split=='{split}'")
        self.src_images = df['src'].values
        self.dst_images = df['dst'].values
        self.conditions = df.iloc[:, 2:-8].values
        self.real_error = MinMaxScaler().fit_transform(df.iloc[:, -8:-2].values)

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.input_size = input_size
        self.split = split

    def __getitem__(self, i):

        src_fpath = os.path.join(self.src_dir, self.src_images[i])
        dst_fpath = os.path.join(self.dst_dir, self.dst_images[i])

        src = Image.open(src_fpath).convert("L")
        dst = Image.open(dst_fpath).convert("RGB")
        dst = dst.resize(size=src.size)

        src = resize_and_pad(src, self.input_size)
        dst = resize_and_pad(dst, self.input_size)

        dst = transforms.functional.to_tensor(dst)
        dst = transforms.functional.normalize(dst, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        src = transforms.functional.to_tensor(src)
        src = transforms.functional.normalize(src, (.5,), (.5,))

        conditions = self.conditions[i]
        real_error = self.real_error[i]

        return src, dst, conditions, real_error

    def __len__(self):

        return len(self.dst_images)


class SLADataset(Dataset):

    def __init__(self, src_dir, dst_dir, csv_fpath, input_size, split):
        """도면 이미지, 출력 이미지, 프린터 파라미터, 변형률 정보"""

        df = pd.read_csv(csv_fpath, encoding='utf-8-sig').query(f"split=='{split}'")
        self.src_images = df['src'].values
        self.dst_images = df['dst'].values
        self.conditions = df.iloc[:, 2:-8].values
        self.real_error = df.iloc[:, -8:-2].values

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.input_size = input_size
        self.split = split

    def __getitem__(self, i):
        src_fpath = os.path.join(self.src_dir, self.src_images[i])
        dst_fpath = os.path.join(self.dst_dir, self.dst_images[i])

        src = Image.open(src_fpath).convert("L")
        dst = Image.open(dst_fpath)
        dst = dst.resize(size=src.size)

        src = resize_and_pad(src, self.input_size)
        dst = resize_and_pad(dst, self.input_size)

        dst = transforms.functional.to_tensor(dst)
        dst = transforms.functional.normalize(dst, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        src = transforms.functional.to_tensor(src)
        src = transforms.functional.normalize(src, (.5,), (.5,))

        conditions = self.conditions[i]
        real_error = self.real_error[i]

        return src, dst, conditions, real_error

    def __len__(self):
        return len(self.dst_images)
