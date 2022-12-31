import argparse
import os
import glob
import logging
import random
import tempfile
import warnings

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

from dataset import get_dataset
from models import get_model


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--args_fpath", type=str, required=True)
    parser.add_argument("--ckpt_fpath", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_threads", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pid = os.getpid()
    command = os.popen(f"ps -p {pid} -o cmd --no-headers").read().strip()
    logging.info(f"starting evaluation job: {command}")

    args = get_args()
    training_args = torch.load(args.args_fpath)
    logging.info("Training arguments") 
    for k, v in vars(training_args).items():
        logging.info(f"{k}: {v}")


    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    _, _, test_dataset = get_dataset(training_args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads)

    # get model
    num_embeddings = test_dataset.conditions.shape[1]
    netG, netD = get_model(num_embeddings, training_args)
    state_dict = torch.load(args.ckpt_fpath, map_location="cpu")
    netG.load_state_dict(state_dict)
    netG.to(args.device)
    netG.eval()

    src_images = []
    real_images = []
    pred_images = []
    real_errors = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        
        for src, dst, cond, real_error in test_loader:

            src = src.to(args.device).float()
            cond = cond.to(args.device).float()
            pred = netG(src, cond)

            src_images.append(src.detach().float().cpu())
            real_images.append(dst.detach().float())
            pred_images.append(pred.detach().float().cpu())
            real_errors.append(real_error.float().detach())

    src_images = torch.cat(src_images) * 0.5 + 0.5
    real_images = torch.cat(real_images) * 0.5 + 0.5
    pred_images = torch.cat(pred_images) * 0.5 + 0.5

    with tempfile.TemporaryDirectory() as save_dir:

        os.makedirs(os.path.join(save_dir, "real"), exist_ok=False)
        os.makedirs(os.path.join(save_dir, "pred"), exist_ok=False)

        for org_filename, pred_img in zip(test_loader.dataset.dst_images, pred_images):
            save_path = os.path.join(save_dir, "pred", os.path.basename(org_filename))
            save_image(pred_img, save_path, normalize=False)

        for org_filename, real_img in zip(test_loader.dataset.dst_images, real_images):
            save_path = os.path.join(save_dir, "real", os.path.basename(org_filename))
            save_image(real_img, save_path, normalize=False)

        fid_value = calculate_fid_given_paths(
            paths = [os.path.join(save_dir, "real"), os.path.join(save_dir, "pred")],
            batch_size = args.batch_size,
            device = args.device,
            dims = 2048,
            num_workers = args.num_threads
        )

        sensitivity = 30
        lower = np.array([0, 0, 255 - sensitivity])
        upper = np.array([255, sensitivity, 255])

        pred_fpaths = glob.glob(os.path.join(save_dir, "pred", "*.png"))
        true_fpaths = glob.glob(os.path.join(save_dir, "real", "*.png"))

        pred_fpaths.sort()
        true_fpaths.sort()

        f1_list = []
        precision_list = []
        recall_list = []

        for pred_fpath, true_fpath in zip(pred_fpaths, true_fpaths):
        
            pred = cv2.imread(pred_fpath)
            true = cv2.imread(true_fpath)
        
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2HSV)
            true = cv2.cvtColor(true, cv2.COLOR_BGR2HSV)
        
            pred_edge = cv2.bitwise_not(cv2.inRange(pred, lower, upper))  # 흰색 이외의 영역 -> 엣지
            true_edge = cv2.bitwise_not(cv2.inRange(true, lower, upper))  # 흰색 이외의 영역 -> 엣지
        
            true_positive = cv2.bitwise_and(pred_edge, true_edge)  # 엣지가 겹치는 영역: 엣지를 엣지라고 맞춘 영역
        
            precision = true_positive.sum() / pred_edge.sum()  # TP / (TP + FP) : 엣지로 예측한 부분 중 정답인 부분의 영역
            recall =  true_positive.sum() / true_edge.sum() # TP / (TP + FN) : 실제 엣지 중 검출된 부분의 영역

            f1 = 2 * (precision * recall) / (precision + recall)
            f1_list.append(f1)
            precision_list.append(precision)

    logging.info(f"Experiment: {training_args.expr_name}")
    logging.info(f"FID: {fid_value:.2f}")
    logging.info(f"ODS: {np.mean(f1_list).round(4)}")
    logging.info(f"Avg.Precision: {np.mean(precision_list).round(4)}")

    logging.info("finished")
