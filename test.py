import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, CustomDataset
from networks import UNetGenerator, ResnetGenerator, get_norm_layer
from torchvision.utils import save_image
from metric import mean_absolute_cri_error, mean_pixel_loss


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./logs')

    # parser.add_argument('--src_dir', type=str, default='./data/base')
    # parser.add_argument('--dst_dir', type=str, default='./data/15')
    # parser.add_argument('--csv_fpath', type=str, default='./data/after_scale.csv')
    parser.add_argument('--src_dir', type=str, default='./data/test/AI_dataset_all')
    parser.add_argument('--csv_fpath', type=str, default='./data/test/label.scaled.csv')

    # model parameters
    parser.add_argument('--input_nc', type=int, default=1,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')

    # dataset parameters
    parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--input_size', type=int, default=256, help='scale images to this size')

    # training parameters
    parser.add_argument('--output_dir', type=str, default='./cri-test')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoints/2022-07-19 02:02:47.145104/checkpoint-final.pth')
    parser.add_argument('--seed', type=int, default=777)

    args = parser.parse_args()
    # args.output_dir = os.path.join(args.output_dir, args.checkpoint_path.split("/")[-2])

    return args


def main(args):

    print("=" * 80)
    for k, v in vars(args).items():
        print(k, ":", v)
    print("=" * 80)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get model
    # netG = UNetGenerator(output_nc=args.output_nc, input_nc=args.input_nc, norm_type='instance')
    norm_layer = get_norm_layer('instance')
    netG = ResnetGenerator(args.input_nc, args.output_nc, args.ngf, norm_layer=norm_layer,
                           use_dropout=False, n_blocks=9).to(device)
    state_dict = torch.load(args.checkpoint_path)
    netG.load_state_dict(state_dict)
    netG.to(device)
    netG.eval()

    test_dataset = TestDataset(args.src_dir, args.csv_fpath, args.input_size)
    # test_dataset = CustomDataset(args.src_dir, args.dst_dir, args.csv_fpath, args.input_size, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # real_images = []
    fake_images = []
    for src, cond in test_loader:
        with torch.no_grad():
            src = src.to(device)
            cond = cond.to(device)
            fake = netG(src, cond).detach().cpu()
        fake_images.append(fake)
        # real_images.append(dst)

    fake_images = (torch.cat(fake_images).squeeze() + 1.0) * 0.5
    # real_images = (torch.cat(real_images).squeeze() + 1.0) * 0.5

    for org_filename, fake_img in zip(test_dataset.src_images, fake_images):
        save_path = os.path.join(args.output_dir, os.path.basename(org_filename))
        save_image(fake_img.float(), save_path)

    fake_images = (fake_images * 255.)
    # real_images = (real_images * 255.)

    # cri_error = mean_absolute_cri_error(fake_images, real_images)
    # pixel_loss = mean_pixel_loss(fake_images, real_images, p=1)
    #
    # print(f"CRI error: {cri_error:.4f}")
    # print(f"Pixel loss: {pixel_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
