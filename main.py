import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataset
from engine import train_one_epoch, evaluate, generate
from losses import GANLoss, VGGPerceptualLoss
from models import get_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--lambda_VGG', type=float, default=100.,
                        help='weight for the content(VGG) loss (originally proposed in SRGAN)')
    parser.add_argument('--style_layers', type=int, nargs="+", default=[])
    parser.add_argument('--max_grad_norm', type=float, default=0.0)
    parser.add_argument('--smoothing', type=float, default=0.1)

    # model parameters
    parser.add_argument('--netG', type=str, default='resnet', choices=['attn', 'unet', 'resnet'])
    parser.add_argument('--norm_type', default='instance', type=str, choices=['batch', 'instance', 'none'])
    parser.add_argument('--no_dropout', action='store_true')
    parser.add_argument('--input_nc', type=int, default=1,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of disc filters in the first conv layer')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers_G', type=int, default=9, help='only used if netD==n_layers')
    parser.add_argument('--n_layers_D', type=int, default=2, help='only used if netD==n_layers')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')

    # dataset parameters
    parser.add_argument('--dataset', type=str, default='tdp-color', choices=['tdp-color', 'tdp-gray', 'hdjoong'])
    parser.add_argument('--src_dir', type=str, default='./data/data/Blueprint')
    parser.add_argument('--dst_dir', type=str, default='./data/data/Mash')
    parser.add_argument('--csv_fpath', type=str, default='./data/data/Metadata/data.csv')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--input_size', type=int, default=[], nargs="+", help='input size. empty for no resizing')

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs with the initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_iters', type=int, default=25)

    # misc
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./logs/translation')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--ckpt_freq', type=int, default=50, help='save model for evey n epochs')
    parser.add_argument('--expr_name', type=str, default=str(datetime.now()))
    parser.add_argument('--seed', type=int, default=777)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.expr_name)
    args.output_dir = os.path.join(args.output_dir, args.expr_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.expr_name)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(os.path.join(args.output_dir, "real")):
        os.makedirs(os.path.join(args.output_dir, "real"))
    if not os.path.exists(os.path.join(args.output_dir, "fake")):
        os.makedirs(os.path.join(args.output_dir, "fake"))
    args.total_epochs = args.n_epochs + args.n_epochs_decay
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

    criterionGAN = GANLoss(args.gan_mode, target_real_label=1 - args.smoothing, target_fake_label=args.smoothing).to(device)
    criterionVGG = VGGPerceptualLoss().to(device)

    # get dataset
    train_dataset, valid_dataset, test_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_threads)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.num_threads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.num_threads)

    # get model
    num_embeddings = train_dataset.conditions.shape[1]
    netG, netD = get_model(num_embeddings, args)
    netG.to(device)
    netD.to(device)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    scheduler_G = StepLR(optimizer_G, step_size=args.lr_decay_iters)
    scheduler_D = StepLR(optimizer_D, step_size=args.lr_decay_iters)

    # get tensorboard log writer
    if args.log_dir is not None:
        log_writer = SummaryWriter(args.log_dir)

    # train pix2pix
    for epoch in range(1, args.total_epochs + 1):

        train_one_epoch(
            netG,
            netD,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            criterionGAN,
            criterionVGG,
            train_loader,
            epoch,
            device,
            log_writer,
            args
        )

        evaluate(netG, valid_loader, epoch, device, log_writer, args)

        if not epoch % args.ckpt_freq:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint-{epoch}.pth')
            torch.save(netG.state_dict(), save_path)

    evaluate(netG, test_loader, args.n_epochs, device, log_writer, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
