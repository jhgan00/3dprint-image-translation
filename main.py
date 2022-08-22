import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CustomDataset
from engine import train_one_epoch, evaluate, generate
from networks import ResnetGenerator, UNetGenerator, init_weights, get_norm_layer
from losses import FocalLoss


def get_args():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--netG', type=str, default='resnet', choices=['unet', 'resnet'])
    parser.add_argument('--norm_type', default='instance', type=str, choices=['batch', 'instance', 'none'])
    parser.add_argument('--no_dropout', action='store_true')
    parser.add_argument('--input_nc', type=int, default=1,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=5,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--n_layers', type=int, default=9, help='only used if netD==n_layers')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')

    # dataset parameters
    # 1% 데이터셋 경로
    parser.add_argument('--src_dir', type=str, default='./data/Blueprint')
    parser.add_argument('--dst_dir', type=str, default='./data/Mash')
    parser.add_argument('--csv_fpath', type=str, default='./data/Metadata/data.csv')
    parser.add_argument('--use_validset', action="store_true")

    parser.add_argument('--num_threads', default=2, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs with the initial learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_iters', type=int, default=25)

    # misc
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--ckpt_freq', type=int, default=50, help='save model for evey n epochs')
    parser.add_argument('--expr_name', type=str, default=str(datetime.now()))
    parser.add_argument('--seed', type=int, default=777)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.expr_name)
    args.output_dir = os.path.join(args.output_dir, args.expr_name)
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.expr_name)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
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

    # 정상 0 / 내부 1 / 수축 2 / 외부 3 / 팽창 4
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1., 1., 1e-1, 1.], device=device))
    # criterion = FocalLoss(alpha=0.25, gamma=2.)

    # get dataset
    train_dataset = CustomDataset(args.src_dir, args.dst_dir, args.csv_fpath, split="train")
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_threads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.num_threads)

    # get model
    num_embeddings = train_dataset.dataset.conditions.shape[1]
    use_dropout = not args.no_dropout
    if args.netG == 'unet':
        netG = UNetGenerator(output_nc=args.output_nc, input_nc=args.input_nc, norm_type=args.norm_type).to(device)
    else:
        norm_layer = get_norm_layer(args.norm_type)
        netG = ResnetGenerator(args.input_nc, args.output_nc, num_embeddings, args.ngf, norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=args.n_layers).to(device)
    init_weights(netG, args.init_type, args.init_gain)
    optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_iters)

    # get tensorboard log writer
    if args.log_dir is not None:
        log_writer = SummaryWriter(args.log_dir)

    # train pix2pix
    best_score = .0
    for epoch in range(1, args.total_epochs + 1):

        train_one_epoch(
            netG,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            epoch,
            device,
            log_writer,
            args
        )

        eval_result = evaluate(netG, test_loader, epoch, device, log_writer, args)
        f1_score = eval_result['macro avg']['f1-score']

        if f1_score >= best_score:
            save_path = os.path.join(args.checkpoints_dir, 'checkpoint-best.pth')
            torch.save(netG.state_dict(), save_path)
            best_score = f1_score

    save_path = os.path.join(args.checkpoints_dir, 'checkpoint-final.pth')
    torch.save(netG.state_dict(), save_path)
    evaluate(netG, test_loader, epoch + 1, device, log_writer=log_writer, args=args)


if __name__ == "__main__":
    args = get_args()
    main(args)
