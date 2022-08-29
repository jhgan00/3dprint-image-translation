import os

import torch
from sklearn.metrics import classification_report
from skimage.color import label2rgb
from torchvision.utils import make_grid, save_image

LABELS = [0, 1, 2, 3]
TARGET_NAMES = 'Background Normal Contraction Expansion'.split()
LABEL_DICT = {k: v for k, v in zip(LABELS, TARGET_NAMES)}


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train_one_epoch(
        model: torch.nn.Module,
        optim: torch.nn.Module,
        sched,
        criterion,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        device: torch.device,
        log_writer: torch.utils.tensorboard.SummaryWriter,
        args
):

    model.train()

    if epoch > args.n_epochs:
        sched.step()

    for iter, (src, dst, cond, _) in enumerate(train_loader, 1):

        src = src.to(device)
        dst = dst.to(device)
        cond = cond.to(device).float()

        optim.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(src, cond)  # G(A)
            optim.zero_grad()  # set G's gradients to zero
            loss = criterion(pred, dst)
        loss.backward()
        optim.step()  # udpate G's weights

        global_step = (epoch - 1) * len(train_loader) + iter

        if not (iter % args.log_freq) or (iter == len(train_loader)):
            loss = loss.item()
            print(f"Epoch [{epoch}/{args.total_epochs}] Batch [{iter}/{len(train_loader)}] Loss: {loss:.4f}")
            log_writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], global_step)
            log_writer.add_scalar('Loss', loss, global_step)

        if not (iter % args.display_freq):

            dst = dst.detach().cpu().numpy()
            dst_rgb = torch.FloatTensor([label2rgb(x, colors=['green', 'blue', 'red'], bg_label=0, bg_color='black') for x in dst])
            dst_rgb = dst_rgb.permute(0, 3, 1, 2)

            pred = pred.argmax(1).detach().cpu().numpy()
            pred_rgb = torch.FloatTensor([label2rgb(x, colors=['green', 'blue', 'red'], bg_label=0, bg_color='black') for x in pred])
            pred_rgb = pred_rgb.permute(0, 3, 1, 2)

            log_writer.add_image('images/src', make_grid(src, value_range=(-1, 1), normalize=True, nrow=2), global_step)
            log_writer.add_image('images/dst', make_grid(dst_rgb, nrow=2, normalize=True), global_step)
            log_writer.add_image('images/pred', make_grid(pred_rgb, nrow=2, normalize=True), global_step)


@torch.no_grad()
def evaluate(G: torch.nn.Module, data_loader: torch.utils.data.DataLoader, epoch: int, device: torch.device,
             log_writer: torch.utils.tensorboard.SummaryWriter, args):

    print("=" * 80)
    print(f"Validation at epoch [{epoch}/{args.total_epochs}]")

    G.eval()
    dsts = []
    preds = []

    for src, dst, cond, src_error in data_loader:

        src = src.to(device)
        cond = cond.to(device).float()
        pred = G(src, cond)

        dsts += dst.detach().cpu().numpy().flatten().tolist()
        preds += pred.argmax(1).detach().cpu().numpy().flatten().tolist()

    result_dict = classification_report(dsts, preds, labels=LABELS, target_names=TARGET_NAMES, output_dict=True, zero_division=0.)
    header_format = "{:15} | {:15} | {:15} | {:15}"
    header = header_format.format("Label", "Precision", "Recall", "F1-score")
    print(header)
    print("-" * len(header))

    row_format = "{:15} | {:15.2%} | {:15.2%} | {:15.2%}"
    for target in TARGET_NAMES:
        result = result_dict.get(target)
        print(row_format.format(target, result["precision"], result["recall"], result["f1-score"]))
        log_writer.add_scalar(f"Precision/{target}", result['precision'], global_step=epoch)
        log_writer.add_scalar(f"Recall/{target}", result['recall'], global_step=epoch)
        log_writer.add_scalar(f"F1-score/{target}", result['f1-score'], global_step=epoch)

    print("=" * 80)

    return result_dict


@torch.no_grad()
def generate(G: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, args):
    G.eval()
    pred_images = []
    for src, _, cond, src_error in data_loader:
        src = src.to(device)
        cond = cond.to(device)
        pred = G(src, cond).detach().cpu()
        pred_rgb = torch.FloatTensor([
            label2rgb(x, colors=['white', 'blue', 'gray', 'red'], bg_label=0, bg_color='black') for x in pred
        ])
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)
        pred_images.append(pred_rgb)

    for org_filename, pred_img in zip(data_loader.dataset.dst_images, pred_images):
        save_path = os.path.join(args.output_dir, os.path.basename(org_filename))
        save_image(pred_img, save_path)

    return pred_images
