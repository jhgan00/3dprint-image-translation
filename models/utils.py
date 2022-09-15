from torch import nn
from torch.nn import init
import functools
from models.unet import UNetGenerator
from models.resnet import ResnetGenerator
from models.attn_resnet import AttentionalResnetGenerator
from models.patch_gan import NLayerDiscriminator


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_model(num_embeddings, args):

    use_dropout = not args.no_dropout
    netG, netD = None, None

    if args.netG == 'unet':
        netG = UNetGenerator(output_nc=args.output_nc, input_nc=args.input_nc, norm_type=args.norm_type)
    elif args.netG == 'resnet':
        norm_layer = get_norm_layer(args.norm_type)
        netG = ResnetGenerator(args.input_nc, args.output_nc, num_embeddings, args.ngf, norm_layer=norm_layer,
                               use_dropout=use_dropout, n_blocks=args.n_layers_G)
    else:
        norm_layer = get_norm_layer(args.norm_type)
        netG = AttentionalResnetGenerator(args.input_nc, args.output_nc, num_embeddings, args.ngf, norm_layer=norm_layer,
                                          use_dropout=use_dropout, n_blocks=args.n_layers_G, n_heads=args.n_heads)

    netD = NLayerDiscriminator(input_nc=args.input_nc + args.output_nc, ndf=args.ndf, n_layers=args.n_layers_D)

    init_weights(netG, args.init_type, args.init_gain)
    init_weights(netD, args.init_type, args.init_gain)

    return netG, netD


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s_ is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
