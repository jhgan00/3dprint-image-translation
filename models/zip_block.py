import math
import torch
from torch import nn
import functools
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ZipBlock(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 c_dim,
                 norm_layer=nn.BatchNorm2d,
                 dropout=0.5,
                 n_blocks=9,
                 padding_type='reflect'
                 ):
        assert (n_blocks >= 0)
        super(ZipBlock, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc + c_dim, ngf, kernel_size=7, padding=0, bias=use_bias),
                # nn.Conv2d(2*input_nc*c_dim, ngf, kernel_size=7, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * mult * 2),
                     nn.ReLU(True)
                     ]
        res = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNeXt blocks
            res += [
                ResnetBlock(dim=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout,
                            use_bias=use_bias)]

        up = []
        mult = 2 ** n_downsampling
        up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1,
                                  bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        for i in range(n_downsampling - 1):  # add upsampling layers
            mult = 2 ** (n_downsampling - i - 1)
            up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        up += [nn.ReflectionPad2d(3)]
        up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up += [nn.Tanh()]

        self.zipping = Concatting()
        # self.zipping = Zipping(dim=input_nc, c_dim=c_dim, use_bias=use_bias)
        self.down = nn.Sequential(*down)
        self.res = nn.Sequential(*res)
        self.up = nn.Sequential(*up)

    def forward(self, x, cond):
        x = self.zipping(x, cond)
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Concatting(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, cond):
        lst_a = []
        for vector in cond:
            lst_b = []
            for j in vector:
                a = torch.ones(1, 1, x.shape[2], x.shape[3]).to(x.device) * j
                lst_b.append(a)
            y = torch.cat([*lst_b], 1)
            lst_a.append(y)
        z = torch.cat([*lst_a], 0)
        return torch.cat([x, z], 1)
