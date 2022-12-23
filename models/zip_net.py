"""lee"""
import torch
import torch.nn as nn  #
from torch.nn import init  #
import functools  #


class ZipUNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, c_dim, ngf=64, norm_layer=nn.BatchNorm2d, dropout=0.5):
        super(ZipUNet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.zipping = Zipping(dim=input_nc, c_dim=c_dim, use_bias=use_bias)  # add ZipBlock Class
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                                 submodule=unet_block, norm_layer=norm_layer, dropout=dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=c_dim, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, x, cond):
        x = self.zipping(x, cond)
        x = self.model(x)
        return x


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Zipping(nn.Module):
    def __init__(self, dim, c_dim, use_bias):
        super(Zipping, self).__init__()
        self.groups = self.build_g(dim, c_dim, use_bias)

    def build_g(self, dim, c_dim, use_bias):
        return nn.Sequential(nn.Conv2d((1 + dim) * c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim, bias=use_bias))

    def forward(self, x, cond):
        g_conv = self.groups
        #         lst_a = []
        for i, (vector, img) in enumerate(zip(cond, x)):
            for j, scalar in enumerate(vector):
                a = torch.ones(1, x.shape[2], x.shape[3]).to(x.device) * scalar
                a = torch.cat([img, a], 0)
                if j == 0:
                    temp_a = a
                else:
                    temp_a = torch.cat([temp_a, a], 0)
            if i == 0:
                temp_b = temp_a.repeat(1, 1, 1, 1)
            else:
                temp_a = temp_a.repeat(1, 1, 1, 1)
                temp_b = torch.cat([temp_b, temp_a], 0)
        return g_conv(temp_b)


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
