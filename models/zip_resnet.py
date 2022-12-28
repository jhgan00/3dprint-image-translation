"""lee"""
import torch
import torch.nn as nn #
import functools #


class ZipBlockResNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, c_dim, norm_layer=nn.BatchNorm2d, dropout=0.5, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ZipBlockResNet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down = [nn.ReflectionPad2d(3),
                nn.Conv2d(c_dim, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]

        res = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add blocks
            res += [ResnetBlock(dim=ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]

        up = []
        mult = 2 ** n_downsampling
        up += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        for i in range(n_downsampling - 1) :  # add upsampling layers
            mult = 2 ** (n_downsampling - i - 1)
            up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        up += [nn.ReflectionPad2d(3)]
        up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up += [nn.Tanh()]

        self.zipping = Zipping(dim=input_nc, c_dim=c_dim, use_bias=use_bias)
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
    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias):
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
        out = x + self.conv_block(x)  # add skip connections
        return out

class Zipping(nn.Module):
    def __init__(self, dim, c_dim, use_bias):
        super(Zipping, self).__init__()
        self.groups = self.build_g(dim, c_dim, use_bias)

    def build_g(self, dim, c_dim, use_bias):
        return nn.Sequential(nn.Conv2d((1+dim)*c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim, bias=use_bias))

    def forward(self, x, cond):
        g_conv = self.groups
        #         lst_a = []
        for i, (vector, img) in enumerate(zip(cond, x)):
            for j, scalar in enumerate(vector):
                a = torch.ones(1, x.shape[2], x.shape[3]).to(x.device) * scalar
                a = torch.cat([img, a], 0)
                if j == 0 :
                    temp_a = a
                else :
                    temp_a = torch.cat([temp_a, a], 0)
            if i == 0 :
                temp_b = temp_a.repeat(1,1,1,1)
            else :
                temp_a = temp_a.repeat(1,1,1,1)
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
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
