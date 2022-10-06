import torch
from torch import nn
import functools
from models.resnet import ResnetBlock


class SelfAttentionConditionEmbedding(nn.Module):

    def __init__(self, num_vocab: int, d_model: int, d_ffn: int, nhead: int, num_layers: int):
        super().__init__()
        w = torch.normal(0., 0.02, size=(1, num_vocab, d_model))
        self.embeddings = nn.Parameter(data=w, requires_grad=True)
        # self.encoder = nn.Sequential(*[
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ffn)
        #     for _ in range(num_layers)
        # ])

    def forward(self, x):
        x = self.embeddings * x.unsqueeze(-1)
        # x = self.encoder(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm([d_model, 128, 128])

    def forward(self, x, cond):
        """
        :param x: batch x ngf x w x h 크기의 텐서 (이미지)
        :param cond:  ngf 크기의 텐서 (환경변수)
        :return:
        """
        batch_size = x.size(0)
        res = x.view(batch_size, -1, self.d_model)
        res, _ = self.attn(res, cond, cond)
        res = res.reshape(*x.shape)
        return self.norm(x + res)


class AttentionalResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_embeddings, ngf=64, norm_layer=nn.BatchNorm2d, dropout=False, n_blocks=6, n_heads=4, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(AttentionalResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        down = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * mult * 2),
                     nn.ReLU(True),
                     # nn.Dropout2d(0.5), # 07/19 customization: 드랍아웃 추가
                     ]

        res = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            res += [
                AttentionBlock(d_model=ngf * mult, nhead=n_heads),
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias),
            ]

        up = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        up += [nn.ReflectionPad2d(3)]
        up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up += [nn.Tanh()]

        self.down = nn.Sequential(*down)
        self.res = nn.ModuleList(res)
        self.up = nn.Sequential(*up)
        self.condition_embedding = SelfAttentionConditionEmbedding(num_embeddings, ngf * 2 ** n_downsampling, 512, n_heads, num_layers=2)

    def forward(self, x, cond):
        """Standard forward w/ condition embeddings"""
        x = self.down(x)
        cond = self.condition_embedding(cond)
        for layer in self.res:
            if isinstance(layer, ResnetBlock): x = layer(x)
            else: x = layer(x, cond)
        x = self.up(x)
        return x