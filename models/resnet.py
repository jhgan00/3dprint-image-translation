import math
import torch
from torch import nn
import functools
import torch.nn.functional as F


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_embeddings, ngf=64, norm_layer=nn.BatchNorm2d, dropout=0.0, n_blocks=6, padding_type='reflect'):
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
        super(ResnetGenerator, self).__init__()
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
                     ]

        res = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            res += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]

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
        self.res = nn.Sequential(*res)
        self.up = nn.Sequential(*up)
        self.condition_embedding = ConditionEmbedding(num_embeddings, 256)
        self.position_embedding = PositionEmbedding(256)

    def forward(self, x, cond):
        """Standard forward w/ condition embeddings"""
        x = self.down(x)
        x = self.position_embedding(x)
        x = self.condition_embedding(x, cond)  # convnet feature - printer parameter attention
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
            use_dropout (bool)  -- if use dropout layers.
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
            conv_block += [nn.Dropout(dropout)]

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


class ConditionEmbedding(nn.Module):

    def __init__(self, num_vocab: int, embed_dim: int):
        super().__init__()
        w = torch.empty(size=(num_vocab, embed_dim))
        nn.init.normal_(w)
        self.embeddings = nn.Parameter(data=w, requires_grad=True)
        self.attn = AttentionBlock(256, 1)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor):
        conditions = self.embeddings.unsqueeze(0) * conditions.unsqueeze(-1)
        return self.attn(x, conditions)


class AttentionBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attn = MultiHeadAttention(d_model, nhead)
        self.norm = nn.LayerNorm([d_model, 128, 128])

    def forward(self, x, cond):
        """
        :param x: batch x ngf x w x h 크기의 텐서 (이미지)
        :param cond:  ngf 크기의 텐서 (환경변수)
        :return:
        """
        batch_size = x.size(0)
        res = x.view(batch_size, -1, self.d_model)
        res = self.attn(res, cond, cond)
        res = res.reshape(*x.shape)
        return self.norm(x + res)


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 3. pass them softmax to make [0, 1] range
        score = F.sigmoid(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v)
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_heads
        tensor = tensor.view(batch_size, length, self.n_heads, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Parameter(data=torch.normal(0, 1e-2, (d_model, 128, 128)), requires_grad=True)

    def forward(self, x):
        return x + self.embedding
