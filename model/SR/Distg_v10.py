import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
import math
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        angRes = args.angRes_in
        self.n_groups = args.n_groups
        self.n_blocks = args.n_blocks
        self.channels = args.channels
        self.angRes = args.angRes_in
        self.upscale_factor = args.scale_factor

        self.FEM = FEM(channels=self.channels)

        self.altblock = nn.Sequential(
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
            BasicWindow(channels=self.channels, window_size=5),
        )

        HybridFeatureExtractionModule = [
            HFEM(angRes, self.n_blocks, self.channels)
            for _ in range(self.n_groups)
        ]
        HybridFeatureExtractionModule.append(nn.Conv2d(self.channels, self.channels, kernel_size=3,stride =1,dilation=1,padding=1, bias=False))


        # self.fuse_conv = nn.Conv2d(2*self.channels, self.channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)
        self.init_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1,
                                   dilation=self.angRes, padding=self.angRes, bias=False)

        upsample = [Upsampler(self.upscale_factor, self.channels, kernel_size=3, stride=1, dilation=1, padding=1, act=False),
                    nn.Conv2d(self.channels, 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)]

        self.disentg = nn.Sequential(*HybridFeatureExtractionModule)
        self.upsample = nn.Sequential(*upsample)


    def forward_disg(self, x, x_feat):
        # b c (u h) (v w) -> b c (h u) (w v)
        buffer = SAI2MacPI(x, self.angRes)
        buffer = self.init_conv(buffer) + x_feat
        buffer = self.disentg(buffer) + x_feat
        "the shape of the buffer is (1, 64, 160, 160)"
        # b c (h u) (w v) -> b c (u h) (v w)
        buffer_SAI = MacPI2SAI(buffer, self.angRes)
        out = self.upsample(buffer_SAI)

        return out



    def forward(self, x, info=None):
        # Bicubic Upscaling
        sr = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        x = SAI2MacPI(x, self.angRes)

        # shallow feature extract
        buffer = self.FEM(x)

        distg_x = MacPI2SAI(buffer, self.angRes)

        # Deep Spatial-Angular Correlation Learning
        "the shape of buffer (1, 64, 160, 160)"
        buffer = self.altblock(buffer) + buffer

        # UP-Sampling
        y = MacPI2SAI(buffer, self.angRes)
        y = self.upsample(y)

        # SSR
        y_1 = self.forward_disg(distg_x, buffer)

        sr = 0.5*y + 0.5*y_1 + sr

        return sr


class HFEM(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(HFEM, self).__init__()
        self.n_blocks = n_blocks
        self.angRes = angRes

        self.SAS_conv = SAS_conv(channels, angRes)

        self.SAC_conv = SAC_conv(channels, angRes)

        self.LeakyRelu = nn.LeakyReLU(0.1, inplace=True)

        self.AFM = AttentionFusion(channels)

        self.SRG = ResidualGroup(self.n_blocks, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self,x):

        fea_SAS = self.SAS_conv(x)

        fea_SAC = self.SAC_conv(x)

        fea_SAS = fea_SAS.unsqueeze(1)
        buffer = torch.cat([fea_SAC.unsqueeze(1), fea_SAS], dim=1)

        buffer = self.AFM(buffer)
        buffer = self.SRG(buffer)


        return buffer + x


class SAS_conv(nn.Module):
    def __init__(self, channels, angRes):
        super(SAS_conv, self).__init__()
        self.spa_intra = nn.Conv2d(channels, channels, kernel_size=3, dilation=int(angRes), padding=int(angRes), bias=False)
        self.ang_intra = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.ang2spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        buffer = self.spa_intra(x)
        buffer = self.act(buffer)

        buffer = self.ang_intra(buffer)
        buffer = self.act(buffer)
        buffer = self.ang2spa(buffer)

        return buffer


class SAC_conv(nn.Module):
    def __init__(self, channels, angRes):
        super(SAC_conv, self).__init__()
        self.angRes = angRes

        self.init_indicator = 'relu'
        self.act = nn.ReLU(inplace=True)
        a = 0

        self.ver_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        init.kaiming_normal_(self.ver_conv.weight, a, 'fan_in', self.init_indicator)
        # init.constant_(self.ver_conv.bias, 0.0)

        self.hor_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        init.kaiming_normal_(self.hor_conv.weight, a, 'fan_in', self.init_indicator)
        # init.constant_(self.hor_conv.bias, 0.0)

    def forward(self, x):
        x = rearrange(x, 'b c (h u) (w v) -> b c u v h w', u=self.angRes, v=self.angRes)
        N, c, U, V, h, w = x.shape

        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N * V * w, c, U, h)

        # vertical
        out = self.act(self.ver_conv(x))
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * U * h, c, V, w)

        # horizontal
        out = self.act(self.hor_conv(out))
        out = out.view(N, U * h, c, V * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, V, w, c, U, h)
        out = rearrange(out, "b v w c u h -> b c (h u) (w v)")

        return out

class AttentionFusion(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(AttentionFusion, self).__init__()

        self.epsilon = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.conv = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)

    def forward(self, x):
        m_batchsize, N, C, height, width = x.size()
        x_reshape = x.view(m_batchsize, N, -1)
        M = C * height * width

        # compute covariance feature
        mean = torch.mean(x_reshape, dim=-1).unsqueeze(-1)
        x_reshape = x_reshape - mean
        cov = (1 / (M - 1) * x_reshape @ x_reshape.transpose(-1, -2)) * self.alpha
        # print(cov)
        norm = cov / ((cov.pow(2).mean((1, 2), keepdim=True) + self.epsilon).pow(0.5))  # l-2 norm

        attention = torch.tanh(self.gamma * norm + self.beta)
        x_reshape = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, x_reshape)
        out = out.view(m_batchsize, N, C, height, width)

        out += x
        out = out.view(m_batchsize, -1, height, width)
        out = self.conv(out)

        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class ResidualBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, stride, dilation, padding, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # # initialization
        # initialize_weights([self.conv1, self.conv2], 0.1)
        self.CALayer = CALayer(n_feat, reduction=int(n_feat // 4))

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.CALayer(out)
        return x + out


## Residual Group
class ResidualGroup(nn.Module):
    def __init__(self, n_blocks, n_feat, kernel_size, stride, dilation, padding, bias=True):
        super(ResidualGroup, self).__init__()

        self.fea_resblock = make_layer(ResidualBlock, n_feat, n_blocks, kernel_size, stride, dilation, padding)
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation,
                              padding=padding, bias=True)

    def forward(self, x):
        res = self.fea_resblock(x)
        res = self.conv(res)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, kernel_size, stride, dilation, padding, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                   padding=padding, bias=True))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding, bias=True))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResASSP(nn.Module):
    def __init__(self, channels):
        super(ResASSP, self).__init__()
        self.conv_d1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.conv_d2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.conv_d4 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channels*3, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        buffer = []
        buffer.append(self.conv_d1(x))
        buffer.append(self.conv_d2(x))
        buffer.append(self.conv_d4(x))
        buffer = torch.cat(buffer, dim=1)
        buffer = self.conv_t(buffer)

        return buffer + x

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)

        return buffer

class FEM(nn.Module):
    def __init__(self, channels):
        super(FEM, self).__init__()
        self.FEconv = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ResASSP_1 = ResASSP(channels)
        self.RB_1 = RB(channels)
        self.ResASSP_2 = ResASSP(channels)
        self.RB_2 = RB(channels)

    def forward(self, x):
        buffer = self.FEconv(x)
        buffer = self.ResASSP_1(buffer)
        buffer = self.RB_1(buffer)
        buffer = self.ResASSP_2(buffer)
        buffer = self.RB_2(buffer)

        return buffer

class MLP(nn.Module):
    def __init__(self, in_features, hidden_feature=None, out_feature=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_feature = out_feature or in_features
        hidden_feature = hidden_feature or in_features
        self.fc1 = nn.Linear(in_features, hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0,):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] -1) * (2 * window_size[1] -1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn +relative_position_bias

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_head=4,
                 window_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(WinTransformerBlock,self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size=to_2tuple(window_size),
                                    num_heads=num_head,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_feature=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):

        shape = x.shape
        H = shape[2]
        W = shape[3]
        x = x.flatten(2).permute(0, 2, 1)

        B, L, C = x.shape

        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 划分窗口
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

        # 计算注意力
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # 将窗口还原为原本的图像
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, H*W, C)
        x = shortcut+x

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x

class BasicWindow(nn.Module):
    def __init__(self, channels, window_size):
        super(BasicWindow, self).__init__()
        self.window_trans_5 = WinTransformerBlock(dim=channels, window_size=window_size)
        self.window_trans_10 = WinTransformerBlock(dim=channels, window_size=2*window_size)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):

        # intra_angel
        buffer_intra = self.window_trans_5(x)

        # inter_angel
        buffer_inter = self.window_trans_10(x)

        # buffer = torch.cat((buffer_intra,buffer_inter), dim=1)
        buffer = buffer_inter+buffer_intra

        buffer = self.conv(buffer)

        return buffer + x


def make_layer(block, nf, n_layers, kernel_size, stride, dilation, padding):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf, kernel_size, stride, dilation, padding))
    return nn.Sequential(*layers)


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def MacPI2EPI(x, angRes):
    data_0 = []
    data_90 = []
    data_45 = []
    data_135 = []

    index_center = int(angRes // 2)
    for i in range(0, angRes, 1):
        img_tmp = x[:, :, index_center::angRes, i::angRes]
        data_0.append(img_tmp)
    data_0 = torch.cat(data_0, 1)

    for i in range(0, angRes, 1):
        img_tmp = x[:, :, i::angRes, index_center::angRes]
        data_90.append(img_tmp)
    data_90 = torch.cat(data_90, 1)

    for i in range(0, angRes, 1):
        img_tmp = x[:, :, i::angRes, i::angRes]
        data_45.append(img_tmp)
    data_45 = torch.cat(data_45, 1)

    for i in range(0, angRes, 1):
        img_tmp = x[:, :, i::angRes, angRes - i - 1::angRes]
        data_135.append(img_tmp)
    data_135 = torch.cat(data_135, 1)

    return data_0, data_90, data_45, data_135


def SAI24DLF(x, angRes):
    uh, vw = x.shape
    h0, w0 = int(uh // angRes), int(vw // angRes)
    LFout = torch.zeros(angRes, angRes, h0, w0)

    for u in range(angRes):
        start_u = u * h0
        end_u = (u + 1) * h0
        for v in range(angRes):
            start_v = v * w0
            end_v = (v + 1) * w0
            img_tmp = x[start_u:end_u, start_v:end_v]
            LFout[u, v, :, :] = img_tmp

    return LFout


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


def Convert4DLF2SAI(x, angRes):
    u, v, h, w = x.shape
    LFout = torch.zeros(1, 1, u * h, v * w)

    for u in range(angRes):
        start_u = u * h
        end_u = (u + 1) * h
        for v in range(angRes):
            start_v = v * w
            end_v = (v + 1) * w
            img_tmp = x[u, v, :, :]
            LFout[:, :, start_u:end_u, start_v:end_v] = img_tmp

    return LFout


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss