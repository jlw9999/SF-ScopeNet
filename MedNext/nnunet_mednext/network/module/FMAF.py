import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]  # Get depth, height, and width dimensions
        return rearrange(self.body(rearrange(x, 'b c d h w -> b (d h w) c')), 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class FMAF(nn.Module):
    def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias'):
        super(FMAF, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1)

        self.conv1_1_1 = nn.Conv3d(dim, dim, (1, 7, 7), padding=(0, 3, 3), groups=dim)
        self.conv1_1_2 = nn.Conv3d(dim, dim, (1, 11, 11), padding=(0, 5, 5), groups=dim)
        self.conv1_1_3 = nn.Conv3d(dim, dim, (1, 21, 21), padding=(0, 10, 10), groups=dim)
        self.conv1_2_1 = nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim)
        self.conv1_2_2 = nn.Conv3d(dim, dim, (11, 1, 1), padding=(5, 0, 0), groups=dim)
        self.conv1_2_3 = nn.Conv3d(dim, dim, (21, 1, 1), padding=(10, 0, 0), groups=dim)

        self.conv2_1_1 = nn.Conv3d(dim, dim, (1, 7, 7), padding=(0, 3, 3), groups=dim)
        self.conv2_1_2 = nn.Conv3d(dim, dim, (1, 11, 11), padding=(0, 5, 5), groups=dim)
        self.conv2_1_3 = nn.Conv3d(dim, dim, (1, 21, 21), padding=(0, 10, 10), groups=dim)
        self.conv2_2_1 = nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim)
        self.conv2_2_2 = nn.Conv3d(dim, dim, (11, 1, 1), padding=(5, 0, 0), groups=dim)
        self.conv2_2_3 = nn.Conv3d(dim, dim, (21, 1, 1), padding=(10, 0, 0), groups=dim)

    def forward(self, x1, x2):
        b, c, d, h, w = x1.shape

        # Normalize input features
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        # Perform FFT on both inputs
        freq_x1 = torch.fft.fftn(x1, dim=(-3, -2, -1))
        freq_x2 = torch.fft.fftn(x2, dim=(-3, -2, -1))

        # Separate low and high frequencies for x1
        low_freq_x1 = freq_x1.clone()
        high_freq_x1 = freq_x1.clone()

        # 高通滤波：清零中心低频区域
        high_freq_x1[..., d//4:3*d//4, h//4:3*h//4, w//4:3*w//4] = 0

        # 低通滤波：清零所有边缘高频区域（6个面）
        low_freq_x1[..., :d//4, :, :] = 0        # D dimension first 1/4
        low_freq_x1[..., 3*d//4:, :, :] = 0      # D dimension rear 1/4
        low_freq_x1[..., :, :h//4, :] = 0        # H dimension first 1/4
        low_freq_x1[..., :, 3*h//4:, :] = 0      # H dimension rear 1/4
        low_freq_x1[..., :, :, :w//4] = 0        # W dimension first 1/4
        low_freq_x1[..., :, :, 3*w//4:] = 0      # W dimension rear 1/4

        # Separate low and high frequencies for x2
        low_freq_x2 = freq_x2.clone()
        high_freq_x2 = freq_x2.clone()

        # 高通滤波：清零中心低频区域
        high_freq_x2[..., d//4:3*d//4, h//4:3*h//4, w//4:3*w//4] = 0

        # 低通滤波：清零所有边缘高频区域（6个面）
        low_freq_x2[..., :d//4, :, :] = 0        # D dimension first 1/4
        low_freq_x2[..., 3*d//4:, :, :] = 0      # D dimension rear 1/4
        low_freq_x2[..., :, :h//4, :] = 0        # H dimension first 1/4
        low_freq_x2[..., :, 3*h//4:, :] = 0      # H dimension rear 1/4
        low_freq_x2[..., :, :, :w//4] = 0        # W dimension first 1/4
        low_freq_x2[..., :, :, 3*w//4:] = 0      # W dimension rear 1/4

        # Reconstruct back to spatial domain
        high_freq_x1 = torch.fft.ifftn(high_freq_x1, dim=(-3, -2, -1)).real
        low_freq_x1 = torch.fft.ifftn(low_freq_x1, dim=(-3, -2, -1)).real

        high_freq_x2 = torch.fft.ifftn(high_freq_x2, dim=(-3, -2, -1)).real
        low_freq_x2 = torch.fft.ifftn(low_freq_x2, dim=(-3, -2, -1)).real

        # Enhance x1 with low-frequency and x2 with high-frequency
        x1 = x1 + low_freq_x1
        x2 = x2 + high_freq_x2

        # Apply convolutions (same structure as 2D, but in 3D)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        out1 = attn_111 + attn_112 + attn_113 + attn_121 + attn_122 + attn_123

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out2 = attn_211 + attn_212 + attn_213 + attn_221 + attn_222 + attn_223

        # Combine outputs and project back
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)

        # Rearranging to match attention mechanism
        k1 = rearrange(out1, 'b (head c) d h w -> b head d (h w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) d h w -> b head d (h w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) d h w -> b head d (h w c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) d h w -> b head d (h w c)', head=self.num_heads)

        q2 = rearrange(out1, 'b (head c) d h w -> b head d (h w c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) d h w -> b head d (h w c)', head=self.num_heads)

        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1

        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2

        # Rearrange the output tensors back to the original shape
        out3 = rearrange(out3, 'b head d (h w c) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)
        out4 = rearrange(out4, 'b head d (h w c) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out3) + self.project_out(out4) + x1 + x2
        return out