import torch
import torch.nn as nn

class SeparableConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, use_bn=True, preact=False):
        super().__init__()

        padding = ksize - stride
        pad_layer = nn.ConstantPad3d(padding // 2 if padding % 2 == 0 else (padding % 2, padding - padding % 2) * 3, 0)

        layers = []
        if preact:
            if use_bn:
                layers.append(nn.BatchNorm3d(in_ch))
            layers.append(nn.ReLU())
        layers.extend([
            pad_layer,
            nn.Conv3d(in_ch, in_ch, kernel_size=ksize, stride=stride, groups=in_ch, bias=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, bias=not use_bn)
        ])
        if use_bn and not preact:
            layers.append(nn.BatchNorm3d(out_ch))
        if not preact:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StageAwareFusionModule(nn.Module):
    def __init__(self, stage_channels, pool_scales, inner_ch, key_ch, val_ch, branches):
        super().__init__()
        self.final_ch = stage_channels[-1]
        self.mid_ch = inner_ch
        self.key_ch = key_ch
        self.val_ch = val_ch
        self.num_branches = branches
        self.concat_ch = self.mid_ch * self.num_branches

        self.fusion_layers = nn.ModuleList([
            SeparableConvUnit(in_ch, self.mid_ch, 3, preact=True)
            for in_ch in stage_channels
        ])

        self.pool_layers = nn.ModuleList([
            nn.MaxPool3d(kernel_size=s, stride=s)
            for s in pool_scales
        ])

        self.ch_attn_q = SeparableConvUnit(self.concat_ch, self.concat_ch, 1, preact=True)
        self.ch_attn_k = SeparableConvUnit(self.concat_ch, 1, 1, preact=True)
        self.ch_attn_v = SeparableConvUnit(self.concat_ch, self.concat_ch, 1, preact=True)
        self.ch_score_layer = nn.Conv3d(self.concat_ch, self.concat_ch, 1)
        self.ch_norm = nn.LayerNorm((self.concat_ch, 1, 1, 1))
        self.act = nn.Sigmoid()

        self.sp_attn_q = SeparableConvUnit(self.concat_ch, self.num_branches * self.key_ch, 1, preact=True)
        self.sp_attn_k = SeparableConvUnit(self.concat_ch, self.num_branches * self.key_ch, 1, preact=True)
        self.sp_attn_v = SeparableConvUnit(self.concat_ch, self.num_branches * self.val_ch, 1, preact=True)
        self.sp_final = SeparableConvUnit(self.num_branches * self.val_ch, self.concat_ch, 1, preact=True)

        self.final_conv = SeparableConvUnit(self.concat_ch, self.final_ch, 3, preact=True)

        self.softmax_ch = nn.Softmax(dim=1)
        self.softmax_sp = nn.Softmax(dim=-1)

    def forward(self, inputs):
        pooled = [pool(inp) for pool, inp in zip(self.pool_layers, inputs)]
        feats = [block(f) for block, f in zip(self.fusion_layers, pooled)]
        x = torch.cat(feats, dim=1)  # (B, C_concat, D, H, W)

        B, C, D, H, W = x.shape

        # Channel attention
        q_c = self.ch_attn_q(x).reshape(B, -1, D * H * W)
        k_c = self.ch_attn_k(x).reshape(B, -1, 1)
        k_c = self.softmax_ch(k_c)
        z_c = torch.matmul(q_c, k_c).view(B, -1, 1, 1, 1)
        score_c = self.act(self.ch_norm(self.ch_score_layer(z_c)))
        x_attn = self.ch_attn_v(x) * score_c

        # Spatial attention
        q_s = self.sp_attn_q(x_attn).reshape(B, self.num_branches, self.key_ch, D, H, W).permute(0, 2, 3, 4, 5, 1).reshape(B, self.key_ch, -1)
        k_s = self.sp_attn_k(x_attn).reshape(B, self.num_branches, self.key_ch, D, H, W).permute(0, 2, 3, 4, 5, 1).mean([-1, -2, -3, -4]).reshape(B, 1, self.key_ch)
        v_s = self.sp_attn_v(x_attn).reshape(B, self.num_branches, self.key_ch, D, H, W).permute(0, 2, 3, 4, 5, 1)

        k_s = self.softmax_sp(k_s)
        z_s = torch.matmul(k_s, q_s).reshape(B, 1, D, H, W, self.num_branches)
        score_s = self.act(z_s)
        out_s = v_s * score_s
        out_s = out_s.permute(0, 5, 1, 2, 3, 4).reshape(B, self.num_branches * self.val_ch, D, H, W)
        fused = self.sp_final(out_s)

        return self.final_conv(fused)