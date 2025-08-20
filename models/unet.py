import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# time embedding module
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        # sin and cos position encoding
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        # embedding layer
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

# physical feature embedding module
class ConditionEmbedding(nn.Module):
    def __init__(self, num_classes, num_continuous_features, dim):
        super().__init__()
        self.num_continuous_features = num_continuous_features

        # category emb
        self.class_embed = nn.Sequential(
            nn.Embedding(num_classes, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # continuous feature processing
        self.continuous_net = nn.Sequential(
            nn.BatchNorm1d(num_continuous_features),  # 自动归一化连续特征
            nn.Linear(num_continuous_features, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # feature fusion (attention mechanism)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, conditions):
        class_idx = conditions[:, -1].long()
        continuous_feats = conditions[:, : self.num_continuous_features]

        # processing class and continuous features
        class_emb = self.class_embed(class_idx)
        continuous_emb = self.continuous_net(continuous_feats)

        # concat and attentional weighted fusion
        combined = torch.cat([class_emb, continuous_emb], dim=-1)
        weights = torch.sigmoid(self.fusion(combined))
        output = weights * class_emb + (1 - weights) * continuous_emb

        return output

# downsampling module
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    # temb and cemb are placeholder for coding convenient
    def forward(self, x, temb, cemb):
        x = self.main(x)
        return x

# upsampling module
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    # temb and cemb are placeholder for coding convenient
    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, cdim, dropout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        # fc that takes in time emb
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        # fc that takes in condition emb
        self.cemb_proj = nn.Sequential(
            Swish(),
            nn.Linear(cdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, cemb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cemb_proj(cemb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        return h


class UNet(nn.Module):
    def __init__(self, T, img_ch, ch, ch_mult, num_res_blocks, dropout, num_classes, num_features):
        super().__init__()
        tdim = ch * 4
        cdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.condition_embedding = ConditionEmbedding(num_classes, num_features, cdim)  # condition embedding
        # head layer controls output to be 5 channel
        self.head = nn.Conv2d(img_ch, ch, kernel_size=3, stride=1, padding=1)
        # create encoder sequential of resnet blocks and downsampling blocks
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when downsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch,
                    out_ch=out_ch,
                    tdim=tdim,
                    cdim=cdim,
                    dropout=dropout
                ))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # bottleneck layer
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, cdim, dropout),
            ResBlock(now_ch, now_ch, tdim, cdim, dropout),
        ])

        # create decoder sequential of resnet blocks and upsampling blocks
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch,
                    out_ch=out_ch,
                    tdim=tdim,
                    cdim=cdim,
                    dropout=dropout
                ))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # tail layer control outputs to be 5 channel
        self.tail = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            Swish(),
            nn.Conv2d(now_ch, img_ch, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, c):
        # Timestep embedding
        temb = self.time_embedding(t)
        cemb = self.condition_embedding(c)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        # Bottleneck
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
