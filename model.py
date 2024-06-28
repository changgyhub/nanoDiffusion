import torch
from torch import nn

from data import get_mnist_tensor_shape


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be divisible by 2"

        pos_seq = torch.linspace(0, seq_len - 1, seq_len)
        dim_seq = torch.linspace(0, dim - 2, dim // 2)
        pos, dim_2i = torch.meshgrid([pos_seq, dim_seq], indexing="ij")
        pe_2i = torch.sin(pos / (10000 ** (dim_2i / dim)))
        pe_2i_plus_1 = torch.cos(pos / (10000 ** (dim_2i / dim)))

        self.embedding = nn.Embedding(seq_len, dim)
        self.embedding.weight.data = torch.stack((pe_2i, pe_2i_plus_1), 2).reshape(seq_len, dim)
        self.embedding.requires_grad_(False)

    def forward(self, x):
        return self.embedding(x)


class ConvBlock(nn.Module):

    def __init__(self, shape, in_channels, out_channels, residual):
        super().__init__()
        self.norm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_channels == out_channels:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv2(self.activation(self.conv1(self.norm(x))))
        if self.residual:
            out += self.residual_conv(x)
        return self.activation(out)


class UNet(nn.Module):

    def __init__(self, n_steps, block_channel_multiplier=4, num_blocks=3, pe_dim=32, residual=True):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(pe_dim, n_steps)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_layer_encoders = nn.ModuleList()
        self.pe_layer_decoders = nn.ModuleList()
        self.downscales = nn.ModuleList()
        self.upscales = nn.ModuleList()
        
        # Pre-compute model architecture.
        C, H, W = get_mnist_tensor_shape()
        channels, Hs, Ws = [block_channel_multiplier], [H], [W]
        for _ in range(num_blocks - 1):
            channels.append(channels[-1]*2)
            Hs.append(Hs[-1]//2)
            Ws.append(Ws[-1]//2)

        # Define encoders.
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            self.pe_layer_encoders.append(nn.Sequential(
                nn.Linear(pe_dim, prev_channel),
                nn.ReLU(),
                nn.Linear(prev_channel, prev_channel))
            )
            self.encoders.append(nn.Sequential(
                ConvBlock((prev_channel, cH, cW), prev_channel, channel, residual),
                ConvBlock((channel, cH, cW), channel, channel, residual))
            )
            self.downscales.append(nn.Conv2d(channel, channel, kernel_size=2, stride=2))
            prev_channel = channel
        self.pe_layer_encoders.append(nn.Linear(pe_dim, prev_channel))
        channel = channels[-1]
        self.encoders.append(nn.Sequential(
            ConvBlock((prev_channel, Hs[-1], Ws[-1]), prev_channel, channel, residual),
            ConvBlock((channel, Hs[-1], Ws[-1]), channel, channel, residual),
        ))
        self.downscales.append(nn.Identity())
        prev_channel = channel

        # Define decoders.
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.pe_layer_decoders.append(nn.Linear(pe_dim, prev_channel))
            self.upscales.append(nn.ConvTranspose2d(prev_channel, channel, kernel_size=2, stride=2))
            self.decoders.append(nn.Sequential(
                ConvBlock((channel * 2, cH, cW), channel * 2, channel, residual),
                ConvBlock((channel, cH, cW), channel, channel, residual))
            )
            prev_channel = channel
        self.conv_out = nn.Conv2d(prev_channel, C, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_layer, encoder, downscale in zip(self.pe_layer_encoders, self.encoders, self.downscales):
            pe = pe_layer(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = downscale(x)
        for pe_layer, decoder, upscale, encoder_out in zip(self.pe_layer_decoders, self.decoders, self.upscales, encoder_outs[-2::-1]):
            pe = pe_layer(t).reshape(n, -1, 1, 1)
            x = upscale(x)
            padH = encoder_out.shape[2] - x.shape[2]
            padW = encoder_out.shape[3] - x.shape[3]
            x = nn.functional.pad(x, (padH // 2, padH - padH // 2, padW // 2, padW - padW // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x
