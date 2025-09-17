import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class InputWrapper(torch.nn.Module):
    """Wrapper to handle time concatenation consistently."""
    def __init__(self, v):
        super(InputWrapper, self).__init__()
        self.v = v

    def net_inp(self, t, x):
        """Concatenate time over the batch dimension."""
        t = t.to(dtype=x.dtype, device=x.device)
        inp = torch.cat((t.repeat(x.shape[0]).unsqueeze(1), x), dim=1)
        return inp

    def forward(self, x, t):
        self.v.to(x)
        tx = self.net_inp(t, x)
        return self.v(tx)


# 1. FOURIER FEATURE NETWORK (Highly Recommended)
def make_fourier_net(
    hidden_sizes,
    in_size,
    out_size,
    fourier_features=256,
    fourier_scale=10.0,
    inner_act='gelu',
    final_act='none',
    use_layernorm=True
):
    """
    Network with random Fourier features for better high-frequency learning.
    This helps capture sharp boundaries like in the checkerboard.
    """

    class FourierFeatures(nn.Module):
        def __init__(self, in_features, out_features, scale=10.0):
            super().__init__()
            assert out_features % 2 == 0
            self.register_buffer(
                'weight',
                torch.randn(in_features, out_features // 2) * scale
            )

        def forward(self, x):
            x_proj = x @ self.weight
            return torch.cat(
                [torch.sin(2 * np.pi * x_proj),
                 torch.cos(2 * np.pi * x_proj)], dim=-1
            )

    net = []

    # Add Fourier feature layer
    net.append(FourierFeatures(in_size, fourier_features, scale=fourier_scale))

    # Build hidden layers
    sizes = [fourier_features] + hidden_sizes + [out_size]
    for i in range(len(sizes) - 1):
        net.append(nn.Linear(sizes[i], sizes[i+1]))

        if i != len(sizes) - 2:  # Not the last layer
            if use_layernorm:
                net.append(nn.LayerNorm(sizes[i+1]))
            net.append(make_activation(inner_act))
        else:  # Last layer
            if final_act != 'none':
                net.append(make_activation(final_act))

    v_net = nn.Sequential(*net)
    return InputWrapper(v_net)


# 2. RESIDUAL NETWORK WITH SKIP CONNECTIONS
def make_resnet(
    hidden_sizes,
    in_size,
    out_size,
    inner_act='gelu',
    final_act='none',
    use_layernorm=True,
    dropout=0.0
):
    """
    Residual network with skip connections for better gradient flow.
    Helps train deeper networks for complex velocity fields.
    """

    class ResBlock(nn.Module):
        def __init__(self, dim, activation='gelu', use_layernorm=True, dropout=0.0):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.activation = make_activation(activation)
            self.layernorm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            residual = x
            x = self.layernorm(x)
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x + residual

    net = []

    # Input projection
    if hidden_sizes:
        hidden_dim = hidden_sizes[0]
        net.append(nn.Linear(in_size, hidden_dim))
        net.append(make_activation(inner_act))

        # Residual blocks
        for _ in range(len(hidden_sizes)):
            net.append(ResBlock(hidden_dim, inner_act, use_layernorm, dropout))

        # Output projection
        net.append(nn.Linear(hidden_dim, out_size))
    else:
        net.append(nn.Linear(in_size, out_size))

    if final_act != 'none':
        net.append(make_activation(final_act))

    v_net = nn.Sequential(*net)
    return InputWrapper(v_net)


# 3. SIREN (Sinusoidal Representation Network)
def make_siren(
    hidden_sizes,
    in_size,
    out_size,
    omega_0=30.0,
    final_act='none'
):
    """
    SIREN: Networks with periodic activations.
    Excellent for representing functions with sharp features.
    """

    class SineLayer(nn.Module):
        def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
            super().__init__()
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_features, out_features)

            # Special initialization for SIREN
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
                else:
                    self.linear.weight.uniform_(
                        -np.sqrt(6 / in_features) / omega_0,
                        np.sqrt(6 / in_features) / omega_0
                    )

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    net = []

    # Build SIREN layers
    sizes = [in_size] + hidden_sizes + [out_size]
    for i in range(len(sizes) - 1):
        is_first = (i == 0)
        is_last = (i == len(sizes) - 2)

        if is_last:
            # Final layer is linear
            net.append(nn.Linear(sizes[i], sizes[i+1]))
            if final_act != 'none':
                net.append(make_activation(final_act))
        else:
            net.append(SineLayer(sizes[i], sizes[i+1], omega_0, is_first))

    v_net = nn.Sequential(*net)
    return InputWrapper(v_net)


# 4. MULTI-SCALE NETWORK
def make_multiscale_net(
    hidden_sizes,
    in_size,
    out_size,
    n_scales=3,
    base_scale=1.0,
    scale_factor=10.0,
    inner_act='gelu',
    final_act='none'
):
    """
    Multi-scale network that processes features at different frequency scales.
    """

    class MultiScaleBlock(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, n_scales, base_scale,
                     scale_factor, activation):
            super().__init__()
            self.scales = [base_scale * (scale_factor ** i) for i in range(n_scales)]
            self.networks = nn.ModuleList()

            for scale in self.scales:
                net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    make_activation(activation),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    make_activation(activation),
                    nn.Linear(hidden_dim, out_dim)
                )
                self.networks.append(net)

            self.combine = nn.Linear(out_dim * n_scales, out_dim)

        def forward(self, x):
            outputs = []
            for scale, net in zip(self.scales, self.networks):
                scaled_x = x * scale
                outputs.append(net(scaled_x))

            combined = torch.cat(outputs, dim=-1)
            return self.combine(combined)

    if hidden_sizes:
        hidden_dim = hidden_sizes[0]
        net = MultiScaleBlock(in_size, hidden_dim, out_size, n_scales,
                              base_scale, scale_factor, inner_act)
    else:
        net = nn.Linear(in_size, out_size)

    if final_act != 'none':
        net = nn.Sequential(net, make_activation(final_act))

    return InputWrapper(net)


# 5. ATTENTION-BASED NETWORK
def make_attention_net(
    hidden_sizes,
    in_size,
    out_size,
    n_heads=4,
    inner_act='gelu',
    final_act='none',
    use_layernorm=True
):
    """
    Network with self-attention for capturing long-range dependencies.
    """

    class AttentionBlock(nn.Module):
        def __init__(self, dim, n_heads=4, activation='gelu'):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                make_activation(activation),
                nn.Linear(dim * 4, dim)
            )

        def forward(self, x):
            # Self-attention with residual
            x_norm = self.norm1(x)
            x_att, _ = self.attention(x_norm.unsqueeze(1),
                                      x_norm.unsqueeze(1),
                                      x_norm.unsqueeze(1))
            x = x + x_att.squeeze(1)

            # Feedforward with residual
            x = x + self.ff(self.norm2(x))
            return x

    net = []

    # Input projection
    if hidden_sizes:
        hidden_dim = hidden_sizes[0]
        net.append(nn.Linear(in_size, hidden_dim))
        net.append(make_activation(inner_act))

        # Attention blocks
        for _ in range(max(1, len(hidden_sizes) // 2)):
            net.append(AttentionBlock(hidden_dim, n_heads, inner_act))

        # Output projection
        net.append(nn.Linear(hidden_dim, out_size))
    else:
        net.append(nn.Linear(in_size, out_size))

    if final_act != 'none':
        net.append(make_activation(final_act))

    v_net = nn.Sequential(*net)
    return InputWrapper(v_net)


def _sinusoidal_time_embedding(t, dim):
    half = dim // 2
    device, dtype = t.device, t.dtype
    freq = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim * 4, 64)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t):
        return self.net(t)


class _ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class _Downsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class _Upsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


class UNetVelocity(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        time_dim=256,
        channel_mults=(1, 2, 4),
        dropout=0.0,
        groups=32
    ):
        super().__init__()
        if len(channel_mults) != 3:
            raise ValueError("channel_mults must contain three entries")
        self.time_dim = time_dim
        self.time_embed = _TimeEmbedding(time_dim)
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        c1 = base_channels * channel_mults[0]
        c2 = base_channels * channel_mults[1]
        c3 = base_channels * channel_mults[2]

        self.down1 = _ResBlock2D(base_channels, c1, time_dim, groups, dropout)
        self.down2 = _ResBlock2D(c1, c2, time_dim, groups, dropout)
        self.down3 = _ResBlock2D(c2, c3, time_dim, groups, dropout)
        self.ds1 = _Downsample2D(c1)
        self.ds2 = _Downsample2D(c2)

        self.mid1 = _ResBlock2D(c3, c3, time_dim, groups, dropout)
        self.mid2 = _ResBlock2D(c3, c3, time_dim, groups, dropout)

        self.up1 = _ResBlock2D(c3 + c2, c2, time_dim, groups, dropout)
        self.up2 = _ResBlock2D(c2 + c1, c1, time_dim, groups, dropout)
        self.us1 = _Upsample2D(c3)
        self.us2 = _Upsample2D(c2)

        self.out_norm = nn.GroupNorm(min(groups, c1), c1)
        self.out = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def _prep_time(self, t, batch, device, dtype):
        if t.dim() > 1:
            t = t.view(t.shape[0], -1)
            if t.shape[1] != 1:
                raise ValueError("Time tensor must have a singleton feature dimension")
            t = t[:, 0]
        t = t.to(device=device, dtype=dtype).reshape(-1)
        if t.shape[0] == 1 and batch > 1:
            t = t.repeat(batch)
        elif t.shape[0] != batch:
            if batch % t.shape[0] == 0:
                t = t.repeat(int(batch / t.shape[0]))
            else:
                raise ValueError("Time tensor batch does not align with input batch")
        return t

    def forward(self, x, t):
        b = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = self._prep_time(t, b, x.device, x.dtype)
        time_emb = self.time_embed(_sinusoidal_time_embedding(t, self.time_dim))

        x0 = self.input(x)
        d1 = self.down1(x0, time_emb)
        x = self.ds1(d1)
        d2 = self.down2(x, time_emb)
        x = self.ds2(d2)
        x = self.down3(x, time_emb)

        x = self.mid1(x, time_emb)
        x = self.mid2(x, time_emb)

        x = self.us1(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up1(x, time_emb)

        x = self.us2(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up2(x, time_emb)

        x = F.silu(self.out_norm(x))
        return self.out(x)


def make_unet_velocity(
    in_channels=3,
    base_channels=128,
    time_dim=256,
    channel_mults=(1, 2, 4),
    dropout=0.0,
    groups=32
):
    """Factory for UNetVelocity so notebooks can instantiate without boilerplate."""
    return UNetVelocity(
        in_channels=in_channels,
        base_channels=base_channels,
        time_dim=time_dim,
        channel_mults=channel_mults,
        dropout=dropout,
        groups=groups,
    )


def make_activation(act):
    """Helper function to create activation layers."""
    if act == 'relu':
        return nn.ReLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'silu' or act == 'swish':
        return nn.SiLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'none' or act is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f'Unknown activation function {act}')
