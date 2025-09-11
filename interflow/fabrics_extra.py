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
