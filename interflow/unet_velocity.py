"""
UNet-based velocity field for 2D stochastic interpolants.
Adapted from MAC project's UNet architecture for velocity field learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# ----------------------------------------------------------------------------
# Utility functions and basic layers
# ----------------------------------------------------------------------------

def weight_init(shape, mode, fan_in, fan_out):
    """Initialize weights and biases."""
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class GroupNorm(nn.Module):
    """Group normalization layer."""
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        # Ensure we have at least 1 group and it divides num_channels
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.num_groups = max(1, self.num_groups)  # At least 1 group
        # Find a valid number of groups that divides num_channels
        while self.num_groups > 1 and num_channels % self.num_groups != 0:
            self.num_groups -= 1
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # For 2D vectors, add dummy spatial dimensions if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = F.group_norm(x, num_groups=self.num_groups,
                           weight=self.weight, bias=self.bias, eps=self.eps)
            return x.squeeze(-1).squeeze(-1)
        return F.group_norm(x, num_groups=self.num_groups,
                          weight=self.weight, bias=self.bias, eps=self.eps)


class PositionalEmbedding(nn.Module):
    """Timestep embedding using sinusoidal positional encoding."""
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1 and x.shape[0] == 1:
            x = x.expand(1)  # Ensure batch dimension

        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs

        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x @ freqs.unsqueeze(0)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(nn.Module):
    """Random Fourier feature embedding for time."""
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x @ (2 * np.pi * self.freqs).unsqueeze(0)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Simplified UNet blocks for 2D velocity field
# ----------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with time conditioning for 2D data."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 emb_channels,
                 dropout=0.0,
                 use_norm=True,
                 activation='gelu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm

        # Activation function
        if activation == 'gelu':
            self.act = F.gelu
        elif activation == 'silu':
            self.act = F.silu
        else:
            self.act = F.relu

        # Normalization
        if use_norm:
            self.norm1 = GroupNorm(in_channels)
            self.norm2 = GroupNorm(out_channels)

        # Main pathway
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

        # Time embedding projection
        self.time_emb_proj = nn.Linear(emb_channels, out_channels * 2)

        # Residual connection
        if in_channels != out_channels:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.skip = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, emb):
        # Store residual
        residual = x

        # First block
        if self.use_norm:
            x = self.norm1(x)
        x = self.act(x)
        x = self.fc1(x)

        # Add time embedding
        emb_out = self.time_emb_proj(self.act(emb))
        scale, shift = emb_out.chunk(2, dim=1)
        x = x * (1 + scale) + shift

        # Second block
        if self.use_norm:
            x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Skip connection
        if self.skip is not None:
            residual = self.skip(residual)

        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block for 2D data."""
    def __init__(self, channels, num_heads=1, head_channels=64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads if num_heads is not None else channels // head_channels
        self.head_dim = channels // self.num_heads

        self.norm = GroupNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B, C = x.shape[:2]
        residual = x

        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        scale = 1 / np.sqrt(self.head_dim)
        attn = torch.einsum('bhd,bHd->bhH', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhH,bHd->bhd', attn, v)
        out = out.reshape(B, C)

        # Output projection
        out = self.proj_out(out)

        return out + residual


# ----------------------------------------------------------------------------
# Main UNet architecture for 2D velocity field
# ----------------------------------------------------------------------------

class UNetVelocityField2D(nn.Module):
    """
    UNet-based velocity field for 2D stochastic interpolants.
    Takes (x, t) as input where x is 2D and outputs 2D velocity.
    """
    def __init__(self,
                 input_dim=2,                    # Dimension of input x
                 hidden_channels=256,             # Base number of hidden channels
                 depth=4,                         # Number of ResBlocks
                 time_emb_dim=256,               # Dimension of time embedding
                 dropout=0.0,                    # Dropout rate
                 use_attention=True,            # Whether to use attention
                 attention_layers=[2, 3],       # Which layers to add attention
                 embedding_type='positional',    # 'positional' or 'fourier'
                 activation='gelu',              # Activation function
                 use_norm=True):                # Whether to use normalization
        super().__init__()

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels

        # Time embedding
        if embedding_type == 'positional':
            self.time_embed = nn.Sequential(
                PositionalEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
            )
        else:  # fourier
            self.time_embed = nn.Sequential(
                FourierEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
            )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_channels)

        # Encoder path (downsampling)
        self.encoder_blocks = nn.ModuleList()
        channels = hidden_channels

        for i in range(depth):
            # Add residual block
            block = ResBlock(
                channels,
                channels,
                time_emb_dim * 4,
                dropout=dropout,
                use_norm=use_norm,
                activation=activation
            )
            self.encoder_blocks.append(block)

            # Add attention if specified
            if use_attention and i in attention_layers:
                attn = AttentionBlock(channels)
                self.encoder_blocks.append(attn)

        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock(channels, channels, time_emb_dim * 4,
                    dropout=dropout, use_norm=use_norm, activation=activation),
            AttentionBlock(channels) if use_attention else nn.Identity(),
            ResBlock(channels, channels, time_emb_dim * 4,
                    dropout=dropout, use_norm=use_norm, activation=activation),
        )

        # Decoder path (upsampling)
        self.decoder_blocks = nn.ModuleList()

        for i in range(depth):
            # Add residual block with skip connection
            block = ResBlock(
                channels * 2,  # Account for skip connections
                channels,
                time_emb_dim * 4,
                dropout=dropout,
                use_norm=use_norm,
                activation=activation
            )
            self.decoder_blocks.append(block)

            # Add attention if specified
            if use_attention and (depth - 1 - i) in attention_layers:
                attn = AttentionBlock(channels)
                self.decoder_blocks.append(attn)

        # Output projection
        self.output_norm = GroupNorm(channels) if use_norm else nn.Identity()
        self.output_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, input_dim),  # Output same dimension as input
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        Forward pass of the velocity field.

        Args:
            x: Input positions of shape (batch_size, input_dim)
            t: Time values of shape (batch_size,) or (batch_size, 1)

        Returns:
            Velocity field v(x, t) of shape (batch_size, input_dim)
        """
        # Ensure correct shapes
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        # Time embedding
        t_emb = self.time_embed(t)

        # Initial projection
        h = self.input_proj(x)

        # Encoder with skip connections
        encoder_outs = []
        for block in self.encoder_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
                encoder_outs.append(h)
            else:  # Attention block
                h = block(h)

        # Middle
        for block in self.middle_block:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:  # Attention block
                h = block(h)

        # Decoder with skip connections
        for block in self.decoder_blocks:
            if isinstance(block, ResBlock):
                # Concatenate skip connection
                if encoder_outs:
                    skip = encoder_outs.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            else:  # Attention block
                h = block(h)

        # Output
        h = self.output_norm(h)
        h = F.gelu(h)
        v = self.output_proj(h)

        return v


class LightweightUNetVelocity2D(nn.Module):
    """
    Lightweight UNet velocity field for 2D problems.
    More suitable for low-dimensional problems like 2D checkerboard.
    """
    def __init__(self,
                 input_dim=2,
                 hidden_dims=[128, 256, 256, 128],
                 time_emb_dim=128,
                 dropout=0.0,
                 activation='gelu'):
        super().__init__()

        self.input_dim = input_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            PositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim * 2),
        )

        # Build network
        layers = []
        in_dim = input_dim

        # Encoder
        self.encoder_layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(
                ResBlock(in_dim, hidden_dim, time_emb_dim * 2,
                        dropout=dropout, activation=activation)
            )
            in_dim = hidden_dim

        # Middle attention
        self.middle_attention = AttentionBlock(hidden_dims[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Skip connections double the input
            in_dim = hidden_dims[-i-1] + hidden_dims[-i-2]
            out_dim = hidden_dims[-i-2]
            self.decoder_layers.append(
                ResBlock(in_dim, out_dim, time_emb_dim * 2,
                        dropout=dropout, activation=activation)
            )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x, t):
        """
        Forward pass computing v(x, t).

        Args:
            x: Positions (batch_size, 2)
            t: Times (batch_size,) or (batch_size, 1)

        Returns:
            Velocities (batch_size, 2)
        """
        # Handle time shape
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        # Time embedding
        t_emb = self.time_embed(t)

        # Encoder with skip connections
        skips = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h, t_emb)
            skips.append(h)

        # Middle attention
        h = self.middle_attention(h)

        # Decoder with skips
        for i, layer in enumerate(self.decoder_layers):
            skip = skips[-(i+2)]  # Get corresponding encoder output
            h = torch.cat([h, skip], dim=1)
            h = layer(h, t_emb)

        # Output
        v = self.output_layer(h)

        return v


# ----------------------------------------------------------------------------
# Wrapper to match stochastic interpolants interface
# ----------------------------------------------------------------------------

class InputWrapper(nn.Module):
    """
    Wrapper to ensure compatibility with stochastic interpolants framework.
    Concatenates x and t, then passes to the velocity network.
    """
    def __init__(self, v_net):
        super().__init__()
        self.v = v_net

    def forward(self, xt, t):
        """
        Args:
            xt: Concatenated [x, t] tensor of shape (batch_size, dim+1)
                or just x of shape (batch_size, dim)
            t: Time tensor (may be already included in xt)

        Returns:
            Velocity v(x, t) of shape (batch_size, dim)
        """
        # If xt already contains time, extract it
        if xt.shape[1] == self.v.input_dim + 1:
            x = xt[:, :self.v.input_dim]
            t_from_xt = xt[:, self.v.input_dim:self.v.input_dim+1]
            # Use provided t if available, otherwise use extracted
            if t is None:
                t = t_from_xt
        else:
            x = xt

        # Ensure t has correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        return self.v(x, t)


# ----------------------------------------------------------------------------
# Factory functions for easy instantiation
# ----------------------------------------------------------------------------

def make_unet_velocity_2d(hidden_channels=256,
                          depth=4,
                          time_emb_dim=256,
                          dropout=0.0,
                          use_attention=True,
                          attention_layers=[2, 3],
                          embedding_type='positional',
                          activation='gelu',
                          wrapped=True):
    """
    Create a UNet velocity field for 2D problems.

    Args:
        hidden_channels: Base number of channels
        depth: Number of ResBlocks in encoder/decoder
        time_emb_dim: Dimension of time embedding
        dropout: Dropout probability
        use_attention: Whether to use self-attention
        attention_layers: Which layers to add attention to
        embedding_type: 'positional' or 'fourier'
        activation: Activation function ('gelu', 'silu', 'relu')
        wrapped: Whether to wrap in InputWrapper for compatibility

    Returns:
        Velocity field network
    """
    v_net = UNetVelocityField2D(
        input_dim=2,
        hidden_channels=hidden_channels,
        depth=depth,
        time_emb_dim=time_emb_dim,
        dropout=dropout,
        use_attention=use_attention,
        attention_layers=attention_layers,
        embedding_type=embedding_type,
        activation=activation
    )

    if wrapped:
        return InputWrapper(v_net)
    return v_net


def make_lightweight_unet_velocity_2d(hidden_dims=[128, 256, 256, 128],
                                      time_emb_dim=128,
                                      dropout=0.0,
                                      activation='gelu',
                                      wrapped=True):
    """
    Create a lightweight UNet velocity field for 2D problems.

    Args:
        hidden_dims: List of hidden dimensions for each layer
        time_emb_dim: Dimension of time embedding
        dropout: Dropout probability
        activation: Activation function
        wrapped: Whether to wrap in InputWrapper

    Returns:
        Velocity field network
    """
    v_net = LightweightUNetVelocity2D(
        input_dim=2,
        hidden_dims=hidden_dims,
        time_emb_dim=time_emb_dim,
        dropout=dropout,
        activation=activation
    )

    if wrapped:
        return InputWrapper(v_net)
    return v_net