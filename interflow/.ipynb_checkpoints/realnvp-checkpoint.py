"""
Time-indexed RealNVP implementation using nflows package.

This module provides a time-conditioned invertible transformation T(t, x)
where t ∈ [0,1] and x ∈ R^d, supporting both vector and image inputs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms
from nflows.nn import nets


class FourierTimeEmbedding(nn.Module):
    """
    Creates smooth, differentiable time embeddings using Fourier features.
    Maps t ∈ [0,1] to high-dimensional representation for conditioning.
    """
    def __init__(self, embed_dim=128, min_freq=1.0, max_freq=1000.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos pairs"
        
        n_freqs = embed_dim // 2
        # Geometric progression of frequencies for multi-scale representation
        freqs = torch.exp(torch.linspace(
            math.log(min_freq), 
            math.log(max_freq), 
            n_freqs
        ))
        self.register_buffer("freqs", freqs)
    
    def forward(self, t):
        """
        Args:
            t: [B] tensor with values in [0,1]
        Returns:
            [B, embed_dim] time embeddings
        """
        t = t.view(-1, 1)
        # Phase = 2π * t * frequency
        phases = 2.0 * math.pi * t * self.freqs[None, :]
        # Concatenate sin and cos for each frequency
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


def create_alternating_mask(features, even_first=True):
    """Creates alternating binary mask for coupling layers."""
    mask = torch.zeros(features)
    mask[int(even_first)::2] = 1.0
    return mask


def create_checkerboard_mask(C, H, W, white_first=True):
    """Creates checkerboard mask for image coupling layers."""
    mask = torch.zeros(C, H, W)
    for i in range(H):
        for j in range(W):
            if ((i + j) % 2 == 0) == white_first:
                mask[:, i, j] = 1.0
    return mask.flatten()


def create_channel_mask(C, H, W, first_half=True):
    """Creates channel-wise mask for image coupling layers."""
    mask = torch.zeros(C, H, W)
    split_point = C // 2
    if first_half:
        mask[:split_point, :, :] = 1.0
    else:
        mask[split_point:, :, :] = 1.0
    return mask.flatten()


class TimeConditionedAffineCoupling(transforms.AffineCouplingTransform):
    """
    Affine coupling layer with time conditioning.
    Inherits from nflows but adds time context support.
    """
    def forward(self, inputs, context=None):
        """Apply coupling transform with time context."""
        return super().forward(inputs, context=context)
    
    def inverse(self, inputs, context=None):
        """Apply inverse coupling transform with time context."""
        return super().inverse(inputs, context=context)


class ConvNet2d(nn.Module):
    """
    Convolutional network for image coupling layers.
    Maps masked image regions to affine parameters for unmasked regions.
    """
    def __init__(self, in_channels, out_features, context_features,
                 hidden_channels=64, num_blocks=3):
        super().__init__()
        
        # Project time context to spatial features
        self.context_net = nn.Sequential(
            nn.Linear(context_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Convolutional blocks
        layers = []
        layers.append(nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(hidden_channels, out_features, 3, padding=1))
        self.conv_blocks = nn.Sequential(*layers)
        
    def forward(self, inputs, context=None):
        """
        Args:
            inputs: [B, in_features] flattened masked image
            context: [B, context_features] time embedding
        Returns:
            [B, out_features] affine parameters
        """
        # This will be wrapped by ImageAffineCouplingLayer
        return inputs  # Placeholder - actual implementation in wrapper


class ImageAffineCouplingLayer(nn.Module):
    """
    Wrapper for image-based affine coupling with proper reshaping.
    """
    def __init__(self, C, H, W, mask_flat, time_embed_dim=128,
                 hidden_channels=64, num_blocks=3):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.D = C * H * W
        
        # Determine masked/unmasked indices
        mask_bool = mask_flat > 0.5
        self.register_buffer("mask_A", torch.nonzero(mask_bool).squeeze(-1))
        self.register_buffer("mask_B", torch.nonzero(~mask_bool).squeeze(-1))
        
        self.n_A = len(self.mask_A)
        self.n_B = len(self.mask_B)
        
        # Context projection
        self.context_proj = nn.Linear(time_embed_dim, hidden_channels)
        
        # Conv network for computing affine params
        in_ch = C + 1  # Image channels + mask channel
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_ch + hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2 * C, 3, padding=1)  # scale + shift
        )
        
        # Create spatial mask
        mask_img = mask_flat.view(C, H, W)
        self.register_buffer("spatial_mask", (mask_img.sum(0, keepdim=True) > 0).float())
        
    def forward(self, inputs, context=None):
        """
        Args:
            inputs: [B, n_A] masked values
            context: [B, time_embed_dim] time embedding
        Returns:
            [B, 2*n_B] affine parameters (shift and log_scale)
        """
        B = inputs.size(0)
        device = inputs.device
        
        # Reconstruct partial image
        img_full = torch.zeros(B, self.D, device=device)
        img_full[:, self.mask_A] = inputs
        img = img_full.view(B, self.C, self.H, self.W)
        
        # Add mask channel
        mask_ch = self.spatial_mask.expand(B, -1, -1, -1)
        
        # Add time context as spatial feature
        ctx = self.context_proj(context)
        ctx_spatial = ctx.view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)
        
        # Concatenate and process
        combined = torch.cat([img, mask_ch, ctx_spatial], dim=1)
        params_2C = self.conv_net(combined)
        
        # Extract parameters for unmasked positions
        params_flat = params_2C.view(B, 2 * self.C, -1).permute(0, 2, 1).contiguous()
        params_flat = params_flat.view(B, -1)
        
        shift = params_flat[:, :self.D][:, self.mask_B]
        log_scale = params_flat[:, self.D:][:, self.mask_B]
        
        return torch.cat([shift, log_scale], dim=1)


class TimeIndexedRealNVP(nn.Module):
    """
    Main time-indexed RealNVP flow T(t, x).
    
    Properties:
    - T: [0,1] × R^d → R^d
    - Invertible w.r.t. x for any fixed t
    - Differentiable w.r.t. both t and x
    - Supports vector and image inputs
    """
    
    def __init__(self,
                 shape,  # int (vector dim) or tuple (C, H, W)
                 num_layers=12,
                 time_embed_dim=128,
                 # Vector mode parameters
                 hidden_features=1024,
                 num_blocks=2,
                 use_batch_norm=False,
                 use_permutation=True,
                 # Image mode parameters  
                 hidden_channels=64,
                 conv_num_blocks=3):
        super().__init__()
        
        # Time embedding network
        self.time_embed = FourierTimeEmbedding(time_embed_dim)
        
        # Determine mode and create appropriate layers
        if isinstance(shape, int):
            self._init_vector_mode(shape, num_layers, time_embed_dim,
                                 hidden_features, num_blocks, 
                                 use_batch_norm, use_permutation)
        else:
            self._init_image_mode(shape, num_layers, time_embed_dim,
                                hidden_channels, conv_num_blocks)
    
    def _init_vector_mode(self, dim, num_layers, time_embed_dim,
                         hidden_features, num_blocks, 
                         use_batch_norm, use_permutation):
        """Initialize vector mode with MLP coupling layers."""
        self.mode = "vector"
        self.dim = dim
        
        layers = []
        mask = create_alternating_mask(dim, even_first=True)
        
        for i in range(num_layers):
            # Create coupling layer with time conditioning
            def create_net(in_features, out_features):
                return nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    context_features=time_embed_dim,
                    num_blocks=num_blocks,
                    activation=F.relu,
                    use_batch_norm=use_batch_norm
                )
            
            coupling = transforms.AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_net
            )
            layers.append(coupling)
            
            # Add permutation for mixing
            if use_permutation and i < num_layers - 1:
                layers.append(transforms.RandomPermutation(features=dim))
            
            # Alternate mask
            mask = 1.0 - mask
        
        self.transform = transforms.CompositeTransform(layers)
    
    def _init_image_mode(self, shape, num_layers, time_embed_dim,
                        hidden_channels, conv_num_blocks):
        """Initialize image mode with convolutional coupling layers."""
        self.mode = "image"
        self.C, self.H, self.W = shape
        self.dim = self.C * self.H * self.W
        
        layers = []
        
        for i in range(num_layers):
            # Alternate between checkerboard and channel masks
            if i % 2 == 0:
                mask = create_checkerboard_mask(self.C, self.H, self.W, 
                                               white_first=(i % 4 == 0))
            else:
                mask = create_channel_mask(self.C, self.H, self.W,
                                         first_half=((i // 2) % 2 == 0))
            
            # Create image coupling layer
            def create_net(in_features, out_features):
                return ImageAffineCouplingLayer(
                    C=self.C, H=self.H, W=self.W,
                    mask_flat=mask,
                    time_embed_dim=time_embed_dim,
                    hidden_channels=hidden_channels,
                    num_blocks=conv_num_blocks
                )
            
            coupling = transforms.AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_net
            )
            layers.append(coupling)
        
        self.transform = transforms.CompositeTransform(layers)
    
    def forward(self, x, t, inverse=False):
        """
        Apply transformation T(t, x) or its inverse.
        
        Args:
            x: [B, D] or [B, C, H, W] input data
            t: [B] or scalar, time values in [0, 1]
            inverse: If True, compute T^{-1}(t, x)
        
        Returns:
            y: Transformed data (same shape as x)
            log_det: [B] log determinant of Jacobian
        """
        # Handle scalar t
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        
        # Flatten if image
        original_shape = x.shape
        if x.dim() == 4:  # [B, C, H, W]
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
        
        # Get time embedding
        t_embed = self.time_embed(t)
        
        # Apply transformation
        if not inverse:
            y_flat, log_det = self.transform.forward(x_flat, context=t_embed)
        else:
            y_flat, log_det = self.transform.inverse(x_flat, context=t_embed)
        
        # Reshape if needed
        if len(original_shape) == 4:
            y = y_flat.view(original_shape)
        else:
            y = y_flat
        
        return y, log_det
    
    def inverse(self, y, t):
        """Convenience method for inverse transformation."""
        return self.forward(y, t, inverse=True)
    
    def sample_trajectory(self, x0, t_steps=None, num_steps=100):
        """
        Sample trajectory of x through time.
        
        Args:
            x0: [B, D] or [B, C, H, W] initial points
            t_steps: Optional tensor of time values
            num_steps: Number of steps if t_steps not provided
        
        Returns:
            trajectory: [T, B, ...] trajectory through time
            t_values: [T] time values used
        """
        if t_steps is None:
            t_steps = torch.linspace(0, 1, num_steps, device=x0.device)
        
        trajectory = []
        with torch.no_grad():
            for t_val in t_steps:
                xt, _ = self.forward(x0, t_val)
                trajectory.append(xt)
        
        return torch.stack(trajectory), t_steps
    
    def compute_time_derivative(self, x, t, method='finite_diff', eps=1e-4):
        """
        Compute ∂T/∂t (t, x).
        
        Args:
            x: [B, D] or [B, C, H, W] input
            t: [B] time values
            method: 'finite_diff' or 'autograd'
            eps: Step size for finite differences
        
        Returns:
            dTdt: [B, ...] time derivative
        """
        if method == 'finite_diff':
            # Forward difference
            t_plus = torch.clamp(t + eps, max=1.0)
            y_plus, _ = self.forward(x, t_plus)
            y, _ = self.forward(x, t)
            return (y_plus - y) / eps
        
        elif method == 'autograd':
            t = t.requires_grad_(True)
            y, _ = self.forward(x, t)
            
            # Compute gradient w.r.t. t
            grads = []
            for i in range(y.shape[0]):
                grad = torch.autograd.grad(
                    y[i].sum(), t[i],
                    retain_graph=True,
                    create_graph=True
                )[0]
                grads.append(grad)
            
            return torch.stack(grads).view(y.shape[0], 1, *([1] * (y.dim() - 2)))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_jacobian_trace(self, x, t, num_samples=1):
        """
        Estimate trace(∂T/∂x) using Hutchinson's estimator.
        
        Args:
            x: [B, D] or [B, C, H, W] input
            t: [B] time values  
            num_samples: Number of random samples for estimation
        
        Returns:
            trace: [B] estimated trace values
        """
        x_flat = x.view(x.shape[0], -1) if x.dim() == 4 else x
        B, D = x_flat.shape
        
        trace_est = torch.zeros(B, device=x.device)
        
        for _ in range(num_samples):
            # Rademacher random vector
            v = torch.randn_like(x_flat).sign()
            
            # Compute JVP: (∂T/∂x) @ v
            x_flat.requires_grad_(True)
            y, _ = self.forward(x_flat.view(x.shape), t)
            y_flat = y.view(B, -1)
            
            jvp = torch.autograd.grad(
                y_flat, x_flat,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False
            )[0]
            
            # Estimate: E[v^T @ J @ v] = tr(J)
            trace_est += (jvp * v).sum(dim=1)
        
        return trace_est / num_samples


# Factory functions for common use cases
def create_mnist_flow(**kwargs):
    """Create RealNVP flow for MNIST (784-dimensional vectors)."""
    defaults = dict(
        shape=784,
        num_layers=8,
        hidden_features=512,
        time_embed_dim=128
    )
    defaults.update(kwargs)
    return TimeIndexedRealNVP(**defaults)


def create_cifar10_flow(**kwargs):
    """Create RealNVP flow for CIFAR-10 (3×32×32 images)."""
    defaults = dict(
        shape=(3, 32, 32),
        num_layers=12,
        hidden_channels=128,
        time_embed_dim=256
    )
    defaults.update(kwargs)
    return TimeIndexedRealNVP(**defaults)


def create_vector_flow(dim, **kwargs):
    """Create RealNVP flow for arbitrary vector dimension."""
    defaults = dict(
        shape=dim,
        num_layers=10,
        hidden_features=1024,
        time_embed_dim=128
    )
    defaults.update(kwargs)
    return TimeIndexedRealNVP(**defaults)