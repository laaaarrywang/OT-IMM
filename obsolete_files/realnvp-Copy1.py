"""
Time-indexed RealNVP implementation using nflows package.

This module provides a time-conditioned invertible transformation T(t, x)
where t \in [0,1] and x ∈ R^d, supporting both vector and image inputs.
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
    Maps t \in [0,1] to high-dimensional representation for conditioning.
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
        t = t.view(-1, 1) # [B] --> [B,1] 
        # Phase = 2π * t * frequency
        phases = 2.0 * math.pi * t * self.freqs[None, :] # [None, : ] adds a dimension for broadcasting
        # Concatenate sin and cos for each frequency
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


def create_coupling_mask(shape, mask_type = 'alternating', **kwargs):
    if isinstance(shape, int):
        # vector mask
        mask = torch.zeros(shape)
        mask[int(kwargs.get("even_first",True))::2] = 1.0
        return mask


    C, H, W = shape
    mask = torch.zeros(C, H, W)

    if mask_type == "checkerboard":
        white_first = kwargs.get("white_first",True)
        for i in range(H):
            for j in range(W):
                if ((i + j) % 2 == 0) == white_first:
                    mask[:, i, j] = 1.0
    elif mask_type == "channel":
        first_half = kwargs.get("first_half", True)
        if C == 1:
            # For single channel, use spatial split instead
            if first_half: # first half is masked
                mask[:, :, :W//2] = 1.0
            else:
                mask[:, :, W//2:] = 1.0
        else:
            # For multiple channels, split by channel
            split_point = C // 2
            if split_point == 0: # single channel
                split_point = 1
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


class ResidualBlock2d(nn.Module):
    """
    Residual block for convolutional networks with optional downsampling.
    Uses pre-activation design (BN -> ReLU -> Conv).
    """
    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Main path
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        self.main_path = nn.Sequential(*layers)
        
        # Skip connection
        self.skip_connection = nn.Identity()
        if in_channels != out_channels or stride != 1:
            # 1x1 conv for channel/spatial adjustment
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
    
    def forward(self, x):
        return self.main_path(x) + self.skip_connection(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Helps the network focus on important channels.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ImageAffineCouplingLayer(nn.Module):
    """
    Wrapper for image-based affine coupling with proper reshaping.
    Now includes residual blocks for better gradient flow and capacity.
    """
    def __init__(self, C, H, W, mask_flat, time_embed_dim=128,
                 hidden_channels=64, num_blocks=3, use_batch_norm=True,
                 use_se_blocks=False, use_residual = True):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.D = C * H * W
        
        # Determine identity/transform indices following TRUE nflows convention:
        # mask=0 → identity features (pass through AND are INPUTS to transform_net!)
        # mask=1 → transform features (get transformed using output from transform_net)
        if not isinstance(mask_flat, torch.Tensor):
            mask_flat = torch.tensor(mask_flat, dtype=torch.float32)
        
        mask_bool = mask_flat > 0.5
        # In nflows: identity_features are where mask <= 0.5, transform_features are where mask > 0.5
        # transform_net receives identity_features as input
        identity_indices = torch.nonzero(~mask_bool, as_tuple=False).view(-1)  # mask=0: pass through
        transform_indices = torch.nonzero(mask_bool, as_tuple=False).view(-1)  # mask=1: get transformed
        
        self.register_buffer("identity_indices", identity_indices)
        self.register_buffer("transform_indices", transform_indices)
        
        self.n_identity = identity_indices.shape[0]
        self.n_transform = transform_indices.shape[0]
        
        # Validate mask
        assert self.n_identity > 0, f"Identity features empty! mask sum: {mask_flat.sum()}"
        assert self.n_transform > 0, f"Transform features empty! mask sum: {mask_flat.sum()}"
        
        # Context projection
        self.context_proj = nn.Linear(time_embed_dim, hidden_channels)
        
        # Input channels
        in_ch = C + 1  # Image channels + mask channel

        if use_residual:
            # Initial convolution to project to hidden_channels
            self.input_conv = nn.Conv2d(in_ch + hidden_channels, hidden_channels, 3, padding=1)
            
            # Build residual blocks
            self.blocks = nn.ModuleList()
            for i in range(num_blocks):
                self.blocks.append(
                    ResidualBlock2d(hidden_channels, hidden_channels, use_batch_norm=use_batch_norm)
                )
                # Optionally add SE blocks for better feature recalibration
                if use_se_blocks and i % 2 == 1:
                    self.blocks.append(SEBlock(hidden_channels))
            
            # Output convolution
            self.output_conv = nn.Conv2d(hidden_channels, 2 * C, 3, padding=1)  # scale + shift
        else:
            self.input_conv = None
            self.blocks = None
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_ch + hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, 2 * C, 3, padding=1)
            )
        
        # Create spatial mask
        mask_img = mask_flat.view(C, H, W)
        self.register_buffer("spatial_mask", (mask_img.sum(0, keepdim=True) > 0).float())
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        if hasattr(self, 'output_conv') and self.output_conv is not None:
            # Small initialization for output layer to start near identity (residual case)
            nn.init.normal_(self.output_conv.weight, 0, 0.01)
            if self.output_conv.bias is not None:
                nn.init.constant_(self.output_conv.bias, 0)
        elif hasattr(self, 'conv_net') and self.conv_net is not None:
            # Initialize the last layer of conv_net (non-residual case)
            last_conv = self.conv_net[-1]  # Last layer is the output conv
            nn.init.normal_(last_conv.weight, 0, 0.01)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)
        
    def forward(self, inputs, context=None):
        """
        IMPORTANT: nflows convention:
        - inputs: [B, n_identity] identity feature values (mask=0 positions!)
        - Returns: [B, 2*n_transform] affine parameters for transform features (mask=1)
        
        Args:
            inputs: [B, n_identity] identity feature values (nflows passes mask=0 features!)
            context: [B, time_embed_dim] time embedding
        Returns:
            [B, 2*n_transform] affine parameters for transform features (mask=1)
        """
        # Handle different input shapes from nflows
        if inputs.dim() == 1:
            # Single sample case
            B = 1
            inputs = inputs.unsqueeze(0)
        else:
            B = inputs.size(0)
        
        device = inputs.device
        
        # Check input dimensions - nflows passes IDENTITY features (mask=0) as input!
        if inputs.size(1) != self.n_identity:
            raise ValueError(f"Expected input size {self.n_identity}, got {inputs.size(1)}")
        
        # Reconstruct partial image with identity features (mask=0 positions)
        img_full = torch.zeros(B, self.D, device=device)
        # Place identity features in their positions
        img_full.scatter_(1, self.identity_indices.unsqueeze(0).expand(B, -1), inputs)
        img = img_full.view(B, self.C, self.H, self.W)
        
        # Add mask channel
        mask_ch = self.spatial_mask.expand(B, -1, -1, -1)
        
        # Add time context as spatial feature
        ctx = self.context_proj(context)
        ctx_spatial = ctx.view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)
        
        # Concatenate and process
        combined = torch.cat([img, mask_ch, ctx_spatial], dim=1)
        if self.blocks is not None:
            # Process through residual network
            x = self.input_conv(combined)
            
            # Apply residual blocks
            for block in self.blocks:
                x = block(x)
            
            # Generate affine parameters
            params_2C = self.output_conv(x)
        else:
            params_2C = self.conv_net(combined)
        
        # Extract parameters for transform positions (mask=1, the ones that get transformed)
        # params_2C is [B, 2*C, H, W]
        params_flat = params_2C.view(B, 2 * self.C, self.H * self.W)
        
        # Split into shift and scale
        shift_all = params_flat[:, :self.C, :].reshape(B, self.D)  # [B, C*H*W]
        log_scale_all = params_flat[:, self.C:, :].reshape(B, self.D)  # [B, C*H*W]
        
        # Clamp log_scale for numerical stability
        log_scale_all = torch.clamp(log_scale_all, min=-10, max=10)
        
        # Extract parameters for TRANSFORM positions (nflows applies these to mask=1 features)
        shift_transform = torch.gather(shift_all, 1, self.transform_indices.unsqueeze(0).expand(B, -1))
        log_scale_transform = torch.gather(log_scale_all, 1, self.transform_indices.unsqueeze(0).expand(B, -1))
        
        return torch.cat([shift_transform, log_scale_transform], dim=1)


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
                 conv_num_blocks=3,
                 use_conv_batch_norm=True,
                 use_se_blocks=False,
                 use_residual = True):
        super().__init__()
        
        # Store additional parameters for image mode
        self.use_conv_batch_norm = use_conv_batch_norm
        self.use_se_blocks = use_se_blocks
        self.use_residual = use_residual
        
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
        #mask = create_alternating_mask(dim, even_first=True)
        mask = create_coupling_mask(dim, even_first = True)
        
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
        masks = []  # Store all masks to avoid closure issues
        
        # Pre-create all masks
        for i in range(num_layers):
            if i % 2 == 0:
                #mask = create_checkerboard_mask(self.C, self.H, self.W, 
                 #                              white_first=(i % 4 == 0))
                mask = create_coupling_mask(shape, "checkerboard",
                                            white_first=(i % 4 == 0))
            else:
                #mask = create_channel_mask(self.C, self.H, self.W,
                 #                        first_half=((i // 2) % 2 == 0))
                mask = create_coupling_mask(shape, "channel",
                                         first_half=((i // 2) % 2 == 0))
            masks.append(mask.clone())
        
        # Create layers with stored masks
        for i, mask in enumerate(masks):
            # Create layer with explicit mask indexing
            class LayerFactory:
                def __init__(self, mask_val, C, H, W, time_embed_dim, hidden_channels, 
                           conv_num_blocks, use_batch_norm, use_se_blocks, use_residual):
                    self.mask_val = mask_val
                    self.C = C
                    self.H = H
                    self.W = W
                    self.time_embed_dim = time_embed_dim
                    self.hidden_channels = hidden_channels
                    self.conv_num_blocks = conv_num_blocks
                    self.use_batch_norm = use_batch_norm
                    self.use_se_blocks = use_se_blocks
                    self.use_residual = use_residual
                
                def __call__(self, in_features, out_features): # so that LayerFactory is callable and, when called, receives in_features and out_features and returns a instance of the class (nn.Module) as required by the transforms.AffineCouplingTransform
                    return ImageAffineCouplingLayer(
                        C=self.C, H=self.H, W=self.W,
                        mask_flat=self.mask_val,
                        time_embed_dim=self.time_embed_dim,
                        hidden_channels=self.hidden_channels,
                        num_blocks=self.conv_num_blocks,
                        use_batch_norm=self.use_batch_norm,
                        use_se_blocks=self.use_se_blocks,
                        use_residual = self.use_residual
                    )
            
            factory = LayerFactory(mask, self.C, self.H, self.W, 
                                 time_embed_dim, hidden_channels, conv_num_blocks,
                                 self.use_conv_batch_norm, self.use_se_blocks, self.use_residual)
            
            coupling = transforms.AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=factory
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
    """Create RealNVP flow for CIFAR-10 (3×32×32 images) with residual blocks."""
    defaults = dict(
        shape=(3, 32, 32),
        num_layers=12,
        hidden_channels=128,
        time_embed_dim=256,
        conv_num_blocks=3,  # Now actually used!
        use_conv_batch_norm=True,
        use_se_blocks=False,  # Can enable for better performance
        use_residual = True
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


def create_imagenet_flow(**kwargs):
    """
    Create RealNVP flow for ImageNet (3×224×224 images) with residual blocks.
    
    Note: ImageNet images are high-resolution, so we use:
    - Fewer layers to manage memory
    - Larger hidden channels for more capacity
    - Larger time embedding for better conditioning
    - SE blocks for better feature recalibration
    
    For full ImageNet, consider:
    - Multi-scale architecture
    - Gradient checkpointing for memory efficiency
    - Mixed precision training
    """
    defaults = dict(
        shape=(3, 224, 224),
        num_layers=6,  # Fewer layers due to high resolution
        hidden_channels=256,  # More capacity per layer
        time_embed_dim=512,  # Richer time conditioning
        conv_num_blocks=4,  # Deeper conv networks with residuals
        use_conv_batch_norm=True,
        use_se_blocks=True,  # Enable SE blocks for ImageNet
        use_residual = True
    )
    defaults.update(kwargs)
    return TimeIndexedRealNVP(**defaults)