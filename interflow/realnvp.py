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
    Lightweight Fourier time embedding:
      t in [0,1] -> R^{embed_dim} with sin/cos features.
    Optionally includes a tiny linear projector to a target dim.
    """
    def __init__(self, embed_dim: int = 64, min_freq: float = 1.0, max_freq: float = 1000.0,
                 project_to: int = None):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even (sin/cos pairs)."
        n_freqs = embed_dim // 2

        freqs = torch.exp(torch.linspace(
            math.log(min_freq), math.log(max_freq), n_freqs
        ))
        self.register_buffer("freqs", freqs)  # non-trainable, moves with .to(device)
        self.embed_dim = embed_dim

        self.proj = None
        if project_to is not None:
            self.proj = nn.Linear(embed_dim, project_to)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape [] or [B] or [B,1], values expected in [0,1].
        returns: [B, embed_dim] (or [B, project_to] if proj is set)
        """
        if t.dim() == 0:
            t = t[None]  # make it [1]
        if t.dim() == 2 and t.size(1) == 1:
            t = t.view(-1)  # [B,1] -> [B]
        t = t.view(-1, 1)  # ensure [B,1]

        # phases: [B, n_freqs]
        phases = 2.0 * math.pi * t * self.freqs[None, :]
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # [B, embed_dim]

        if self.proj is not None:
            emb = self.proj(emb)
        return emb


def create_coupling_mask(shape, mask_type='alternating', **kwargs):
    if isinstance(shape, int):
        # Vector mask - alternating pattern
        mask = torch.zeros(shape)
        even_first = kwargs.get("even_first", True)
        start_idx = 0 if even_first else 1
        mask[start_idx::2] = 1.0
        return mask

    C, H, W = shape
    mask = torch.zeros(C, H, W)

    if mask_type == "checkerboard":
        white_first = kwargs.get("white_first", True)
        for i in range(H):
            for j in range(W):
                if ((i + j) % 2 == 0) == white_first:
                    mask[:, i, j] = 1.0
    elif mask_type == "channel":
        first_half = kwargs.get("first_half", True)
        if C == 1:
            # For single channel, use spatial split instead
            if first_half:  # first half is masked
                mask[:, :, :W//2] = 1.0
            else:
                mask[:, :, W//2:] = 1.0
        else:
            # For multiple channels, split by channel
            split_point = C // 2
            if split_point == 0:  # single channel
                split_point = 1
            if first_half:
                mask[:split_point, :, :] = 1.0
            else:
                mask[split_point:, :, :] = 1.0
    else: 
        raise ValueError(f"Invalid mask type: {mask_type}")
    return mask.flatten()

class FiLM2d(nn.Module):
    """
    Per-channel FiLM module for 2D images.
    """
    def __init__(self, channels: int, context_dim: int):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.gamma = nn.Linear(context_dim, channels)
        self.beta = nn.Linear(context_dim, channels)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(context).view(-1, self.channels, 1, 1)
        beta = self.beta(context).view(-1, self.channels, 1, 1)
        return x * (1.0 + gamma) + beta

class PreActResBlock(nn.Module):
    """
    Pre-activation ResNet block with GroupNorm + FiLM.
    Keeps spatial resolution (no down/upsampling).
    """
    def __init__(self, channels: int, context_dim: int, groups: int = 32):
        super().__init__()
        g = min(groups, channels)  # guard when channels < groups
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1) # with kernel size = 3 and padding = 1 and stride = 1, the output will have the same spatial dimensions as the input.
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = FiLM2d(channels, context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, context)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class ImageCouplingResNet(nn.Module):
    """
    Conv coupling-net for RealNVP (image mode).
    Expects identity features as a flat vector (from nflows) and a time/context vector.
    Uses a mask channel and FiLM-conditioned ResNet blocks.

    Mask semantics and nflows interaction:
        - nflows mask convention: 1=identity, 0=transform
        - nflows passes IDENTITY features (mask=1) to us as inputs  
        - We output transformation parameters for TRANSFORM features (mask=0)
        
    However, our internal logic maps:
        - identity_indices: mask=0 positions (what we receive from nflows)
        - transform_indices: mask=1 positions (what we generate parameters for)

    Args:
      shape: (C, H, W)
      mask_flat: [C*H*W] binary mask (1=identity, 0=transform)
      context_dim: time embedding dim
      base_channels: width of the conv stack
      num_blocks: number of residual blocks
      log_scale_clamp: clamp for stability
    """
    def __init__(self, shape: tuple[int,int,int], mask_flat, context_dim: int, base_channels=64, num_blocks=4, groups: int = 32, log_scale_clamp: float = 10.0):
        super().__init__()
        self.C, self.H, self.W = shape
        self.D = self.C * self.H * self.W
        self.context_dim = context_dim
        self.log_scale_clamp = float(log_scale_clamp)
        
        if not isinstance(mask_flat, torch.Tensor):
            mask_flat = torch.tensor(mask_flat, dtype=torch.float32)
        mask_flat = mask_flat.to(torch.float32).view(-1)
        
        # Interpret mask: nflows convention is 1=identity, 0=transform
        # But our scatter/gather operations need the opposite interpretation
        mask_bool = mask_flat > 0.5
        id_idx = torch.nonzero(~mask_bool, as_tuple=False).view(-1)  # mask=0 → identity (what we receive)
        tr_idx = torch.nonzero(mask_bool, as_tuple=False).view(-1)   # mask=1 → transform (what we output params for)

        assert id_idx.numel() > 0, "Identity features empty!"
        assert tr_idx.numel() > 0, "Transform features empty!"
        
        self.register_buffer("identity_indices", id_idx)
        self.register_buffer("transform_indices", tr_idx)
        self.register_buffer("mask_flat", mask_flat)

        # A single binary channel that marks identity positions (1 at identity, 0 at transform)
        mask_3d = mask_bool.to(torch.float32).view(self.C, self.H, self.W)
        # Create summary mask: 1 where ANY channel has identity, 0 where ALL are transform
        id_mask_img = (mask_3d.sum(dim=0, keepdim=True) > 0).float()  # Shape: [1, H, W]
        self.register_buffer("identity_mask_img", id_mask_img)

        # image (C) + identity mask (1)
        self.stem = nn.Conv2d(self.C + 1, base_channels, 3, padding=1)

        self.blocks = nn.ModuleList([PreActResBlock(base_channels, context_dim, groups=groups) for _ in range(num_blocks)]) # stack of residual blocks. when used in forward pass, each block will receive the output of the previous block as input and the context as additional input.

        self.head = nn.Conv2d(base_channels, 2 * self.C, 3, padding=1)
        # nn.init.zeros_(self.head.weight)
        # nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, n_identity] (identity slice from nflows)
        context: [B, context_dim] (time embedding)
        returns: [B, 2 * n_transform] (shift and log_scale for transform features)
        """
        B = inputs.size(0)
        device = inputs.device
        D = self.D

        # Reconstruct partial image with identity features
        img_full = torch.zeros(B, D, device = device, dtype = inputs.dtype)
        img_full.scatter_(1, self.identity_indices[None, :].expand(B, -1), inputs) # scatter_ usage: scatter_(dim, index, src), the underscore indicates in-place operation.
        img = img_full.view(B, self.C, self.H, self.W)

        # Append identity-mask channel
        id_mask = self.identity_mask_img.expand(B, -1, -1, -1)
        x = torch.cat([img, id_mask], dim=1)

        h = self.stem(x)
        for block in self.blocks:
            h = block(h, context)

        params = self.head(h) # [B, 2C, H, W]
        params = params.view(B, 2 * self.C, -1) # [B, 2C, D]

        shift_all = params[:, :self.C, :].reshape(B, self.D)
        log_all = params[:, self.C:, :].reshape(B, self.D)

        log_all = log_all.clamp(min=-self.log_scale_clamp, max=self.log_scale_clamp)
        
        shift = torch.gather(shift_all, 1, self.transform_indices[None, :].expand(B, -1)) # collects values from specific positions in the tensor.
        log = torch.gather(log_all, 1, self.transform_indices[None, :].expand(B, -1))

        return torch.cat([shift, log], dim=1) # [B, 2 * n_transform]

def make_image_coupling_factory(
    shape: tuple[int, int, int],
    mask_flat: torch.Tensor,
    context_dim: int,
    base_channels: int = 128,
    num_blocks: int = 4,
    groups: int = 32,
    log_scale_clamp: float = 10.0,
):
    """
    Returns a callable with signature (in_features, out_features) -> nn.Module
    to be passed to nflows.AffineCouplingTransform.
    (in/out are ignored here; nflows requires the signature.)
    """
    def factory(_in_features: int, _out_features: int) -> nn.Module:
        return ImageCouplingResNet(
            shape=shape,
            mask_flat=mask_flat,
            context_dim=context_dim,
            base_channels=base_channels,
            num_blocks=num_blocks,
            groups=groups,
            log_scale_clamp=log_scale_clamp,
        )
    return factory

    # --- add this near the top of time_indexed_realnvp.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorCouplingMLP(nn.Module):
    """
    MLP for vector-mode coupling nets.
    inputs:  [B, in_features]    (identity slice)
    context: [B, context_dim]    (time embedding)
    returns: [B, out_features]   (usually 2 * n_transform)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        context_dim: int,
        hidden_features: int = 512,
        num_hidden_layers: int = 3,
        activation: str = "gelu",
        use_layernorm: bool = False,
        zero_init_last: bool = True,
    ):
        super().__init__()
        act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }.get(activation, nn.GELU)

        layers = [nn.Linear(in_features + context_dim, hidden_features)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_features))
        layers.append(act())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_features))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_features, out_features)

        if zero_init_last:
            # Small random init instead of zeros for testing
            nn.init.normal_(self.out.weight, 0, 0.01)
            nn.init.zeros_(self.out.bias)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([inputs, context], dim=1)
        h = self.backbone(x)
        return self.out(h)


def make_vector_coupling_factory(
    context_dim: int,
    hidden_features: int = 512,
    num_hidden_layers: int = 3,
    activation: str = "gelu",
    use_layernorm: bool = False,
    zero_init_last: bool = True,
):
    """
    Returns a callable (in_features, out_features) -> nn.Module
    matching nflows' transform_net_create_fn signature.
    """
    def factory(in_features: int, out_features: int) -> nn.Module:
        return VectorCouplingMLP(
            in_features=in_features,
            out_features=out_features,
            context_dim=context_dim,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            use_layernorm=use_layernorm,
            zero_init_last=zero_init_last,
        )
    return factory


class TimeIndexedRealNVP(nn.Module):
    """
    T(t, x): time-indexed RealNVP flow.
      - Invertible w.r.t. x for any fixed t
      - Differentiable in both x and t (t enters only through TimeEmbedding)
      - Supports vector inputs [B, D] and image inputs [B, C, H, W]

    Mask convention (your function):
      create_coupling_mask(...): 1 = TRANSFORM, 0 = IDENTITY
    nflows convention for AffineCouplingTransform:
      mask passed in must be 1 = IDENTITY, 0 = TRANSFORM
    -> We therefore flip your mask: mask_nflows = 1 - mask_user
    """
    def __init__(self,
        shape,                          # int for vectors, or (C,H,W) for images
        num_layers: int = 8,
        time_embed_dim: int = 128,      # context dim for coupling nets
        use_permutation: bool = True,   # add RandomPermutation between layers
        # vector-coupling kwargs (used if shape is int)
        vector_hidden: int = 512,
        vector_layers: int = 3,
        vector_activation: str = "gelu",
        vector_layernorm: bool = False,
        # image-coupling kwargs (used if shape is (C,H,W))
        img_base_channels: int = 128,
        img_num_blocks: int = 4,
        img_groups: int = 32,
        img_log_scale_clamp: float = 10.0,
        ):
        super().__init__()
        self.shape = shape
        self.is_image = not isinstance(shape, int)
        self.num_layers = int(num_layers)

        # time embedding
        self.time_embed = FourierTimeEmbedding(embed_dim=time_embed_dim)

        # build the stack of RealNVP coupling layers
        layers = []
        if not self.is_image:
            #  --------- vector mode ---------
            D = int(shape)
            mask_nf = 1.0 - create_coupling_mask(D, mask_type="alternating", even_first=True)
            
            factory = make_vector_coupling_factory(
                context_dim=time_embed_dim,
                hidden_features=vector_hidden,
                num_hidden_layers=vector_layers,
                activation=vector_activation,
                use_layernorm=vector_layernorm,
                zero_init_last=True,
            )
            for i in range(self.num_layers):
                layers.append(transforms.AffineCouplingTransform(
                    mask=mask_nf,
                    transform_net_create_fn=factory,
                ))
                if use_permutation and i < self.num_layers - 1:
                    layers.append(transforms.RandomPermutation(features=D))
                mask_nf = 1.0 - mask_nf # flip the mask for the next layer
        else:
            #  --------- image mode ---------
            C, H, W = shape
            D = C * H * W
            def make_img_mask_nf(i: int):
                """
                Build nflows-convention mask directly:
                  1 = IDENTITY, 0 = TRANSFORM
                We flip the output of your create_coupling_mask (which returns 1=TRANSFORM).
                """
                if i % 2 == 0:
                    m_user = create_coupling_mask((C, H, W), mask_type="checkerboard",
                                                  white_first=(i % 4 == 0))
                else:
                    m_user = create_coupling_mask((C, H, W), mask_type="channel",
                                                  first_half=((i // 2) % 2 == 0))
                return 1.0 - m_user  # flip to nflows convention
            for i in range(self.num_layers):
                # Layer 0: Checkerboard (white_first=True)  
                # Layer 1: Channel (first_half=True)          
                # Layer 2: Checkerboard (white_first=False) 
                # Layer 3: Channel (first_half=False)       
                # Layer 4: Checkerboard (white_first=True)  
                mask_nf = make_img_mask_nf(i)  # 1=IDENTITY, 0=TRANSFORM
                factory = make_image_coupling_factory(
                    shape=(C, H, W),
                    mask_flat=mask_nf,               # pass the SAME (nflows) mask to the net
                    context_dim=time_embed_dim,
                    base_channels=img_base_channels,
                    num_blocks=img_num_blocks,
                    groups=img_groups,
                    log_scale_clamp=img_log_scale_clamp,
                )
                layers.append(transforms.AffineCouplingTransform(
                    mask=mask_nf,
                    transform_net_create_fn=factory,
                ))
                if use_permutation and i < self.num_layers - 1:
                    layers.append(transforms.RandomPermutation(features=D))

        self.transform = transforms.CompositeTransform(layers)
    def forward(self, x: torch.Tensor, t, inverse: bool = False):
        """
        Apply T(t, x) or its inverse.
        x: [B,D] (vector) or [B,C,H,W] (image)
        t: scalar, [B], or [B,1] in [0,1]
        returns: (y, log_det) with y same shape as x, log_det [B]
        """
        # handle time
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_emb = self.time_embed(t)  # [B, time_embed_dim]

        # flatten images for nflows
        orig_shape = x.shape
        if self.is_image:
            B = x.shape[0]
            x_in = x.view(B, -1)
        else:
            x_in = x

        # apply coupling stack
        if not inverse:
            y_flat, log_det = self.transform.forward(x_in, context=t_emb)
        else:
            y_flat, log_det = self.transform.inverse(x_in, context=t_emb)

        # restore image shape
        if self.is_image:
            y = y_flat.view(orig_shape)
        else:
            y = y_flat

        return y, log_det

    def inverse(self, y: torch.Tensor, t):
        """Convenience: compute T^{-1}(t, y)."""
        return self.forward(y, t, inverse=True)


# ---------- builders / factory helpers ----------

def create_vector_flow(
    dim: int,
    *,
    num_layers: int = 8,
    time_embed_dim: int = 128,
    hidden: int = 512,
    mlp_blocks: int = 3,
    activation: str = "gelu",
    use_layernorm: bool = False,
    use_permutation: bool = True,
) -> TimeIndexedRealNVP:
    """
    RealNVP for vectors x ∈ R^dim (multivariate Gaussians, tabular, etc.).
    """
    return TimeIndexedRealNVP(
        shape=dim,
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        use_permutation=use_permutation,
        vector_hidden=hidden,
        vector_layers=mlp_blocks,
        vector_activation=activation,
        vector_layernorm=use_layernorm,
    ).double()


def create_mnist_flow(
    *,
    image_mode: bool = True,   # set False to use flat 784-d vector path
    num_layers: int = 8,
    time_embed_dim: int = 128,
    # image-mode params
    img_base_channels: int = 96,
    img_blocks: int = 4,
    img_groups: int = 32,
    img_log_scale_clamp: float = 10.0,
    # vector-mode params
    hidden: int = 512,
    mlp_blocks: int = 3,
    activation: str = "gelu",
    use_layernorm: bool = False,
    use_permutation: bool = True,
) -> TimeIndexedRealNVP:
    """
    RealNVP for MNIST. image_mode=True → use (1,28,28) conv coupling;
    otherwise use flat 784-d vector coupling.
    """
    if image_mode:
        return TimeIndexedRealNVP(
            shape=(1, 28, 28),
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            use_permutation=use_permutation,
            img_base_channels=img_base_channels,
            img_num_blocks=img_blocks,
            img_groups=img_groups,
            img_log_scale_clamp=img_log_scale_clamp,
        ).double()
    else:
        return create_vector_flow(
            28 * 28,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            hidden=hidden,
            mlp_blocks=mlp_blocks,
            activation=activation,
            use_layernorm=use_layernorm,
            use_permutation=use_permutation,
        )

# Convenience builders for common image flows
def create_cifar10_flow(
    *,
    num_layers: int = 12,
    time_embed_dim: int = 256,
    img_base_channels: int = 128,
    img_blocks: int = 4,
    img_groups: int = 32,
    img_log_scale_clamp: float = 10.0,
    use_permutation: bool = True,
) -> TimeIndexedRealNVP:
    """
    RealNVP for CIFAR-10 images (3×32×32) with ResNet coupling.
    """
    return TimeIndexedRealNVP(
        shape=(3, 32, 32),
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        use_permutation=use_permutation,
        img_base_channels=img_base_channels,
        img_num_blocks=img_blocks,
        img_groups=img_groups,
        img_log_scale_clamp=img_log_scale_clamp,
    ).double()


def create_imagenet_flow(
    *,
    resolution: int = 224,           # 128/224/256...
    num_layers: int = 6,             # fewer layers at high res
    time_embed_dim: int = 512,
    img_base_channels: int = 256,    # wider trunk
    img_blocks: int = 4,
    img_groups: int = 32,
    img_log_scale_clamp: float = 10.0,
    use_permutation: bool = True,
) -> TimeIndexedRealNVP:
    """
    RealNVP for ImageNet-like images (3×R×R). Increase base_channels and keep depth moderate.
    """
    return TimeIndexedRealNVP(
        shape=(3, resolution, resolution),
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        use_permutation=use_permutation,
        img_base_channels=img_base_channels,
        img_num_blocks=img_blocks,
        img_groups=img_groups,
        img_log_scale_clamp=img_log_scale_clamp,
    ).double()


def create_image_flow(
    C: int, H: int, W: int,
    *,
    num_layers: int = 8,
    time_embed_dim: int = 256,
    img_base_channels: int = 128,
    img_blocks: int = 4,
    img_groups: int = 32,
    img_log_scale_clamp: float = 10.0,
    use_permutation: bool = True,
) -> TimeIndexedRealNVP:
    """
    Generic image flow for arbitrary shapes (C,H,W).
    """
    return TimeIndexedRealNVP(
        shape=(C, H, W),
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        use_permutation=use_permutation,
        img_base_channels=img_base_channels,
        img_num_blocks=img_blocks,
        img_groups=img_groups,
        img_log_scale_clamp=img_log_scale_clamp,
    ).double()

# Create a numerically stable ImageNet flow variant
def create_imagenet_flow_stable(
    *,
    resolution: int = 224,
    num_layers: int = 4,          # Fewer layers to reduce error accumulation  
    time_embed_dim: int = 256,
    img_base_channels: int = 128,
    img_blocks: int = 3,
    img_groups: int = 32,
    img_log_scale_clamp: float = 5.0,  # REDUCED from 10.0 to 5.0
    use_permutation: bool = True,
) -> TimeIndexedRealNVP:
    """
    Numerically stable ImageNet flow with:
    - Reduced log_scale_clamp for better reversibility
    - Fewer coupling layers to minimize error accumulation
    - Conservative architecture for numerical stability
    """
    flow = TimeIndexedRealNVP(
        shape=(3, resolution, resolution),
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        use_permutation=use_permutation,
        img_base_channels=img_base_channels,
        img_num_blocks=img_blocks,
        img_groups=img_groups,
        img_log_scale_clamp=img_log_scale_clamp,  # Key change: 5.0 instead of 10.0
    ).double()
    
    # Replace with more stable time embedding
    flow.time_embed = FourierTimeEmbedding(
        embed_dim=time_embed_dim, 
        min_freq=1.0, 
        max_freq=50.0  # Reduced from 1000 to 50
    ).double()
    
    return flow


