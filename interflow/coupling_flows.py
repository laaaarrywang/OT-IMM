# pip install nflows
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms
from nflows.nn import nets

# ---------- Time embedding (Fourier features) ----------
class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim=128, f_min=1.0, f_max=1000.0):
        super().__init__()
        assert time_embed_dim % 2 == 0, "time_embed_dim must be even"
        n = time_embed_dim // 2
        freqs = torch.exp(torch.linspace(math.log(f_min), math.log(f_max), n))
        self.register_buffer("freqs", freqs)

    def forward(self, t):   # t: [B]
        t = t.view(-1, 1)
        phases = 2.0 * math.pi * t * self.freqs[None, :]
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # [B, 2n]


# ---------- Vector (MLP) RealNVP bits ----------
def alternating_mask(features, start_with_one=True, device=None, dtype=torch.float32):
    m = torch.zeros(features, dtype=dtype, device=device)
    m[start_with_one::2] = 1.0
    return m

def make_affine_coupling_vector(features, time_embed_dim, hidden_features=1024, num_blocks=2,
                                mask=None, activation=F.relu, use_batch_norm=False):
    if mask is None:
        mask = alternating_mask(features, start_with_one=True)

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features=in_features,      # = sum(mask)
            out_features=out_features,    # = 2 * |B|
            hidden_features=hidden_features,
            context_features=time_embed_dim,
            num_blocks=num_blocks,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    return transforms.AffineCouplingTransform(mask=mask, transform_net_create_fn=create_net)


# ---------- Image (Conv) RealNVP bits ----------
def checkerboard_mask_chw(C, H, W, on_white=True, device=None, dtype=torch.float32):
    grid = torch.zeros(1, H, W, device=device, dtype=dtype)
    for i in range(H):
        for j in range(W):
            if ((i + j) % 2 == 0) == on_white:
                grid[0, i, j] = 1.0
    m = grid.expand(C, H, W).clone()
    return m.flatten()  # [D]

def channel_mask_chw(C, H, W, first_half=True, device=None, dtype=torch.float32):
    m = torch.zeros(C, H, W, device=device, dtype=dtype)
    if first_half:
        m[: C // 2, :, :] = 1.0
    else:
        m[C // 2 :, :, :] = 1.0
    return m.flatten()  # [D]

class ImageAffineConditioner(nn.Module):
    """
    Conv conditioner that:
      - gets A-coordinates vector [B, nA] from nflows,
      - rebuilds image with A filled, B zero,
      - adds 1 mask channel and broadcast time-context channels,
      - predicts [shift, log_scale] per pixel (2*C channels),
      - returns only B positions packed as [B, 2*nB] (shift_B, logscale_B).
    """
    def __init__(self, in_features, out_features, C, H, W, mask_flat,
                 time_embed_dim=128, hidden_channels=64, num_blocks=3, ctx_channels=16):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        D = C * H * W
        self.D = D

        # nflows convention: mask==1 -> TRANSFORMED (B), mask==0 -> IDENTITY/conditioner (A)
        mask_bool = mask_flat > 0.5
        idx_B = torch.nonzero(mask_bool, as_tuple=False).view(-1)      # transformed
        idx_A = torch.nonzero(~mask_bool, as_tuple=False).view(-1)     # identity/conditioner

        # Register buffers BEFORE using them
        self.register_buffer("idx_A", idx_A.long())
        self.register_buffer("idx_B", idx_B.long())

        self.nB = int(mask_bool.sum().item())
        self.nA = D - self.nB

        # Sanity checks with what nflows passes to transform_net_create_fn
        assert in_features == self.nA, f"in_features={in_features} but expected {self.nA}"
        assert out_features == 2 * self.nB, f"out_features={out_features} but expected {2*self.nB}"

        # One-hot basis to scatter A coords into flat image without in-place ops (vmap-safe)
        A_basis = F.one_hot(idx_A.long(), num_classes=D).to(torch.float32)  # [nA, D]
        self.register_buffer("A_basis", A_basis)

        # Single mask channel (any channel marked as A becomes 1)
        mask_img = mask_flat.view(C, H, W)
        mask_single = (mask_img.sum(0, keepdim=True) > 0).to(mask_img.dtype)  # [1,H,W]
        self.register_buffer("mask_channel", mask_single)

        # Time context projection -> ctx_channels (broadcast over HxW)
        self.ctx_proj = nn.Linear(time_embed_dim, ctx_channels, bias=True)

        in_ch = C + 1 + ctx_channels
        ch = hidden_channels
        blocks = [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks - 1):
            blocks += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        blocks += [nn.Conv2d(ch, 2 * C, 3, padding=1)]  # outputs [shift, log_scale] per channel
        self.trunk = nn.Sequential(*blocks)

    def forward(self, x_A_vec, context):
        """
        x_A_vec: [B, nA]  (only A-coordinates)
        context: [B, time_embed_dim]
        returns: [B, 2*nB] packed as [shift_B, log_scale_B]
        """
        B = x_A_vec.size(0)
        dtype = x_A_vec.dtype

        # vmap-safe scatter via matmul: [B, nA] @ [nA, D] -> [B, D]
        img_A = torch.matmul(x_A_vec, self.A_basis.to(dtype=dtype))
        img_A = img_A.view(B, self.C, self.H, self.W)

        mask = self.mask_channel.expand(B, -1, -1, -1)                 # [B,1,H,W]
        ctx  = self.ctx_proj(context).view(B, -1, 1, 1).expand(-1, -1, self.H, self.W)

        feats = torch.cat([img_A, mask, ctx], dim=1)                   # [B, C+1+ctx, H, W]
        params_full = self.trunk(feats)                                 # [B, 2*C, H, W]
        params_flat = params_full.view(B, 2 * self.D)                   # [B, 2D]

        shift_full    = params_flat[:, : self.D]                        # [B, D]
        logscale_full = params_flat[:, self.D :]                        # [B, D]

        # vmap-safe gathers
        shift_B    = torch.index_select(shift_full,    1, self.idx_B)   # [B, nB]
        logscale_B = torch.index_select(logscale_full, 1, self.idx_B)   # [B, nB]

        return torch.cat([shift_B, logscale_B], dim=1)                  # [B, 2*nB]

        return torch.cat([shift_B, logscale_B], dim=1)    # [B, 2*nB]

def make_affine_coupling_image(C, H, W, time_embed_dim, mask_flat,
                               hidden_channels=64, num_blocks=3, ctx_channels=16):
    def create_net(in_features, out_features):
        return ImageAffineConditioner(
            in_features=in_features, out_features=out_features,
            C=C, H=H, W=W, mask_flat=mask_flat,
            time_embed_dim=time_embed_dim,
            hidden_channels=hidden_channels, num_blocks=num_blocks, ctx_channels=ctx_channels
        )
    return transforms.AffineCouplingTransform(mask=mask_flat, transform_net_create_fn=create_net)


# ---------- Unified class ----------
class TimeIndexedAffineFlow(nn.Module):
    """
    Unified RealNVP-style, time-conditioned flow:
      - If shape is int D  -> vector mode (MLP conditioner, optional permutations).
      - If shape is (C,H,W)-> image mode (conv conditioner, checkerboard/channel masks).
    Supports both forward/inverse and derivative helpers for either mode.
    """
    def __init__(self,
                 shape,                        # int D  or tuple (C,H,W)
                 num_layers=12,
                 time_embed_dim=128,
                 # vector (MLP) conditioner opts
                 vec_hidden_features=1024,
                 vec_blocks=2,
                 vec_use_batch_norm=False,
                 vec_mix_with_permutations=True,
                 # image (Conv) conditioner opts
                 img_hidden_channels=64,
                 img_blocks=3,
                 img_ctx_channels=16):
        super().__init__()
        self.time_embed = TimeEmbedding(time_embed_dim=time_embed_dim)

        if isinstance(shape, int):
            # -------- vector mode --------
            self.mode = "vector"
            self.D = int(shape)
            self.C = self.H = self.W = None

            layers = []
            mask = alternating_mask(self.D, start_with_one=True)
            for i in range(num_layers):
                layers.append(
                    make_affine_coupling_vector(
                        features=self.D,
                        time_embed_dim=time_embed_dim,
                        hidden_features=vec_hidden_features,
                        num_blocks=vec_blocks,
                        mask=mask,
                        use_batch_norm=vec_use_batch_norm,
                    )
                )
                if vec_mix_with_permutations:
                    layers.append(transforms.RandomPermutation(features=self.D))
                mask = 1.0 - mask
            self.transform = transforms.CompositeTransform(layers)

        else:
            # -------- image mode --------
            assert len(shape) == 3, "shape must be int D or tuple (C,H,W)."
            self.mode = "image"
            self.C, self.H, self.W = map(int, shape)
            self.D = self.C * self.H * self.W

            layers = []
            # alternate checkerboard and channel masks
            for i in range(num_layers):
                if i % 2 == 0:
                    m = checkerboard_mask_chw(self.C, self.H, self.W, on_white=(i % 4 == 0))
                else:
                    m = channel_mask_chw(self.C, self.H, self.W, first_half=((i // 2) % 2 == 0))
                layers.append(
                    make_affine_coupling_image(
                        C=self.C, H=self.H, W=self.W,
                        time_embed_dim=time_embed_dim,
                        mask_flat=m,
                        hidden_channels=img_hidden_channels,
                        num_blocks=img_blocks,
                        ctx_channels=img_ctx_channels,
                    )
                )
            self.transform = transforms.CompositeTransform(layers)

    # ----- flatten/unflatten helpers -----
    def _maybe_flatten(self, x):
        if x.dim() == 4:   # [B,C,H,W] -> [B,D]
            B = x.size(0)
            return x.view(B, -1), True
        elif x.dim() == 2:
            return x, False
        else:
            raise ValueError("x must be [B,D] or [B,C,H,W].")

    def _maybe_unflatten(self, y, was_4d):
        if was_4d and self.mode == "image":
            B = y.size(0)
            return y.view(B, self.C, self.H, self.W)
        elif was_4d and self.mode == "vector":
            # if user fed [B,C,H,W] to vector mode, just reshape back to the same
            B = y.size(0)
            # try to infer shape from D (best-effort)
            raise ValueError("In vector mode, please pass [B,D]; no image shape known to unflatten.")
        return y

    # ----- pure forward/inverse (x->y / y->x) -----
    def forward(self, x, t, reverse=False):
        """
        x: [B,D] or [B,C,H,W] (image mode supports either; vector mode expects [B,D])
        t: [B]
        returns: y (same shape as input if image mode), logabsdet [B]
        """
        if self.mode == "image":
            x_flat, was_4d = self._maybe_flatten(x)
        else:
            # vector mode: allow [B,D]; if [B,C,H,W], flatten to [B,D] but cannot unflatten back
            if x.dim() == 4:
                x_flat = x.view(x.size(0), -1)
                was_4d = True
            else:
                x_flat, was_4d = x, False

        t_embed = self.time_embed(t)
        if not reverse:
            y_flat, logabsdet = self.transform.forward(x_flat, context=t_embed)
        else:
            y_flat, logabsdet = self.transform.inverse(x_flat, context=t_embed)

        if self.mode == "image":
            y = self._maybe_unflatten(y_flat, was_4d)
        else:
            if was_4d:
                # cannot unflatten without (C,H,W); warn user explicitly
                raise ValueError("Vector-mode flow cannot unflatten to image; pass [B,D] instead.")
            y = y_flat
        return y, logabsdet

    # ---------- internal flat maps ----------
    def _f_x_flat(self, x_flat, t):
        t_embed = self.time_embed(t)
        y_flat, _ = self.transform.forward(x_flat, context=t_embed)
        return y_flat

    def _g_y_flat(self, y_flat, t):
        t_embed = self.time_embed(t)
        x_flat, _ = self.transform.inverse(y_flat, context=t_embed)
        return x_flat

    # ---------- d/dt T_t(x) ----------
    def dTdt(self, x, t, method="finite_diff", eps=1e-3):
        if self.mode == "image":
            x_flat, was_4d = self._maybe_flatten(x)
        else:
            if x.dim() == 4:
                x_flat = x.view(x.size(0), -1)
                was_4d = True
            else:
                x_flat, was_4d = x, False

        if method == "finite_diff":
            t_plus  = (t + eps).clamp(0.0, 1.0)
            t_minus = (t - eps).clamp(0.0, 1.0)
            y_plus  = self._f_x_flat(x_flat, t_plus)
            y_minus = self._f_x_flat(x_flat, t_minus)
            dydt_flat = (y_plus - y_minus) / (2.0 * eps)
        else:
            from torch.func import jacrev, vmap
            def f_single(t_i, x_i_flat):
                return self._f_x_flat(x_i_flat.unsqueeze(0), t_i.unsqueeze(0)).squeeze(0)
            dydt_flat = vmap(jacrev(f_single, argnums=0))(t, x_flat)

        if self.mode == "image":
            return self._maybe_unflatten(dydt_flat, was_4d)
        else:
            if was_4d:
                raise ValueError("Vector-mode dTdt got [B,C,H,W]; pass [B,D] instead.")
            return dydt_flat

    # ---------- d/dt T_t^{-1}(y) ----------
    def dInvTdt(self, y, t, method="autograd", eps=1e-3):
        if self.mode == "image":
            y_flat, was_4d = self._maybe_flatten(y)
        else:
            if y.dim() == 4:
                y_flat = y.view(y.size(0), -1)
                was_4d = True
            else:
                y_flat, was_4d = y, False

        if method == "finite_diff":
            t_plus  = (t + eps).clamp(0.0, 1.0)
            t_minus = (t - eps).clamp(0.0, 1.0)
            x_plus  = self._g_y_flat(y_flat, t_plus)
            x_minus = self._g_y_flat(y_flat, t_minus)
            dxdt_flat = (x_plus - x_minus) / (2.0 * eps)

        elif method == "autograd":
            from torch.func import jacrev, vmap
            def inv_single(t_i, y_i_flat):
                return self._g_y_flat(y_i_flat.unsqueeze(0), t_i.unsqueeze(0)).squeeze(0)
            dxdt_flat = vmap(jacrev(inv_single, argnums=0))(t, y_flat)

        elif method == "implicit":
            from torch.autograd.functional import jvp
            x_flat = self._g_y_flat(y_flat, t)
            v_flat = self.dTdt(x_flat, t, method="finite_diff", eps=eps)  # [B,D]
            outs = []
            for i in range(y_flat.size(0)):
                def S_y_only(yy_flat):
                    return self._g_y_flat(yy_flat.unsqueeze(0), t[i].unsqueeze(0)).squeeze(0)
                Ji_v, = jvp(S_y_only, (y_flat[i],), (v_flat[i],), create_graph=False, strict=True)
                outs.append(-Ji_v)
            dxdt_flat = torch.stack(outs, 0)

        else:
            raise ValueError(f"Unknown method: {method}")

        if self.mode == "image":
            return self._maybe_unflatten(dxdt_flat, was_4d)
        else:
            if was_4d:
                raise ValueError("Vector-mode dInvTdt got [B,C,H,W]; pass [B,D] instead.")
            return dxdt_flat

    # ---------- Dense Jacobian (debug only) ----------
    def jacobian_x_dense(self, x, t):
        if self.mode == "image":
            x_flat, _ = self._maybe_flatten(x)
        else:
            x_flat = x if x.dim() == 2 else x.view(x.size(0), -1)

        from torch.func import jacrev, vmap
        def f_single(x_i_flat, t_i):
            return self._f_x_flat(x_i_flat.unsqueeze(0), t_i.unsqueeze(0)).squeeze(0)
        return vmap(jacrev(f_single, argnums=0))(x_flat, t)  # [B,D,D]

    # ---------- JVP: (dT/dx) @ v ----------
    def jvp_x(self, x, t, v):
        if self.mode == "image":
            x_flat, was_4d = self._maybe_flatten(x)
            v_flat, _      = self._maybe_flatten(v)
        else:
            x_flat = x if x.dim() == 2 else x.view(x.size(0), -1)
            v_flat = v if v.dim() == 2 else v.view(v.size(0), -1)
            was_4d = False

        try:
            from torch.func import jvp, vmap
            def body(xi, ti, vi):
                f = lambda z: self._f_x_flat(z.unsqueeze(0), ti.unsqueeze(0)).squeeze(0)
                _, jvp_out = jvp(f, (xi,), (vi,))
                return jvp_out
            Jv_flat = vmap(body)(x_flat, t, v_flat)
        except Exception:
            from torch.autograd.functional import jvp as jvp_old
            outs = []
            for i in range(x_flat.size(0)):
                f = lambda z: self._f_x_flat(z.unsqueeze(0), t[i].unsqueeze(0)).squeeze(0)
                _, jvp_out = jvp_old(f, (x_flat[i],), (v_flat[i],), create_graph=False, strict=True)
                outs.append(jvp_out)
            Jv_flat = torch.stack(outs, 0)

        if self.mode == "image":
            return self._maybe_unflatten(Jv_flat, was_4d)
        else:
            return Jv_flat

    # ---------- VJP: (dT/dx)^T @ u ----------
    def vjp_x(self, x, t, u):
        if self.mode == "image":
            x_flat, was_4d = self._maybe_flatten(x)
            u_flat, _      = self._maybe_flatten(u)
        else:
            x_flat = x if x.dim() == 2 else x.view(x.size(0), -1)
            u_flat = u if u.dim() == 2 else u.view(u.size(0), -1)
            was_4d = False

        try:
            from torch.func import vjp, vmap
            def body(xi, ti, ui):
                f = lambda z: self._f_x_flat(z.unsqueeze(0), ti.unsqueeze(0)).squeeze(0)
                _, pullback = vjp(f, xi)
                (vjp_out,) = pullback(ui)
                return vjp_out
            JT_u_flat = vmap(body)(x_flat, t, u_flat)
        except Exception:
            x_req = x_flat.detach().requires_grad_(True)
            y = self._f_x_flat(x_req, t)
            s = (y * u_flat).sum()
            JT_u_flat, = torch.autograd.grad(s, x_req, retain_graph=False, create_graph=False)

        if self.mode == "image":
            return self._maybe_unflatten(JT_u_flat, was_4d)
        else:
            return JT_u_flat

    # ---------- trace(dT/dx) via Hutchinson ----------
    def trace_jacobian_x(self, x, t, num_samples=1, distribution="rademacher"):
        if self.mode == "image":
            x_flat, _ = self._maybe_flatten(x)
        else:
            x_flat = x if x.dim() == 2 else x.view(x.size(0), -1)

        B, D = x_flat.shape
        est = torch.zeros(B, device=x_flat.device, dtype=x_flat.dtype)
        for _ in range(num_samples):
            if distribution == "rademacher":
                v = torch.empty(B, D, device=x_flat.device, dtype=x_flat.dtype).bernoulli_(0.5).mul_(2).sub_(1)
            else:
                v = torch.randn(B, D, device=x_flat.device, dtype=x_flat.dtype)
            Jv = self.jvp_x(x_flat, t, v)  # returns flat since we pass flat
            est = est + (Jv * v).sum(dim=1)
        return est / float(num_samples)
