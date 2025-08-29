import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Any, Tuple, Optional
from torchdiffeq import odeint_adjoint as odeint
from torch import vmap
import math
from . import fabrics
from .stochastic_interpolant import Interpolant, compute_div


class TimeModulatedFlow(nn.Module):
    """
    Simple time-modulated flow that interpolates between identity (t=0) and learned transformation (t=1)
    """
    def __init__(self, dim: int, hidden_dims: int = 128, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        
        # Network to produce flow parameters conditioned on time
        self.time_net = nn.Sequential(
            nn.Linear(1, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, dim * 2)  # Output scale and shift
        )
        
        # Additional learnable transformation
        self.transform_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dims),  # Input: x + time
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, dim)
        )
        
        # Initialize networks to be close to identity
        with torch.no_grad():
            # Initialize last layer with small weights
            self.time_net[-1].weight.data *= 0.01
            self.time_net[-1].bias.data.zero_()
            self.transform_net[-1].weight.data *= 0.01
            self.transform_net[-1].bias.data.zero_()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, inverse: bool = False):
        """
        Apply T_t or T_t^{-1} to x
        Args:
            x: [batch, dim]
            t: scalar or [batch] time values
            inverse: if True, compute T_t^{-1}(x)
        """
        batch_size = x.shape[0]
        
        # Ensure t is properly shaped [batch, 1]
        if t.dim() == 0:
            t_batch = t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t_batch = t.expand(batch_size).unsqueeze(1)
            else:
                t_batch = t.unsqueeze(1)
        else:
            t_batch = t
        
        if not inverse:
            # Forward transformation
            # Get time-dependent parameters
            time_params = self.time_net(t_batch)
            scale = 1.0 + torch.tanh(time_params[:, :self.dim]) * 0.5  # Scale in [0.5, 1.5]
            shift = time_params[:, self.dim:] * 0.1  # Small shift
            
            # Apply affine transformation
            z = x * scale + shift * t_batch
            
            # Apply additional nonlinear transformation
            x_t = torch.cat([z, t_batch], dim=1)
            residual = self.transform_net(x_t) * t_batch  # Scale by time
            z = z + residual
            
            # Ensure identity at t=0 by interpolation
            z = (1 - t_batch) * x + t_batch * z
            
            # Simple log determinant (approximation)
            log_det = torch.sum(torch.log(torch.abs(scale) + 1e-8), dim=1) * t_batch.squeeze()
        else:
            # Inverse transformation (approximate)
            # For simplicity, use fixed-point iteration
            z = x.clone()
            
            # Iterate to find inverse
            for _ in range(5):
                # Get parameters
                time_params = self.time_net(t_batch)
                scale = 1.0 + torch.tanh(time_params[:, :self.dim]) * 0.5
                shift = time_params[:, self.dim:] * 0.1
                
                # Compute forward transform of current z
                z_scaled = z * scale + shift * t_batch
                x_t = torch.cat([z_scaled, t_batch], dim=1)
                residual = self.transform_net(x_t) * t_batch
                z_forward = z_scaled + residual
                z_forward = (1 - t_batch) * z + t_batch * z_forward
                
                # Update z towards inverse
                z = z - 0.5 * (z_forward - x)
            
            log_det = -torch.sum(torch.log(torch.abs(scale) + 1e-8), dim=1) * t_batch.squeeze()
        
        return z, log_det
    
    def inverse(self, x: torch.Tensor, t: torch.Tensor):
        """Compute T_t^{-1}(x)"""
        return self.forward(x, t, inverse=True)


class NonlinearInterpolantSimpleFixed(Interpolant):
    """
    Fixed version of nonlinear stochastic interpolant that avoids vmap issues
    """
    def __init__(
        self,
        dim: int,
        path: str = 'linear',
        gamma_type: str = 'brownian',
        hidden_dims: int = 128,
        n_layers: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_nonlinear: bool = True  # Flag to enable/disable nonlinear transform
    ):
        # Initialize base interpolant structure
        super().__init__(path=path, gamma_type=gamma_type)
        
        self.dim = dim
        self.device = device
        self.use_nonlinear = use_nonlinear
        
        if use_nonlinear:
            # Initialize time-modulated flow T_t
            self.T_t = TimeModulatedFlow(
                dim=dim,
                hidden_dims=hidden_dims,
                n_layers=n_layers
            ).to(device)
        else:
            self.T_t = None
        
        # Store original functions
        self.linear_It = self.It
        self.linear_dtIt = self.dtIt
        
        # Only override if using nonlinear
        if use_nonlinear:
            self._setup_nonlinear_interpolant()
    
    def _setup_nonlinear_interpolant(self):
        """Setup the nonlinear interpolant functions without conditional logic"""
        
        # For one-sided interpolants, we need to handle this carefully
        is_one_sided = (self.path in ['one-sided-linear', 'one-sided-trig'])
        
        def nonlinear_It(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # Ensure tensors are on correct device
            t = t.to(self.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device)
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            
            # Always compute both branches, then select based on path
            # This avoids conditional logic that breaks vmap
            
            # Branch 1: One-sided (T_0 is identity)
            T0_inv_x0_onesided = x0  # Identity at t=0
            t_ones = torch.ones(x1.shape[0], device=self.device)
            T1_inv_x1, _ = self.T_t.inverse(x1, t_ones)
            
            # Branch 2: General case
            t_zeros = torch.zeros(x0.shape[0], device=self.device)
            T0_inv_x0_general, _ = self.T_t.inverse(x0, t_zeros)
            
            # Select appropriate branch without conditional
            # Use multiplication by boolean mask
            use_onesided = float(is_one_sided)
            T0_inv_x0 = use_onesided * T0_inv_x0_onesided + (1 - use_onesided) * T0_inv_x0_general
            
            # Compute coefficients - handle different tensor shapes
            if t.dim() == 0:
                t_coeff = t
            elif t.dim() == 1:
                t_coeff = t[0] if t.shape[0] == 1 else t
            else:
                t_coeff = t.squeeze()
            
            a_t = self.a(t_coeff).to(self.device)
            b_t = self.b(t_coeff).to(self.device)
            
            # Ensure proper broadcasting
            if a_t.dim() == 0:
                a_t = a_t.unsqueeze(0).unsqueeze(1).expand(x0.shape[0], 1)
            elif a_t.dim() == 1:
                if a_t.shape[0] == 1:
                    a_t = a_t.unsqueeze(1).expand(x0.shape[0], 1)
                else:
                    a_t = a_t.unsqueeze(1)
            
            if b_t.dim() == 0:
                b_t = b_t.unsqueeze(0).unsqueeze(1).expand(x1.shape[0], 1)
            elif b_t.dim() == 1:
                if b_t.shape[0] == 1:
                    b_t = b_t.unsqueeze(1).expand(x1.shape[0], 1)
                else:
                    b_t = b_t.unsqueeze(1)
            
            # Linear combination in latent space
            z = a_t * T0_inv_x0 + b_t * T1_inv_x1
            
            # Apply T_t
            t_batch = t if t.dim() > 0 else t.unsqueeze(0)
            if t_batch.shape[0] == 1 and z.shape[0] > 1:
                t_batch = t_batch.expand(z.shape[0])
            
            result, _ = self.T_t.forward(z, t_batch)
            return result
        
        def nonlinear_dtIt(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # Simple finite difference approximation for time derivative
            eps = 1e-4
            
            t = t.to(self.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device, dtype=torch.float32)
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            
            # Compute I_t and I_{t+eps}
            t_next = torch.clamp(t + eps, max=1.0)
            actual_eps = t_next - t
            
            It_current = nonlinear_It(t, x0, x1)
            It_next = nonlinear_It(t_next, x0, x1)
            
            # Avoid division by zero
            safe_eps = torch.where(actual_eps > 0, actual_eps, torch.ones_like(actual_eps))
            dtIt = torch.where(actual_eps > 0, (It_next - It_current) / safe_eps, torch.zeros_like(It_current))
            
            return dtIt
        
        # Override the interpolant functions
        self.It = nonlinear_It
        self.dtIt = nonlinear_dtIt
    
    def calc_xt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        """Override calc_xt to use appropriate interpolant"""
        if self.path in ['one-sided-linear', 'one-sided-trig', 'mirror']:
            return self.It(t, x0, x1)
        else:
            z = torch.randn(x0.shape).to(t)
            return self.It(t, x0, x1) + self.gamma(t)*z, z
    
    def calc_antithetic_xts(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        """Override for antithetic sampling"""
        is_one_sided = (self.path in ['one-sided-linear', 'one-sided-trig'])
        
        if is_one_sided:
            # For one-sided, use the interpolant
            It_base = self.It(t, x0, x1)
            
            # Create antithetic samples
            if t.dim() == 0:
                t_coeff = t
            elif t.dim() == 1:
                t_coeff = t[0] if t.shape[0] == 1 else t
            else:
                t_coeff = t.squeeze()
            
            a_t = self.a(t_coeff).to(self.device)
            
            # Ensure proper shape for multiplication
            if a_t.dim() == 0:
                a_t = a_t.unsqueeze(0).unsqueeze(1).expand(x0.shape[0], 1)
            elif a_t.dim() == 1:
                if a_t.shape[0] == 1:
                    a_t = a_t.unsqueeze(1).expand(x0.shape[0], 1)
                else:
                    a_t = a_t.unsqueeze(1)
            
            It_p = It_base + 0.1 * a_t * x0  # Small perturbation
            It_m = It_base - 0.1 * a_t * x0
            
            return It_p, It_m, x0
        else:
            z = torch.randn(x0.shape).to(t)
            gam = self.gamma(t)
            It = self.It(t, x0, x1)
            return It + gam*z, It - gam*z, z
    
    def parameters(self):
        """Get parameters of the flow for optimization"""
        if self.use_nonlinear and self.T_t is not None:
            return self.T_t.parameters()
        else:
            return iter([])  # Return empty iterator if no parameters
    
    def to(self, device):
        """Move the interpolant to a specific device"""
        if self.T_t is not None:
            self.T_t = self.T_t.to(device)
        self.device = device
        return self


# Import the trainer from the simple version
from .nonlinear_stochastic_interpolant_simple import NonlinearSITrainerSimple