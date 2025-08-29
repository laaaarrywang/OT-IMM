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
        
        # Handle time tensor
        if t.dim() == 0:
            t_batch = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                t_batch = t.expand(batch_size).unsqueeze(1)
            elif t.shape[0] == batch_size:
                t_batch = t.unsqueeze(1)
            else:
                raise ValueError(f"Time tensor has incompatible shape: {t.shape} for batch size {batch_size}")
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


class NonlinearInterpolantSimple(Interpolant):
    """
    Simplified nonlinear stochastic interpolant using custom time-modulated flows
    """
    def __init__(
        self,
        dim: int,
        path: str = 'linear',
        gamma_type: str = 'brownian',
        hidden_dims: int = 128,
        n_layers: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Initialize base interpolant structure
        super().__init__(path=path, gamma_type=gamma_type)
        
        self.dim = dim
        self.device = device
        
        # Initialize time-modulated flow T_t
        self.T_t = TimeModulatedFlow(
            dim=dim,
            hidden_dims=hidden_dims,
            n_layers=n_layers
        ).to(device)
        
        # Store original functions
        self.linear_It = self.It
        self.linear_dtIt = self.dtIt
        
        # Override with nonlinear versions
        self._setup_nonlinear_interpolant()
    
    def _setup_nonlinear_interpolant(self):
        """Setup the nonlinear interpolant functions"""
        
        def nonlinear_It(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # Ensure tensors are on correct device
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device)
            else:
                t = t.to(self.device)
            
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            
            # For one-sided interpolants, T_0 is identity (Gaussian base)
            if self.path in ['one-sided-linear', 'one-sided-trig']:
                # x0 ~ N(0,1), so T_0^{-1}(x0) = x0 (identity at t=0)
                T0_inv_x0 = x0
                
                # T_1^{-1}(x1) - transform target back to Gaussian space
                t_ones = torch.ones(x1.shape[0], device=self.device)
                T1_inv_x1, _ = self.T_t.inverse(x1, t_ones)
                
                # Linear combination in latent space
                if t.dim() == 0:
                    a_t = self.a(t).to(self.device)
                    b_t = self.b(t).to(self.device)
                else:
                    a_t = self.a(t.squeeze()).to(self.device)
                    b_t = self.b(t.squeeze()).to(self.device)
                
                z = a_t * T0_inv_x0 + b_t * T1_inv_x1
                
                # Apply T_t to the interpolated latent
                if t.dim() == 0:
                    t_batch = t.unsqueeze(0).expand(z.shape[0])
                elif t.dim() == 1 and t.shape[0] == 1:
                    t_batch = t.expand(z.shape[0])
                else:
                    t_batch = t
                
                result, _ = self.T_t.forward(z, t_batch)
            else:
                # General case (both endpoints transformed)
                t_zeros = torch.zeros(x0.shape[0], device=self.device)
                t_ones = torch.ones(x1.shape[0], device=self.device)
                
                T0_inv_x0, _ = self.T_t.inverse(x0, t_zeros)
                T1_inv_x1, _ = self.T_t.inverse(x1, t_ones)
                
                if t.dim() == 0:
                    a_t = self.a(t).to(self.device)
                    b_t = self.b(t).to(self.device)
                else:
                    a_t = self.a(t.squeeze()).to(self.device)
                    b_t = self.b(t.squeeze()).to(self.device)
                
                z = a_t * T0_inv_x0 + b_t * T1_inv_x1
                
                if t.dim() == 0:
                    t_batch = t.unsqueeze(0).expand(z.shape[0])
                elif t.dim() == 1 and t.shape[0] == 1:
                    t_batch = t.expand(z.shape[0])
                else:
                    t_batch = t
                    
                result, _ = self.T_t.forward(z, t_batch)
            
            return result
        
        def nonlinear_dtIt(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # Simple finite difference approximation for time derivative
            eps = 1e-4
            
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device, dtype=torch.float32)
            else:
                t = t.to(self.device)
            
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            
            # Ensure we don't go outside [0, 1]
            t_next = torch.clamp(t + eps, max=1.0)
            actual_eps = t_next - t
            
            # Compute I_t and I_{t+eps}
            It_current = nonlinear_It(t, x0, x1)
            It_next = nonlinear_It(t_next, x0, x1)
            
            # Finite difference
            if actual_eps > 0:
                dtIt = (It_next - It_current) / actual_eps
            else:
                dtIt = torch.zeros_like(It_current)
            
            return dtIt
        
        # Override the interpolant functions
        self.It = nonlinear_It
        self.dtIt = nonlinear_dtIt
    
    def calc_xt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        """Override calc_xt to use nonlinear interpolant"""
        if self.path in ['one-sided-linear', 'one-sided-trig', 'mirror']:
            return self.It(t, x0, x1)
        else:
            z = torch.randn(x0.shape).to(t)
            return self.It(t, x0, x1) + self.gamma(t)*z, z
    
    def calc_antithetic_xts(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
        """Override for antithetic sampling with nonlinear interpolant"""
        if self.path in ['one-sided-linear', 'one-sided-trig']:
            # For one-sided, use the nonlinear interpolant
            It_base = self.It(t, x0, x1)
            
            # Create antithetic samples around the interpolated point
            # Use the original linear coefficients for the perturbation
            a_t = self.a(t).to(self.device)
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
        return self.T_t.parameters()
    
    def to(self, device):
        """Move the interpolant to a specific device"""
        self.T_t = self.T_t.to(device)
        self.device = device
        return self


class NonlinearSITrainerSimple:
    """
    Simple trainer for nonlinear stochastic interpolants
    """
    def __init__(
        self,
        interpolant: NonlinearInterpolantSimple,
        velocity_net: nn.Module,
        score_net: nn.Module,
        lr_interpolant: float = 1e-4,
        lr_velocity: float = 2e-3,
        lr_score: float = 2e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.interpolant = interpolant.to(device)
        self.velocity_net = velocity_net.to(device)
        self.score_net = score_net.to(device)
        self.device = device
        
        # Optimizers
        self.opt_velocity = torch.optim.Adam(velocity_net.parameters(), lr=lr_velocity)
        self.opt_score = torch.optim.Adam(score_net.parameters(), lr=lr_score)
        
        # Only create interpolant optimizer if it has parameters
        if list(interpolant.parameters()):
            self.opt_interpolant = torch.optim.Adam(
                interpolant.parameters(), lr=lr_interpolant
            )
        else:
            self.opt_interpolant = None
        
        # Learning rate schedulers
        self.sched_velocity = torch.optim.lr_scheduler.StepLR(
            self.opt_velocity, step_size=1500, gamma=0.4
        )
        self.sched_score = torch.optim.lr_scheduler.StepLR(
            self.opt_score, step_size=1500, gamma=0.4
        )
        
        if self.opt_interpolant:
            self.sched_interpolant = torch.optim.lr_scheduler.StepLR(
                self.opt_interpolant, step_size=1500, gamma=0.4
            )
        else:
            self.sched_interpolant = None
    
    def train_step(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        loss_fn_v: Callable,
        loss_fn_s: Callable,
        n_inner_steps: int = 1,
        train_interpolant: bool = True
    ):
        """
        Perform one training step
        """
        bs = x0.shape[0]
        
        # Move data to device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        
        losses = {}
        
        # Train velocity and score networks
        for _ in range(n_inner_steps):
            self.opt_velocity.zero_grad()
            self.opt_score.zero_grad()
            
            # Sample time uniformly
            ts = torch.rand(bs, device=self.device)
            
            # Compute losses with error handling
            try:
                loss_v = loss_fn_v(self.velocity_net, x0, x1, ts, self.interpolant)
                loss_s = loss_fn_s(self.score_net, x0, x1, ts, self.interpolant)
                
                # Check for NaN
                if torch.isnan(loss_v) or torch.isnan(loss_s) or torch.isinf(loss_v) or torch.isinf(loss_s):
                    print("Warning: NaN or Inf loss detected, skipping update")
                    losses['velocity'] = 0.0
                    losses['score'] = 0.0
                    losses['interpolant'] = 0.0
                    losses['total'] = 0.0
                    return losses
                
                # Backward and update
                loss_v.backward()
                loss_s.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
                
                self.opt_velocity.step()
                self.opt_score.step()
                
            except Exception as e:
                print(f"Error in training step: {e}")
                losses['velocity'] = 0.0
                losses['score'] = 0.0
                losses['interpolant'] = 0.0
                losses['total'] = 0.0
                return losses
        
        # Optionally train interpolant
        if train_interpolant and self.opt_interpolant:
            self.opt_interpolant.zero_grad()
            
            try:
                # Sample new time points
                ts = torch.rand(bs, device=self.device)
                
                # Compute losses for interpolant update
                loss_v = loss_fn_v(self.velocity_net, x0, x1, ts, self.interpolant)
                loss_s = loss_fn_s(self.score_net, x0, x1, ts, self.interpolant)
                
                # Adversarial loss for interpolant (maximize difficulty)
                loss_interpolant = -(loss_v + loss_s) * 0.01  # Very small scale for stability
                
                if not torch.isnan(loss_interpolant) and not torch.isinf(loss_interpolant):
                    loss_interpolant.backward()
                    torch.nn.utils.clip_grad_norm_(self.interpolant.parameters(), 0.1)
                    self.opt_interpolant.step()
                else:
                    loss_interpolant = torch.tensor(0.0)
            except Exception as e:
                print(f"Error training interpolant: {e}")
                loss_interpolant = torch.tensor(0.0)
        else:
            loss_interpolant = torch.tensor(0.0)
        
        # Update learning rates
        self.sched_velocity.step()
        self.sched_score.step()
        if train_interpolant and self.sched_interpolant:
            self.sched_interpolant.step()
        
        losses['velocity'] = loss_v.item() if 'loss_v' in locals() and not torch.isnan(loss_v) else 0.0
        losses['score'] = loss_s.item() if 'loss_s' in locals() and not torch.isnan(loss_s) else 0.0
        losses['interpolant'] = -loss_interpolant.item() if not torch.isnan(loss_interpolant) else 0.0
        losses['total'] = losses['velocity'] + losses['score']
        
        return losses