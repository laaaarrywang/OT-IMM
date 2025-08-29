import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Any, Tuple, Optional
from torchdiffeq import odeint_adjoint as odeint
from torch import vmap
import math
from . import fabrics
from .stochastic_interpolant import Interpolant, compute_div
import normflows as nf


class SimpleNeuralSplineFlow(nn.Module):
    """
    Simplified time-indexed neural spline flow T_t
    Uses a simpler architecture that's more compatible with normflows
    """
    def __init__(self, dim: int, n_layers: int = 4, hidden_dims: int = 128, 
                 n_bins: int = 8, tail_bound: float = 3.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        
        # Build a sequence of coupling layers
        self.flows = nn.ModuleList()
        
        for i in range(n_layers):
            # Alternate masks for coupling
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[dim//2:] = 1
            else:
                mask[:dim//2] = 1
            
            # Create coupling layer
            # Using AffineCoupling as it's simpler and more stable
            coupling = nf.flows.AffineCoupling(
                mask=mask.bool(),
                affine_type='sigmoid',
                normalize=True
            )
            
            self.flows.append(coupling)
        
        # Add learnable time modulation
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, dim * 2)  # scale and shift for each dimension
        )
    
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
            t = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Get time-dependent modulation
        time_params = self.time_embed(t)
        scale = torch.sigmoid(time_params[:, :self.dim])  # [0, 1]
        shift = time_params[:, self.dim:] * 0.1  # Small shift
        
        # Modulate input based on time (identity at t=0)
        if not inverse:
            # Forward: gradually apply transformation
            z = x * (1 - t + t * scale) + t * shift
            
            # Apply flows
            log_det = torch.zeros(batch_size, device=x.device)
            for flow in self.flows:
                z, ld = flow(z)
                log_det = log_det + ld
            
            # Final time modulation to ensure identity at t=0
            z = (1 - t) * x + t * z
        else:
            # Inverse: undo time modulation first
            z = (x - (1 - t) * x) / (t + 1e-8)
            
            # Apply inverse flows in reverse order
            log_det = torch.zeros(batch_size, device=x.device)
            for flow in reversed(self.flows):
                z, ld = flow.inverse(z)
                log_det = log_det - ld
            
            # Undo initial modulation
            z = (z - t * shift) / (1 - t + t * scale + 1e-8)
        
        return z, log_det
    
    def inverse(self, x: torch.Tensor, t: torch.Tensor):
        """Compute T_t^{-1}(x)"""
        return self.forward(x, t, inverse=True)


class NonlinearInterpolantV2(Interpolant):
    """
    Simplified nonlinear stochastic interpolant using neural flows
    """
    def __init__(
        self,
        dim: int,
        path: str = 'linear',
        gamma_type: str = 'brownian',
        n_layers: int = 4,
        hidden_dims: int = 128,
        n_bins: int = 8,
        tail_bound: float = 3.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Initialize base interpolant structure
        super().__init__(path=path, gamma_type=gamma_type)
        
        self.dim = dim
        self.device = device
        
        # Initialize simplified neural spline flow T_t
        self.T_t = SimpleNeuralSplineFlow(
            dim=dim,
            n_layers=n_layers,
            hidden_dims=hidden_dims,
            n_bins=n_bins,
            tail_bound=tail_bound
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
            t = t.to(self.device)
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            
            # For one-sided interpolants, T_0 is identity (Gaussian base)
            if self.path in ['one-sided-linear', 'one-sided-trig']:
                # x0 ~ N(0,1), so T_0^{-1}(x0) = x0
                T0_inv_x0 = x0
                
                # T_1^{-1}(x1) - transform target back to Gaussian
                t_ones = torch.ones(x1.shape[0], device=self.device)
                T1_inv_x1, _ = self.T_t.inverse(x1, t_ones)
                
                # Linear combination in latent space
                a_t = self.a(t).to(self.device)
                b_t = self.b(t).to(self.device)
                z = a_t * T0_inv_x0 + b_t * T1_inv_x1
                
                # Apply T_t
                if t.dim() == 0:
                    t_batch = t.expand(z.shape[0])
                else:
                    t_batch = t
                result, _ = self.T_t.forward(z, t_batch)
            else:
                # General case
                t_zeros = torch.zeros(x0.shape[0], device=self.device)
                t_ones = torch.ones(x1.shape[0], device=self.device)
                
                T0_inv_x0, _ = self.T_t.inverse(x0, t_zeros)
                T1_inv_x1, _ = self.T_t.inverse(x1, t_ones)
                
                a_t = self.a(t).to(self.device)
                b_t = self.b(t).to(self.device)
                z = a_t * T0_inv_x0 + b_t * T1_inv_x1
                
                if t.dim() == 0:
                    t_batch = t.expand(z.shape[0])
                else:
                    t_batch = t
                result, _ = self.T_t.forward(z, t_batch)
            
            return result
        
        def nonlinear_dtIt(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # Simple finite difference approximation for time derivative
            eps = 1e-5
            
            t = t.to(self.device)
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            
            # Compute I_t and I_{t+eps}
            It_current = nonlinear_It(t, x0, x1)
            It_next = nonlinear_It(t + eps, x0, x1)
            
            # Finite difference
            dtIt = (It_next - It_current) / eps
            
            return dtIt
        
        # Override the interpolant functions
        self.It = nonlinear_It
        self.dtIt = nonlinear_dtIt
    
    def parameters(self):
        """Get parameters of the neural spline flow for optimization"""
        return self.T_t.parameters()
    
    def to(self, device):
        """Move the interpolant to a specific device"""
        self.T_t = self.T_t.to(device)
        self.device = device
        return self


class NonlinearSITrainerV2:
    """
    Simplified trainer for nonlinear stochastic interpolants
    """
    def __init__(
        self,
        interpolant: NonlinearInterpolantV2,
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
        self.opt_interpolant = torch.optim.Adam(
            interpolant.parameters(), lr=lr_interpolant
        )
        
        # Learning rate schedulers
        self.sched_velocity = torch.optim.lr_scheduler.StepLR(
            self.opt_velocity, step_size=1500, gamma=0.4
        )
        self.sched_score = torch.optim.lr_scheduler.StepLR(
            self.opt_score, step_size=1500, gamma=0.4
        )
        self.sched_interpolant = torch.optim.lr_scheduler.StepLR(
            self.opt_interpolant, step_size=1500, gamma=0.4
        )
    
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
        
        Args:
            x0: Base samples [batch, dim]
            x1: Target samples [batch, dim]
            loss_fn_v: Loss function for velocity
            loss_fn_s: Loss function for score
            n_inner_steps: Number of inner minimization steps
            train_interpolant: Whether to update interpolant
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
            
            # Compute losses
            loss_v = loss_fn_v(self.velocity_net, x0, x1, ts, self.interpolant)
            loss_s = loss_fn_s(self.score_net, x0, x1, ts, self.interpolant)
            
            # Check for NaN
            if torch.isnan(loss_v) or torch.isnan(loss_s):
                print("Warning: NaN loss detected, skipping update")
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
        
        # Optionally train interpolant
        if train_interpolant:
            self.opt_interpolant.zero_grad()
            
            # Sample new time points
            ts = torch.rand(bs, device=self.device)
            
            # Compute losses for interpolant update
            loss_v = loss_fn_v(self.velocity_net, x0, x1, ts, self.interpolant)
            loss_s = loss_fn_s(self.score_net, x0, x1, ts, self.interpolant)
            
            # Adversarial loss for interpolant (maximize difficulty)
            loss_interpolant = -(loss_v + loss_s) * 0.1  # Scale down for stability
            
            if not torch.isnan(loss_interpolant):
                loss_interpolant.backward()
                torch.nn.utils.clip_grad_norm_(self.interpolant.parameters(), 1.0)
                self.opt_interpolant.step()
        else:
            loss_interpolant = torch.tensor(0.0)
        
        # Update learning rates
        self.sched_velocity.step()
        self.sched_score.step()
        if train_interpolant:
            self.sched_interpolant.step()
        
        losses['velocity'] = loss_v.item() if not torch.isnan(loss_v) else 0.0
        losses['score'] = loss_s.item() if not torch.isnan(loss_s) else 0.0
        losses['interpolant'] = -loss_interpolant.item() if not torch.isnan(loss_interpolant) else 0.0
        losses['total'] = losses['velocity'] + losses['score']
        
        return losses