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
from normflows import flows, nets, utils


class NeuralSplineFlow(nn.Module):
    """
    Time-indexed neural spline flow T_t that satisfies:
    - T_0 is identity (for Gaussian base)
    - Supports inverse operation
    - Supports differentiation w.r.t. time
    """
    def __init__(self, dim: int, n_layers: int = 4, hidden_dims: int = 128, 
                 n_bins: int = 8, tail_bound: float = 3.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        
        # Create time-conditioned coupling layers
        self.flows = nn.ModuleList()
        for i in range(n_layers):
            # Create masks for coupling layers
            if i % 2 == 0:
                mask = torch.zeros(dim)
                mask[dim//2:] = 1
            else:
                mask = torch.ones(dim)
                mask[dim//2:] = 0
            
            # Time-conditioned neural network for the spline parameters
            param_net = self._create_conditioned_net(dim, hidden_dims, n_bins)
            
            # Create the coupling layer with rational quadratic splines
            coupling = flows.CoupledRationalQuadraticSpline(
                n_dims=dim,
                n_bins=n_bins,
                tail_bound=tail_bound,
                mask=mask,
                param_net=param_net
            )
            
            self.flows.append(coupling)
    
    def _create_conditioned_net(self, dim: int, hidden_dims: int, n_bins: int):
        """Create a time-conditioned network for spline parameters"""
        # Input: masked dimensions + time (1D)
        # Output: parameters for rational quadratic spline
        input_dim = dim // 2 + 1  # Half dims from mask + time
        
        # Each dimension needs: n_bins widths, n_bins heights, n_bins-1 derivatives
        output_dim = (dim - dim // 2) * (3 * n_bins - 1)
        
        class TimeConditionedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, output_dim)
                )
                
                # Initialize last layer with small weights for stability
                nn.init.normal_(self.net[-1].weight, std=0.01)
                nn.init.zeros_(self.net[-1].bias)
            
            def forward(self, x, t):
                # x: [batch, dim//2] (masked dimensions)
                # t: [batch, 1] or scalar
                if t.dim() == 0:
                    t = t.unsqueeze(0).expand(x.shape[0], 1)
                elif t.dim() == 1:
                    t = t.unsqueeze(1)
                    
                inp = torch.cat([x, t], dim=-1)
                return self.net(inp)
        
        return TimeConditionedNet()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, inverse: bool = False):
        """
        Apply T_t or T_t^{-1} to x
        Args:
            x: [batch, dim]
            t: scalar or [batch] time values
            inverse: if True, compute T_t^{-1}(x)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        
        # Scale t to control the strength of the transformation
        # At t=0, scale=0 (identity), at t=1, scale=1 (full transformation)
        scale = t
        
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x.clone()
        
        if not inverse:
            # Forward pass through flows
            for flow in self.flows:
                # Apply scaled transformation
                z_temp, ld = flow(z, context=scale.unsqueeze(1))
                z = (1 - scale.unsqueeze(1)) * z + scale.unsqueeze(1) * z_temp
                log_det = log_det + scale * ld.squeeze()
        else:
            # Inverse pass through flows (reversed order)
            for flow in reversed(self.flows):
                # Inverse of scaled transformation
                z_temp = (z - (1 - scale.unsqueeze(1)) * z) / scale.unsqueeze(1)
                z_temp, ld = flow.inverse(z_temp, context=scale.unsqueeze(1))
                z = z_temp
                log_det = log_det - scale * ld.squeeze()
        
        return z, log_det
    
    def inverse(self, x: torch.Tensor, t: torch.Tensor):
        """Compute T_t^{-1}(x)"""
        return self.forward(x, t, inverse=True)
    
    def differentiate_wrt_time(self, x: torch.Tensor, t: torch.Tensor):
        """
        Compute d/dt T_t(x) using automatic differentiation
        """
        t = t.requires_grad_(True)
        z, _ = self.forward(x, t)
        
        # Compute gradient w.r.t. time for each dimension
        dt_Tt = []
        for i in range(self.dim):
            grad = torch.autograd.grad(
                z[:, i].sum(), t, 
                create_graph=True, 
                retain_graph=True
            )[0]
            dt_Tt.append(grad)
        
        dt_Tt = torch.stack(dt_Tt, dim=1)
        return dt_Tt


class NonlinearInterpolant(Interpolant):
    """
    Nonlinear stochastic interpolant using neural spline flows
    I_t(x_0, x_1) = T_t(A_t T_0^{-1}(x_0) + B_t T_1^{-1}(x_1))
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
        
        # Initialize neural spline flow T_t
        self.T_t = NeuralSplineFlow(
            dim=dim,
            n_layers=n_layers,
            hidden_dims=hidden_dims,
            n_bins=n_bins,
            tail_bound=tail_bound
        ).to(device)
        
        # Override the interpolant functions
        self._setup_nonlinear_interpolant()
    
    def _setup_nonlinear_interpolant(self):
        """Setup the nonlinear interpolant functions"""
        
        # Store original linear functions
        self.linear_It = self.It
        self.linear_dtIt = self.dtIt
        
        # Define new nonlinear interpolant
        def nonlinear_It(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # For one-sided interpolants, T_0 is identity (Gaussian base)
            if self.path in ['one-sided-linear', 'one-sided-trig']:
                # x0 ~ N(0,1), so T_0^{-1}(x0) = x0
                T0_inv_x0 = x0
                # T_1^{-1}(x1) 
                T1_inv_x1, _ = self.T_t.inverse(x1, torch.ones_like(t))
                
                # Linear combination in latent space
                z = self.a(t) * T0_inv_x0 + self.b(t) * T1_inv_x1
                
                # Apply T_t
                result, _ = self.T_t.forward(z, t)
            else:
                # General case with both endpoints transformed
                T0_inv_x0, _ = self.T_t.inverse(x0, torch.zeros_like(t))
                T1_inv_x1, _ = self.T_t.inverse(x1, torch.ones_like(t))
                
                # Linear combination in latent space
                z = self.a(t) * T0_inv_x0 + self.b(t) * T1_inv_x1
                
                # Apply T_t
                result, _ = self.T_t.forward(z, t)
            
            return result
        
        def nonlinear_dtIt(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor):
            # Compute time derivative of the nonlinear interpolant
            if self.path in ['one-sided-linear', 'one-sided-trig']:
                T0_inv_x0 = x0
                T1_inv_x1, _ = self.T_t.inverse(x1, torch.ones_like(t))
            else:
                T0_inv_x0, _ = self.T_t.inverse(x0, torch.zeros_like(t))
                T1_inv_x1, _ = self.T_t.inverse(x1, torch.ones_like(t))
            
            # Linear combination and its time derivative
            z = self.a(t) * T0_inv_x0 + self.b(t) * T1_inv_x1
            dz_dt = self.adot(t) * T0_inv_x0 + self.bdot(t) * T1_inv_x1
            
            # Apply T_t and get its time derivative
            Tt_z, _ = self.T_t.forward(z, t)
            
            # Compute dT_t/dt(z) using automatic differentiation
            dt_Tt_z = self.T_t.differentiate_wrt_time(z, t)
            
            # Compute Jacobian of T_t w.r.t. z
            # For efficiency, we use the fact that the flow is invertible
            # and compute via the chain rule
            t_expanded = t.requires_grad_(True)
            z_expanded = z.requires_grad_(True)
            
            Tt_z_temp, log_det = self.T_t.forward(z_expanded, t_expanded)
            
            # Jacobian-vector product
            jac_dz = torch.autograd.grad(
                Tt_z_temp, z_expanded,
                grad_outputs=dz_dt,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Total derivative: dI_t/dt = dT_t/dt(z) + J_T_t(z) * dz/dt
            result = dt_Tt_z + jac_dz
            
            return result
        
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


class NonlinearSITrainer:
    """
    Trainer for nonlinear stochastic interpolants with max-min optimization
    """
    def __init__(
        self,
        interpolant: NonlinearInterpolant,
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
        
        # Optimizers for max-min problem
        # Minimize over velocity and score networks
        self.opt_velocity = torch.optim.Adam(velocity_net.parameters(), lr=lr_velocity)
        self.opt_score = torch.optim.Adam(score_net.parameters(), lr=lr_score)
        
        # Maximize over interpolant (minimize negative loss)
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
        n_inner_steps: int = 1
    ):
        """
        Perform one training step with max-min optimization
        
        Args:
            x0: Base samples [batch, dim]
            x1: Target samples [batch, dim]
            loss_fn_v: Loss function for velocity
            loss_fn_s: Loss function for score
            n_inner_steps: Number of inner minimization steps
        """
        bs = x0.shape[0]
        
        # Move data to device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        
        losses = {}
        
        # Inner loop: minimize over velocity and score
        for _ in range(n_inner_steps):
            self.opt_velocity.zero_grad()
            self.opt_score.zero_grad()
            
            # Sample time uniformly
            ts = torch.rand(bs, device=self.device)
            
            # Compute losses
            loss_v = loss_fn_v(self.velocity_net, x0, x1, ts, self.interpolant)
            loss_s = loss_fn_s(self.score_net, x0, x1, ts, self.interpolant)
            
            # Backward and update
            loss_v.backward(retain_graph=True)
            loss_s.backward(retain_graph=True)
            
            self.opt_velocity.step()
            self.opt_score.step()
        
        # Outer step: maximize over interpolant (minimize negative loss)
        self.opt_interpolant.zero_grad()
        
        # Sample new time points
        ts = torch.rand(bs, device=self.device)
        
        # Compute losses again for interpolant update
        loss_v = loss_fn_v(self.velocity_net, x0, x1, ts, self.interpolant)
        loss_s = loss_fn_s(self.score_net, x0, x1, ts, self.interpolant)
        
        # We want to maximize the loss w.r.t. interpolant
        # This encourages harder interpolation paths
        loss_interpolant = -(loss_v + loss_s)
        
        loss_interpolant.backward()
        self.opt_interpolant.step()
        
        # Update learning rates
        self.sched_velocity.step()
        self.sched_score.step()
        self.sched_interpolant.step()
        
        losses['velocity'] = loss_v.item()
        losses['score'] = loss_s.item()
        losses['interpolant'] = -loss_interpolant.item()
        losses['total'] = losses['velocity'] + losses['score']
        
        return losses