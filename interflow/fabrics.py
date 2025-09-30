import numpy as np
import torch
from torch.func import jvp
from . import util
from .realnvp import TimeIndexedRealNVP, create_vector_flow, create_mnist_flow, create_cifar10_flow, create_imagenet_flow, create_imagenet_flow_stable, create_image_flow
import math
import hashlib
import os


class InputWrapper(torch.nn.Module):
    def __init__(self, v):
        super(InputWrapper, self).__init__()
        self.v = v
        
    def net_inp(
        self,
        t: torch.tensor,  # [1]
        x: torch.tensor   # [batch x dim]
    ) -> torch.tensor:    # [batch x (1 + dim)]
        """Concatenate time over the batch dimension."""
        # Ensure t matches x's dtype and device
        t = t.to(dtype=x.dtype, device=x.device)
        inp = torch.cat((t.repeat(x.shape[0]).unsqueeze(1), x), dim = 1)
        return inp
    
    def forward(self, x, t):
        # Ensure wrapped module matches x's dtype/device
        self.v.to(x)
        tx = self.net_inp(t,x)
        return self.v(tx)

def make_fc_net(hidden_sizes, in_size, out_size, inner_act, final_act, **config):
    sizes = [in_size] + hidden_sizes + [out_size]
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Linear(
            sizes[i], sizes[i+1]))
        if i != len(sizes) - 2:
            net.append(make_activation(inner_act))
            continue
        else:
            if make_activation(final_act):
                net.append(make_activation(final_act))
                
    v_net = torch.nn.Sequential(*net)
    return InputWrapper(v_net)





def make_It(path='linear', gamma = None, gamma_dot = None, gg_dot = None,
           # New parameters for nonlinear interpolant
           flow_config = None, data_type = 'vector', data_dim = None,
           # New parameter for diagonal matrix coefficients (for debugging rectangular targets)
           diagonal_scale = None):
    """gamma function must be specified if using the trigonometric interpolant
    
       For nonlinear interpolant:
       - flow_config: dict with flow parameters
       - data_type: 'vector', 'mnist', 'cifar10', 'image', 'imagenet'
       - data_dim: dimension for vector data or (C, H, W) for image data
    """

    if path == 'linear':
        
        
        a      = lambda t: (1-t)
        adot   = lambda t: -1.0
        b      = lambda t: t
        bdot   = lambda t: 1.0
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'trig':
        if gamma == None:
            raise TypeError("Gamma function must be provided for trigonometric interpolant!")
        a    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)
        b    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        adot = lambda t: -gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t) \
                                - 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        bdot = lambda t: -gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t) \
                                + 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)

        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'encoding-decoding':

        a    = lambda t: torch.where(t <= 0.5, torch.cos(math.pi*t)**2, torch.tensor(0.))
        adot = lambda t: torch.where(t <= 0.5, -2*math.pi*torch.cos(math.pi*t)*torch.sin(math.pi*t), torch.tensor(0.))
        b    = lambda t: torch.where(t > 0.5,  torch.cos(math.pi*t)**2, 0.)
        bdot = lambda t: torch.where(t > 0.5,  -2*math.pi*torch.cos(math.pi*t)*torch.sin(math.pi*t), torch.tensor(0.))
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
    
    elif path == 'one-sided-linear':
        # Original scalar implementation (kept for backward compatibility)
        # a      = lambda t: (1-t)
        # adot   = lambda t: -1.0
        # b      = lambda t: t
        # bdot   = lambda t: 1.0
        # It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        # dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1

        # New implementation with optional diagonal scaling
        if diagonal_scale is not None:
            # diagonal_scale should be a tensor of shape [data_dim]
            # Different interpolation speeds for each dimension
            if not isinstance(diagonal_scale, torch.Tensor):
                diagonal_scale = torch.tensor(diagonal_scale, dtype=torch.float32)

            # Ensure diagonal_scale is positive
            diagonal_scale = torch.abs(diagonal_scale)

            def a(t):
                # a(t) could also be diagonal, but keeping scalar for simplicity
                return (1-t)

            def adot(t):
                return -1.0

            def b(t):
                # We want b(1) = 1 for all dimensions (identity behavior)
                # But different dimensions can reach 1 at different rates
                # Use a power function: b_i(t) = t^(1/scale_i)
                # This ensures b_i(1) = 1 for all i, but with different speeds
                if isinstance(t, torch.Tensor):
                    t = t.view(-1, 1) if t.dim() > 0 else t
                # Each dimension interpolates at rate controlled by 1/diagonal_scale
                # Higher scale = slower interpolation for that dimension
                return torch.pow(t, 1.0 / diagonal_scale)

            def bdot(t):
                # Derivative of t^(1/scale) is (1/scale) * t^(1/scale - 1)
                if isinstance(t, torch.Tensor):
                    t = t.view(-1, 1) if t.dim() > 0 else t
                return (1.0 / diagonal_scale) * torch.pow(t, 1.0 / diagonal_scale - 1.0)

            def It(t, x0, x1):
                # Interpolation with diagonal b
                # Each dimension has its own interpolation coefficient
                b_diag = b(t)
                if b_diag.dim() == 1:
                    # Expand for broadcasting with batch
                    b_diag = b_diag.unsqueeze(0)
                return a(t)*x0 + b_diag*x1

            def dtIt(t, x0, x1):
                bdot_diag = bdot(t)
                if bdot_diag.dim() == 1:
                    bdot_diag = bdot_diag.unsqueeze(0)
                return adot(t)*x0 + bdot_diag*x1
        else:
            # Original scalar implementation
            a      = lambda t: (1-t)
            adot   = lambda t: -1.0
            b      = lambda t: t
            bdot   = lambda t: 1.0
            It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
            dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1

    elif path == 'one-sided-trig':

        a      = lambda t: torch.cos(0.5*math.pi*t)
        adot   = lambda t: -0.5*math.pi*torch.sin(0.5*math.pi*t)
        b      = lambda t: torch.sin(0.5*math.pi*t)
        bdot   = lambda t: 0.5*math.pi*torch.cos(0.5*math.pi*t)

        
        It   = lambda t, x0, x1: a(t)*x0 + b(t)*x1
        dtIt = lambda t, x0, x1: adot(t)*x0 + bdot(t)*x1
        
    elif path == 'mirror':
        if gamma == None:
            raise TypeError("Gamma function must be provided for mirror interpolant!")
        
        a     = lambda t: gamma(t)
        adot  = lambda t: gamma_dot(t)
        b     = lambda t: torch.tensor(1.0)
        bdot  = lambda t: torch.tensor(0.0)
        
        It    = lambda t, x0, x1: b(t)*x1 + a(t)*x0
        dtIt  = lambda t, x0, x1: adot(t)*x0

    elif path == "nonlinear":
        # Nonlinear interpolant using TimeIndexedAffineFlow
        # I_t = T_t(a_t * T_0^{-1}(x0) + b_t * T_1^{-1}(x1))
        
        a      = lambda t: torch.cos(0.5*math.pi*t)
        adot   = lambda t: -0.5*math.pi*torch.sin(0.5*math.pi*t)
        b      = lambda t: torch.sin(0.5*math.pi*t)
        bdot   = lambda t: 0.5*math.pi*torch.cos(0.5*math.pi*t)
        # Initialize the flow model based on data type
        if flow_config is None:
            flow_config = {}  # Use defaults

        if data_type == 'vector':
            if data_dim is None:
                raise ValueError("data_dim must be specified for vector data")
            flow_model = create_vector_flow(data_dim, **flow_config)
        elif data_type == 'mnist':
            flow_model = create_mnist_flow(**flow_config)
        elif data_type == 'cifar10':
            flow_model = create_cifar10_flow(**flow_config)
        elif data_type == 'imagenet':
            flow_model = create_imagenet_flow(**flow_config)
        elif data_type == 'imagenet_stable':
            flow_model = create_imagenet_flow_stable(**flow_config)
        elif data_type == 'image':
            if data_dim is None or len(data_dim) != 3:
                raise ValueError("data_dim must be (C,H,W) for custom images")
            C, H, W = data_dim
            flow_model = create_image_flow(C, H, W, **flow_config)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        def It_method(self, t, x0, x1):
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype = x0.dtype, device = x0.device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            
            # Get batch size
            B = x0.shape[0]

            # Expand t to match batch size
            if t.shape[0] == 1 and B>1:
                t_expanded = t.expand(B)
            else:
                t_expanded = t

            # Create boundary time tensors
            t_0 = torch.zeros_like(t_expanded, dtype = x0.dtype, device = x0.device)
            t_1 = torch.ones_like(t_expanded, dtype = x1.dtype, device = x1.device)
            
            # Apply inverse flow to get z0 and z1
            z0, _  = self(x0, t_0, inverse=True) # T_0^{-1}(x0)
            z1, _  = self(x1, t_1, inverse=True) # T_1^{-1}(x1)

            # Step 2: Linear combination in latent space
            a_t = a(t)  # Coefficient at time t
            b_t = b(t)  # Coefficient at time t

            # Handle broadcasting for batch dimension
            if a_t.dim() == 0:
                a_t = a_t.unsqueeze(0)
            if b_t.dim() == 0:
                b_t = b_t.unsqueeze(0)

            # Expand coefficients to match data shape
            if data_type == 'vector':
                a_t = a_t.view(-1, 1).expand(B, -1)  # [B, 1] -> [B, D]
                b_t = b_t.view(-1, 1).expand(B, -1)
                if a_t.shape[1] == 1:
                    a_t = a_t.expand_as(z0)
                    b_t = b_t.expand_as(z1)
            else:  # Image data
                while a_t.dim() < z0.dim():
                    a_t = a_t.unsqueeze(-1)
                    b_t = b_t.unsqueeze(-1)
                a_t = a_t.expand_as(z0)
                b_t = b_t.expand_as(z1)

            z_interp = a_t * z0 + b_t * z1

            # Step 3: Apply forward flow at time t
            y, _ = self(z_interp, t_expanded, inverse=False)  # T_t(z_interp)

            return y 
        def dtIt_method(self, t, x0, x1):
            """
            Time derivative of nonlinear interpolation.
            
            d/dt[I_t] = ∂T_t/∂t(z_t) + (∂T_t/∂z)(z_t) * (a'_t*z0 + b'_t*z1)
            
            where z_t = a_t*T_0^{-1}(x0) + b_t*T_1^{-1}(x1)
            """
            # Ensure t is a tensor with gradient tracking
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            
            B = x0.shape[0]
            
            # Expand t if needed
            if t.shape[0] == 1 and B > 1:
                t_expanded = t.expand(B)
            else:
                t_expanded = t
            
            # Get latent representations at boundaries
            t_0 = torch.zeros(B, dtype=x0.dtype, device=x0.device)
            t_1 = torch.ones(B, dtype=x1.dtype, device=x1.device)
            
            z0, _ = self(x0, t_0, inverse=True)  # T_0^{-1}(x0)
            z1, _ = self(x1, t_1, inverse=True)  # T_1^{-1}(x1)
            
            # Compute coefficients and their derivatives
            a_t = a(t)
            b_t = b(t)
            adot_t = adot(t)
            bdot_t = bdot(t)
            
            # Handle broadcasting
            for coef in [a_t, b_t, adot_t, bdot_t]:
                if coef.dim() == 0:
                    coef = coef.unsqueeze(0)
            
            # Expand coefficients to match data shape
            if data_type == 'vector':
                a_t = a_t.view(-1, 1).expand(B, -1)
                b_t = b_t.view(-1, 1).expand(B, -1)
                adot_t = adot_t.view(-1, 1).expand(B, -1)
                bdot_t = bdot_t.view(-1, 1).expand(B, -1)
                if a_t.shape[1] == 1:
                    a_t = a_t.expand_as(z0)
                    b_t = b_t.expand_as(z1)
                    adot_t = adot_t.expand_as(z0)
                    bdot_t = bdot_t.expand_as(z1)
            else:  # Image data
                for coef in [a_t, b_t, adot_t, bdot_t]:
                    while coef.dim() < z0.dim():
                        coef = coef.unsqueeze(-1)
                a_t = a_t.expand_as(z0)
                b_t = b_t.expand_as(z1)
                adot_t = adot_t.expand_as(z0)
                bdot_t = bdot_t.expand_as(z1)
            
            # Interpolated latent point
            z_interp = a_t * z0 + b_t * z1
            
            # Define forward at time t
            def F(z, tt):
                out, _ = self(z, tt, inverse=False)
                return out
            
            # Tangents wrt (z, t):
            # v_z = dz/dt = a'_t z0 + b'_t z1
            # v_t = dt/dt = 1
            v_z = adot_t * z0 + bdot_t * z1
            v_t = torch.ones_like(t_expanded)
            
            # Single JVP over (z, t) gives full d/dt[T_t(z_t)]
            _, ydot = jvp(F, (z_interp, t_expanded), (v_z, v_t))
            return ydot
            
        flow_model.It = It_method.__get__(flow_model, flow_model.__class__)
        flow_model.dtIt = dtIt_method.__get__(flow_model, flow_model.__class__)
        
        return flow_model.It, flow_model.dtIt, (a, adot, b, bdot), flow_model # early return
    elif path == 'custom':
        return None, None, None

    else:
        raise NotImplementedError("The interpolant you specified is not implemented.")
    return It, dtIt, (a, adot, b, bdot)
    
    


def make_gamma(gamma_type = 'brownian', aval = None):
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    if gamma_type == 'brownian':
        gamma = lambda t: torch.sqrt(t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
        gg_dot = lambda t: (1/2)*(1-2*t)
        
    elif gamma_type == 'a-brownian':
        gamma = lambda t: torch.sqrt(a*t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(a*t*(1-t)))) * a*(1 -2*t)
        gg_dot = lambda t: (a/2)*(1-2*t)
        
    elif gamma_type == 'zero':
        gamma = gamma_dot = gg_dot = lambda t: torch.zeros_like(t)

    elif gamma_type == 'bsquared':
        gamma = lambda t: t*(1-t)
        gamma_dot = lambda t: 1 -2*t
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sinesquared':
        gamma = lambda t: torch.sin(math.pi * t)**2
        gamma_dot = lambda t: 2*math.pi*torch.sin(math.pi * t)*torch.cos(math.pi*t)
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sigmoid':
        f = torch.tensor(10.0)
        gamma = lambda t: torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f*(t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
        gamma_dot = lambda t: (-f)*( 1 - torch.sigmoid(-1 + f*(t - (1/2))) )*torch.sigmoid(-1 + f*(t - (1/2)))  + f*(1 - torch.sigmoid(1 + f*(t - (1/2)))  )*torch.sigmoid(1 + f*(t - (1/2)))
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == None:
        gamma     = lambda t: torch.zeros(1) ### no gamma
        gamma_dot = lambda t: torch.zeros(1) ### no gamma
        gg_dot    = lambda t: torch.zeros(1) ### no gamma
        
    else:
        raise NotImplementedError("The gamma you specified is not implemented.")
        
                
    return gamma, gamma_dot, gg_dot



def make_activation(act):
    if act == 'elu':
        return torch.nn.ELU()
    if act == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act == 'elu':
        return torch.nn.ELU()
    elif act == 'relu':
        return torch.nn.ReLU()
    elif act == 'tanh':
        return torch.nn.Tanh()
    elif act =='sigmoid':
        return torch.nn.Sigmoid()
    elif act == 'softplus':
        return torch.nn.Softplus()
    elif act == 'silu':
        return torch.nn.SiLU()
    elif act == 'Sigmoid2Pi':
        class Sigmoid2Pi(torch.nn.Sigmoid):
            def forward(self, input):
                return 2*np.pi*super().forward(input) - np.pi
        return Sigmoid2Pi()
    elif act == 'none' or act is None:
        return None
    else:
        raise NotImplementedError(f'Unknown activation function {act}')
