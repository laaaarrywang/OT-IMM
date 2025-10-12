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
           # New parameters for multivariate (matrix-coefficient) interpolant
           matrix_config = None):
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
        # Simple scalar implementation
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

    elif path == 'multivariate':
        """
        Multivariate stochastic interpolant with matrix coefficients.

        Interpolation: x_t = A(t) @ x_0 + B(t) @ x_1

        Where A(t) and B(t) are matrices (can be diagonal or full).

        Parametrization:
        - A(t) = M_A ⊙ f_A(t)  where f_A(t) = (1-t)^p
        - B(t) = M_B ⊙ f_B(t)  where f_B(t) = t^q

        Boundary conditions:
        - A(0) = M_A, B(0) = 0
        - A(1) = 0, B(1) = M_B

        matrix_config: dict with the following keys:
          - 'matrix_A': Initial matrix M_A (default: I)
          - 'matrix_B': Initial matrix M_B (default: I)
          - 'exponent_p': Exponent(s) for A(t), scalar or per-dimension (default: 1.0)
          - 'exponent_q': Exponent(s) for B(t), scalar or per-dimension (default: 1.0)
          - 'matrix_type': 'diagonal' or 'full' (default: 'diagonal')
          - 'trainable': Whether matrices should be trainable (future feature, default: False)

        For hyperparameter tuning: Pass fixed matrices and vary exponents.
        For future training: Set trainable=True to make matrices learnable.
        """

        if data_dim is None:
            raise ValueError("data_dim must be specified for multivariate interpolant")

        if matrix_config is None:
            matrix_config = {}

        # Extract configuration
        matrix_type = matrix_config.get('matrix_type', 'diagonal')
        trainable = matrix_config.get('trainable', False)

        # Parse exponents (can be scalar or per-dimension)
        exponent_p = matrix_config.get('exponent_p', 1.0)
        exponent_q = matrix_config.get('exponent_q', 1.0)

        # Convert to tensors
        if isinstance(exponent_p, (list, tuple)):
            p_exp = torch.tensor(exponent_p, dtype=torch.float32)
        elif isinstance(exponent_p, torch.Tensor):
            p_exp = exponent_p.float()
        else:  # scalar
            p_exp = torch.tensor([exponent_p] * data_dim, dtype=torch.float32)

        if isinstance(exponent_q, (list, tuple)):
            q_exp = torch.tensor(exponent_q, dtype=torch.float32)
        elif isinstance(exponent_q, torch.Tensor):
            q_exp = exponent_q.float()
        else:  # scalar
            q_exp = torch.tensor([exponent_q] * data_dim, dtype=torch.float32)

        # Parse matrices M_A and M_B
        matrix_A = matrix_config.get('matrix_A', None)
        matrix_B = matrix_config.get('matrix_B', None)

        if matrix_type == 'diagonal':
            # Diagonal matrices represented as vectors
            if matrix_A is None:
                M_A = torch.ones(data_dim, dtype=torch.float32)  # Identity
            else:
                M_A = torch.tensor(matrix_A, dtype=torch.float32) if not isinstance(matrix_A, torch.Tensor) else matrix_A.float()
                assert M_A.shape == (data_dim,), f"matrix_A must be shape ({data_dim},) for diagonal type"

            if matrix_B is None:
                M_B = torch.ones(data_dim, dtype=torch.float32)  # Identity
            else:
                M_B = torch.tensor(matrix_B, dtype=torch.float32) if not isinstance(matrix_B, torch.Tensor) else matrix_B.float()
                assert M_B.shape == (data_dim,), f"matrix_B must be shape ({data_dim},) for diagonal type"

        elif matrix_type == 'full':
            # Full matrices
            if matrix_A is None:
                M_A = torch.eye(data_dim, dtype=torch.float32)  # Identity
            else:
                M_A = torch.tensor(matrix_A, dtype=torch.float32) if not isinstance(matrix_A, torch.Tensor) else matrix_A.float()
                assert M_A.shape == (data_dim, data_dim), f"matrix_A must be shape ({data_dim}, {data_dim}) for full type"

            if matrix_B is None:
                M_B = torch.eye(data_dim, dtype=torch.float32)  # Identity
            else:
                M_B = torch.tensor(matrix_B, dtype=torch.float32) if not isinstance(matrix_B, torch.Tensor) else matrix_B.float()
                assert M_B.shape == (data_dim, data_dim), f"matrix_B must be shape ({data_dim}, {data_dim}) for full type"
        else:
            raise ValueError(f"matrix_type must be 'diagonal' or 'full', got {matrix_type}")

        # TODO: If trainable=True, wrap as nn.Parameter (for future training support)

        # Define matrix coefficient functions
        if matrix_type == 'diagonal':
            def A_matrix(t):
                """A(t) = M_A ⊙ (1-t)^p for diagonal case"""
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)  # [1]
                # Move tensors to correct device/dtype
                p = p_exp.to(t.device, t.dtype)
                M = M_A.to(t.device, t.dtype)
                # Compute (1-t)^p element-wise, shape: [batch, dim]
                return M.unsqueeze(0) * ((1 - t).unsqueeze(-1) ** p)

            def B_matrix(t):
                """B(t) = M_B ⊙ t^q for diagonal case"""
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                q = q_exp.to(t.device, t.dtype)
                M = M_B.to(t.device, t.dtype)
                return M.unsqueeze(0) * (t.unsqueeze(-1) ** q)

            def A_matrix_dot(t):
                """dA/dt = M_A ⊙ [-p * (1-t)^(p-1)]"""
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                p = p_exp.to(t.device, t.dtype)
                M = M_A.to(t.device, t.dtype)
                eps = 1e-8  # Avoid 0^negative
                return M.unsqueeze(0) * (-p * ((1 - t + eps).unsqueeze(-1) ** (p - 1)))

            def B_matrix_dot(t):
                """dB/dt = M_B ⊙ [q * t^(q-1)]"""
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                q = q_exp.to(t.device, t.dtype)
                M = M_B.to(t.device, t.dtype)
                eps = 1e-8
                return M.unsqueeze(0) * (q * ((t + eps).unsqueeze(-1) ** (q - 1)))

            # For diagonal: use element-wise (Hadamard) product
            It   = lambda t, x0, x1: A_matrix(t) * x0 + B_matrix(t) * x1
            dtIt = lambda t, x0, x1: A_matrix_dot(t) * x0 + B_matrix_dot(t) * x1

        else:  # full matrices
            def A_matrix(t):
                """A(t) = M_A @ diag((1-t)^p) for full case"""
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                p = p_exp.to(t.device, t.dtype)
                M = M_A.to(t.device, t.dtype)
                # Diagonal scaling: (1-t)^p per dimension
                scale_diag = (1 - t).unsqueeze(-1) ** p  # [batch, dim]
                # Apply: M @ diag(scale) equivalent to M * scale (broadcasting)
                return M.unsqueeze(0) * scale_diag.unsqueeze(1)  # [batch, dim, dim]

            def B_matrix(t):
                """B(t) = M_B @ diag(t^q) for full case"""
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                q = q_exp.to(t.device, t.dtype)
                M = M_B.to(t.device, t.dtype)
                scale_diag = t.unsqueeze(-1) ** q
                return M.unsqueeze(0) * scale_diag.unsqueeze(1)

            def A_matrix_dot(t):
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                p = p_exp.to(t.device, t.dtype)
                M = M_A.to(t.device, t.dtype)
                eps = 1e-8
                scale_diag = -p * ((1 - t + eps).unsqueeze(-1) ** (p - 1))
                return M.unsqueeze(0) * scale_diag.unsqueeze(1)

            def B_matrix_dot(t):
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                q = q_exp.to(t.device, t.dtype)
                M = M_B.to(t.device, t.dtype)
                eps = 1e-8
                scale_diag = q * ((t + eps).unsqueeze(-1) ** (q - 1))
                return M.unsqueeze(0) * scale_diag.unsqueeze(1)

            # For full matrices: use matrix-vector product
            def It(t, x0, x1):
                A_t = A_matrix(t)  # [batch, dim, dim]
                B_t = B_matrix(t)
                # Matrix-vector product: einsum or matmul
                return torch.einsum('bij,bj->bi', A_t, x0) + torch.einsum('bij,bj->bi', B_t, x1)

            def dtIt(t, x0, x1):
                A_t_dot = A_matrix_dot(t)
                B_t_dot = B_matrix_dot(t)
                return torch.einsum('bij,bj->bi', A_t_dot, x0) + torch.einsum('bij,bj->bi', B_t_dot, x1)

        # For compatibility, define scalar a/b (these won't be used directly)
        a = A_matrix
        adot = A_matrix_dot
        b = B_matrix
        bdot = B_matrix_dot

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
