# Chat Session: Stochastic Interpolants Analysis

**Date:** 2025-09-24
**Topic:** Analysis of diagonal scaling issues in checker-nonlinear-2D.ipynb

## User Question

The user is working on stochastic interpolants and has added `diagonal_scale` to the `b(t)` function to handle different scales at different dimensions of target_rect density in checker-nonlinear-2D.ipynb. The goal is to transform rectangular density into square checkerboard, but the current approach with predetermined diagonal_scale is not working well.

## Problem Analysis

After examining the code in `interflow/fabrics.py` and the notebook context, the core issue is:

### Current Approach Problems

1. **Mismatched geometry**: The rectangular checkerboard target has different aspect ratios in X and Y directions, but the current diagonal scaling only changes interpolation speed, not the geometry transformation needed.

2. **Wrong transformation approach**: Trying to make rectangles become squares by changing interpolation timing, but what's really needed is a geometric transformation that maps rectangles to squares.

3. **Limited scope of diagonal scaling**: The current `diagonal_scale` only affects the `b(t)` function (how fast each dimension interpolates), but doesn't address the fundamental geometric mismatch between the source (Gaussian) and target (rectangular checker) distributions.

### Code Context

The current implementation in `fabrics.py` (lines 98-162) modifies the `one-sided-linear` path:

```python
def b(t):
    # Different dimensions can reach 1 at different rates
    # Use a power function: b_i(t) = t^(1/scale_i)
    if isinstance(t, torch.Tensor):
        t = t.view(-1, 1) if t.dim() > 0 else t
    return torch.pow(t, 1.0 / diagonal_scale)
```

This changes **timing** but not **geometry**.

## Recommended Solutions

### Option 1: Geometric Preprocessing (Recommended)
Instead of modifying `make_It`, preprocess the target density:

```python
def transform_target_to_square_checker(target_rect_fn):
    """Transform rectangular checker to square checker"""
    def square_checker_target(bs):
        samples = target_rect_fn(bs)  # Get rectangular samples
        # Apply geometric transform: stretch Y, compress X
        transform_matrix = torch.tensor([[0.4, 0.0],   # compress X by 0.4
                                       [0.0, 2.5]])    # stretch Y by 2.5
        return samples @ transform_matrix.T
    return square_checker_target
```

### Option 2: Learned Transformation Flow
Use a separate normalizing flow to transform rectangular checker → square checker before applying the main interpolant.

### Option 3: Anisotropic Base Distribution
Modify the base distribution to better match the target geometry:

```python
base_cov = torch.tensor([[9.0, 0.0],    # wider in X
                        [0.0, 1.0]])     # narrower in Y
base = prior.MultivariateNormal(base_loc, base_cov)
```

### Option 4: Nonlinear Interpolant with Geometric Awareness
If using the nonlinear interpolant, modify the flow architecture to be aware of the anisotropic geometry.

## Key Insight

The fundamental insight is that **diagonal scaling changes timing but not geometry**. The problem requires a transformation that addresses the geometric mismatch between distributions, not just different interpolation speeds per dimension.

## Next Steps

1. Try Option 1 first - it's simplest and most direct
2. Analyze the target geometry - compute actual aspect ratios of rectangular vs square checkers
3. If preprocessing works, keep using the simpler linear interpolant
4. If not, consider the learned transformation approach

## Files Involved

- `/home/wang6559/Desktop/stochastic-interpolants/notebooks/checker-nonlinear-2D.ipynb`
- `/home/wang6559/Desktop/stochastic-interpolants/interflow/fabrics.py` (lines 53-359, specifically 98-162)
- `/home/wang6559/Desktop/stochastic-interpolants/interflow/stochastic_interpolant.py`

The user is currently using a `WarmupInterpolant` that combines nonlinear and one-sided-linear interpolants with diagonal scaling `[0.1, 1.0]`.

Read the chat_session_stochastic_interpolants to continue our chat first.

● I'll read the chat session file to understand the context of our previous conversation.

● Read(chat_session_stochastic_interpolants.md)
  ⎿  Read 86 lines (ctrl+o to expand)

● I can see from the chat session that you're working on stochastic interpolants with diagonal scaling issues. The problem is that the current diagonal scaling approach only
   changes interpolation timing but not the geometry transformation needed to map rectangular density to square checkerboard.

  Let me check the current state of your notebook to see what progress has been made since our last conversation.

● Read(notebooks/checker-nonlinear-2D.ipynb)
  ⎿  Read 23 cells

● Let me check the current state of the code to see what improvements have been made since our last conversation:

● Let me look at the recent changes in the fabrics.py file to understand the diagonal scaling implementation:

● Read(interflow/fabrics.py)
  ⎿  Read 80 lines (ctrl+o to expand)

● Now let me check the visualization results in the notebook to see what's happening:

● Bash(jupyter nbconvert --to script notebooks/checker-nonlinear-2D.ipynb --stdout 2>/dev/null | grep -A5 "Phase 1: Solve SBP" | head -15)
  ⎿  (No content)

● Based on my analysis, I can see that:

  1. Current diagonal scaling approach: You're using diagonal_scale = [0.1, 1.0] which makes the X-dimension interpolate much slower (power of 10) than Y. This is
  implemented through b(t) = t^(1/scale) in the one-sided-linear path in fabrics.py:133.
  2. The problem: The diagonal scaling only changes timing (how fast each dimension interpolates) but doesn't address the geometric mismatch between rectangular checkerboard
   and square checkerboard.
  3. Your current setup: Using a WarmupInterpolant that combines nonlinear and one-sided-linear interpolants, with the rectangular target (target_rect) that has different
  aspect ratios.

● Let me implement the geometric preprocessing solution that was recommended in the chat session. I'll add it to the notebook right after the target definitions:

● Read(notebooks/checker-nonlinear-2D.ipynb)
  ⎿  Read 23 cells

● Now let me add the geometric preprocessing solution to transform the rectangular checkerboard to a square checkerboard. I'll insert this right after the target
  definitions:
  ⎿  Interrupted · What should Claude do instead?

> Forget about warmup. We use warmup for nonlinear but for rectangular we don't need that. The real issue is that the linear interpolant works good for target but bad for 
  target_rect.

● You're absolutely right! The core issue is the geometric mismatch. The linear interpolant works well for the square checkerboard (target) but fails for rectangular
  (target_rect) because:

  1. Square target: X and Y dimensions have similar ranges (both ~[-4, 4]), so uniform linear interpolation works fine
  2. Rectangular target: X spans [-10, 10] while Y spans [-4, 4] - a 2.5:1 aspect ratio mismatch

  The diagonal scaling approach won't fix this because it only changes timing, not geometry. Let me implement the geometric preprocessing solution that transforms
  rectangular samples to square:

● Read(notebooks/checker-nonlinear-2D.ipynb)
  ⎿  Read 23 cells

● Now let me add the geometric preprocessing solution. I'll insert a new cell after the target definitions that creates a transformed target function:

● Edit Notebook(notebooks/checker-nonlinear-2D.ipynb@745b2f2e-5100-4947-80f1-a8d3ba113b47)
  ⎿  User rejected insert cell in notebooks/checker-nonlinear-2D.ipynb at cell 745b2f2e-5100-4947-80f1-a8d3ba113b47

     # Geometric preprocessing: Transform rectangular checkerboard to square
     def transform_rect_to_square(target_rect_fn, 
                                  x_compress=0.4,  # Compress X by factor of 0.4 (from 20 to 8)
                                  y_stretch=2.0,   # Stretch Y by factor of 2.0 (from 4 to 8)
                                  device=None):
         """
         Transform rectangular checker samples to square checker samples.
         
         The rectangular checker has X in [-10, 10] and Y in [-4, 4] (aspect ratio 2.5:1).
         We transform it to approximately square by:
         - Compressing X dimension: X_new = X * 0.4 → [-4, 4]
         - Stretching Y dimension: Y_new = Y * 2.0 → [-8, 8]
         
         This makes both dimensions have similar ranges, which works well with linear interpolant.
         """
         if device is None:
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         def square_checker_target(bs):
             # Get rectangular samples
             samples = target_rect_fn(bs)
             
             # Apply geometric transformation
             transform_matrix = torch.tensor([[x_compress, 0.0],     # compress X 
                                              [0.0, y_stretch]],      # stretch Y
                                             device=device, dtype=torch.float32)
             
             # Transform samples to square geometry
             transformed = samples @ transform_matrix.T
             return transformed
         
         return square_checker_target

     # Create the transformed target
     target_square = transform_rect_to_square(target_rect, x_compress=0.4, y_stretch=2.0, device=device)

     # Visualize both rectangular and square targets
     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

     # Original rectangular target
     rect_samples = grab(target_rect(10000))
     axes[0].hist2d(rect_samples[:, 0], rect_samples[:, 1], bins=100, range=[[-10, 10], [-10, 10]])
     axes[0].set_title("Original Rectangular Checker\nX:[-10,10], Y:[-4,4]")
     axes[0].set_xlabel("X")
     axes[0].set_ylabel("Y")
     axes[0].set_aspect('equal')

     # Transformed square target
     square_samples = grab(target_square(10000))
     axes[1].hist2d(square_samples[:, 0], square_samples[:, 1], bins=100, range=[[-10, 10], [-10, 10]])
     axes[1].set_title("Transformed Square Checker\nX:[-4,4], Y:[-8,8]")
     axes[1].set_xlabel("X")
     axes[1].set_ylabel("Y")
     axes[1].set_aspect('equal')

     # Compare with original square target for reference
     orig_square_samples = grab(target(10000))
     axes[2].hist2d(orig_square_samples[:, 0], orig_square_samples[:, 1], bins=100, range=[[-10, 10], [-10, 10]])
     axes[2].set_title("Original Square Target\n(for reference)")
     axes[2].set_xlabel("X")
     axes[2].set_ylabel("Y")
     axes[2].set_aspect('equal')

     plt.tight_layout()
     plt.show()

     print(f"Rectangular target range: X∈[{rect_samples[:,0].min():.1f}, {rect_samples[:,0].max():.1f}], Y∈[{rect_samples[:,1].min():.1f}, {rect_samples[:,1].max():.1f}]")
     print(f"Square target range: X∈[{square_samples[:,0].min():.1f}, {square_samples[:,0].max():.1f}], Y∈[{square_samples[:,1].min():.1f}, {square_samples[:,1].max():.1f}]")
     print(f"Aspect ratio - Rectangular: {(rect_samples[:,0].max()-rect_samples[:,0].min())/(rect_samples[:,1].max()-rect_samples[:,1].min()):.2f}:1")
     print(f"Aspect ratio - Square: {(square_samples[:,0].max()-square_samples[:,0].min())/(square_samples[:,1].max()-square_samples[:,1].min()):.2f}:1")

> In real practice, you never know the shape of target, so you can't do this. You need to suggest how we can design the stochastic interpolant to solve this issue. Give me 
  some thoughts first

● You're absolutely right - in practice, we don't know the target geometry ahead of time. Here are several approaches to design stochastic interpolants that can handle
  arbitrary target geometries:

  1. Adaptive/Learnable Interpolant Path

  Instead of fixed linear interpolation I_t = (1-t)x_0 + t·x_1, learn the interpolation path:
  - Learnable time warping: I_t = (1-β(t))x_0 + β(t)·x_1 where β(t) is a learned monotonic function
  - Dimension-wise adaptive rates: Learn different β_i(t) for each dimension based on data
  - Neural ODE interpolant: Let the interpolant itself be parameterized by a neural network that learns the optimal path

  2. Anisotropic Diffusion Interpolant

  Instead of isotropic interpolation, use an adaptive covariance matrix:
  - Learn a time-dependent covariance Σ(t) that captures the target's geometry
  - The interpolant becomes: I_t = (1-t)x_0 + t·x_1 + √(t(1-t))·Σ(t)^{1/2}·ε
  - Σ(t) can be learned from early samples to capture aspect ratios

  3. Multi-Scale Hierarchical Interpolant

  Different dimensions might need different "speeds":
  - Decompose the interpolation into multiple scales/frequencies
  - Fast-moving components handle large-scale geometry changes
  - Slow-moving components handle fine details
  - Similar to wavelet decomposition but learnable

  4. Geometry-Aware Flow Matching

  The key insight: the flow model T_t should compensate for geometric mismatches:
  - If linear interpolant creates distortions, T_t learns to correct them
  - Instead of fighting this, explicitly parameterize T_t to handle affine transformations
  - Add a learnable affine layer: T_t(x) = A(t)·x + f(x,t) where A(t) is a learnable matrix

  5. Optimal Transport-Inspired Interpolant

  Use ideas from optimal transport to find better interpolation paths:
  - Approximate the Wasserstein geodesic between source and target
  - Use Monge map estimation to find the optimal coupling
  - The interpolant follows the transport plan rather than straight lines

  6. Attention-Based Adaptive Interpolant

  Use attention mechanisms to learn which dimensions need special treatment:
  - Self-attention on dimensions to discover correlations
  - Cross-attention between x_0 and x_1 to learn optimal coupling
  - Time-conditioned attention weights that adapt the interpolation path

  Practical Recommendation

  For your rectangular checkerboard problem, I'd suggest starting with:

  1. Learnable diagonal scaling in T_t: Instead of fixing diagonal_scale in the interpolant, make it part of the flow model that's learned adversarially
  2. Adaptive base distribution: Learn the covariance of the base distribution jointly with the interpolant
  3. Time-dependent affine correction: Add a simple learnable affine transformation A(t) to the flow model output

  The key principle: Let the model discover the geometry mismatch and correct for it, rather than pre-specifying the correction.

> I have few thoughts: (1)The current learnable nonlinear stochastic interpolant is based on learnable T_t. But it currently also doesn't work for this type of density. I 
  don't want to change the structure of it for now so do you have some ideas on tuning the hyperparameters of it? (2) suggestion 5 is already realized by inner-outer max-min 
  problem. (3) For high-dimensional images, I think it is hard to use nonlinear SI because there is jvp module and complex structure of T_t. Can you theoretically inspect the 
  architecture and tell me whether we can use it? (4) If we can't use nonlinear SI, my though is to apply simpler T_t/just learnable linear interpolant with matrix 
  coefficients. Consider my thoughts 1 and 2 thoroughly first, then we go to next two thoughts.

● Let me address your thoughts systematically:

  (1) Hyperparameter Tuning for Nonlinear SI on Rectangular Target

  The current nonlinear SI fails because the T_t network isn't expressive enough to handle the anisotropic geometry. Here are hyperparameter adjustments:

  Key Issues & Solutions:

  a) Frequency representation:
  - Current: fourier_max_freq: 10.0 - too low for sharp rectangular boundaries
  - Suggestion: Increase to 30-50 for sharper feature capture
  - Add more Fourier features (128-256) to handle different scales per dimension

  b) Network capacity:
  - Current: num_layers: 2, hidden: 128 - insufficient for geometric transformation
  - Suggestion:
    - Increase num_layers: 4-6
    - Increase hidden: 256-512
    - More mlp_blocks: 3-4

  c) Training dynamics:
  - Key insight: The adversarial training might be unstable with geometric mismatch
  - Suggestions:
    - Reduce learning rate ratio: lr_flow = 1e-4 (instead of 2e-4)
    - Increase n_outer: 10-20 to give T_t more updates
    - Use gradient clipping: clip_flow: 0.1-0.5
    - Add warmup: Start with n_outer: 1 and gradually increase

  d) Spectral normalization:
  - Current: vector_use_spectral_norm: True - might be too restrictive
  - Try: Setting to False or using lighter regularization
  - vector_log_scale_clamp: 2.0-3.0 (instead of 1.0) for more flexibility

  e) Architecture modifications without changing structure:
  flow_config = {
      "num_layers": 4,          # More layers
      "time_embed_dim": 256,    # Richer time encoding
      "hidden": 512,            # Wider network
      "mlp_blocks": 3,          # Deeper blocks
      "activation": "gelu",
      "use_layernorm": True,    # Keep for stability
      "use_permutation": True,
      "fourier_min_freq": 0.1,  # Lower min for large-scale
      "fourier_max_freq": 50.0, # Higher max for sharp edges
      "vector_use_spectral_norm": False,  # Less restriction
      "vector_log_scale_clamp": 3.0,      # More flexibility
      "vector_use_soft_clamp": True,
  }

  (2) Why Current Nonlinear SI Struggles with Rectangular Geometry

● The current T_t learns a diffeomorphism but:

  1. Scale mismatch: T_t must simultaneously learn:
    - Small corrections for Y dimension (4 units range)
    - Large corrections for X dimension (20 units range)
    - Single network struggles with this 5:1 scale difference
  2. Fourier basis limitation: The Fourier features encode position globally, but rectangular checker needs different frequencies for X vs Y
  3. Adversarial training issue: The min-max optimization finds the Wasserstein geodesic, BUT:
    - The geodesic assumes T_t can represent any diffeomorphism
    - In practice, T_t is constrained by network capacity
    - The rectangular geometry needs a "shearing" transformation that's hard for isotropic networks

  Solution approach: Add dimension-specific processing in T_t:
  - Use separate sub-networks for X and Y initially
  - Combine with cross-dimension interactions later
  - Or use positional encoding scales that adapt per dimension

  (3) Computational Feasibility for High-Dimensional Images

  Let me analyze the computational complexity:

  Current implementation uses JVP (Jacobian-vector products):
  - For dimension d, each JVP costs O(d) forward passes
  - Total cost: O(d²) for full Jacobian
  - For images (d=3×256×256 ≈ 200k), this is prohibitive

  Memory requirements:
  - Storing intermediate activations for backprop through JVP
  - For high-dim, this explodes quickly

  Theoretical assessment:
  - Nonlinear SI is NOT feasible for raw high-dim images due to JVP computation
  - Need either:
    a. Work in latent space (encode to ~512-2048 dims first)
    b. Use Hutchinson trace estimator (stochastic approximation of trace)
    c. Switch to linear/affine interpolants that avoid JVP

  (4) Simpler Alternatives for High-Dim

  For images, consider:

  1. Learnable Linear Interpolant with Matrix Coefficients:
  I_t(x0, x1) = A(t)·x0 + B(t)·x1
  1. where A(t), B(t) are small neural networks outputting diagonal or low-rank matrices
  2. Block-diagonal approach: Process image in patches, avoiding full JVP
  3. Latent space interpolation: Use VAE/encoder, interpolate in latent space

  Recommended approach for rectangles: Try the enhanced hyperparameters first, focusing on:
  - Higher Fourier frequencies
  - Deeper network
  - Asymmetric learning rates
  - Longer training with more outer steps

> Can you speak more on the possible architecture for learnable linear interpolants? As concrete  as possible. Use mathematics to clearly define your models

● Learnable Linear Interpolant Architectures

  Let me define several concrete mathematical models for learnable linear interpolants:

  1. Time-Dependent Affine Interpolant

  Mathematical Definition:
  I_t(x₀, x₁) = A(t)x₀ + B(t)x₁

  where A(t), B(t) ∈ ℝ^(d×d) satisfy:
  - Boundary conditions: A(0) = I, B(0) = 0, A(1) = 0, B(1) = I
  - Constraint: A(t) + B(t) = I (ensures interpolation)

  Parameterization:
  A(t) = (1 - α(t)) · I
  B(t) = α(t) · I

  where α: [0,1] → [0,1]^d is a learnable monotonic function

  Neural Network Architecture:
  class AlphaNetwork(nn.Module):
      def __init__(self, d, hidden=128):
          self.net = nn.Sequential(
              nn.Linear(1, hidden),
              nn.SiLU(),
              nn.Linear(hidden, hidden),
              nn.SiLU(),
              nn.Linear(hidden, d)
          )

      def forward(self, t):
          # Ensure monotonicity and boundaries
          raw = self.net(t)
          alpha = torch.sigmoid(raw) * t.unsqueeze(-1)  # Ensures α(0)=0, α(1)≤1
          return alpha

  Derivative for training:
  dI_t/dt = -α'(t) ⊙ x₀ + α'(t) ⊙ x₁ = α'(t) ⊙ (x₁ - x₀)

  2. Low-Rank Time-Dependent Interpolant

  Mathematical Definition:
  I_t(x₀, x₁) = x₀ + t(x₁ - x₀) + U(t)V(t)ᵀ(x₁ - x₀)

  where:
  - U(t) ∈ ℝ^(d×r), V(t) ∈ ℝ^(d×r), r ≪ d (rank)
  - Boundary: U(0)V(0)ᵀ = 0, U(1)V(1)ᵀ = 0

  Parameterization:
  class LowRankInterpolant(nn.Module):
      def __init__(self, d, rank=32, hidden=128):
          self.d = d
          self.rank = rank
          self.u_net = nn.Sequential(
              nn.Linear(1, hidden),
              nn.SiLU(),
              nn.Linear(hidden, d * rank)
          )
          self.v_net = nn.Sequential(
              nn.Linear(1, hidden),
              nn.SiLU(),
              nn.Linear(hidden, d * rank)
          )

      def forward(self, t, x0, x1):
          u = self.u_net(t).view(-1, self.d, self.rank)
          v = self.v_net(t).view(-1, self.d, self.rank)

          # Apply boundary condition
          gate = t * (1 - t) * 4  # Peak at t=0.5
          u = u * gate.unsqueeze(-1).unsqueeze(-1)

          diff = x1 - x0
          correction = torch.bmm(u, v.transpose(-1, -2))
          correction = torch.bmm(correction, diff.unsqueeze(-1)).squeeze(-1)

          return x0 + t * diff + correction

  3. Diagonal + Residual Interpolant

  Mathematical Definition:
  I_t(x₀, x₁) = (1-D(t))⊙x₀ + D(t)⊙x₁ + R(t,x₀,x₁)

  where:
  - D(t) ∈ ℝ^d is a diagonal scaling (per-dimension rates)
  - R(t,x₀,x₁) is a small residual correction

  Concrete Form:
  D_i(t) = t^(β_i), where β_i > 0 are learnable exponents
  R(t,x₀,x₁) = γ(t) · MLP(concat[x₀, x₁, t])
  γ(t) = t(1-t) ensures R vanishes at boundaries

  Implementation:
  class DiagonalResidualInterpolant(nn.Module):
      def __init__(self, d, hidden=256):
          self.d = d
          self.log_beta = nn.Parameter(torch.zeros(d))  # Learnable exponents
          self.residual = nn.Sequential(
              nn.Linear(2*d + 1, hidden),
              nn.SiLU(),
              nn.Linear(hidden, hidden),
              nn.SiLU(),
              nn.Linear(hidden, d)
          )

      def forward(self, t, x0, x1):
          # Diagonal interpolation with per-dim rates
          beta = torch.exp(self.log_beta).clamp(0.1, 10.0)
          D = t.unsqueeze(-1).pow(beta)

          # Main interpolation
          interpolated = (1 - D) * x0 + D * x1

          # Residual correction
          gamma = 4 * t * (1 - t)  # Boundary condition
          features = torch.cat([x0, x1, t.expand_as(x0[..., :1])], dim=-1)
          residual = gamma * self.residual(features)

          return interpolated + residual

  4. Block-Diagonal Learnable Interpolant (for High-Dim)

  Mathematical Definition:
  For high-dimensional data, partition dimensions into blocks:
  x = [x^(1), x^(2), ..., x^(B)]

  Each block has its own interpolation matrix:
  I_t(x₀, x₁) = ⊕ᵢ₌₁^B [A^(i)(t)x₀^(i) + B^(i)(t)x₁^(i)]

  For Images Specifically:
  class ImageBlockInterpolant(nn.Module):
      def __init__(self, channels=3, block_size=16, hidden=64):
          self.channels = channels
          self.block_size = block_size

          # Per-block interpolation networks
          self.block_nets = nn.ModuleList([
              nn.Sequential(
                  nn.Linear(1, hidden),
                  nn.SiLU(),
                  nn.Linear(hidden, 2 * block_size * block_size)
              ) for _ in range(channels)
          ])

      def forward(self, t, x0, x1):
          # x0, x1 shape: [batch, channels, height, width]
          b, c, h, w = x0.shape
          output = []

          for ch in range(c):
              # Process each channel
              x0_ch = x0[:, ch].view(b, -1, self.block_size, self.block_size)
              x1_ch = x1[:, ch].view(b, -1, self.block_size, self.block_size)

              # Get interpolation weights for this channel
              weights = self.block_nets[ch](t).view(b, 2, -1, 1, 1)
              alpha, beta = weights[:, 0], weights[:, 1]

              # Normalize to ensure alpha + beta = 1
              total = alpha + beta + 1e-8
              alpha, beta = alpha / total, beta / total

              interpolated = alpha * x0_ch + beta * x1_ch
              output.append(interpolated.view(b, h, w))

          return torch.stack(output, dim=1)

  5. Optimal Transport-Inspired Linear Interpolant

  Mathematical Definition:
  Learn a transport matrix M(t) that approximates Monge map:
  I_t(x₀, x₁) = x₀ + t · T(x₀, x₁)
  T(x₀, x₁) = M(θ) · (x₁ - x₀)

  where M(θ) is constrained to be:
  - Doubly stochastic (for measure preservation)
  - Or orthogonal (for distance preservation)

  Parameterization via Sinkhorn:
  class OptimalTransportInterpolant(nn.Module):
      def __init__(self, d, n_iter=10):
          self.d = d
          self.n_iter = n_iter
          # Learn log of transport cost matrix
          self.log_C = nn.Parameter(torch.randn(d, d) * 0.01)

      def sinkhorn(self, log_C, n_iter=10):
          # Sinkhorn algorithm to get doubly stochastic matrix
          K = torch.exp(-log_C)
          for _ in range(n_iter):
              K = K / K.sum(dim=1, keepdim=True)
              K = K / K.sum(dim=0, keepdim=True)
          return K

      def forward(self, t, x0, x1):
          M = self.sinkhorn(self.log_C, self.n_iter)
          transport = torch.matmul(M, (x1 - x0).unsqueeze(-1)).squeeze(-1)
          return x0 + t * transport

  Computational Complexity Comparison

  | Model             | Forward Pass | Backward Pass | Memory | Best For                |
  |-------------------|--------------|---------------|--------|-------------------------|
  | Nonlinear SI      | O(d²)        | O(d²)         | O(d²)  | Low-dim (d<100)         |
  | Diagonal+Residual | O(d·h)       | O(d·h)        | O(d·h) | Medium-dim              |
  | Low-Rank          | O(d·r²)      | O(d·r²)       | O(d·r) | High-dim with structure |
  | Block-Diagonal    | O(d·b)       | O(d·b)        | O(d)   | Images                  |
  | OT-Linear         | O(d²)        | O(d²)         | O(d²)  | Known geometry          |

  For rectangular checker problem, I recommend starting with Diagonal+Residual as it can learn different interpolation rates per dimension while maintaining computational
  efficiency.

> Why increase the fourier_max_freq can improve sharpeness?

● Great question! Let me explain why increasing fourier_max_freq improves sharpness, especially for rectangular checkerboard patterns.

  Fourier Features and Sharp Boundaries

  1. Fourier Basis Representation

  When using Fourier features, the neural network first maps input coordinates to:
  γ(x) = [sin(2πf₁x), cos(2πf₁x), ..., sin(2πfₖx), cos(2πfₖx)]
  where f₁, ..., fₖ are frequencies sampled between fourier_min_freq and fourier_max_freq.

  2. Sharp Edges Need High Frequencies

  According to Fourier analysis:
  - Smooth functions = mostly low frequencies
  - Sharp edges/discontinuities = need high frequencies

  For a rectangular checkerboard:
  - The boundaries are step functions (discontinuous jumps from black to white tiles)
  - Step functions have Fourier decomposition: f(x) = Σ(1/n)sin(nx) for odd n
  - The series needs terms up to very high n to approximate sharp edges

  3. Mathematical Intuition

  Consider a 1D step function at x=0:
  Step function: f(x) = {0 if x<0, 1 if x>0}
  Fourier series: f(x) ≈ 1/2 + (2/π)Σ(1/n)sin(nx) for odd n

  With only low frequencies (n ≤ 10):
  - The approximation is smooth and wavy (Gibbs phenomenon)
  - The transition takes ~π/n_max width

  With high frequencies (n ≤ 50):
  - The transition becomes much sharper
  - Width of transition ~ π/50 (5x sharper!)

  4. For Rectangular Checkerboard Specifically

  The rectangular checker has:
  - Sharp boundaries between tiles
  - Different spatial scales: X-dimension tiles are wider than Y-dimension tiles

  With fourier_max_freq=10:
  Smallest feature size capturable ≈ π/10 ≈ 0.3 units

  With fourier_max_freq=50:
  Smallest feature size capturable ≈ π/50 ≈ 0.06 units

  Since your rectangular tiles have height h=0.1, you need frequencies high enough to resolve this scale!

  5. Visual Example

  # Low frequency (smooth approximation):
  sin(2πx) + sin(4πx) + sin(6πx)  → Blurry edges

  # High frequency (sharp approximation):  
  sin(2πx) + ... + sin(100πx)     → Sharp edges

  6. Why This Matters for T_t

  The flow model T_t needs to:
  1. Detect the sharp boundaries (requires high-freq input features)
  2. Correct geometric distortions near edges
  3. Preserve the sharpness during interpolation

  Without high frequencies, T_t sees a "blurred" version of the checker pattern and cannot properly learn the necessary corrections.

  7. Practical Trade-off

  - Too low (e.g., 10): Cannot represent sharp features
  - Too high (e.g., 1000): May overfit to noise, harder to optimize
  - Sweet spot for 2D checker: 30-50 gives good sharpness without overfitting

  This is why increasing fourier_max_freq from 10 to 30-50 should significantly improve the model's ability to handle the sharp rectangular boundaries in your checkerboard
  pattern.

> But I think this is for time embedding

● You're absolutely right! I apologize for the confusion. The fourier_max_freq in your flow_config is for time embedding, not spatial features. Let me correct my analysis:

  Time Embedding Fourier Features

  What the Config Actually Does:

  flow_config = {
      "fourier_min_freq": 1.0,
      "fourier_max_freq": 10.0,  # This is for TIME encoding, not spatial!
      ...
  }

  This creates a time embedding:
  γ(t) = [sin(2πf₁t), cos(2πf₁t), ..., sin(2πfₖt), cos(2πfₖt)]
  where t ∈ [0,1] is the interpolation time.

  Why Higher Time Frequencies Matter for Rectangular Geometry:

  1. Different Correction Schedules:
    - The rectangular checker needs different corrections at different times
    - X-dimension (20 units span) needs large corrections early
    - Y-dimension (4 units span) needs smaller, steadier corrections
    - T_t must learn time-varying behavior that differs per dimension
  2. Rapid Transitions in Optimal Transport Path:
    - The optimal transport from Gaussian to rectangular checker isn't smooth in time
    - There might be rapid changes at certain t values where the geometry "snaps" into place
    - Low time frequencies → T_t can only learn smooth time evolution
    - High time frequencies → T_t can learn sharp temporal transitions
  3. Example Scenario:
  At t=0.3: T_t needs to start stretching X direction
  At t=0.5: T_t needs to quickly adjust Y direction
  At t=0.7: T_t needs to sharpen boundaries
  3. These rapid changes in behavior require high-frequency time components.

  The Real Issue with Rectangular Target:

  The actual problem isn't about representing sharp spatial features - it's that:

  1. T_t needs different time-dependent strategies for X vs Y dimensions
  2. Single time embedding is shared across all dimensions
  3. Low frequency time embedding (max_freq=10) means T_t can only learn slowly-varying corrections

  Better Solutions:

  Instead of just increasing fourier_max_freq, consider:

  1. Dimension-specific time processing:
  # In your T_t network, process time differently per dimension
  time_embed_x = fourier_features(t, max_freq=50)  # Needs rapid changes
  time_embed_y = fourier_features(t, max_freq=20)  # Smoother evolution
  2. Learnable frequency selection:
  # Let the network learn which frequencies it needs
  self.freq_weights = nn.Parameter(torch.ones(n_frequencies))
  weighted_features = fourier_features(t) * self.freq_weights
  3. Multi-scale time embedding:
  # Combine multiple frequency ranges
  low_freq = fourier_features(t, max_freq=10)   # Global evolution
  high_freq = fourier_features(t, max_freq=50)  # Fine corrections
  time_embed = concat([low_freq, high_freq])

  So you're correct - the fourier_max_freq is about temporal resolution of how T_t changes over time, not spatial sharpness. The rectangular problem needs T_t to make rapid
  time-dependent adjustments, which is why higher time frequencies might help, but for different reasons than I initially stated.

> there are min and max freq, do i need to change min freq?

● Good question! The fourier_min_freq controls the slowest/smoothest time variations that T_t can learn. Here's when and why you might want to adjust it:

  Current Setting:

  "fourier_min_freq": 1.0,   # Slowest oscillation: 1 cycle over [0,1]
  "fourier_max_freq": 10.0,   # Fastest oscillation: 10 cycles over [0,1]

  What Min Frequency Controls:

  1. Global, smooth evolution:
    - min_freq=1.0 → sin(2πt) completes exactly 1 cycle from t=0 to t=1
    - This captures the overall "arc" of the interpolation
  2. Lower min_freq (e.g., 0.1-0.5):
    - Allows even smoother, more global changes
    - sin(0.2πt) → only 1/5 of a cycle over [0,1]
    - Good for: Gentle, monotonic corrections
  3. Higher min_freq (e.g., 2-5):
    - Removes very smooth components
    - Forces all corrections to have some oscillation
    - Usually bad: You lose the ability to learn simple linear-like corrections

  For Rectangular Checkerboard:

  Keep min_freq low (0.5-1.0) because:

  1. Need smooth base trajectory: The overall transformation from Gaussian to rectangular should be smooth
  2. Dimension-specific rates: X and Y need different but steady progression rates
  3. Avoid oscillatory artifacts: High min_freq might force unnecessary wiggling

  Recommended settings:
  "fourier_min_freq": 0.5,    # Allow smoother global evolution
  "fourier_max_freq": 30.0,   # Allow sharp temporal corrections

  Why This Combination Works:

  - Low frequencies (0.5-2): Handle the smooth, monotonic "transport" from source to target
  - Mid frequencies (2-10): Adjust for geometric differences between dimensions
  - High frequencies (10-30): Make rapid corrections near critical time points

  Quick Test:

  Try these configurations and compare:
  # Config A: Original
  "fourier_min_freq": 1.0, "fourier_max_freq": 10.0

  # Config B: Wider range (recommended)
  "fourier_min_freq": 0.5, "fourier_max_freq": 30.0

  # Config C: Only high freq (probably bad)
  "fourier_min_freq": 5.0, "fourier_max_freq": 50.0

  Config B should work best as it gives T_t maximum flexibility to learn both smooth global transport and sharp local corrections.