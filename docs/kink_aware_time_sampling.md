# Kink-Aware Time Sampling

This note records the mathematical motivation behind the kink-aware time sampler
that now powers `notebooks/checker-multivariateSI-2D.ipynb`.

## Flow Geometry

For the polynomial multivariate interpolant we study, each coordinate evolves as

\[
 x_t^{(i)} = A_i(t) x_0^{(i)} + B_i(t) x_1^{(i)}, \qquad
 A_i(t) = 1 - t^{p_i}, \qquad B_i(t) = t^{q_i},
\]

with exponents \(p_i, q_i > 0\). The first two derivatives are

\[
 \dot{x}_t^{(i)} = -p_i t^{p_i-1} x_0^{(i)} + q_i t^{q_i-1} x_1^{(i)},
 \qquad
 \ddot{x}_t^{(i)} = -p_i (p_i-1) t^{p_i-2} x_0^{(i)} + q_i (q_i-1) t^{q_i-2} x_1^{(i)}.
\]

Key observations:

- **Boundary cusps.** If \(p_i < 1\) (resp. \(q_i < 1\)) the source (resp. target)
  contribution produces an infinite slope at \(t=1\) (resp. \(t=0\)).
- **Boundary snap.** When \(1 < p_i < 2\) or \(1 < q_i < 2\), the velocity is
  finite at the boundary but the acceleration diverges, creating a sharp “snap”.
- **Interior switching.** For \(p_i \neq q_i\) and \(x_0^{(i)} x_1^{(i)} > 0\) the
  velocity component that dominates flips sign around
  \[
    t_{\text{vel}}^{(i)} = \left(\frac{p_i |x_0^{(i)}|}{q_i |x_1^{(i)}|}\right)^{1/(q_i - p_i)}.
  \]
  Before \(t_{\text{vel}}^{(i)}\) the source drives the motion, afterwards the target does.
- **Balance point.** Solving
  \( |x_0^{(i)}|(1 - t^{p_i}) = |x_1^{(i)}| t^{q_i} \) by bisection yields
  \(t_{\text{bal}}^{(i)}\), the time where magnitudes from source and target match. The
  curvature \(\kappa(t) = \| \dot{x}(t) \times \ddot{x}(t) \| / \| \dot{x}(t) \|^3\)
  typically peaks near these balance points.

The kink-aware sampler pools \(t_{\text{vel}}\) and \(t_{\text{bal}}\) over the current
mini-batch to discover where trajectories bend sharply.

## Mixture Proposal

For a batch of size \(B\) with dimension \(d\):

1. Draw \(x_0, x_1\) and compute \(t_{\text{vel}}^{(i)}\) analytically plus a bisection
   approximation of \(t_{\text{bal}}^{(i)}\). We retain times lying strictly inside
   \([\delta, 1-\delta]\) with \(\delta \approx 0.03\).
2. Form a four-component importance distribution
   \[
     q(t) = w_1 f_{\text{Beta}(\alpha,1)}(t)
           + w_2 f_{\text{Beta}(1,\beta)}(t)
           + w_3 f_{\text{Unif}}(t)
           + w_4 f_{\text{kink}}(t),
   \]
   where
   - \(f_{\text{Beta}(\alpha,1)}\) concentrates mass near \(t=1\) to resolve A(t) kinks;
   - \(f_{\text{Beta}(1,\beta)}\) concentrates mass near \(t=0\) for B(t) kinks;
   - \(f_{\text{Unif}}\) is the unit-density baseline on \([0,1]\);
   - \(f_{\text{kink}}\) is a uniform mixture of slabs of width \(2\delta\) centred on
     the discovered kink times (if no kink survives the filtering, the weight is
     reassigned to the uniform term).

   The Beta shape parameters \(\alpha = \max_i p_i\) and \(\beta = \max_i q_i\) match
   the steepest exponent along any coordinate. We empirically fix
   \((w_1, w_2, w_3, w_4) \approx (0.25, 0.25, 0.20, 0.30)\) and let the last
   component fold back into \(w_3\) when no kink times are available, ensuring
   \(\sum w_j = 1\).
3. Sampling from the kink component proceeds by choosing a kink centre at random
   and adding uniform jitter in \([-\delta, \delta]\), clamped to \([\varepsilon, 1-\varepsilon]\)
   with machine epsilon \(\varepsilon\).

## Importance Weights

Given sampled times \(t_b\) and loss values \(\ell_b\), we accumulate

\[
  \mathcal{L} = \frac{1}{B} \sum_{b=1}^B \frac{\ell_b}{q(t_b)}.
\]

The density \(q(t_b)\) is evaluated explicitly by summing the four component
pdfs:

- Beta pdfs via \(f_{\text{Beta}}(t) = \text{Beta}(t; a,b)\).
- Uniform pdf equals 1.
- Kink pdf equals the number of active slabs covering \(t_b\) divided by the total
  slab measure \(2\delta \times N_{\text{kink}}\).

Clamping ensures numerical stability and prevents division by zero.

## Practical Notes

- The sampler defaults to the Beta/Uniform triad when no matrix configuration is
  supplied or when the interpolant is non-polynomial.
- The same time/weight tensor can be reused for both the velocity and flow
  updates, keeping the outer maximisation aligned with the inner minimisation.
- The additional per-batch computation (two logarithms and a handful of bisection
  steps) is cheap compared with neural forward passes yet removes the need for
  ad-hoc learning-rate tweaks when exponents become highly anisotropic.



## Function Glossary

- `_as_float_list(values, default, dim=None)`: normalises matrix-config entries (scalars, lists, tensors) into Python float lists, optionally padding/truncating to match the dimensionality of the interpolant.
- `_compute_static_kink_summary(matrix_cfg, eps=1e-4)`: inspects the polynomial exponents, predicts boundary and interior kink landmarks assuming |x₀|≈|x₁|, and aggregates them with heuristic importance weights for logging.
- `_get_beta_parameters_from_matrix_config(matrix_cfg)`: extracts the steepest exponents across coordinates to set the Beta(α,1) / Beta(1,β) shapes that emphasise late-time or early-time regions.
- `_prepare_exponent_tensors(matrix_cfg, data_dim, device, dtype)`: converts the per-dimension exponent lists into torch tensors aligned with the current batch device/dtype so the sampler can broadcast them safely.
- `_compute_batch_kink_times(x0_batch, x1_batch, p_exponents, q_exponents, eps, max_points)`: given the actual mini-batch, solves for velocity-switch and balance times per coordinate (analytically plus short bisection), filters them into the interior, and returns a deduplicated tensor of kink candidates.
- `_sample_mixed_times_and_weights(bs, device, dtype, alpha, beta, mix_prob, keepdim, strategy, x0_batch, x1_batch, p_exponents, q_exponents, kink_delta, max_kink_points)`: draws times from the four-component mixture (two Betas, uniform baseline, kink slabs), evaluates the proposal density, and returns both samples and inverse-pdf importance weights.
- `train_step(...)`: wraps the SGD/Adam inner-outer loops, wiring the kink-aware sampler into the velocity and flow updates (with batch reuse when enabled) and keeping the existing optimisation/logging contract intact.
