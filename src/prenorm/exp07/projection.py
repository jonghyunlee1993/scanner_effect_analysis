"""Scalar identity-direction composite range projection (Exp-07 plan §5.3).

Input ``x = R(l, b)`` with a low-band-corrected candidate coarse ``l'`` gives the
unsafe correction ``x + d`` where ``d = R(l' - l, 0)`` touches only the low
coefficients -- the copied detail bands ``b`` are reused unchanged.  Because the
synthesis ``R`` is linear,

    x + alpha * d = R(l + alpha*(l' - l), b),

so scaling the correction by a single per-image ``alpha`` never recomputes ``b``.
We pick the largest ``alpha in [0, 1]`` that keeps the whole composite inside
``[lo, hi]`` everywhere.  Range safety and exact high-band preservation then both
hold by construction and no output clamp is ever applied.
"""

from __future__ import annotations

import torch


def scalar_range_alpha(x: torch.Tensor, d: torch.Tensor,
                       lo: float = -1.0, hi: float = 1.0):
    """Largest per-image ``alpha`` in ``[0, 1]`` with ``x + alpha*d`` in range.

    Returns ``(alpha, alpha_raw, binding)`` where

    - ``alpha`` has shape ``[N, 1, 1, 1]`` for broadcasting,
    - ``alpha_raw`` has shape ``[N]`` and is the unclamped line-search bound
      (``alpha_raw >= 1`` means the correction is not range-binding),
    - ``binding`` is a dict of per-image diagnostics at the constraining element:
      ``abs_x`` (``|x|`` at the binding pixel) and ``abs_d`` (``|d|`` there),
      plus ``d_inf`` (``max|d|`` per image) and ``frac_extreme``
      (fraction of elements with ``|x| > 0.98``).
    """
    if x.shape != d.shape or x.ndim != 4:
        raise ValueError("x and d must share shape [N, C, H, W]")
    n = x.shape[0]
    inf = torch.full_like(d, float("inf"))
    # Per-element upper bound on alpha from whichever range face d points toward.
    # torch.where evaluates both arms, so guard the denominators to stay finite;
    # the guarded value is only ever selected where the sign condition is false.
    up_pos = (hi - x) / d.clamp_min(1e-12)   # used where d > 0
    up_neg = (lo - x) / d.clamp_max(-1e-12)  # used where d < 0
    bound = torch.where(d > 0, up_pos, inf)
    bound = torch.where(d < 0, up_neg, bound)  # d == 0 stays inf (no constraint)

    flat = bound.reshape(n, -1)
    alpha_raw, arg = flat.min(dim=1)
    alpha = alpha_raw.clamp(0.0, 1.0).reshape(n, 1, 1, 1)

    x_flat = x.reshape(n, -1)
    d_flat = d.reshape(n, -1)
    idx = arg.reshape(n, 1)
    binding = {
        "abs_x": x_flat.gather(1, idx).squeeze(1).abs(),
        "abs_d": d_flat.gather(1, idx).squeeze(1).abs(),
        "d_inf": d_flat.abs().amax(dim=1),
        "frac_extreme": (x_flat.abs() > 0.98).float().mean(dim=1),
    }
    return alpha, alpha_raw, binding


def project_correction(x: torch.Tensor, d: torch.Tensor,
                       lo: float = -1.0, hi: float = 1.0):
    """Range-safe corrected image ``x + alpha*d`` and its diagnostics."""
    alpha, alpha_raw, binding = scalar_range_alpha(x, d, lo, hi)
    return x + alpha * d, alpha, alpha_raw, binding


def low_correction_delta(pyramid, source: torch.Tensor,
                         corrected_low: torch.Tensor):
    """Image-space delta ``d = R(l' - l, 0)`` from a coarse-band correction.

    Returns ``(d, low, bands)`` where ``low``/``bands`` are ``source``'s
    decomposition and ``d`` reuses zeroed detail bands so only the low
    coefficients move.
    """
    low, bands = pyramid.decompose(source)
    zeros = [torch.zeros_like(band) for band in bands]
    d = pyramid.reconstruct(corrected_low - low, zeros)
    return d, low, bands


def local_range_project(pyramid, source: torch.Tensor,
                        corrected_low: torch.Tensor,
                        lo: float = -1.0, hi: float = 1.0,
                        steps: int = 150, lr: float = 0.03,
                        penalty: float = 100.0):
    """Coefficient-wise range projection (Exp-07 plan §5.3, second step).

    The scalar projection scales the *whole* low correction by one per-image
    ``alpha`` and is therefore choked by a single output pixel that the copied
    detail already drives near a range face.  The exact fix is the coarse-space
    QP

        minimize  1/2 || c - l' ||^2   s.t.   R(c, bands) in [lo, hi] everywhere,

    where ``bands`` are the copied detail bands.  Crucially ``R(c, bands)`` reuses
    the *same* bands for any ``c`` -- so ``R(c, bands) - R(c, 0) = R(0, bands)``
    is the copied detail regardless of ``c`` -- meaning the high-band identity
    never constrains ``c``.  Only the range box does.

    We solve it by a few dozen steps of penalised gradient descent (autograd
    supplies the exact synthesis adjoint), letting the correction back off only
    where the composite would leave range, then apply the scalar backstop on the
    *achieved* direction ``c - l`` so strict range safety holds by construction
    (``alpha = 0``, i.e. no correction, is always feasible).

    Returns a dict with the range-safe ``output``, the range-safe ``coarse``
    ``= l + alpha*(c - l)``, ``alpha`` ``[N,1,1,1]``, ``alpha_raw`` ``[N]``,
    ``binding`` (scalar-projection diagnostics), and pre-backstop feasibility
    diagnostics ``pre_max_violation`` ``[N]`` and ``pre_feasible`` ``[N]`` bool.
    """
    low, bands = pyramid.decompose(source)
    low = low.detach()
    bands = [band.detach() for band in bands]
    target = corrected_low.detach()

    with torch.enable_grad():
        c = target.clone().requires_grad_(True)
        opt = torch.optim.Adam([c], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            out = pyramid.reconstruct(c, bands)
            over = (out - hi).clamp_min(0.0)
            under = (lo - out).clamp_min(0.0)
            prox = 0.5 * (c - target).pow(2).flatten(1).sum(1)
            pen = 0.5 * penalty * (over.pow(2) + under.pow(2)).flatten(1).sum(1)
            (prox + pen).sum().backward()
            opt.step()
    c = c.detach()

    with torch.no_grad():
        out = pyramid.reconstruct(c, bands)
        viol = torch.maximum(out - hi, lo - out).clamp_min(0.0)
        pre_max_violation = viol.flatten(1).amax(1)
        pre_feasible = pre_max_violation <= 1e-6

    d, _, _ = low_correction_delta(pyramid, source, c)
    output, alpha, alpha_raw, binding = project_correction(source, d, lo, hi)
    coarse = low + alpha.reshape(-1, 1, 1, 1) * (c - low)
    return {
        "output": output,
        "coarse": coarse,
        "alpha": alpha,
        "alpha_raw": alpha_raw,
        "binding": binding,
        "pre_max_violation": pre_max_violation,
        "pre_feasible": pre_feasible,
    }
