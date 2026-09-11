"""Regression tests: ``eps`` behavior of RelAdamW on LARGE tensors (>= 4M elements).

Companion to ``test_reladamw_eps.py`` (typical LoRA shapes) and
``probe_reladamw_eps_large.py`` (the diagnostic probe these thresholds come
from).

Findings locked in here:

* The only ``eps`` site with leverage at scale is the denominator RMS
  normalization ``s_var = sqrt(mean(v_hat) + eps)``. ``v_hat`` scales like
  ``(target_norm / sqrt(numel))**2``, so for multi-million-element tensors
  with a weight norm floored at ``unit_clamp`` (zero-init / tiny weights),
  ``v_hat`` sits at 1e-9..1e-7 -- right at the default ``eps = 1e-8``.
* In that regime ``eps = 1e-8`` systematically INFLATES the update norm
  (measured: +8% at 2048x2048, +29% at 4096x4096 for steady gradients),
  breaking the intended ``||update|| ~ lr * target_norm`` semantics.
* ``eps <= 1e-10`` is flat (1e-12 matches the eps->0 limit to <0.5%),
  and ``eps >= 1e-6`` inflates the update 4-8x.
* For typical LoRA shapes (numel <= ~500k) the default 1e-8 remains in the
  flat regime -- see ``test_reladamw_eps.py``.

All tests run on CUDA.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.RelAdamW import RelAdamW
except ImportError:
    import importlib.util

    _p = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer", "RelAdamW.py"
    )
    _spec = importlib.util.spec_from_file_location("reladamw_standalone", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    RelAdamW = _m.RelAdamW

DEV = "cuda"


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _final_update(shape, p_norm_target, eps, steps=40, seed=0, steady=True):
    """Run the core step on a large tensor; return the final update."""
    torch.manual_seed(seed)
    numel = shape[0] * shape[1]
    p = torch.randn(shape, device=DEV) * (p_norm_target / numel ** 0.5)
    if steady:
        grads = [torch.randn(shape, device=DEV) * 0.02]
    else:
        gen = torch.Generator(device=DEV).manual_seed(seed + 1)
        grads = [torch.randn(shape, device=DEV, generator=gen) * 0.02 for _ in range(steps)]

    momentum = torch.zeros_like(p)
    v = torch.zeros_like(p)
    step_t = torch.zeros((), device=DEV, dtype=torch.float32)
    upd = None
    for i in range(steps):
        step_t.add_(1)
        upd = RelAdamW._reladamw_step_core(
            p, grads[i % len(grads)], momentum, v, step_t,
            0.95, 0.999, 0.0,
            3, "multi_axis", 1.0,
            True, True, True, True, True, eps,
        )
    return upd


def _rel_norm(upd, target=1.0):
    return upd.norm().item() / target


# ---------------------------------------------------------------------------
# Large tensor + unit_clamp-floored target: the eps-leveraged regime
# ---------------------------------------------------------------------------
def test_large_steady_eps_1e8_inflates_update():
    """2048x2048, p_norm floored at unit_clamp, steady grad: eps=1e-8 must
    inflate the update norm by >= 5% relative to the eps->0 limit (eps=1e-12)."""
    _requires_cuda()
    base = _rel_norm(_final_update((2048, 2048), 0.5, 1e-12))
    biased = _rel_norm(_final_update((2048, 2048), 0.5, 1e-8))
    assert biased / base > 1.05, f"expected >5% inflation, got {biased / base:.4f}"


def test_large_steady_flat_below_1e10():
    """eps in [1e-14, 1e-10] is within 0.5% of each other (flat regime)."""
    _requires_cuda()
    ref = _rel_norm(_final_update((2048, 2048), 0.5, 1e-12))
    for e in (1e-14, 1e-10):
        rel = _rel_norm(_final_update((2048, 2048), 0.5, e))
        assert abs(rel / ref - 1.0) < 5e-3, f"eps={e} not flat: {rel / ref:.5f}"


def test_large_steady_monotonic_inflation_with_eps():
    """Update norm grows monotonically with eps once the floor is active."""
    _requires_cuda()
    norms = [
        _rel_norm(_final_update((4096, 4096), 0.5, e))
        for e in (1e-12, 1e-10, 1e-8, 1e-7, 1e-6)
    ]
    assert norms[0] < norms[2] < norms[3] < norms[4], norms


def test_large_steady_eps_1e6_blowup():
    """eps=1e-6 inflates the update several-fold (probe measured 4-8x)."""
    _requires_cuda()
    base = _rel_norm(_final_update((4096, 4096), 0.5, 1e-12))
    biased = _rel_norm(_final_update((4096, 4096), 0.5, 1e-6))
    assert biased / base > 3.0, f"expected >3x inflation, got {biased / base:.4f}"


def test_large_noisy_milder_but_present():
    """Noisy gradients keep v_hat higher, so the eps=1e-8 bias is milder
    (probe: +2% at 2048x2048, +8% at 4096x4096) but still monotonically
    increasing in eps."""
    _requires_cuda()
    base = _rel_norm(_final_update((4096, 4096), 0.5, 1e-12, steady=False))
    biased = _rel_norm(_final_update((4096, 4096), 0.5, 1e-8, steady=False))
    assert biased / base > 1.02, f"expected >2% inflation, got {biased / base:.4f}"


def _final_update_fixed(p, grads, eps, steps):
    momentum = torch.zeros_like(p)
    v = torch.zeros_like(p)
    step_t = torch.zeros((), device=DEV, dtype=torch.float32)
    upd = None
    for i in range(steps):
        step_t.add_(1)
        upd = RelAdamW._reladamw_step_core(
            p, grads[i % len(grads)], momentum, v, step_t,
            0.95, 0.999, 0.0,
            3, "multi_axis", 1.0,
            True, True, True, True, True, eps,
        )
    return upd


def test_ultralow_eps_no_nan_on_degenerate_inputs():
    """eps as low as 1e-20 stays finite on exact-zero and denormal gradients.

    eps>0 is the only guard against rsqrt(0) -> inf -> 0*inf = NaN when the
    gradient is exactly zero, and against underflowing mean-squares for
    denormal-scale gradients. Verified: exact-zero update and no NaN/inf.
    """
    _requires_cuda()
    for e in (1e-16, 1e-20):
        torch.manual_seed(3)
        p0 = torch.zeros(128, 512, device=DEV)
        g0 = torch.zeros(128, 512, device=DEV)
        upd = _final_update_fixed(p0, [g0], e, steps=5)
        assert not torch.isnan(upd).any() and not torch.isinf(upd).any()
        assert upd.abs().max().item() == 0.0

        p = torch.randn(128, 512, device=DEV) * 0.01
        g = torch.randn(128, 512, device=DEV) * 1e-20
        upd = _final_update_fixed(p, [g], e, steps=5)
        assert not torch.isnan(upd).any() and not torch.isinf(upd).any()
        # Scale-invariant by design: update norm tracks the weight norm, not
        # the (denormal) gradient scale.
        assert upd.norm().item() == pytest.approx(p.norm().item(), rel=0.1)


def test_ultralow_eps_tracks_zero_limit_long_steady():
    """eps=1e-16 matches the eps->0 limit exactly on a 5000-step steady run
    where eps=1e-12 already drifts by ~3% (fp32 stall regime)."""
    _requires_cuda()
    torch.manual_seed(0)
    shape = (2048, 2048)
    numel = shape[0] * shape[1]
    p = torch.randn(shape, device=DEV) * (0.5 / numel ** 0.5)
    g = torch.randn(shape, device=DEV) * 0.02
    upd = _final_update_fixed(p, [g], 1e-16, steps=5000)
    assert not torch.isnan(upd).any() and not torch.isinf(upd).any()
    assert abs(upd.norm().item() - 1.0) < 1e-2  # ||u||/target == 1


def test_design_semantics_restored_at_low_eps():
    """At eps <= 1e-10 the intended semantics hold exactly: for a steady
    gradient the final update norm equals the target norm (||u||/target ~ 1)."""
    _requires_cuda()
    for shape in ((2048, 2048), (4096, 4096)):
        rel = _rel_norm(_final_update(shape, 0.5, 1e-10))
        assert abs(rel - 1.0) < 5e-3, f"shape={shape}: ||u||/target = {rel:.5f}"
