"""Regression tests locking in the observed ``eps`` behavior of RelAdamW.

These tests document *where* ``eps`` has leverage and where it does not, based
on the empirical probe in ``probe_reladamw_eps.py``:

* The Sinkhorn mean-square floor is a pure numerical guard: for gradients with
  O(1) energy the output is insensitive to ``eps`` over many orders of magnitude.
* The weight/grad norm rescaling ``sqrt(sum x^2 + eps)`` is insensitive because
  the sum of squares dwarfs ``eps`` (and ``unit_clamp`` floors the target).
* The denominator ``+ eps`` (RMS-normalized to ~1) is the only site with real
  leverage: ``eps <= 1e-8`` is a flat regime, while ``eps >= 1e-4`` measurably
  distorts the trajectory (much more so for a zero-init LoRA-B style param).
* An all-zero gradient yields an exactly zero update for every ``eps``.

All tests run on CUDA.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.RelAdamW import RelAdamW, sinkhorn_rms_balance
except ImportError:
    import importlib.util

    _p = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer", "RelAdamW.py"
    )
    _spec = importlib.util.spec_from_file_location("reladamw_standalone", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    RelAdamW = _m.RelAdamW
    sinkhorn_rms_balance = _m.sinkhorn_rms_balance

DEV = "cuda"
EPS_GRID = [1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _run_core(p, g, steps, eps):
    momentum = torch.zeros_like(p)
    v = torch.zeros_like(p)
    step_t = torch.zeros((), device=DEV, dtype=torch.float32)
    upd = None
    for _ in range(steps):
        step_t.add_(1)
        upd = RelAdamW._reladamw_step_core(
            p, g, momentum, v, step_t,
            0.95, 0.999, 0.0,
            3, "multi_axis", 1.0,
            True, True, True, True, True, eps,
        )
    return upd


def _trajectory(eps, shapes, steps=30, seed=1234):
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(s, device=DEV) * 0.05) for s in shapes]
    with torch.no_grad():
        params[1].zero_()
    opt = RelAdamW(params, lr=1e-2, betas=(0.95, 0.999), eps=eps, weight_decay=0.1)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    grads = [torch.randn(s, device=DEV, generator=gen) * 0.02 for s in shapes]
    for _ in range(steps):
        for p, gg in zip(params, grads):
            p.grad = gg.clone()
        opt.step()
        for p in params:
            p.grad = None
    return [p.detach().clone() for p in params]


# ---------------------------------------------------------------------------
# Sinkhorn mean-square floor: insensitive (pure guard)
# ---------------------------------------------------------------------------
def test_sinkhorn_eps_insensitive_for_normal_grad():
    _requires_cuda()
    torch.manual_seed(0)
    g = torch.randn(512, 512, device=DEV)
    base = sinkhorn_rms_balance(g.clone(), num_iter=3, shape_mode="multi_axis", eps=1e-8)
    for e in EPS_GRID:
        out = sinkhorn_rms_balance(g.clone(), num_iter=3, shape_mode="multi_axis", eps=e)
        torch.testing.assert_close(out, base, rtol=1e-4, atol=1e-6)


def test_sinkhorn_near_dead_slice_not_blown_up():
    _requires_cuda()
    torch.manual_seed(0)
    g = torch.randn(512, 512, device=DEV)
    g[0] = g[0] * 1e-6
    out = sinkhorn_rms_balance(g.clone(), num_iter=3, shape_mode="multi_axis", eps=1e-8)
    # Sinkhorn equalizes energy; nothing should explode even for a dead slice.
    assert out.abs().max().item() < 1e2


# ---------------------------------------------------------------------------
# Weight/grad norm rescaling: insensitive (isolated formula, no denom mixing)
# ---------------------------------------------------------------------------
def _g_normed(p, g, e):
    g_sink = sinkhorn_rms_balance(g.clone(), num_iter=3, shape_mode="multi_axis", eps=e)
    p_norm = torch.sqrt(torch.sum(p.square()) + e)
    s_norm = torch.sqrt(torch.sum(g_sink.square()) + e)
    return g_sink * (torch.clamp_min(p_norm, 1.0) / (s_norm + e))


def test_norm_rescale_eps_insensitive():
    _requires_cuda()
    torch.manual_seed(1)
    p = torch.full((128, 256), 0.05, device=DEV)
    g = torch.randn(128, 256, device=DEV) * 0.02
    base = _g_normed(p, g, 1e-8)
    for e in EPS_GRID:
        # sums of squares (~80 here) dwarf eps by ~9 orders of magnitude.
        torch.testing.assert_close(_g_normed(p, g, e), base, rtol=1e-6, atol=1e-8)


def test_single_step_update_norm_insensitive_up_to_1e5():
    """The bulk single-step update is stable for eps <= 1e-5.

    Only the low-variance tail coordinates shift element-wise; the aggregate
    update magnitude is essentially unchanged.
    """
    _requires_cuda()
    torch.manual_seed(1)
    p = torch.full((128, 256), 0.05, device=DEV)
    g = torch.randn(128, 256, device=DEV) * 0.02
    base = _run_core(p.clone(), g.clone(), 1, 1e-8)
    base_norm = base.norm().item()
    for e in [1e-12, 1e-10, 1e-8, 1e-6, 1e-5]:
        upd = _run_core(p.clone(), g.clone(), 1, e)
        rel = (upd - base).norm().item() / (base_norm + 1e-12)
        assert rel < 1e-2, f"eps={e} changed aggregate update by {rel}"


# ---------------------------------------------------------------------------
# Denominator: flat regime below 1e-8; distortion at >= 1e-4
# ---------------------------------------------------------------------------
def test_denominator_eps_flat_regime():
    _requires_cuda()
    shapes = [(64, 512), (512, 64)]
    base = _trajectory(1e-8, shapes)
    for e in (1e-12, 1e-10):
        tr = _trajectory(e, shapes)
        rel = (tr[0] - base[0]).norm().item() / (base[0].norm().item() + 1e-12)
        assert rel < 1e-4, f"eps={e} should be in the flat regime, got {rel}"


def test_denominator_eps_distorts_at_1e4():
    _requires_cuda()
    shapes = [(64, 512), (512, 64)]
    base = _trajectory(1e-8, shapes)
    tr = _trajectory(1e-4, shapes)
    rel_zero_init = (tr[1] - base[1]).norm().item() / (base[1].norm().item() + 1e-12)
    # Zero-init LoRA-B style params are the most eps-sensitive.
    assert rel_zero_init > 1e-2, f"eps=1e-4 should distort, got {rel_zero_init}"


def test_denominator_eps_monotonic_sensitivity():
    _requires_cuda()
    shapes = [(64, 512), (512, 64)]
    base = _trajectory(1e-8, shapes)
    rels = []
    for e in [1e-10, 1e-8, 1e-6, 1e-5, 1e-4]:
        tr = _trajectory(e, shapes)
        rels.append((tr[1] - base[1]).norm().item() / (base[1].norm().item() + 1e-12))
    # 1e-8 is the self-comparison (0); larger eps strictly increases distortion.
    assert rels[2] < rels[3] < rels[4]


# ---------------------------------------------------------------------------
# Zero gradient -> zero update for every eps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", EPS_GRID)
def test_zero_grad_zero_update_any_eps(eps):
    _requires_cuda()
    p = torch.zeros(128, 512, device=DEV)
    g = torch.zeros(128, 512, device=DEV)
    upd = _run_core(p, g, 5, eps)
    assert upd.abs().max().item() == 0.0
