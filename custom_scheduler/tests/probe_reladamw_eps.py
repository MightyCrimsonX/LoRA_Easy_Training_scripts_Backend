"""Empirical probe: how much does ``eps`` matter at each usage site in RelAdamW?

This is a diagnostics script (not a pass/fail test). It quantifies the
sensitivity of the four distinct ``eps`` roles in ``RelAdamW.py``:

  A. Sinkhorn mean-square floor        -> ``rsqrt(mean_sq + eps)``
  B. Weight/grad norm rescaling        -> ``sqrt(sum(x^2) + eps)`` / ``n + eps``
  C. Denominator normalization         -> ``sqrt(mean(v) + eps)`` / ``d + eps``
  D. Relative-WD norm and p_norm floor -> ``sqrt(sum(u^2) + eps)`` / ``clamp_min(p, eps)``

Run:  python tests/probe_reladamw_eps.py
"""

import os
import sys

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


def _hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# A. Sinkhorn mean-square floor: amplification of a near-zero slice
# ---------------------------------------------------------------------------
def probe_sinkhorn_degenerate():
    _hdr("A. Sinkhorn mean-square floor (near-zero slice amplification)")
    print("Grad: realistic 512x512, but row 0 scaled to 1e-6 (near-dead slice).")
    print("Reported: max|out| and out RMS. Larger amplification = worse blow-up.")
    print(f"{'eps':>10} | {'max|out|':>14} | {'out_rms':>12} | {'amp=max/rms':>12}")

    g = torch.randn(512, 512, device=DEV)
    g[0] = g[0] * 1e-6
    for e in EPS_GRID:
        out = sinkhorn_rms_balance(g.clone(), num_iter=3, shape_mode="multi_axis", eps=e)
        mx = out.abs().max().item()
        rms = out.square().mean().sqrt().item()
        print(f"{e:>10.0e} | {mx:>14.4e} | {rms:>12.4e} | {mx / rms:>12.4e}")


# ---------------------------------------------------------------------------
# B. Weight/grad norm rescaling sensitivity
# ---------------------------------------------------------------------------
def probe_norm_rescale():
    _hdr("B. Weight/grad norm rescaling (sqrt(sum x^2 + eps))")
    print("Compare g_normed produced for a normal param vs an all-zero param.")
    print("zero-param => p_norm=sqrt(eps); nonzero => p_norm dominates eps.")

    def g_normed_for(p_norm_target, g, e):
        g_sink = sinkhorn_rms_balance(g.clone(), num_iter=3, shape_mode="multi_axis", eps=e)
        p_norm = torch.sqrt(torch.sum((torch.full_like(g, p_norm_target)).square()) + e)
        s_norm = torch.sqrt(torch.sum(g_sink.square()) + e)
        return g_sink * (torch.clamp_min(p_norm, 1.0) / (s_norm + e))

    g = torch.randn(256, 256, device=DEV)
    print(f"{'eps':>10} | {'||g_normed|| (live)':>22} | {'||g_normed|| (x1e-9)':>22}")
    for e in EPS_GRID:
        live = g_normed_for(0.05, g, e).norm().item()
        tiny = g_normed_for(1e-9, g, e).norm().item()
        print(f"{e:>10.0e} | {live:>22.6e} | {tiny:>22.6e}")


# ---------------------------------------------------------------------------
# C. Denominator normalization: steady-gradient "surprise -> 0" case
# ---------------------------------------------------------------------------
def _run_core(p, g, steps, eps, momentum=None, v=None, steady=False):
    """Run _reladamw_step_core repeatedly; return final update and states."""
    shape = p.shape
    if momentum is None:
        momentum = torch.zeros(shape, device=DEV, dtype=torch.float32)
    if v is None:
        v = torch.zeros(shape, device=DEV, dtype=torch.float32)
    step_t = torch.zeros((), device=DEV, dtype=torch.float32)
    upd = None
    for _ in range(steps):
        step_t.add_(1)
        upd = RelAdamW._reladamw_step_core(
            p, g, momentum, v, step_t,
            0.95, 0.999, 0.0,          # beta1, beta2, weight_decay
            3, "multi_axis", 1.0,       # sinkhorn_iter, shape_mode, unit_clamp
            True, True, True, True,     # nesterov, cautious_update, cautious_wd, overshoot
            True,                       # debias
            eps,
        )
    return upd, momentum, v


def probe_denominator_steady():
    _hdr("C. Denominator normalization (steady gradient -> surprise ~ 0)")
    print("Constant gradient across 40 steps. As surprise d->0, sqrt(v)->0 and the")
    print("RMS-normalized denom approaches the +eps floor. Measure ||u_final||.")

    p = torch.full((256, 256), 0.05, device=DEV)
    g = torch.randn(256, 256, device=DEV) * 0.01
    print(f"{'eps':>10} | {'||u_final|| (step40)':>22} | {'max|u_raw|':>14}")
    for e in EPS_GRID:
        upd, _, _ = _run_core(p.clone(), g.clone(), 40, e, steady=True)
        print(f"{e:>10.0e} | {upd.norm().item():>22.6e} | {upd.abs().max().item():>14.4e}")


def probe_denominator_deadcoord():
    _hdr("C'. Denominator + eps: dead coordinate (zero-init LoRA-B style)")
    print("Gradient is exactly zero on the whole tensor (nothing to learn).")
    print("A correct result is a ~zero update regardless of eps (0/eps == 0).")
    p = torch.zeros(128, 512, device=DEV)
    g = torch.zeros(128, 512, device=DEV)
    print(f"{'eps':>10} | {'||u_final||':>14} | {'max|u_final|':>14}")
    for e in EPS_GRID:
        upd, _, _ = _run_core(p.clone(), g.clone(), 5, e)
        print(f"{e:>10.0e} | {upd.norm().item():>14.4e} | {upd.abs().max().item():>14.4e}")


# ---------------------------------------------------------------------------
# D. Full end-to-end trajectory sensitivity (realistic LoRA-ish shapes)
# ---------------------------------------------------------------------------
def _trajectory(eps, shapes, steps=30, seed=1234):
    torch.manual_seed(seed)
    params = [
        torch.nn.Parameter(torch.randn(s, device=DEV) * 0.05) for s in shapes
    ]
    # LoRA-B style zero-init on the second param.
    with torch.no_grad():
        params[1].zero_()
    opt = RelAdamW(
        params, lr=1e-2, betas=(0.95, 0.999), eps=eps,
        weight_decay=0.1, foreach=False,
    )
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    grads = [
        torch.randn(s, device=DEV, generator=gen) * 0.02 for s in shapes
    ]
    for _ in range(steps):
        for p, gg in zip(params, grads):
            p.grad = gg.clone()
        opt.step()
        for p in params:
            p.grad = None
    return [p.detach().clone() for p in params]


def probe_e2e():
    _hdr("D. End-to-end trajectory sensitivity (normal and zero-init params)")
    print("30 steps, lr=1e-2, wd=0.1, shapes [(64,512), (512,64) zero-init].")
    print("Delta reported relative to the eps=1e-8 baseline trajectory.")
    shapes = [(64, 512), (512, 64)]
    base = _trajectory(1e-8, shapes)
    print(f"{'eps':>10} | {'rel.Delta p0':>14} | {'rel.Delta p1(zero)':>18}")
    for e in EPS_GRID:
        tr = _trajectory(e, shapes)
        d0 = (tr[0] - base[0]).norm().item() / (base[0].norm().item() + 1e-12)
        d1 = (tr[1] - base[1]).norm().item() / (base[1].norm().item() + 1e-12)
        print(f"{e:>10.0e} | {d0:>14.4e} | {d1:>18.4e}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cuda.matmul.allow_tf32 = True
    probe_sinkhorn_degenerate()
    probe_norm_rescale()
    probe_denominator_steady()
    probe_denominator_deadcoord()
    probe_e2e()
    print("\nDone.")
