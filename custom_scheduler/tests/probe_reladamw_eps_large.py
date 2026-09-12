"""Supplementary probe: ``eps`` headroom for LARGE tensors with small weight norms.

The original ``probe_reladamw_eps.py`` used 256x256 tensors where the surprise
variance ``v_hat`` stays around 1e-5 -- comfortably above any sane ``eps``. The
theoretically exposed regime for the denominator-normalization site

    s_var = sqrt(mean(v_hat) + eps);  denom = sqrt(v_hat)/s_var + eps

is when ``RMS(g_normed) = target_norm / sqrt(numel)`` is small, i.e.:

  * large tensors (LoRA-up on wide layers, full fine-tune projections), and
  * small ``p_norm`` floored at ``unit_clamp`` (fresh/zero-ish weights).

Then ``v_hat ~ (target/sqrt(numel))^2`` can sit at or below ``eps = 1e-8``, the
floor dominates ``s_var``, and the update RMS gets pinned near ``sqrt(eps)``
instead of tracking ``RMS(g_normed)``.

This probe measures, for 2048x2048 (4.2M) and 4096x4096 (16.7M) tensors:

  E. steady gradient (worst case: surprise decays as beta1^t)
  F. iid noisy gradient (v_hat tracks the full gradient variance)

Reported per eps: ||u_final|| / target_norm (intended to be ~O(1)), the RMS of
u_final, and the ratio RMS(u_final)/sqrt(eps) (>> 1 means the floor is NOT
active; ~O(1) means the update is pinned by eps).

Run:  python tests/probe_reladamw_eps_large.py
"""

import os
import sys

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
EPS_GRID = [1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6]


def _run_steps(p, grads, steps, eps):
    """Feed a sequence of gradients to the core; return the final update."""
    momentum = torch.zeros_like(p)
    v = torch.zeros_like(p)
    step_t = torch.zeros((), device=DEV, dtype=torch.float32)
    upd = None
    for i in range(steps):
        step_t.add_(1)
        upd = RelAdamW._reladamw_step_core(
            p, grads[i % len(grads)], momentum, v, step_t,
            0.95, 0.999, 0.0,            # beta1, beta2, weight_decay
            3, "multi_axis", 1.0,        # sinkhorn_iter, shape_mode, unit_clamp
            True, True, True, True,      # nesterov, cautious, cautious_wd, overshoot
            True, eps,                   # debias, eps
        )
    return upd


def _hdr(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def probe_large(shape, mode, steps=40, seed=0, p_norm_target=None):
    torch.manual_seed(seed)
    numel = shape[0] * shape[1]
    # Build p with an EXACT prescribed norm. p_norm_target < unit_clamp pins
    # target_norm at 1.0 (fresh/zero-init LoRA-B worst case for v_hat scale).
    if p_norm_target is None:
        p = torch.randn(shape, device=DEV) * 0.01
    else:
        p = torch.randn(shape, device=DEV) * (p_norm_target / numel ** 0.5)
    p_norm = p.norm().item()
    target = max(p_norm, 1.0)

    if mode == "steady":
        g = torch.randn(shape, device=DEV) * 0.02
        grads = [g]
    else:  # noisy: iid gradient each step
        gen = torch.Generator(device=DEV).manual_seed(seed + 1)
        grads = [torch.randn(shape, device=DEV, generator=gen) * 0.02 for _ in range(steps)]

    rms_g_normed = target / numel ** 0.5
    print(f"\nshape={shape} numel={numel:.2e} p_norm={p_norm:.4f} target={target:.2f}"
          f" -> RMS(g_normed)={rms_g_normed:.3e}, v_hat scale ~ {rms_g_normed**2:.3e}")
    print(f"{'eps':>10} | {'||u||/target':>13} | {'RMS(u_final)':>13} | {'RMS/sqrt(eps)':>13} | {'max|u|':>12}")

    for e in EPS_GRID:
        upd = _run_steps(p.clone(), grads, steps, e)
        u_rms = upd.square().mean().sqrt().item()
        ratio = u_rms / (e ** 0.5)
        print(f"{e:>10.0e} | {upd.norm().item() / target:>13.4f} | "
              f"{u_rms:>13.4e} | {ratio:>13.3e} | {upd.abs().max().item():>12.4e}")


def _run_steps_ok(p, grads, steps, eps):
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


def probe_ultralow():
    """Is eps=1e-16 (or 1e-20) safe? Checks the degenerate inputs where eps is
    the only guard against inf/NaN, plus parity with 1e-12 on normal inputs.

    Hazards specific to ultra-low eps:
      * rsqrt(0 + eps): an exactly-zero gradient would give 0*inf = NaN at
        eps=0; any eps > 0 keeps it finite. Check the update is exactly 0.
      * Denormal-scale gradients (RMS ~1e-20): squares underflow toward 0.
      * Deep-stall surprise variance: mean(v_hat) below eps engages the floor;
        verify no inf/NaN and bounded output.
    """
    eps_list = [1e-12, 1e-16, 1e-20]

    print("\nH1. Zero gradient (all sites see exact zeros).")
    print(f"{'eps':>10} | {'||u_final||':>14} | {'any NaN/inf':>12}")
    for e in eps_list:
        p = torch.zeros(128, 512, device=DEV)
        g = torch.zeros(128, 512, device=DEV)
        upd = _run_steps_ok(p, [g], 5, e)
        bad = bool(torch.isnan(upd).any() or torch.isinf(upd).any())
        print(f"{e:>10.0e} | {upd.norm().item():>14.4e} | {str(bad):>12}")

    print("\nH2. Denormal-scale gradient (RMS ~1e-20; squares underflow toward 0).")
    print(f"{'eps':>10} | {'||u_final||':>14} | {'any NaN/inf':>12} | {'max|u|':>12}")
    for e in eps_list:
        torch.manual_seed(3)
        p = torch.randn(128, 512, device=DEV) * 0.01
        g = torch.randn(128, 512, device=DEV) * 1e-20
        upd = _run_steps_ok(p, [g], 5, e)
        bad = bool(torch.isnan(upd).any() or torch.isinf(upd).any())
        print(f"{e:>10.0e} | {upd.norm().item():>14.4e} | {str(bad):>12} | {upd.abs().max().item():>12.4e}")

    print("\nH3. Parity vs 1e-12 on a normal large-tensor steady run (2048x2048,")
    print("    p_norm floored at unit_clamp, 5000 identical steps).")
    print(f"{'eps':>10} | {'||u||/target':>13} | {'any NaN/inf':>12}")
    torch.manual_seed(0)
    shape = (2048, 2048)
    numel = shape[0] * shape[1]
    p = torch.randn(shape, device=DEV) * (0.5 / numel ** 0.5)
    g = torch.randn(shape, device=DEV) * 0.02
    for e in eps_list + [1e-14]:
        pp = p.clone()
        momentum = torch.zeros_like(pp)
        v = torch.zeros_like(pp)
        step_t = torch.zeros((), device=DEV, dtype=torch.float32)
        upd = None
        for _ in range(5000):
            step_t.add_(1)
            upd = RelAdamW._reladamw_step_core(
                pp, g, momentum, v, step_t,
                0.95, 0.999, 0.0,
                3, "multi_axis", 1.0,
                True, True, True, True, True, e,
            )
        bad = bool(torch.isnan(upd).any() or torch.isinf(upd).any())
        print(f"{e:>10.0e} | {upd.norm().item():>13.4f} | {str(bad):>12}")


def probe_long_steady(shape, steps_list, p_norm_target=0.5, seed=0):
    """Perfectly constant gradient for thousands of steps.

    The surprise d = g - m decays as beta1^t until fp32 quantization stalls
    the momentum lerp; mean(v_hat) can then fall below ANY fixed eps. This is
    the only regime where the choice of the s_var floor changes character
    rather than just shifting a threshold.
    """
    torch.manual_seed(seed)
    numel = shape[0] * shape[1]
    p = torch.randn(shape, device=DEV) * (p_norm_target / numel ** 0.5)
    g = torch.randn(shape, device=DEV) * 0.02
    target = max(p.norm().item(), 1.0)
    rms_g = target / numel ** 0.5
    print(f"\nshape={shape} target={target:.2f} RMS(g_normed)={rms_g:.3e} "
          f"v_hat scale ~ {rms_g**2:.3e}; perfectly steady gradient.")
    print(f"{'steps':>7} |" + "".join(f" {'eps=' + format(e, '.0e'):>13} |" for e in EPS_GRID)
          + "   (cells: ||u||/target)")

    results = {e: {} for e in EPS_GRID}
    for e in EPS_GRID:
        momentum = torch.zeros_like(p)
        v = torch.zeros_like(p)
        step_t = torch.zeros((), device=DEV, dtype=torch.float32)
        for i in range(1, max(steps_list) + 1):
            step_t.add_(1)
            upd = RelAdamW._reladamw_step_core(
                p, g, momentum, v, step_t,
                0.95, 0.999, 0.0,
                3, "multi_axis", 1.0,
                True, True, True, True, True, e,
            )
            if i in steps_list:
                results[e][i] = upd.norm().item() / target
    for s in steps_list:
        print(f"{s:>7} |" + "".join(f" {results[e][s]:>13.4f} |" for e in EPS_GRID))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cuda.matmul.allow_tf32 = True

    _hdr("E. LARGE tensor, TINY p_norm (target=unit_clamp=1), STEADY gradient")
    probe_large((2048, 2048), "steady", p_norm_target=0.5)
    probe_large((4096, 4096), "steady", p_norm_target=0.5)

    _hdr("F. LARGE tensor, TINY p_norm (target=unit_clamp=1), NOISY gradient")
    probe_large((2048, 2048), "noisy", p_norm_target=0.5)
    probe_large((4096, 4096), "noisy", p_norm_target=0.5)

    _hdr("G. LONG-HORIZON perfectly steady gradient (d decays to fp32 stall)")
    probe_long_steady((2048, 2048), [40, 200, 1000, 5000])
    probe_long_steady((512, 64), [40, 200, 1000, 5000])

    _hdr("H. Ultra-low eps safety (1e-16 / 1e-20): degenerate inputs + parity")
    probe_ultralow()

    print("\nDone.")
