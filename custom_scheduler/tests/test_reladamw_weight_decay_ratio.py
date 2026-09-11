"""Tests for the per-parameter weight-decay logic ported into RelAdamW from WarpAINO.

WarpAINO resolves an effective per-parameter weight decay:

* an explicit ``p.weight_decay_ratio`` attribute scales the group's
  ``weight_decay`` (``weight_decay * ratio``);
* without it, parameters flagged ``is_bias`` / ``is_norm`` / ``is_scalar`` /
  ``_is_dora_scale`` skip weight decay entirely (ratio 0.0);
* all other parameters use the full coefficient (ratio 1.0).

These tests cover the ``RelAdamW._parameter_weight_decay`` helper directly and
verify end-to-end behaviour of the native and foreach step paths on CUDA:

* flagged parameters follow the exact ``weight_decay=0`` trajectory;
* ``weight_decay_ratio=r`` follows the exact group-level ``weight_decay * r``
  trajectory;
* an explicit ratio overrides the flag-based default (including forcing WD on
  a bias parameter);
* native and foreach paths agree on a mixed group.
"""

import os
import sys

import pytest
import torch

# Ensure the custom_scheduler package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.RelAdamW import RelAdamW
except ImportError:
    # The package __init__ pulls in optional heavy dependencies
    # (pytorch_optimizer, adv_optm, ...) that may be absent; RelAdamW.py
    # itself only requires torch, so load it directly as a fallback.
    import importlib.util

    _module_path = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer", "RelAdamW.py"
    )
    _spec = importlib.util.spec_from_file_location("reladamw_standalone", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    RelAdamW = _module.RelAdamW

DEVICE = "cuda"

LR = 1e-2
EPS = 1e-8
STEPS = 5
SEED = 20240613


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _group(weight_decay: float) -> dict:
    return {"weight_decay": weight_decay}


def _make_param(shape, seed: int = 0) -> torch.nn.Parameter:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.nn.Parameter(torch.randn(shape, device=DEVICE, generator=gen) * 0.1)


def _make_grads(params, seeds) -> list:
    """Per-parameter gradient tensors, each from its own seeded generator."""
    return [
        torch.randn(
            p.shape,
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(s),
        )
        for p, s in zip(params, seeds)
    ]


def _run(params, weight_decay: float, foreach: bool, grad_seeds=None) -> list:
    """Run STEPS optimizer steps on ``params`` with a fixed gradient stream.

    Gradient seeds default to ``[SEED, SEED+1, ...]`` so each parameter's
    gradient is independent of the other parameters in the group (a parameter
    run alone gets the same gradient as when it is run in a group).
    """
    params = list(params)
    if grad_seeds is None:
        grad_seeds = [SEED + i for i in range(len(params))]
    grads = _make_grads(params, grad_seeds)
    opt = RelAdamW(
        params,
        lr=LR,
        betas=(0.95, 0.999),
        eps=EPS,
        weight_decay=weight_decay,
        foreach=foreach,
    )
    for t in range(STEPS):
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()
        for p in params:
            p.grad = None
    return [p.detach().clone() for p in params]


# ---------------------------------------------------------------------------
# Unit tests for the _parameter_weight_decay helper
# ---------------------------------------------------------------------------


def test_parameter_weight_decay_default_full():
    _requires_cuda()
    p = _make_param((4,))
    assert RelAdamW._parameter_weight_decay(p, _group(0.1)) == 0.1


@pytest.mark.parametrize("flag", ["is_bias", "is_norm", "is_scalar", "_is_dora_scale"])
def test_parameter_weight_decay_flagged_skip(flag):
    _requires_cuda()
    p = _make_param((4,))
    setattr(p, flag, True)
    assert RelAdamW._parameter_weight_decay(p, _group(0.1)) == 0.0


def test_parameter_weight_decay_explicit_ratio():
    _requires_cuda()
    p = _make_param((4,))
    p.weight_decay_ratio = 0.5
    assert RelAdamW._parameter_weight_decay(p, _group(0.1)) == pytest.approx(0.05)


def test_parameter_weight_decay_explicit_ratio_overrides_flag():
    _requires_cuda()
    p = _make_param((4,))
    p.is_bias = True
    p.weight_decay_ratio = 2.0
    assert RelAdamW._parameter_weight_decay(p, _group(0.1)) == pytest.approx(0.2)


def test_parameter_weight_decay_explicit_zero_ratio():
    _requires_cuda()
    p = _make_param((4,))
    p.weight_decay_ratio = 0.0
    assert RelAdamW._parameter_weight_decay(p, _group(0.1)) == 0.0


def test_parameter_weight_decay_zero_group():
    _requires_cuda()
    p = _make_param((4,))
    p.weight_decay_ratio = 0.5
    assert RelAdamW._parameter_weight_decay(p, _group(0.0)) == 0.0


# ---------------------------------------------------------------------------
# Functional tests: native path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["is_bias", "is_norm", "is_scalar", "_is_dora_scale"])
def test_flagged_param_skips_wd_native(flag):
    """A flagged parameter follows the exact weight_decay=0 trajectory."""
    _requires_cuda()
    p_a = _make_param((16, 8), seed=1)
    p_b = _make_param((16, 8), seed=1)
    setattr(p_a, flag, True)

    res_a = _run([p_a], weight_decay=0.1, foreach=False)
    res_b = _run([p_b], weight_decay=0.0, foreach=False)

    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)


def test_unflagged_param_applies_wd_native():
    """An unflagged parameter with weight_decay > 0 diverges from the no-WD run."""
    _requires_cuda()
    p_a = _make_param((16, 8), seed=1)
    p_b = _make_param((16, 8), seed=1)

    res_a = _run([p_a], weight_decay=0.1, foreach=False)
    res_b = _run([p_b], weight_decay=0.0, foreach=False)

    assert not torch.equal(res_a[0], res_b[0])
    # With decay, the parameter norm should shrink relative to the no-WD run.
    assert res_a[0].norm().item() < res_b[0].norm().item()


def test_explicit_ratio_matches_group_level_native():
    """weight_decay_ratio=0.5 with group wd=0.1 matches group wd=0.05 exactly."""
    _requires_cuda()
    p_a = _make_param((16, 8), seed=2)
    p_b = _make_param((16, 8), seed=2)
    p_a.weight_decay_ratio = 0.5

    res_a = _run([p_a], weight_decay=0.1, foreach=False)
    res_b = _run([p_b], weight_decay=0.05, foreach=False)

    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)


def test_explicit_zero_ratio_skips_wd_native():
    """An explicit ratio of 0.0 forces no decay even for an unflagged param."""
    _requires_cuda()
    p_a = _make_param((16, 8), seed=3)
    p_b = _make_param((16, 8), seed=3)
    p_a.weight_decay_ratio = 0.0

    res_a = _run([p_a], weight_decay=0.1, foreach=False)
    res_b = _run([p_b], weight_decay=0.0, foreach=False)

    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)


def test_explicit_ratio_overrides_flag_native():
    """A bias param with weight_decay_ratio=2.0 matches group wd=0.2 exactly."""
    _requires_cuda()
    p_a = _make_param((16, 8), seed=4)
    p_b = _make_param((16, 8), seed=4)
    p_a.is_bias = True
    p_a.weight_decay_ratio = 2.0

    res_a = _run([p_a], weight_decay=0.1, foreach=False)
    res_b = _run([p_b], weight_decay=0.2, foreach=False)

    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Functional tests: foreach path
# ---------------------------------------------------------------------------


def test_flagged_param_skips_wd_foreach():
    _requires_cuda()
    p_a = _make_param((16, 8), seed=1)
    p_b = _make_param((16, 8), seed=1)
    p_a.is_bias = True

    res_a = _run([p_a], weight_decay=0.1, foreach=True)
    res_b = _run([p_b], weight_decay=0.0, foreach=True)

    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)


def test_unflagged_param_applies_wd_foreach():
    _requires_cuda()
    p_a = _make_param((16, 8), seed=1)
    p_b = _make_param((16, 8), seed=1)

    res_a = _run([p_a], weight_decay=0.1, foreach=True)
    res_b = _run([p_b], weight_decay=0.0, foreach=True)

    assert not torch.equal(res_a[0], res_b[0])
    assert res_a[0].norm().item() < res_b[0].norm().item()


def test_explicit_ratio_matches_group_level_foreach():
    _requires_cuda()
    p_a = _make_param((16, 8), seed=2)
    p_b = _make_param((16, 8), seed=2)
    p_a.weight_decay_ratio = 0.5

    res_a = _run([p_a], weight_decay=0.1, foreach=True)
    res_b = _run([p_b], weight_decay=0.05, foreach=True)

    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)


def test_mixed_group_foreach():
    """A mixed group (plain, bias, ratio=0.5) gets per-parameter WD in foreach."""
    _requires_cuda()
    params = [
        _make_param((16, 8), seed=10),
        _make_param((8,), seed=11),
        _make_param((16, 8), seed=12),
    ]
    params[1].is_bias = True
    params[2].weight_decay_ratio = 0.5

    res_mixed = _run(params, weight_decay=0.1, foreach=True)

    # Reference: each parameter run alone at its effective group-level WD,
    # with the same per-parameter gradient seed as in the mixed group.
    ref_plain = _run(
        [_make_param((16, 8), seed=10)], weight_decay=0.1, foreach=True, grad_seeds=[SEED + 0]
    )
    ref_bias = _run(
        [_make_param((8,), seed=11)], weight_decay=0.0, foreach=True, grad_seeds=[SEED + 1]
    )
    ref_ratio = _run(
        [_make_param((16, 8), seed=12)], weight_decay=0.05, foreach=True, grad_seeds=[SEED + 2]
    )

    torch.testing.assert_close(res_mixed[0], ref_plain[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(res_mixed[1], ref_bias[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(res_mixed[2], ref_ratio[0], rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Path consistency and compile smoke test
# ---------------------------------------------------------------------------


def test_native_and_foreach_agree_mixed_group():
    """Native and foreach produce the same results on a mixed group."""
    _requires_cuda()

    def _mixed():
        params = [
            _make_param((16, 8), seed=10),
            _make_param((8,), seed=11),
            _make_param((16, 8), seed=12),
        ]
        params[1].is_bias = True
        params[2].weight_decay_ratio = 0.5
        return params

    res_native = _run(_mixed(), weight_decay=0.1, foreach=False)
    res_foreach = _run(_mixed(), weight_decay=0.1, foreach=True)

    for a, b in zip(res_native, res_foreach):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def _run_compiled(params, weight_decay: float) -> list:
    """Run STEPS steps on ``params`` with compile_step=True."""
    params = list(params)
    grads = _make_grads(params, [SEED + i for i in range(len(params))])
    opt = RelAdamW(
        params,
        lr=LR,
        betas=(0.95, 0.999),
        eps=EPS,
        weight_decay=weight_decay,
        foreach=False,
        compile_step=True,
    )
    for _ in range(STEPS):
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()
        for p in params:
            p.grad = None
    return [p.detach().clone() for p in params]


def test_compile_step_smoke():
    """The compiled core path also honors per-parameter weight decay.

    Both runs are compiled so the comparison isolates the skip logic from
    torch.compile-induced floating-point differences between compiled and
    eager execution.
    """
    _requires_cuda()
    p_a = _make_param((16, 8), seed=1)
    p_b = _make_param((16, 8), seed=1)
    p_a.is_bias = True

    res_a = _run_compiled([p_a], weight_decay=0.1)
    res_b = _run_compiled([p_b], weight_decay=0.0)

    # A flagged param at weight_decay=0.1 must track the no-decay trajectory,
    # even through the compiled core.
    torch.testing.assert_close(res_a[0], res_b[0], rtol=0.0, atol=0.0)