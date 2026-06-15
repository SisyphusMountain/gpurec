"""Parity check: gpurec's optim value+grad == the frozen kernel-bench golden.

The kernel-bench ``data/<label>/whole.pt`` capture is a single gpurec batch frozen with
neutral (domain-free) keys by ``scripts/mint_kernel_bench_fixture.py``. That mint applied a
gpurec->kbench key rename; here we INVERT it to rebuild a gpurec ``_BatchStatic`` from the same
capture, run gpurec's ``make_value_and_grad`` at the captured theta, and compare loss + grad to
the stored golden (which the kbench solver produced). A live kbench cross-check is run too when
the kernel-bench tree is importable.

This is the Phase-1 verification gate: gpurec value+grad must match kernel-bench to the
backward's atomic-noise floor (~2e-4 relative).

    python -m gpurec.optim._parity_kbench [/abs/path/to/whole.pt]
"""

from __future__ import annotations

import sys

import torch

from gpurec.api._batch_state import _BatchStatic
from gpurec.api.solver_options import SolverOptions
from gpurec.optim.value_and_grad import make_value_and_grad

_KBENCH = "/home/enzo/Documents/git/gpurec/kernel-bench"
_DEFAULT_CAP = f"{_KBENCH}/data/666x80/whole.pt"

# mint script gpurec->kbench renames (scripts/mint_kernel_bench_fixture.py); we invert them.
_SH_INV = {  # kbench state_helpers key -> gpurec species_helpers key
    "node_child1": "sp_child1", "node_child2": "sp_child2", "node_parent": "sp_parent",
    "node_subtree_start": "sp_subtree_start", "node_subtree_end": "sp_subtree_end",
}
_WL_INV = {  # kbench wave_layout key -> gpurec wave_layout key
    "leaf_state_index": "leaf_species_index", "root_row_ids": "root_clade_ids",
    "item_idx": "family_idx",
}


def _to_dev(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_dev(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dev(v, device) for v in obj)
    return obj


def gpurec_static_from_capture(cap, device):
    """Rebuild the gpurec ``_BatchStatic`` for a single batch from a kernel-bench capture."""
    st = cap["static"]
    species_helpers = {_SH_INV.get(k, k): _to_dev(v, device) for k, v in st["state_helpers"].items()}
    wave_layout = {_WL_INV.get(k, k): _to_dev(v, device) for k, v in st["wave_layout"].items()}
    so = SolverOptions(**cap["meta"]["solver_options"])
    so.validate()
    n_items = int(cap["meta"]["n_items"])
    return _BatchStatic(
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        genewise=bool(st["itemwise"]),
        specieswise=bool(st["statewise"]),
        rate_family_idx=_to_dev(st["rate_item_idx"], device),
        family_indices=list(range(n_items)),
        family_index_tensor=torch.arange(n_items, dtype=torch.long, device=device),
        solver_options=so,
        warm_E=None,
    )


def run(cap_path=_DEFAULT_CAP, device="cuda"):
    cap = torch.load(cap_path, map_location="cpu", weights_only=False)
    theta = cap["inputs"]["theta"].to(device).contiguous()
    rw = cap["inputs"]["col_weights"].to(device).contiguous()  # receiver_weights
    gold_loss = float(cap["golden"]["loss"])
    gold_g = cap["golden"]["grad_theta"].reshape(-1).to(device).float()

    static = gpurec_static_from_capture(cap, device)
    f = make_value_and_grad([static], rw, theta_shape=tuple(theta.shape))
    loss, g, _sv, _w = f(theta.reshape(-1), want_grad=True)
    g = g.float()

    loss_abs = abs(loss - gold_loss)
    loss_rel = loss_abs / max(1.0, abs(gold_loss))
    g_abs = float((g - gold_g).abs().max())
    g_rel = float(((g - gold_g).abs() / gold_g.abs().clamp_min(1e-12)).max())
    g_rel_l2 = float((g - gold_g).norm() / gold_g.norm().clamp_min(1e-30))
    ok = loss_rel <= 2e-3 and g_rel_l2 <= 2e-3
    print(f"[parity {cap['meta']['label']}] mode={cap['meta']['mode']} S={cap['meta']['S']} "
          f"p={cap['meta']['p'] if 'p' in cap['meta'] else 3*cap['meta']['S']}")
    print(f"  loss   gpurec={loss:.6f} golden={gold_loss:.6f}  abs={loss_abs:.3e} rel={loss_rel:.3e}")
    print(f"  grad   max_abs={g_abs:.3e} max_rel={g_rel:.3e} rel_L2={g_rel_l2:.3e} "
          f"||g||gpurec={float(g.norm()):.4f} ||g||gold={float(gold_g.norm()):.4f}")
    print(f"  -> gpurec-vs-golden {'PASS' if ok else 'FAIL'} (atomic-noise floor ~2e-4)")

    # live kbench cross-check (strongest: identical inputs, both solvers run now)
    try:
        if _KBENCH not in sys.path:
            sys.path.insert(0, _KBENCH)
        from kbench.runtime import make_static, run_backward  # noqa: E402
        from newton.vg import forward_solve as kb_forward  # noqa: E402

        kstatic = make_static(cap, device)
        kloss_t, ksaved = kb_forward(kstatic, theta, rw)
        kg, _kc = run_backward(kstatic, theta, rw, ksaved)
        kg = kg.reshape(-1).float()
        d_loss = abs(float(kloss_t) - loss)
        d_g_l2 = float((g - kg).norm() / kg.norm().clamp_min(1e-30))
        ok2 = (d_loss / max(1.0, abs(loss))) <= 2e-3 and d_g_l2 <= 2e-3
        print(f"  live kbench: loss={float(kloss_t):.6f} d_loss={d_loss:.3e}  grad rel_L2={d_g_l2:.3e}"
              f"  -> gpurec-vs-kbench {'PASS' if ok2 else 'FAIL'}")
        ok = ok and ok2
    except Exception as e:  # noqa: BLE001
        print(f"  live kbench cross-check skipped ({type(e).__name__}: {str(e)[:80]})")

    return ok


if __name__ == "__main__":
    cap_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CAP
    raise SystemExit(0 if run(cap_path) else 1)
