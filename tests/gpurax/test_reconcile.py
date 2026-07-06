import pathlib

from gpurax.reconcile.sample import reconcile_family

FX = pathlib.Path(__file__).parent / "fixtures"


def test_reconcile_returns_events():
    nodes = reconcile_family(str(FX / "species.nwk"), str(FX / "gene.nwk"),
                             (0.2, 0.3, 0.1), seed=0, device="cuda")
    assert len(nodes) >= 1
    kinds = {n[0] for n in nodes}
    assert kinds <= {"speciation", "duplication", "transfer", "leaf", "loss"}
