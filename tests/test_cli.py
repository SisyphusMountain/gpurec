import math
import pytest
from gpurec.cli.main import build_parser, main
from gpurec.cli import _common


def test_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--help"])
    assert e.value.code == 0


def test_no_subcommand_returns_2():
    assert main([]) == 2


def test_reconcile_arg_parsing():
    args = build_parser().parse_args(
        ["reconcile", "--species", "sp.nwk", "--gene", "g.nwk",
         "--delta", "0.1", "--tau", "0.2", "--lambda", "0.3"])
    assert args.command == "reconcile"
    assert args.species == "sp.nwk" and args.gene == ["g.nwk"]
    assert (args.delta, args.tau, args.lambda_) == (0.1, 0.2, 0.3)
    assert args.mode == "global" and args.dtype == "float64"


def test_reconcile_requires_species():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["reconcile", "--gene", "g.nwk"])


def test_bits_to_nats():
    assert math.isclose(_common.bits_to_nats(1.0), math.log(2.0))
    assert math.isclose(_common.bits_to_nats(3.0), 3.0 * math.log(2.0))


@pytest.mark.gpu
def test_reconcile_smoke_gpu(tmp_path, capsys):
    from pathlib import Path
    data = Path(__file__).parent / "data" / "alerax" / "test_trees_1"
    if not data.exists():
        pytest.skip("fixtures not vendored yet (Task 5)")
    rc = main(["reconcile", "--species", str(data / "sp.nwk"),
               "--gene", str(data / "g.nwk"), "--device", "cuda"])
    assert rc == 0
    # gpurec keys by gene-file basename ("g"), not the AleRax family name ("my_family"),
    # which only appears in AleRax's own files.
    out = capsys.readouterr().out.strip()
    parts = out.split()
    assert len(parts) == 2, out
    assert math.isfinite(float(parts[1]))


def test_write_outputs_alerax_format_and_sidecar(tmp_path):
    import json
    from gpurec.cli.fit import _write_outputs
    from gpurec.bench.alerax_io import parse_alerax_parameters, AleraxRates
    out = tmp_path / "rates.txt"
    # node_rates rows are (node, D, L, T) — gpurec order == AleRax D L T column order
    _write_outputs(str(out), [("GLOBAL", 0.1, 0.2, 0.3)],
                   nll_bits=-1.0, nll_nats=-0.6931, elapsed_s=1.0, mode="global", n_families=1)
    parsed = parse_alerax_parameters(out)          # reads columns as D L T
    assert parsed["GLOBAL"] == AleraxRates(D=0.1, L=0.2, T=0.3)   # written verbatim, no reorder
    side = json.loads((tmp_path / "rates.txt.json").read_text())
    assert side["mode"] == "global" and side["nll_nats"] == -0.6931 and side["n_families"] == 1


@pytest.mark.gpu
def test_fit_global_smoke_gpu(tmp_path):
    from pathlib import Path
    data = Path(__file__).parent / "data" / "alerax" / "test_mixed_200"
    if not data.exists():
        pytest.skip("fixtures not vendored yet (Task 5)")
    out = tmp_path / "fit_rates.txt"
    rc = main(["fit", "--species", str(data / "sp.nwk"), "--gene", str(data / "g.nwk"),
               "--mode", "global", "--steps", "5", "--device", "cuda", "--out", str(out)])
    assert rc == 0
    import json
    side = json.loads((tmp_path / "fit_rates.txt.json").read_text())
    assert math.isfinite(side["nll_nats"]) and side["mode"] == "global"
