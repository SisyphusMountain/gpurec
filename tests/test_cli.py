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
    assert args.mode == "global" and args.dtype is None


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


@pytest.mark.gpu
def test_fit_genewise_with_config_without_rates_table_gpu(tmp_path):
    """`gpurec fit --mode genewise --config run.toml` must run when the file has no `[rates]` table.

    fit_genewise's own rate box is 1e-6 / 2.0; a config that never sets `[rates]` carries the
    library-default box, whose upper bound is None (no cap). Substituting that here used to reach the
    Newton bound test `theta >= hi - bounds.bound_active_eps` with `hi` = None, and the whole fit died
    with `TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'`. The unit-level
    precedence checks live in tests/test_config_wiring.py; this is the end-to-end gate.
    """
    from pathlib import Path
    data = Path(__file__).parent / "data" / "alerax" / "test_trees_3"
    if not data.exists():
        pytest.skip("fixtures not vendored yet (Task 5)")
    cfg = tmp_path / "run.toml"
    cfg.write_text("[solver]\npi_iters = 16\n", encoding="utf-8")   # a real config, [rates] absent
    out = tmp_path / "fit_rates.txt"
    rc = main(["fit", "--species", str(data / "sp.nwk"), "--gene", str(data / "g.nwk"),
               "--mode", "genewise", "--device", "cuda",
               "--config", str(cfg), "--out", str(out)])
    assert rc == 0
    import json
    side = json.loads((tmp_path / "fit_rates.txt.json").read_text())
    assert math.isfinite(side["nll_nats"]) and side["mode"] == "genewise"


def test_gene_recon_model_dtype_defaults_to_float32():
    """Back-compat: GeneReconModel(sp, [g]) with no dtype still builds float32 (unchanged callers)."""
    import torch
    from pathlib import Path
    import tempfile
    from gpurec.api.model import GeneReconModel

    with tempfile.TemporaryDirectory() as tmp:
        species_tree = Path(tmp) / "species.nwk"
        gene_tree = Path(tmp) / "family_0.nwk"
        species_tree.write_text("(A:1,B:1)Root:1;\n", encoding="utf-8")
        gene_tree.write_text("(A_1:1,B_1:1)GeneRoot:1;\n", encoding="utf-8")
        model = GeneReconModel(
            species_tree, [gene_tree], device="cpu",
            family_chunk_size=1, clade_budget=None,
            batch_packing="sequential", max_wave_size=4,
        )
        assert model.theta.dtype == torch.float32
        assert model.receiver_weights.dtype == torch.float32
        assert model.origination_weights.dtype == torch.float32
        assert model.dtype == torch.float32


@pytest.mark.gpu
def test_dtype_float64_flows_from_cli_to_model_and_changes_loglik():
    """Proves --dtype float64 actually flows CLI arg -> build_model -> GeneReconModel -> forward().

    Builds two models from the SAME fixture/rates differing only in --dtype, and asserts the
    resulting log-likelihoods differ (float32 vs float64 arithmetic on the same wave computation
    is not bit-identical) and that the model's parameter dtype matches what was requested.
    """
    import torch
    from pathlib import Path
    from gpurec.cli.main import build_parser
    from gpurec.cli import _common

    data = Path(__file__).parent / "data" / "alerax" / "test_trees_1"
    if not data.exists():
        pytest.skip("fixtures not vendored yet (Task 5)")

    def _build(dtype_name):
        args = build_parser().parse_args(
            ["reconcile", "--species", str(data / "sp.nwk"), "--gene", str(data / "g.nwk"),
             "--dtype", dtype_name, "--device", "cuda"])
        model, _genes = _common.build_model(args)
        return model

    model32 = _build("float32")
    model64 = _build("float64")
    assert model32.theta.dtype == torch.float32
    assert model64.theta.dtype == torch.float64

    with torch.no_grad():
        loss32 = float(model32())
        loss64 = float(model64())
    assert math.isfinite(loss32) and math.isfinite(loss64)
    assert loss32 != loss64, "float64 dtype had no effect on the forward computation"
