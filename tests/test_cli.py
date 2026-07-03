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
    out = capsys.readouterr().out
    assert "my_family" in out
