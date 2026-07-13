import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).parent.parent / "benchmark"


def test_shell_scripts_parse():
    for sh in ["config.sh", "lib.sh", "bin/00_preflight.sh", "bin/40_run.sh",
               "bin/70_check_fidelity.sh"]:
        r = subprocess.run(["bash", "-n", str(BENCH / sh)], capture_output=True, text=True)
        assert r.returncode == 0, f"{sh}: {r.stderr}"


def test_python_drivers_help():
    for py in [
        "bench_gpurec_fit.py",
        "eval_at_alerax_rates.py",
    ]:
        r = subprocess.run([sys.executable, str(BENCH / py), "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{py}: {r.stderr}"
