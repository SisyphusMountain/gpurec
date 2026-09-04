import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BENCH = REPO_ROOT / "benchmark"


def test_shell_scripts_parse():
    for sh in ["config.sh", "lib.sh", "bin/00_preflight.sh", "bin/40_run.sh",
               "bin/70_check_fidelity.sh"]:
        r = subprocess.run(["bash", "-n", str(BENCH / sh)], capture_output=True, text=True)
        assert r.returncode == 0, f"{sh}: {r.stderr}"


def test_python_drivers_help():
    # Both drivers import gpurec while building their parser, and a subprocess launched on a script
    # path gets that script's own directory (benchmark/) as sys.path[0] -- not the repo root. The
    # test session itself only imports gpurec because `python -m pytest` puts the working directory
    # on sys.path, which the child never inherits. So hand the child the repo root explicitly,
    # PREPENDED to any PYTHONPATH already set, and the check works from a source checkout whether or
    # not gpurec also happens to be pip-installed in this environment.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    for py in [
        "bench_gpurec_fit.py",
        "eval_at_alerax_rates.py",
    ]:
        r = subprocess.run([sys.executable, str(BENCH / py), "--help"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"{py}: {r.stderr}"
