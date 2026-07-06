import sys
import pathlib

_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)  # repo gpurec, not .venv (project rule)

from gpurec.core.scheduling.batching import preprocess_dataset


def preprocess_families(species_path, topology_paths):
    # Phase-1: use the whole-dataset preprocess as the baseline path.
    # (Incremental build_family_ccp/replan wiring is an optimization gated on the
    #  benchmark below; keep the correct path first.)
    return preprocess_dataset(str(species_path), list(topology_paths))
