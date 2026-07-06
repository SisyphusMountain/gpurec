"""Python wrapper package around the compiled `_impl` pybind11 extension.

`_impl` is a compiled CMake target (see CMakeLists.txt) that links coraxlib
and a minimal set of GeneRaxCore sources. It is not committed to the repo —
rebuild it with:

    gpurax/_seqlik/build.sh

which configures+builds into `build/_seqlik/` (outside the source tree) and
places the resulting `_impl.<ext-suffix>.so` here, next to this file, so it
is importable in-place.
"""

from ._impl import *  # noqa: F401,F403
from ._impl import corax_version  # noqa: F401
