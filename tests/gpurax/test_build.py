"""A2 build gate: the pybind11 _seqlik extension must be importable.

Rebuild with: gpurax/_seqlik/build.sh
"""


def test_extension_imports():
    from gpurax import _seqlik

    assert isinstance(_seqlik.corax_version(), str)
