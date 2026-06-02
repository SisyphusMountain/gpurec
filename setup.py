from setuptools import setup
from setuptools_rust import Binding, RustExtension


setup(
    rust_extensions=[
        RustExtension(
            "gpurec.gpurec_preprocess",
            path="crates/gpurec-preprocess/Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        ),
        RustExtension(
            "gpurec.gpurec_backtrack",
            path="crates/gpurec-backtrack/Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        ),
    ],
    zip_safe=False,
)
