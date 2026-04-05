# R Bindings for rustree

The R bindings in this directory are contributed scripts, not a formal R package.

`rustree.R` provides wrapper functions that call into the compiled Rust shared library.

## Usage

1. Compile rustree with R support: `cargo build --release --features r`
2. Load the shared library in R with `dyn.load("target/release/librustree.so")`
3. Source the wrapper: `source("R/rustree.R")`

For detailed usage instructions and examples, see [docs/R_TUTORIAL.md](../docs/R_TUTORIAL.md).
