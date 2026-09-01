# Input contract

## Species and gene leaf names

The Rust preprocessor maps a gene-tree leaf to a species as follows:

1. if the leaf contains an underscore, use the text before the first
   underscore;
2. otherwise, use the complete leaf name.

Therefore `A_gene42` maps to species `A`, and an exact leaf `A` also maps to
species `A`. Names are case-sensitive. A tree containing leaves `a` and `b`
does not match a species tree containing `A` and `B`.

The current gpurec preprocessing path does not consume an AleRax `mapping =`
file to rename leaves. Inputs should satisfy the naming rule directly.

## Family manifests and paths

`fit_genewise` accepts a list of paths, a glob, a directory, or an AleRax-style
`[FAMILIES]` listing. At present, `starting_gene_tree =` values are interpreted
relative to the process working directory, not automatically relative to the
manifest file. Prefer absolute paths or launch from the directory expected by
the manifest until the resolver backlog item is fixed.

## Rates and units

The parameter tensor stores `[log2(D), log2(L), log2(T)]`. The library returns
negative log-likelihood in bits; the CLI converts the reported value to nats.
See `docs/cli.md` for command-line input and output formats.
