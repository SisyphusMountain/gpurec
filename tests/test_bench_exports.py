def test_bench_public_exports():
    import gpurec.bench as b
    for name in ["AleraxRates", "parse_alerax_likelihoods", "parse_alerax_parameters",
                 "norm_family_name", "global_rates", "FidelityReport", "compare",
                 "reconcile_at_alerax_rates"]:
        assert hasattr(b, name), name


def test_cli_entrypoint_importable():
    from gpurec.cli.main import main, build_parser
    assert callable(main) and build_parser().prog == "gpurec"
