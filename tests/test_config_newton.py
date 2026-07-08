import inspect

import pytest

from gpurec.config.newton import NewtonOptions


def test_newton_options_defaults_pinned():
    """Every NewtonOptions default must equal the literal it replaces (see task-4 brief)."""
    opts = NewtonOptions()
    assert opts.sigma == 0.01
    assert opts.sigma_floor == 1e-4
    assert opts.lanczos_m == 10
    assert opts.nu == 1.5
    assert opts.decrease == 1.5
    assert opts.max_bumps == 3
    assert opts.max_cg == 40
    assert opts.c1 == 1e-4
    assert opts.ls_max == 25
    assert opts.gtol == 1e-2
    assert opts.max_newton == 40
    assert opts.ftol == 1e-9
    assert opts.seed == 0
    assert opts.fd_eps_blockwise == 1e-2
    assert opts.fd_eps_hvp == 1e-5
    assert opts.lam_ceil_factor == 10.0
    assert opts.forcing_eta == 0.1
    assert opts.certify_m == 200
    assert opts.cg_tol == 1e-7
    assert opts.cg_max == 400


def test_newton_options_validate_passes():
    NewtonOptions().validate()


@pytest.mark.parametrize("field,bad", [
    ("lanczos_m", 0),
    ("max_cg", 0),
    ("ls_max", 0),
    ("max_newton", 0),
    ("max_bumps", 0),
    ("certify_m", 0),
    ("cg_max", 0),
    ("sigma", 0.0),
    ("sigma_floor", -1.0),
    ("c1", 0.0),
    ("gtol", -1e-3),
    ("ftol", 0.0),
    ("fd_eps_blockwise", 0.0),
    ("fd_eps_hvp", -1e-5),
    ("cg_tol", 0.0),
])
def test_newton_options_validate_rejects_bad_values(field, bad):
    opts = NewtonOptions(**{field: bad})
    with pytest.raises(ValueError):
        opts.validate()


def test_newton_options_importable_from_config_package():
    from gpurec.config import NewtonOptions as ReExported
    assert ReExported is NewtonOptions


def test_newton_wired_into_curvature_functions():
    """The 4 curvature files' newton_*/certify_* entry points must accept a `newton: NewtonOptions`
    override, with the previously copy-pasted individual kwargs turned into None sentinels
    (deprecation shim) that resolve to NewtonOptions() defaults -- so existing callers passing
    e.g. `max_newton=4` explicitly keep working unchanged."""
    from gpurec.solver import curvature, genewise_curvature, origination_curvature, receiver_curvature

    sig = inspect.signature(curvature.newton_min).parameters
    assert "newton" in sig and sig["newton"].default is None

    shared_fields = ("sigma", "sigma_floor", "lanczos_m", "nu", "max_bumps", "max_cg", "c1",
                     "ls_max", "gtol", "max_newton", "ftol", "seed")
    for mod, fn_name in [
        (origination_curvature, "newton_joint"),
        (receiver_curvature, "newton_joint"),
        (genewise_curvature, "newton_joint_genewise"),
    ]:
        fn = getattr(mod, fn_name)
        params = inspect.signature(fn).parameters
        assert "newton" in params and params["newton"].default is None
        for name in shared_fields:
            assert params[name].default is None, f"{fn_name}.{name} default should be None"

    # `decrease` (origination/genewise) / `omega` (receiver, back-compat alias) map to
    # NewtonOptions.decrease -- both must be None sentinels too.
    assert inspect.signature(receiver_curvature.newton_joint).parameters["omega"].default is None
    assert inspect.signature(origination_curvature.newton_joint).parameters["decrease"].default is None
    assert inspect.signature(
        genewise_curvature.newton_joint_genewise).parameters["decrease"].default is None
