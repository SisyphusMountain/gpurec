"""Fail-closed first-gradient reseed adapters; production source remains untouched."""
from __future__ import annotations

import hashlib
import inspect

import gpurec.fit.genewise_fit as production


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"reseed adapter expected one {label} site, found {count}")
    return source.replace(old, new)


def compile_native_reseed(reseeder):
    source = inspect.getsource(production.fit_genewise)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    source = _replace_once(
        source,
        '''                _track_best(active, lv, sub, live, n_steps)
                # Trust-region ratio test on the step that led here (see ``trust_max`` above).
''',
        '''                _track_best(active, lv, sub, live, n_steps)
                if pi_idx == 0 and it == 0:
                    B_fam.index_copy_(0, active, _fresh_reseed(sub, g))
                # Trust-region ratio test on the step that led here (see ``trust_max`` above).
''',
        "native first-gradient reseed",
    )
    namespace = dict(vars(production))
    namespace["_fresh_reseed"] = reseeder
    exec(compile(source, f"<native-reseed:{source_hash[:12]}>", "exec"), namespace)
    fit = namespace["fit_genewise"]
    return fit, source_hash


def compile_hierarchical_reseed(reseeder):
    # Reuse the already reviewed experiment-only hierarchy transformation, injecting one extra
    # statement after it has constructed phi and g_phi at the paid first Newton evaluation.
    import source_adapter as hierarchy

    original_replace = hierarchy._replace_once

    def intercept(source: str, old: str, new: str, label: str) -> str:
        result = original_replace(source, old, new, label)
        if label == "point gradient transform":
            augmented = new + '''                if pi_idx == 0 and it == 0:
                    B_fam.index_copy_(0, active, _fresh_reseed(sub, g))
'''
            result = _replace_once(result, new, augmented, "hierarchical first-gradient reseed")
        return result

    hierarchy._replace_once = intercept
    try:
        fit, source_hash = hierarchy._compile_hierarchical_fit("native")
    finally:
        hierarchy._replace_once = original_replace
    fit.__globals__["_fresh_reseed"] = reseeder
    return fit, source_hash

