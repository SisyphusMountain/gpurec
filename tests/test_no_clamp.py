"""Ensure clamp/clip never appear in the codebase.

NEVER use clamp, clamp_, clip, or clip_ on tensors. It silently corrupts
numerical results. Use _safe_log2 or handle the math correctly instead.

The only allowed exception is theta_optimizer.py which clamps optimizer
parameters to a minimum bound (not log-space numerics).
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

ALLOWED = {
    "src/optimization/theta_optimizer.py",
}

PATTERN = re.compile(r"\.(clamp|clamp_|clip|clip_)\(")


def test_no_clamp_or_clip():
    violations = []
    for py in sorted(SRC.rglob("*.py")):
        rel = str(py.relative_to(SRC.parent))
        if rel in ALLOWED:
            continue
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if PATTERN.search(line):
                violations.append(f"  {rel}:{i}: {line.strip()}")

    if violations:
        msg = (
            "NEVER use clamp/clip on tensors — it silently corrupts results.\n"
            "Use _safe_log2 or correct math instead.\n"
            "Violations:\n" + "\n".join(violations)
        )
        raise AssertionError(msg)
