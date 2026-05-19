from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_backtracking_alerax_events import load_rates


def _rate_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output"
    (output_dir / "model_parameters").mkdir(parents=True)
    return output_dir


def test_compare_backtracking_load_rates_reads_global_model_parameters(
    tmp_path: Path,
):
    output_dir = _rate_output_dir(tmp_path)
    (output_dir / "model_parameters" / "model_parameters.txt").write_text(
        "node D L T\n16 0.1 0.2 0.3\n",
        encoding="utf-8",
    )

    assert load_rates(output_dir) == (0.1, 0.2, 0.3)


def test_compare_backtracking_load_rates_prefers_family_specific_rates(
    tmp_path: Path,
):
    output_dir = _rate_output_dir(tmp_path)
    (output_dir / "model_parameters" / "model_parameters.txt").write_text(
        "node D L T\n16 0.1 0.2 0.3\n",
        encoding="utf-8",
    )
    (output_dir / "model_parameters" / "family_0001_rates.txt").write_text(
        "D L T\n0.4 0.5 0.6\n",
        encoding="utf-8",
    )

    assert load_rates(output_dir, "family_0001") == (0.4, 0.5, 0.6)


def test_compare_backtracking_load_rates_reports_missing_global_file(
    tmp_path: Path,
):
    output_dir = _rate_output_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="model_parameters.txt"):
        load_rates(output_dir)


@pytest.mark.parametrize(
    ("filename", "text", "family_name", "message"),
    [
        (
            "model_parameters.txt",
            "node D L T\n",
            None,
            "header row and at least one rate row",
        ),
        (
            "model_parameters.txt",
            "node D L T\n16 0.1 0.2\n",
            None,
            "node D L T",
        ),
        (
            "model_parameters.txt",
            "node D L T\n16 bad 0.2 0.3\n",
            None,
            "could not parse D/L/T rates",
        ),
        (
            "family_0001_rates.txt",
            "D L T\n0.1 0.2\n",
            "family_0001",
            "D L T",
        ),
        (
            "family_0001_rates.txt",
            "D L T\n0.1 bad 0.3\n",
            "family_0001",
            "could not parse D/L/T rates",
        ),
    ],
)
def test_compare_backtracking_load_rates_reports_malformed_files(
    tmp_path: Path,
    filename: str,
    text: str,
    family_name: str | None,
    message: str,
):
    output_dir = _rate_output_dir(tmp_path)
    path = output_dir / "model_parameters" / filename
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as exc_info:
        load_rates(output_dir, family_name)

    assert str(path) in str(exc_info.value)
