from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurec.api.model import GeneReconModel
from gpurec.backtracking import (
    ensure_backtracking_available,
    recphyloxml_event_counts,
    sample_recphyloxmls,
)
from gpurec.recphyloxml import (
    EVENT_KEYS,
    direct_children,
    events_for_clade,
    iter_rec_gene_tree_clades,
    local_name,
)

from .checkpoint import (
    load_checkpoint,
    restore_model_theta,
    validate_checkpoint_model_compatibility,
)
from .config import RunConfig, SamplingConfig, _UINT64_MAX
from .model_factory import build_alerax_workflow_model


SPECIES_COLUMNS = (
    "speciations",
    "duplications",
    "losses",
    "transfers",
    "presence",
    "origination",
    "copies",
    "singletons",
    "transfers_to",
)

_UNSAFE_FILENAME_STEM = re.compile(r"[^A-Za-z0-9._-]+")
_SAMPLING_AGGREGATE_FILES = (
    "event_counts.tsv",
    "summary.json",
    "totalSpeciesEventCounts.txt",
    "totalTransfers.txt",
)
_SAMPLING_ALL_PATTERNS = (
    "*_eventCounts_*.txt",
    "*_sample_*.xml",
)


@dataclass
class SamplingResult:
    out_dir: Path
    families_sampled: int
    samples_per_family: int
    xml_files: int


def _family_file_stem(family_index: int, family_name: str) -> str:
    safe_name = _UNSAFE_FILENAME_STEM.sub("_", str(family_name)).strip("._-")
    if not safe_name:
        safe_name = "family"
    return f"{int(family_index):06d}_{safe_name}"


def _xml_species_and_transfer_counts(
    xml: str,
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    species_counts: dict[str, dict[str, float]] = defaultdict(
        lambda: {column: 0.0 for column in SPECIES_COLUMNS}
    )
    transfers: dict[tuple[str, str], float] = defaultdict(float)
    saw_origin = False

    def species_row(species: str) -> dict[str, float]:
        return species_counts[str(species)]

    def visit(clade: Any, active_donor: str | None = None) -> None:
        nonlocal saw_origin
        events = events_for_clade(clade)
        donor = active_donor
        branch_donor: str | None = None
        destination: str | None = None
        for event in events:
            name = local_name(event.tag)
            if name == "speciation":
                species = event.attrib.get("speciesLocation")
                if species:
                    species_row(species)["speciations"] += 1.0
                    species_row(species)["presence"] = 1.0
                    if not saw_origin:
                        species_row(species)["origination"] += 1.0
                        saw_origin = True
            elif name == "duplication":
                species = event.attrib.get("speciesLocation")
                if species:
                    species_row(species)["duplications"] += 1.0
                    species_row(species)["presence"] = 1.0
            elif name == "loss":
                species = event.attrib.get("speciesLocation")
                if species:
                    species_row(species)["losses"] += 1.0
            elif name == "branchingOut":
                branch_donor = event.attrib.get("speciesLocation")
                donor = branch_donor
                if branch_donor:
                    species_row(branch_donor)["transfers"] += 1.0
                    species_row(branch_donor)["presence"] = 1.0
            elif name == "transferBack":
                destination = event.attrib.get("destinationSpecies")
                if destination:
                    species_row(destination)["transfers_to"] += 1.0
                    species_row(destination)["presence"] = 1.0
            elif name == "leaf":
                species = event.attrib.get("speciesLocation")
                if species:
                    row = species_row(species)
                    row["presence"] = 1.0
                    row["copies"] += 1.0
        if donor and destination:
            transfers[(donor, destination)] += 1.0
        child_donor = branch_donor or active_donor
        for child in direct_children(clade, "clade"):
            visit(child, child_donor)

    for clade in iter_rec_gene_tree_clades(xml):
        visit(clade)

    for row in species_counts.values():
        row["singletons"] = 1.0 if row["copies"] == 1.0 else 0.0
    return dict(species_counts), dict(transfers)


def _write_event_counts(path: Path, counts: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key in EVENT_KEYS:
            handle.write(f"{key}:{int(counts.get(key, 0))}\n")


def _write_total_species_counts(
    path: Path,
    species_counts: dict[str, dict[str, float]],
    *,
    divisor: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("species_label, " + ", ".join(SPECIES_COLUMNS) + "\n")
        for species in sorted(species_counts):
            row = species_counts[species]
            values = [row.get(column, 0.0) / max(divisor, 1) for column in SPECIES_COLUMNS]
            handle.write(
                f"{species}, "
                + ", ".join(f"{value:.12g}" for value in values)
                + "\n"
            )


def _write_total_transfers(
    path: Path,
    transfers: dict[tuple[str, str], float],
    *,
    divisor: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(transfers.items(), key=lambda item: (-item[1], item[0]))
    with path.open("w", encoding="utf-8") as handle:
        for (donor, recipient), count in rows:
            value = count / max(divisor, 1)
            if value > 0.0:
                handle.write(f"{donor} {recipient} {value:.12g}\n")


def _write_event_counts_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["family", "sample", *EVENT_KEYS])
        for row in rows:
            writer.writerow(
                [row["family"], row["sample"]]
                + [int(row.get(key, 0)) for key in EVENT_KEYS]
            )


def _generated_sampling_outputs(out_dir: Path) -> list[Path]:
    recon_dir = out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    paths: list[Path] = []
    if all_dir.exists():
        for pattern in _SAMPLING_ALL_PATTERNS:
            paths.extend(
                sorted(path for path in all_dir.glob(pattern) if path.is_file())
            )
    for name in _SAMPLING_AGGREGATE_FILES:
        path = recon_dir / name
        if path.is_file():
            paths.append(path)
    return paths


def _clear_sampling_outputs(out_dir: Path) -> None:
    for path in _generated_sampling_outputs(out_dir):
        path.unlink()
    (out_dir / "reconciliations" / "all").mkdir(parents=True, exist_ok=True)


def _create_sampling_temp_dir(recon_dir: Path, *, prefix: str) -> Path:
    recon_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=recon_dir))


def _cleanup_sampling_temp_dir(path: Path | None) -> OSError | None:
    if path is None:
        return None
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc
    return None


def _move_sampling_outputs_to_backup(
    out_dir: Path,
    backup_dir: Path,
) -> list[tuple[Path, Path]]:
    recon_dir = out_dir / "reconciliations"
    moved: list[tuple[Path, Path]] = []
    try:
        for final_path in _generated_sampling_outputs(out_dir):
            backup_path = backup_dir / final_path.relative_to(recon_dir)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.replace(backup_path)
            moved.append((backup_path, final_path))
    except BaseException as exc:
        restore_error: BaseException | None = None
        for backup_path, final_path in reversed(moved):
            try:
                if backup_path.is_file():
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    backup_path.replace(final_path)
            except BaseException as rollback_error:
                if restore_error is None:
                    restore_error = rollback_error
        if restore_error is not None:
            raise exc from restore_error
        raise
    return moved


def _restore_sampling_backup(moved: list[tuple[Path, Path]]) -> None:
    for backup_path, final_path in reversed(moved):
        if backup_path.is_file():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.replace(final_path)


def _publish_sampling_outputs(
    out_dir: Path,
    staged_outputs: list[tuple[Path, Path]],
) -> None:
    recon_dir = out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    backup_dir = _create_sampling_temp_dir(
        recon_dir,
        prefix=".gpurec-sampling-backup-",
    )
    try:
        moved = _move_sampling_outputs_to_backup(out_dir, backup_dir)
    except BaseException:
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise

    try:
        all_dir.mkdir(parents=True, exist_ok=True)
        for staged_path, final_path in staged_outputs:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(final_path)
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        try:
            _clear_sampling_outputs(out_dir)
        except BaseException as clear_error:
            cleanup_error = clear_error
        try:
            _restore_sampling_backup(moved)
        except BaseException as restore_error:
            if cleanup_error is None:
                cleanup_error = restore_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        if cleanup_error is not None:
            raise exc from cleanup_error
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _validate_sampling_seed_range(
    config: SamplingConfig,
    *,
    start: int,
    stop: int,
) -> None:
    last_family_index = stop - 1
    max_seed = config.seed + last_family_index * config.samples + config.samples - 1
    if max_seed > _UINT64_MAX:
        raise ValueError(
            "sampling seed range exceeds u64 maximum: "
            f"seed={config.seed}, last_family_index={last_family_index}, "
            f"samples={config.samples}, max_seed={max_seed}"
        )


def _close_model_after_error(model: Any, primary_error: BaseException) -> None:
    try:
        model.close()
    except Exception as close_error:
        raise primary_error from close_error


class SamplingRunner:
    def __init__(self, config: SamplingConfig):
        self.config = config

    def _load_model(self) -> tuple[RunConfig, GeneReconModel]:
        payload = load_checkpoint(self.config.checkpoint, map_location="cpu")
        run_config = RunConfig.from_dict(payload["config"])
        model = build_alerax_workflow_model(
            run_config,
            refresh_preprocess_cache=False,
            prefetch_batches=0,
        )
        try:
            validate_checkpoint_model_compatibility(
                path=self.config.checkpoint,
                config=run_config,
                model=model,
                payload=payload,
            )
            restore_model_theta(model, payload)
        except BaseException as exc:
            _close_model_after_error(model, exc)
            raise
        return run_config, model

    def run(self) -> SamplingResult:
        ensure_backtracking_available(self.config.backtrack_binary)
        run_config, model = self._load_model()
        stage_dir: Path | None = None
        try:
            out_dir = self.config.out_dir or run_config.out_dir
            recon_dir = out_dir / "reconciliations"
            all_dir = recon_dir / "all"

            family_names = model.family_names
            start = self.config.family_start
            stop = len(family_names)
            if self.config.max_families is not None:
                stop = min(stop, start + self.config.max_families)
            if start >= stop:
                raise ValueError(
                    f"empty sampling family selection: start={start}, stop={stop}, "
                    f"available={len(family_names)}"
                )
            _validate_sampling_seed_range(self.config, start=start, stop=stop)

            stage_dir = _create_sampling_temp_dir(
                recon_dir,
                prefix=".gpurec-sampling-stage-",
            )
            stage_all_dir = stage_dir / "all"
            stage_all_dir.mkdir(parents=True, exist_ok=True)
            staged_outputs: list[tuple[Path, Path]] = []

            species_totals: dict[str, dict[str, float]] = defaultdict(
                lambda: {column: 0.0 for column in SPECIES_COLUMNS}
            )
            transfer_totals: dict[tuple[str, str], float] = defaultdict(float)
            event_rows: list[dict[str, Any]] = []
            xml_count = 0

            for family_index in range(start, stop):
                family = family_names[family_index]
                family_file_stem = _family_file_stem(family_index, family)
                xmls = sample_recphyloxmls(
                    model,
                    family_index=family_index,
                    num_samples=self.config.samples,
                    seed=self.config.seed + family_index * self.config.samples,
                    max_events=self.config.max_events,
                    backtrack_binary=self.config.backtrack_binary,
                )
                for sample_index, xml in enumerate(xmls):
                    xml_path = all_dir / f"{family_file_stem}_sample_{sample_index}.xml"
                    staged_xml_path = stage_all_dir / xml_path.name
                    staged_outputs.append((staged_xml_path, xml_path))
                    staged_xml_path.write_text(xml, encoding="utf-8")
                    xml_count += 1

                    event_counts = recphyloxml_event_counts(xml)
                    event_counts_path = (
                        all_dir / f"{family_file_stem}_eventCounts_{sample_index}.txt"
                    )
                    staged_event_counts_path = stage_all_dir / event_counts_path.name
                    staged_outputs.append((staged_event_counts_path, event_counts_path))
                    _write_event_counts(staged_event_counts_path, event_counts)
                    event_rows.append(
                        {"family": family, "sample": sample_index, **event_counts}
                    )

                    species_counts, transfers = _xml_species_and_transfer_counts(xml)
                    for species, row in species_counts.items():
                        target = species_totals[species]
                        for key, value in row.items():
                            target[key] += value
                    for pair, value in transfers.items():
                        transfer_totals[pair] += value

            event_counts_table = recon_dir / "event_counts.tsv"
            staged_event_counts_table = stage_dir / "event_counts.tsv"
            staged_outputs.append((staged_event_counts_table, event_counts_table))
            _write_event_counts_table(staged_event_counts_table, event_rows)
            total_species_path = recon_dir / "totalSpeciesEventCounts.txt"
            staged_total_species_path = stage_dir / "totalSpeciesEventCounts.txt"
            staged_outputs.append((staged_total_species_path, total_species_path))
            _write_total_species_counts(
                staged_total_species_path,
                dict(species_totals),
                divisor=self.config.samples,
            )
            total_transfers_path = recon_dir / "totalTransfers.txt"
            staged_total_transfers_path = stage_dir / "totalTransfers.txt"
            staged_outputs.append((staged_total_transfers_path, total_transfers_path))
            _write_total_transfers(
                staged_total_transfers_path,
                dict(transfer_totals),
                divisor=self.config.samples,
            )
            summary = {
                "families_sampled": stop - start,
                "samples_per_family": self.config.samples,
                "xml_files": xml_count,
                "checkpoint": str(self.config.checkpoint),
                "out_dir": str(out_dir),
            }
            summary_path = recon_dir / "summary.json"
            staged_summary_path = stage_dir / "summary.json"
            staged_outputs.append((staged_summary_path, summary_path))
            staged_summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _publish_sampling_outputs(out_dir, staged_outputs)
            cleanup_error = _cleanup_sampling_temp_dir(stage_dir)
            stage_dir = None
            if cleanup_error is not None:
                raise cleanup_error
            result = SamplingResult(
                out_dir=out_dir,
                families_sampled=stop - start,
                samples_per_family=self.config.samples,
                xml_files=xml_count,
            )
        except BaseException as exc:
            cleanup_error = _cleanup_sampling_temp_dir(stage_dir)
            try:
                model.close()
            except Exception as close_error:
                raise exc from close_error
            if cleanup_error is not None:
                raise exc from cleanup_error
            raise
        else:
            model.close()
            return result


def sample(config: SamplingConfig) -> SamplingResult:
    return SamplingRunner(config).run()
