from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpurec.api.model import GeneReconModel
from gpurec.backtracking import (
    EVENT_KEYS,
    recphyloxml_event_counts,
    sample_recphyloxmls,
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


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _events_for_clade(clade: ET.Element) -> list[ET.Element]:
    for child in clade:
        if _local_name(child.tag) == "eventsRec":
            return list(child)
    return []


def _family_file_stem(family_index: int, family_name: str) -> str:
    safe_name = _UNSAFE_FILENAME_STEM.sub("_", str(family_name)).strip("._-")
    if not safe_name:
        safe_name = "family"
    return f"{int(family_index):06d}_{safe_name}"


def _xml_species_and_transfer_counts(
    xml: str,
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    root = ET.fromstring(xml)
    species_counts: dict[str, dict[str, float]] = defaultdict(
        lambda: {column: 0.0 for column in SPECIES_COLUMNS}
    )
    transfers: dict[tuple[str, str], float] = defaultdict(float)
    saw_origin = False

    def species_row(species: str) -> dict[str, float]:
        return species_counts[str(species)]

    def visit(clade: ET.Element, active_donor: str | None = None) -> None:
        nonlocal saw_origin
        events = _events_for_clade(clade)
        donor = active_donor
        branch_donor: str | None = None
        destination: str | None = None
        for event in events:
            name = _local_name(event.tag)
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
        for child in _direct_children(clade, "clade"):
            visit(child, child_donor)

    for rec_gene_tree in root.iter():
        if _local_name(rec_gene_tree.tag) != "recGeneTree":
            continue
        first_clade = next(
            (
                element
                for element in rec_gene_tree.iter()
                if _local_name(element.tag) == "clade"
            ),
            None,
        )
        if first_clade is not None:
            visit(first_clade)

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
        handle.write("family\tsample\t" + "\t".join(EVENT_KEYS) + "\n")
        for row in rows:
            handle.write(
                f"{row['family']}\t{row['sample']}\t"
                + "\t".join(str(int(row.get(key, 0))) for key in EVENT_KEYS)
                + "\n"
            )


def _clear_sampling_outputs(out_dir: Path) -> None:
    recon_dir = out_dir / "reconciliations"
    all_dir = recon_dir / "all"
    if all_dir.exists():
        for pattern in _SAMPLING_ALL_PATTERNS:
            for path in all_dir.glob(pattern):
                if path.is_file():
                    path.unlink()
    all_dir.mkdir(parents=True, exist_ok=True)
    for name in _SAMPLING_AGGREGATE_FILES:
        path = recon_dir / name
        if path.is_file():
            path.unlink()


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
        except Exception:
            model.close()
            raise
        return run_config, model

    def run(self) -> SamplingResult:
        run_config, model = self._load_model()
        try:
            out_dir = self.config.out_dir or run_config.out_dir
            all_dir = out_dir / "reconciliations" / "all"

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
            _clear_sampling_outputs(out_dir)

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
                    xml_path.write_text(xml, encoding="utf-8")
                    xml_count += 1

                    event_counts = recphyloxml_event_counts(xml)
                    _write_event_counts(
                        all_dir / f"{family_file_stem}_eventCounts_{sample_index}.txt",
                        event_counts,
                    )
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

            recon_dir = out_dir / "reconciliations"
            _write_event_counts_table(recon_dir / "event_counts.tsv", event_rows)
            _write_total_species_counts(
                recon_dir / "totalSpeciesEventCounts.txt",
                dict(species_totals),
                divisor=self.config.samples,
            )
            _write_total_transfers(
                recon_dir / "totalTransfers.txt",
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
            (recon_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return SamplingResult(
                out_dir=out_dir,
                families_sampled=stop - start,
                samples_per_family=self.config.samples,
                xml_files=xml_count,
            )
        finally:
            model.close()


def sample(config: SamplingConfig) -> SamplingResult:
    return SamplingRunner(config).run()
