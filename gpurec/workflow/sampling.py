from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel
from gpurec.backtracking import recphyloxml_event_counts, sample_recphyloxmls

from .checkpoint import load_checkpoint, restore_model_theta
from .config import RunConfig, SamplingConfig


EVENT_KEYS = ("S", "SL", "D", "DL", "T", "TL", "L", "Leaf")
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
                    row["singletons"] += 1.0
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


class SamplingRunner:
    def __init__(self, config: SamplingConfig):
        self.config = config

    def _load_model(self) -> tuple[RunConfig, GeneReconModel]:
        payload = load_checkpoint(self.config.checkpoint, map_location="cpu")
        run_config = RunConfig.from_dict(payload["config"])
        if not str(run_config.device).startswith("cuda"):
            raise RuntimeError("gpurec production sampling currently requires CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        model = GeneReconModel.from_alerax_families(
            str(run_config.species_tree),
            run_config.families_file,
            mode=run_config.mode,
            start=run_config.start,
            max_families=run_config.max_families,
            device=run_config.device,
            dtype=run_config.torch_dtype,
            theta_init_rates=run_config.theta_init_rates,
            preprocess_cache_dir=run_config.preprocess_cache,
            fixed_iters_E=run_config.fixed_iters_e,
            max_iters_E=run_config.max_iters_e,
            tol_E=run_config.tol_e,
            fixed_iters_Pi=run_config.fixed_iters_pi,
            neumann_terms=run_config.neumann_terms,
            adaptive_iters=run_config.adaptive_iters,
            convergence_check_interval=run_config.convergence_check_interval,
            e_logsumexp_tol=run_config.e_logsumexp_tol,
            pi_max_diff_tol=run_config.pi_max_diff_tol,
            gradient_change_tol=run_config.gradient_change_tol,
            gradient_change_rtol=run_config.gradient_change_rtol,
            family_chunk_size=run_config.family_chunk_size,
            clade_budget=run_config.clade_budget,
            batch_packing=run_config.batch_packing,
            max_wave_size=run_config.max_wave_size,
            lazy_preprocess=True,
            prefetch_batches=0,
        )
        restore_model_theta(model, payload)
        return run_config, model

    def run(self) -> SamplingResult:
        run_config, model = self._load_model()
        out_dir = self.config.out_dir or run_config.out_dir
        all_dir = out_dir / "reconciliations" / "all"
        all_dir.mkdir(parents=True, exist_ok=True)

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

        species_totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {column: 0.0 for column in SPECIES_COLUMNS}
        )
        transfer_totals: dict[tuple[str, str], float] = defaultdict(float)
        event_rows: list[dict[str, Any]] = []
        xml_count = 0

        try:
            for family_index in range(start, stop):
                family = family_names[family_index]
                xmls = sample_recphyloxmls(
                    model,
                    family_index=family_index,
                    num_samples=self.config.samples,
                    seed=self.config.seed + family_index * self.config.samples,
                    max_events=self.config.max_events,
                    backtrack_binary=self.config.backtrack_binary,
                )
                for sample_index, xml in enumerate(xmls):
                    xml_path = all_dir / f"{family}_sample_{sample_index}.xml"
                    xml_path.write_text(xml, encoding="utf-8")
                    xml_count += 1

                    event_counts = recphyloxml_event_counts(xml)
                    _write_event_counts(
                        all_dir / f"{family}_eventCounts_{sample_index}.txt",
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
