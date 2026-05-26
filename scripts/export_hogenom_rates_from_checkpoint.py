#!/usr/bin/env python3
"""Checkout-local HOGENOM checkpoint rate exporter.

This is a local analysis utility for extracting rate tables from historical
HOGENOM checkpoints. Supported production artifact inspection should use the
installed ``gpurec checkpoint-info`` and ``gpurec summary-info`` commands, or
the files written by ``gpurec.workflow`` runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TreeNode:
    name: str
    children: list["TreeNode"]


def parse_newick(path: Path) -> TreeNode:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty Newick file: {path}")
    idx = 0

    def skip_ws() -> None:
        nonlocal idx
        while idx < len(text) and text[idx].isspace():
            idx += 1

    def read_label() -> str:
        nonlocal idx
        skip_ws()
        if idx < len(text) and text[idx] in "'\"":
            quote = text[idx]
            idx += 1
            start = idx
            while idx < len(text) and text[idx] != quote:
                idx += 1
            label = text[start:idx]
            if idx < len(text):
                idx += 1
            return label
        start = idx
        while idx < len(text) and text[idx] not in ":,();":
            idx += 1
        return text[start:idx].strip()

    def read_length() -> None:
        nonlocal idx
        skip_ws()
        if idx >= len(text) or text[idx] != ":":
            return
        idx += 1
        while idx < len(text) and text[idx] not in ",();":
            idx += 1

    def parse_subtree() -> TreeNode:
        nonlocal idx
        skip_ws()
        if idx < len(text) and text[idx] == "(":
            idx += 1
            children: list[TreeNode] = []
            while True:
                children.append(parse_subtree())
                skip_ws()
                if idx < len(text) and text[idx] == ",":
                    idx += 1
                    continue
                if idx < len(text) and text[idx] == ")":
                    idx += 1
                    break
                raise ValueError(f"invalid Newick near offset {idx} in {path}")
            label = read_label()
            read_length()
            return TreeNode(label, children)
        label = read_label()
        read_length()
        return TreeNode(label, [])

    root = parse_subtree()
    skip_ws()
    if idx < len(text) and text[idx] == ";":
        idx += 1
    skip_ws()
    if idx != len(text):
        raise ValueError(f"trailing Newick content near offset {idx} in {path}")
    return root


def is_numeric(text: str) -> bool:
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def unique_label(seen: set[str], base: str) -> str:
    suffix = 0
    while True:
        label = f"{base}_{suffix}"
        if label not in seen:
            return label
        suffix += 1


def species_order_labels(root: TreeNode) -> list[str]:
    """Replicate the postorder labels used by gpurec/AleRax species helpers."""
    labels: list[str] = []
    seen: set[str] = set()

    def visit(node: TreeNode) -> tuple[str, str]:
        if not node.children:
            label = node.name
            labels.append(label)
            seen.add(label)
            return label, label

        child_leaf_labels = [visit(child)[1] for child in node.children]
        label = node.name or ""
        if (not label) or label in seen or is_numeric(label):
            if len(child_leaf_labels) >= 2:
                base = f"Node_{child_leaf_labels[0]}_{child_leaf_labels[1]}"
            else:
                base = f"Node_{child_leaf_labels[0]}"
            label = unique_label(seen, base)
        labels.append(label)
        seen.add(label)
        return label, child_leaf_labels[0]

    visit(root)
    return labels


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"could not safely load checkpoint {path}; regenerate the artifact "
            "or migrate it from a trusted source before retrying"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint {path} must contain a dictionary payload")
    return checkpoint


def _checkpoint_tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"checkpoint is missing tensor {key!r}")
    value = payload[key]
    if not torch.is_tensor(value):
        raise ValueError(f"checkpoint tensor {key!r} must be a tensor")
    return value.detach().cpu()


def _reshape_theta_rows(theta: torch.Tensor, *, key: str) -> torch.Tensor:
    if theta.numel() == 0 or theta.numel() % 3 != 0:
        raise ValueError(f"checkpoint tensor {key!r} must contain D/L/T triples")
    return theta.reshape(-1, 3)


def load_effective_theta(
    checkpoint: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    branch = checkpoint.get("branchscaled")
    if branch is not None:
        if not isinstance(branch, dict):
            raise ValueError("checkpoint branchscaled payload must be a dictionary")
        shared_theta = _checkpoint_tensor(branch, "shared_theta")
        if shared_theta.numel() != 3:
            raise ValueError("checkpoint tensor 'shared_theta' must contain one D/L/T triple")
        shared_theta = shared_theta.reshape(3)
        branch_log_l = _checkpoint_tensor(branch, "branch_log_l").reshape(-1)
        if branch_log_l.numel() == 0:
            raise ValueError("checkpoint tensor 'branch_log_l' must not be empty")
        theta = shared_theta.reshape(1, 3) + branch_log_l.reshape(-1, 1) / math.log(2.0)
        return theta, {
            "shared_theta": shared_theta,
            "branch_log_l": branch_log_l,
        }
    theta = _reshape_theta_rows(_checkpoint_tensor(checkpoint, "theta"), key="theta")
    return theta, None


def write_rates(path: Path, labels: list[str], theta: torch.Tensor, branch: dict[str, torch.Tensor] | None) -> None:
    theta_rows = int(theta.shape[0])
    if theta_rows not in {1, len(labels)}:
        raise ValueError(f"theta rows ({theta_rows}) do not match species-tree labels ({len(labels)})")
    if branch is not None and int(branch["branch_log_l"].numel()) != len(labels):
        raise ValueError(
            f"branchscaled branch rows ({int(branch['branch_log_l'].numel())}) do "
            f"not match species-tree labels ({len(labels)})"
        )

    rates = torch.exp2(theta)
    p_s = 1.0 / (1.0 + rates.sum(dim=1))

    columns = ["row", "label", "D", "T", "L", "pS", "theta_D", "theta_T", "theta_L"]
    if branch is not None:
        columns.extend(
            [
                "l_e",
                "log_l_e",
                "shared_D",
                "shared_T",
                "shared_L",
                "shared_theta_D",
                "shared_theta_T",
                "shared_theta_L",
            ]
        )
        branch_log_l = branch["branch_log_l"]
        branch_l = torch.exp(branch_log_l)
        shared_theta = branch["shared_theta"]
        shared_rates = torch.exp2(shared_theta)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row, label in enumerate(labels):
            theta_row = 0 if theta_rows == 1 else row
            output_row = [
                row,
                label,
                float(rates[theta_row, 0]),
                float(rates[theta_row, 2]),
                float(rates[theta_row, 1]),
                float(p_s[theta_row]),
                float(theta[theta_row, 0]),
                float(theta[theta_row, 2]),
                float(theta[theta_row, 1]),
            ]
            if branch is not None:
                output_row.extend(
                    [
                        float(branch_l[row]),
                        float(branch_log_l[row]),
                        float(shared_rates[0]),
                        float(shared_rates[2]),
                        float(shared_rates[1]),
                        float(shared_theta[0]),
                        float(shared_theta[2]),
                        float(shared_theta[1]),
                    ]
                )
            writer.writerow(output_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export D/T/L rates from an optimize_hogenom_ccp_wandb.py checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        checkpoint = load_checkpoint(args.checkpoint)
        labels = species_order_labels(parse_newick(args.tree))
        theta, branch = load_effective_theta(checkpoint)
        write_rates(args.out, labels, theta, branch)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {args.out} from step={checkpoint.get('step', 'unknown')} "
        f"phase={checkpoint.get('optimizer_phase', 'unknown')}"
    )


if __name__ == "__main__":
    main()
