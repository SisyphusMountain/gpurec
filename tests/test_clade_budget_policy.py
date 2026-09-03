"""``clade_budget_for_device`` sizes a fit's batches to the card it is running on.

The genewise fit's peak GPU memory is one batch's transient [clades x species] buffers plus the
per-batch static index/probability arrays that stay resident for every batch. The first scales with
the per-batch clade budget, the second with the dataset. These tests pin the three properties the
fit relies on: a card with room keeps the tuned batch size (so nothing changes where nothing needs
to), a small card gets a budget whose predicted peak actually fits, and a dataset whose statics
alone overflow the card fails immediately with a message that says so rather than OOM-ing later.

Reference measurement throughout (job 57680456, H100 NVL, the full Coleman set): 5123 families,
22,926,670 clades, 60,422,454 splits, S=2013, fp32, 79 batches of <=315,000 clades, peak 23.85 GiB
of which 2.22 GiB was static -- 1.77 GiB once the unread fp64 split-probability copy is no longer
materialized.
"""
import pytest
import torch

from gpurec.core import memory_policy
from gpurec.core.memory_policy import (
    GIB,
    batch_static_bytes,
    batch_working_set_bytes,
    clade_budget_for_device,
)
from gpurec.core.scheduling.batching import DEFAULT_CLADE_BUDGET

COLEMAN_CLADES = 22_926_670
COLEMAN_SPLITS = 60_422_454
COLEMAN_S = 2013
SCRATCH_TENSORS = 10


def _budget(monkeypatch, budget_bytes, **overrides):
    """Run the policy against a stubbed device budget, so no CUDA device is needed."""
    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", lambda device=None: budget_bytes)
    kwargs = dict(
        total_clades=COLEMAN_CLADES, total_splits=COLEMAN_SPLITS, S=COLEMAN_S,
        dtype=torch.float32, device=None, fixed_clade_budget=DEFAULT_CLADE_BUDGET,
        scratch_tensors=SCRATCH_TENSORS,
    )
    kwargs.update(overrides)
    return clade_budget_for_device(**kwargs)


def test_static_bytes_match_the_measured_coleman_statics():
    """The per-clade / per-split byte constants must reproduce the 1.77 GiB actually measured."""
    predicted = batch_static_bytes(COLEMAN_CLADES, COLEMAN_SPLITS) / GIB
    assert predicted == pytest.approx(1.77, abs=0.10)


def test_static_bytes_do_not_depend_on_how_clades_are_batched():
    """Statics scale with the dataset, not the batch size -- that is why they are a separate term."""
    assert batch_static_bytes(0, 0) == 0
    assert batch_static_bytes(2 * COLEMAN_CLADES, 2 * COLEMAN_SPLITS) == \
        2 * batch_static_bytes(COLEMAN_CLADES, COLEMAN_SPLITS)


def test_working_set_is_linear_in_the_clade_budget():
    half = batch_working_set_bytes(157_500, COLEMAN_S, torch.float32, scratch_tensors=SCRATCH_TENSORS)
    full = batch_working_set_bytes(315_000, COLEMAN_S, torch.float32, scratch_tensors=SCRATCH_TENSORS)
    assert full == 2 * half


def test_a_large_card_keeps_the_tuned_budget(monkeypatch):
    """94 GiB H100: 315,000 fits with room to spare, so the fit must be the one it always was."""
    chosen, detail = _budget(monkeypatch, int(79.1 * GIB))
    assert chosen == DEFAULT_CLADE_BUDGET
    assert detail["automatic"] is False


def test_the_budget_is_never_raised_above_the_tuned_value(monkeypatch):
    """Even an absurdly large card must not grow the batch past the size the packer is tuned for."""
    chosen, _ = _budget(monkeypatch, 1000 * GIB)
    assert chosen == DEFAULT_CLADE_BUDGET


def test_a_24gb_card_gets_a_smaller_budget_that_fits(monkeypatch):
    """A 24 GB RTX 4090 with ~19 GiB free must shrink the batch, and the prediction must fit."""
    device_budget = int(18.25 * GIB)
    chosen, detail = _budget(monkeypatch, device_budget)
    assert chosen < DEFAULT_CLADE_BUDGET
    assert detail["automatic"] is True
    # The allocated peak it predicts must leave room for the caching allocator's larger reservation.
    assert detail["predicted_peak_bytes"] * memory_policy._RESERVED_PER_ALLOCATED <= device_budget + 1
    assert detail["predicted_peak_bytes"] == detail["static_bytes"] + detail["working_set_bytes"]


def test_no_cuda_device_keeps_the_tuned_budget(monkeypatch):
    """With no budget to size against (CPU run), the tuned value stands."""
    chosen, detail = _budget(monkeypatch, None)
    assert chosen == DEFAULT_CLADE_BUDGET
    assert detail["automatic"] is False


def test_statics_larger_than_the_card_fail_loudly(monkeypatch):
    """No batch size rescues a dataset whose resident statics alone overflow the card."""
    with pytest.raises(ValueError, match="no per-batch clade budget can make this fit"):
        _budget(monkeypatch, int(1.0 * GIB))


def test_an_explicit_budget_is_used_as_given(monkeypatch):
    """A caller-supplied fixed_clade_budget is the ceiling, whatever the tuned default is."""
    chosen, _ = _budget(monkeypatch, 1000 * GIB, fixed_clade_budget=50_000)
    assert chosen == 50_000
