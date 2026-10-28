"""Integration boundaries, using temporary files and synthetic data only."""
import copy
import json

import pandas as pd
import pytest

import configs
import confirmation
import data
import experiments
import run


@pytest.fixture
def guard_root(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    for name in confirmation.SOURCE_FILES:
        (tmp_path / name).write_text(f"synthetic source: {name}\n", encoding="utf-8")
    sales = tmp_path / "listings.bin"
    sales.write_bytes(b"synthetic training file; no real data is evaluated")
    monkeypatch.setattr(data, "find_file", lambda name: sales)
    monkeypatch.setattr(configs, "CONFIGS", copy.deepcopy(configs.CONFIGS))
    return tmp_path


def test_default_validation_keeps_confirmation_closed():
    assert list(map(str, run.VALIDATION_MONTHS)) == ["2025-12", "2026-01", "2026-02", "2026-03"]
    assert run.validation_month_label(pd.DataFrame({"month": ["2026-04", "2026-05"]})) == "2026-04 to 2026-05"


def test_confirmation_and_all_cannot_bypass_reservation():
    with pytest.raises(RuntimeError, match="reserved"):
        experiments.run(configs.FINAL["off_market"], "confirm", frozen=True)
    with pytest.raises(RuntimeError, match="disabled"):
        experiments.run(configs.FINAL["off_market"], "all")


def test_confirmation_requires_saved_freeze(guard_root):
    with pytest.raises(RuntimeError, match="freeze"):
        with confirmation.campaign(list(configs.FINAL.values())):
            pytest.fail("Must not evaluate any holdout")


@pytest.mark.parametrize("changed", ["source", "data", "config"])
def test_freeze_detects_changes(guard_root, changed):
    confirmation.freeze()
    if changed == "source":
        (guard_root / "models.py").write_text("changed source")
    elif changed == "data":
        (guard_root / "listings.bin").write_bytes(b"changed data")
    else:
        spec = configs.CONFIGS[configs.FINAL["off_market"]]["off_market"]
        spec["weights"] = [2] + [1] * (len(spec["members"]) - 1)
    with pytest.raises(RuntimeError, match="changed"):
        confirmation.validate_freeze()


def test_failed_confirmation_cannot_be_repeated(guard_root):
    confirmation.freeze()
    names = list(configs.FINAL.values())
    with pytest.raises(ValueError, match="synthetic failure"):
        with confirmation.campaign(names) as reservation:
            confirmation.require_reservation(reservation)
            raise ValueError("synthetic failure")
    record = json.loads((guard_root / "outputs/final_confirmation.json").read_text())
    assert record["status"] == "failed"
    with pytest.raises(RuntimeError, match="already been reserved"):
        with confirmation.campaign(names):
            pytest.fail("Must not evaluate twice")
    with pytest.raises(RuntimeError, match="already reserved"):
        confirmation.freeze()


def test_complete_confirmation_checks_saved_artifacts(guard_root):
    confirmation.freeze()
    names = list(configs.FINAL.values())
    with confirmation.campaign(names):
        for name in names:
            folder = guard_root / "outputs/experiments" / name / "confirm"
            folder.mkdir(parents=True)
            for filename in ("predictions.csv", "metrics.json", "spec.json"):
                (folder / filename).write_text("synthetic result\n")
    assert confirmation.completed_campaign()["status"] == "complete"
    (folder / "metrics.json").write_text("changed result\n")
    with pytest.raises(RuntimeError, match="artifact changed"):
        confirmation.completed_campaign()


def test_confirmation_rejects_partial_or_overridden_campaign(guard_root):
    confirmation.freeze()
    names = list(configs.FINAL.values())
    for selected, track, seed in [(names[:1], None, None), (names, "off_market", None),
                                  (names, None, 3)]:
        with pytest.raises(RuntimeError, match="both frozen"):
            with confirmation.campaign(selected, track=track, seed=seed):
                pytest.fail("Must not evaluate an altered campaign")



def test_imported_confirmation_blocks_new_campaign_without_a_local_freeze(guard_root):
    folder = guard_root / "outputs"
    folder.mkdir()
    (folder / "final_confirmation.json").write_text(json.dumps({
        "status": "complete", "origin": "imported_lab_confirmation"}))
    with pytest.raises(RuntimeError, match="already been reserved"):
        with confirmation.campaign(list(configs.FINAL.values())):
            pytest.fail("An imported completed confirmation must never rerun")
    with pytest.raises(RuntimeError, match="already reserved"):
        confirmation.freeze()


def test_imported_summary_cannot_masquerade_as_full_validation_predictions(guard_root):
    folder = guard_root / "outputs"
    folder.mkdir()
    (folder / "final_confirmation.json").write_text(json.dumps({
        "status": "complete", "origin": "imported_lab_confirmation"}))
    with pytest.raises(RuntimeError, match="Per-home predictions"):
        confirmation.completed_campaign()
