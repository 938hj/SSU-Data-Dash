"""Collect official six-month validation from matching synthetic run artifacts."""
import json

import pandas as pd
import pytest

import configs
import confirmation
import data
import run


@pytest.fixture
def completed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    for name in confirmation.SOURCE_FILES:
        (tmp_path / name).write_text("synthetic source\n")
    sales = tmp_path / "listings.bin"
    sales.write_bytes(b"synthetic data")
    monkeypatch.setattr(data, "find_file", lambda name: sales)
    confirmation.freeze()
    identity = confirmation.snapshot()

    with confirmation.campaign(list(configs.FINAL.values())):
        for track, name in configs.FINAL.items():
            for split, months in [("dev", ["2025-12", "2026-01", "2026-02", "2026-03"]),
                                  ("confirm", ["2026-04", "2026-05"])]:
                folder = tmp_path / "outputs/experiments" / name / split
                folder.mkdir(parents=True)
                pd.DataFrame({"ml_num": [f"{track}-{month}" for month in months],
                              "track": track, "month": months, "sold_price": 100.,
                              "predicted_price": 102., "house_type_name": "Condo Apt"}).to_csv(
                                  folder / "predictions.csv", index=False)
                (folder / "spec.json").write_text(json.dumps(identity, default=str))
                (folder / "metrics.json").write_text("{}")
    return tmp_path


def test_collects_six_months_without_retraining(completed_runs, monkeypatch):
    import models
    monkeypatch.setattr(models, "make_model", lambda *args: pytest.fail("Collection must not train"))
    run.collect_validation()
    result = pd.read_csv(completed_runs / "outputs/validation_predictions.csv")
    assert len(result) == 12
    assert result.groupby("track")["month"].nunique().to_dict() == {
        "off_market": 6, "list_aware": 6}


@pytest.mark.parametrize("changed", ["fingerprint", "months"])
def test_rejects_incompatible_development_artifacts(completed_runs, changed):
    folder = completed_runs / "outputs/experiments" / configs.FINAL["off_market"] / "dev"
    if changed == "fingerprint":
        (folder / "spec.json").write_text("{}")
    else:
        frame = pd.read_csv(folder / "predictions.csv")
        frame.iloc[:-1].to_csv(folder / "predictions.csv", index=False)
    with pytest.raises(ValueError):
        run.collect_validation()


@pytest.fixture
def imported_runs(tmp_path, monkeypatch):
    """Like the lab: confirmation ran once before import (no fingerprints), dev runs here."""
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    for name in confirmation.SOURCE_FILES:
        (tmp_path / name).write_text("synthetic source\n")
    sales = tmp_path / "listings.bin"
    sales.write_bytes(b"synthetic data")
    monkeypatch.setattr(data, "find_file", lambda name: sales)
    identity = json.loads(json.dumps(confirmation.snapshot(), default=str))
    artifacts = {}
    for track, name in configs.FINAL.items():
        for split, months in [("dev", ["2025-12", "2026-01", "2026-02", "2026-03"]),
                              ("confirm", ["2026-04", "2026-05"])]:
            folder = tmp_path / "outputs/experiments" / name / split
            folder.mkdir(parents=True)
            pd.DataFrame({"ml_num": [f"{track}-{month}" for month in months],
                          "track": track, "month": months, "sold_price": 100.,
                          "predicted_price": 102., "house_type_name": "Condo Apt"}).to_csv(
                              folder / "predictions.csv", index=False)
            spec = {"spec": configs.CONFIGS[name]}
            if split == "dev":
                spec.update(identity)
            (folder / "spec.json").write_text(json.dumps(spec, default=str))
            (folder / "metrics.json").write_text(json.dumps(
                {track: {"overall": {"n": len(months), "mdape": 2.0}}}))
            if split == "confirm":
                for filename in ("metrics.json", "spec.json"):
                    artifacts[(folder / filename).relative_to(tmp_path).as_posix()] = \
                        confirmation.sha256(folder / filename)
    (tmp_path / "outputs/final_confirmation.json").write_text(json.dumps(
        {"status": "complete", "origin": "imported_lab_confirmation", "artifacts_sha256": artifacts}))
    return tmp_path


def test_collects_with_imported_lab_confirmation(imported_runs, monkeypatch):
    import models
    monkeypatch.setattr(models, "make_model", lambda *args: pytest.fail("Collection must not train"))
    run.collect_validation()
    result = pd.read_csv(imported_runs / "outputs/validation_predictions.csv")
    assert result.groupby("track")["month"].nunique().to_dict() == {"off_market": 6, "list_aware": 6}


@pytest.mark.parametrize("changed", ["confirm_metrics", "confirm_predictions", "dev_fingerprint"])
def test_imported_confirmation_rejects_changed_files(imported_runs, changed):
    name = configs.FINAL["off_market"]
    folder = imported_runs / "outputs/experiments" / name
    if changed == "confirm_metrics":
        (folder / "confirm/metrics.json").write_text("{}")
    elif changed == "confirm_predictions":
        frame = pd.read_csv(folder / "confirm/predictions.csv")
        frame.assign(predicted_price=110.).to_csv(folder / "confirm/predictions.csv", index=False)
    else:
        (folder / "dev/spec.json").write_text(json.dumps({"spec": configs.CONFIGS[name]}))
    with pytest.raises(ValueError):
        run.collect_validation()
