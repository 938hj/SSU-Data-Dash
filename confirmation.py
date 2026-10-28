"""Freeze final configurations and reserve one April-May confirmation campaign."""
import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import configs
import data

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("configs.py", "data.py", "features.py", "history.py", "market.py",
                "text.py", "models.py", "experiments.py", "run.py", "confirmation.py",
                "requirements.txt")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot():
    return {"final": dict(configs.FINAL),
            "specs": {track: configs.CONFIGS[name][track]
                      for track, name in configs.FINAL.items()},
            "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
            "listings_sha256": sha256(data.find_file("listings"))}


def _path(name):
    return ROOT / "outputs" / name


def _write(path, value, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")


def freeze():
    """Record configuration, source and data identity without evaluating any holdout."""
    if _path("final_confirmation.json").exists():
        raise RuntimeError("Confirmation was already reserved; do not overwrite its history.")
    path = _path("final_freeze.json")
    current = json.loads(json.dumps(snapshot(), default=str))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["snapshot"] != current:
            raise RuntimeError("A different freeze already exists. Review it before replacing it.")
        return path
    _write(path, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                  "snapshot": current}, mode="x")
    return path


def validate_freeze():
    path = _path("final_freeze.json")
    if not path.exists():
        raise RuntimeError("Run `python experiments.py freeze` after final development.")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen["snapshot"] != json.loads(json.dumps(snapshot(), default=str)):
        raise RuntimeError("Frozen configuration, source or listings changed; confirmation refused.")
    return frozen


def require_reservation(reservation):
    path = _path("final_confirmation.json")
    if not reservation or not path.exists():
        raise RuntimeError("Confirmation requires the reserved final campaign.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["reservation"] != reservation or saved["status"] != "started":
        raise RuntimeError("The confirmation reservation is not active.")


@contextmanager
def campaign(names, track=None, seed=None):
    """Reserve both final tracks before scoring; a failed attempt remains consumed."""
    expected = set(configs.FINAL.values())
    if track is not None or seed is not None or len(names) != len(expected) or set(names) != expected:
        raise RuntimeError("Confirm both frozen final configurations together, without track/seed overrides.")
    path = _path("final_confirmation.json")
    if path.exists():
        raise RuntimeError("Confirmation has already been reserved; a second look is refused.")
    frozen = validate_freeze()
    state = {"reservation": uuid.uuid4().hex, "status": "started",
             "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "freeze": frozen, "final": dict(configs.FINAL)}
    try:
        _write(path, state, mode="x")
    except FileExistsError as error:
        raise RuntimeError("Confirmation has already been reserved; a second look is refused.") from error
    try:
        yield state["reservation"]
        artifacts = {}
        for name in expected:
            for filename in ("predictions.csv", "metrics.json", "spec.json"):
                artifact = ROOT / "outputs" / "experiments" / name / "confirm" / filename
                artifacts[artifact.relative_to(ROOT).as_posix()] = sha256(artifact)
        # Catch source/data changes during the campaign as well as before it.
        validate_freeze()
        state.update(status="complete", artifacts_sha256=artifacts)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(path, state)


def completed_campaign():
    path = _path("final_confirmation.json")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("origin") == "imported_lab_confirmation":
        raise RuntimeError("Imported lab confirmation is complete. Per-home predictions and original "
                           "source/data fingerprints were not transferred; use the saved metrics in "
                           "reports/validation_evidence.json. Do not rerun confirmation.")
    validate_freeze()
    path = _path("final_confirmation.json")
    if not path.exists():
        raise RuntimeError("No final confirmation campaign exists.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise RuntimeError("Final confirmation did not complete.")
    for relative, expected in state["artifacts_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Confirmation artifact changed: {relative}")
    return state
