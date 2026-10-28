"""Run named configurations (configs.py) under fixed time splits and log every result.

    python experiments.py run A2_cat_hist                 dev months: 2025-12 to 2026-03
    python experiments.py run A2_cat_hist --split summer25  train before 2025-06-01, predict
                                                          Jun-Sep 2025 in one go (same season as
                                                          the hidden test); also summer24
    python experiments.py run A2_cat_hist --split backtest  train before 2025-12-01, predict
                                                          Dec-Mar in one go (longer horizon,
                                                          like the real Jun-Sep test)
    python experiments.py run A2_cat_hist --split confirm --frozen
                                                          2026-04 to 2026-05: only after the
                                                          final configuration is frozen
    python experiments.py compare                         latest result of every config
    python experiments.py breakdown A2_cat_hist           errors by month, type, price decile

Every run writes outputs/experiments/<config>/<split>/predictions.csv (one row per home,
predicted out of time), metrics.json and spec.json, and appends a line to
outputs/experiments/log.csv. Report numbers come from these saved files.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import configs
import data
import models
import confirmation

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "experiments"
# Each machine keeps its own log so copies never overwrite each other: the cloud runner
# writes log.csv; set SSU_RUNNER (e.g. "alienware") on another machine to get log_<name>.csv.
RUNNER = os.environ.get("SSU_RUNNER", "cloud")
LOG = OUT / ("log.csv" if RUNNER == "cloud" else f"log_{RUNNER}.csv")
TRACKS = ["off_market", "list_aware"]
SPLITS = {
    "dev": pd.period_range("2025-12", "2026-03", freq="M"),
    "confirm": pd.period_range("2026-04", "2026-05", freq="M"),
    "all": pd.period_range("2025-12", "2026-05", freq="M"),
}
# Fixed-cutoff backtests: train once, predict four months ahead in one go, like the real
# test (data to May 2026, homes sold Jun-Sep 2026). "summer" ones match the test season.
BACKTESTS = {
    "backtest": ("2025-12-01", pd.period_range("2025-12", "2026-03", freq="M")),
    "summer25": ("2025-06-01", pd.period_range("2025-06", "2025-09", freq="M")),
    "summer24": ("2024-06-01", pd.period_range("2024-06", "2024-09", freq="M")),
    "summer23": ("2023-06-01", pd.period_range("2023-06", "2023-09", freq="M")),
    "rise21": ("2021-10-01", pd.period_range("2021-10", "2022-01", freq="M")),
}

_LISTINGS = None
_LISTINGS_SOURCE = None


def listings():
    global _LISTINGS, _LISTINGS_SOURCE
    path = data.find_file("listings").resolve()
    source = (str(path), confirmation.sha256(path))
    if _LISTINGS is None or source != _LISTINGS_SOURCE:
        import history
        history.disable_cache()
        _LISTINGS = data.load("listings")
        _LISTINGS_SOURCE = source
    return _LISTINGS


def needs_history(config):
    return any(s.get("model") == "tabular" and ("hist" in s.get("groups", ()) or
               s.get("target") == "log_ref_ratio") for s in config.values())


def warm_cache():
    """History features of every listing row, computed once (see history.enable_cache)."""
    import history
    if not history._CACHE:
        start = time.time()
        history.enable_cache(listings())
        print(f"  history cache built in {time.time() - start:.0f}s", flush=True)


# ------------------------------------------------------------------ scoring --

def score(frame):
    signed = 100 * (frame["predicted_price"] - frame["sold_price"]) / frame["sold_price"]
    absolute = signed.abs()
    interval = {}
    if "lower80" in frame and frame["lower80"].notna().any():
        f = frame[frame["lower80"].notna()]
        inside = (f["sold_price"] >= f["lower80"]) & (f["sold_price"] <= f["upper80"])
        interval = {"coverage80": round(float(inside.mean() * 100), 2),
                    "width80_pct": round(float((100 * (f["upper80"] - f["lower80"]) / f["predicted_price"]).median()), 2)}
    return {**interval, "n": int(len(frame)), "mdape": round(float(absolute.median()), 3),
            "mape": round(float(absolute.mean()), 3),
            "median_signed": round(float(signed.median()), 3),
            "within_5": round(float((absolute <= 5).mean() * 100), 2),
            "within_10": round(float((absolute <= 10).mean() * 100), 2)}


def metrics(pred):
    out = {"overall": score(pred), "by_month": {}, "by_type": {}}
    for month, g in pred.groupby("month"):
        out["by_month"][str(month)] = score(g)
    for kind, g in pred.groupby("house_type_name"):
        out["by_type"][kind] = score(g)
    return out


# ------------------------------------------------------------------ running --

def folds(split):
    sales = listings()
    if split in BACKTESTS:
        cutoff, months = BACKTESTS[split]
        train = sales[sales["date_sold"] < pd.Timestamp(cutoff)]
        homes = sales[sales["date_sold"].dt.to_period("M").isin(months)]
        yield f"{months[0]}..{months[-1]}", train, homes
        return
    for month in SPLITS[split]:
        train = sales[sales["date_sold"] < month.start_time]
        homes = sales[sales["date_sold"].dt.to_period("M") == month]
        yield str(month), train, homes


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def run(name, split="dev", tracks=None, frozen=False, seed=None, reservation=None):
    if split == "all":
        raise RuntimeError("Use dev, then one frozen confirm campaign; --split all is disabled.")
    if split == "confirm" and not frozen:
        sys.exit("The confirm months are kept for the end. Freeze the final configuration first, "
                 "then rerun with --frozen. (Every confirm run is logged.)")
    if split == "confirm":
        confirmation.require_reservation(reservation)
    identity = confirmation.snapshot()
    config = configs.CONFIGS[name]
    tracks = [t for t in (tracks or TRACKS) if t in config]
    if needs_history(config):
        warm_cache()
    rows, timing = [], {}
    for fold, train, homes in folds(split):
        for track in tracks:
            spec = dict(config[track])
            if seed is not None and spec.get("model") == "tabular":
                spec["seed"] = seed
            start = time.time()
            model = models.build_model(spec, track).fit(train)
            visible = data.hide_answers(homes, track)
            lower = upper = None
            if hasattr(model, "predict_with_interval"):
                predicted, lower, upper = model.predict_with_interval(visible, coverage=0.8)
            else:
                predicted = model.predict(visible)
            seconds = time.time() - start
            timing[f"{fold}/{track}"] = round(seconds, 1)
            if not np.all(np.isfinite(predicted)) or (np.asarray(predicted) <= 0).any():
                raise ValueError(f"{name} {fold} {track}: predictions must be finite and positive")
            rows.append(pd.DataFrame({
                "ml_num": homes["ml_num"].to_numpy(), "track": track,
                "month": homes["date_sold"].dt.to_period("M").astype(str).to_numpy(), "fold": fold,
                "house_type_name": homes["house_type_name"].to_numpy(),
                "community": homes["community"].to_numpy(),
                "municipality": homes["municipality"].to_numpy(),
                "sold_price": homes["sold_price"].to_numpy(),
                "list_price": homes["list_price"].to_numpy(),
                "predicted_price": np.asarray(predicted, float),
                "lower80": np.nan if lower is None else np.asarray(lower, float),
                "upper80": np.nan if upper is None else np.asarray(upper, float),
                "best_rounds": getattr(model, "best_rounds", None)}))
            print(f"  {fold} {track:10s} train {len(train):>7,}  homes {len(homes):>6,}  {seconds:6.1f}s",
                  flush=True)
    pred = pd.concat(rows, ignore_index=True)
    folder = OUT / name / split
    folder.mkdir(parents=True, exist_ok=True)
    pred.to_csv(folder / "predictions.csv", index=False)
    result = {t: metrics(pred[pred["track"] == t]) for t in tracks}
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    import catboost, lightgbm, sklearn
    (folder / "spec.json").write_text(json.dumps({
        "config": name, "split": split, "spec": config, "seed": seed, "git": git_hash(),
        "source_sha256": identity["source_sha256"], "listings_sha256": identity["listings_sha256"],
        "timing_seconds": timing, "python": platform.python_version(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "catboost": catboost.__version__, "lightgbm": lightgbm.__version__},
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runner": RUNNER,
        "machine": platform.node()}, indent=2, default=str))
    log_rows = [{"ran_at": time.strftime("%Y-%m-%d %H:%M:%S"), "config": name, "split": split,
                 "track": t, **{k: result[t]["overall"][k] for k in ["n", "mdape", "mape", "median_signed"]},
                 "worst_month_mdape": max(m["mdape"] for m in result[t]["by_month"].values()),
                 "seconds": round(sum(v for k, v in timing.items() if k.endswith(t)), 1),
                 "seed": seed, "git": git_hash()} for t in tracks]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    for t in tracks:
        o = result[t]["overall"]
        months = "  ".join(f"{m}:{v['mdape']:.2f}" for m, v in result[t]["by_month"].items())
        cover = f"  80% interval: covers {o['coverage80']:.1f}%, width {o['width80_pct']:.1f}%" \
            if "coverage80" in o else ""
        print(f"{name} [{split}] {t:10s} MdAPE {o['mdape']:6.2f}%  MAPE {o['mape']:6.2f}%  "
              f"signed {o['median_signed']:+.2f}%   {months}{cover}")
    if confirmation.snapshot() != identity:
        raise RuntimeError("Source, final configuration or listings changed during this experiment.")
    return result


def compare(split=None):
    if not LOG.exists():
        sys.exit("no results yet")
    log = pd.read_csv(LOG)
    if split:
        log = log[log["split"] == split]
    latest = log.groupby(["config", "split", "track"]).tail(1)
    table = latest.pivot_table(index=["split", "config"], columns="track",
                               values=["mdape", "mape"], aggfunc="last")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(2))


def breakdown(name, split="dev"):
    pred = pd.read_csv(OUT / name / split / "predictions.csv")
    pred["ape"] = 100 * (pred["predicted_price"] - pred["sold_price"]).abs() / pred["sold_price"]
    # deciles of the *predicted* price: grouping by the sold price exaggerates errors at the
    # ends (regression to the mean), and the predicted price is what a user would know
    pred["decile"] = pred.groupby("track")["predicted_price"].transform(lambda s: pd.qcut(s, 10, labels=False) + 1)
    pred["signed"] = 100 * (pred["predicted_price"] - pred["sold_price"]) / pred["sold_price"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for by in ["month", "house_type_name", "decile", "municipality"]:
            table = pred.groupby([by, "track"]).agg(mdape=("ape", "median"), signed=("signed", "median"),
                                                    n=("ape", "size")).unstack("track")
            print(table.round(2), "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="freeze final configs, source and training data without scoring")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+")
    r.add_argument("--split", default="dev", choices=["dev", "confirm", "all", *BACKTESTS])
    r.add_argument("--track", choices=TRACKS)
    r.add_argument("--frozen", action="store_true")
    r.add_argument("--seed", type=int)
    c = sub.add_parser("compare")
    c.add_argument("--split")
    b = sub.add_parser("breakdown")
    b.add_argument("name")
    b.add_argument("--split", default="dev")
    args = parser.parse_args()
    if args.command == "freeze":
        print(f"Frozen: {confirmation.freeze()}")
    elif args.command == "run" and args.split == "confirm":
        if not args.frozen:
            parser.error("Confirmation requires --frozen and a saved freeze manifest.")
        with confirmation.campaign(args.names, args.track, args.seed) as reservation:
            for name in args.names:
                run(name, "confirm", frozen=True, reservation=reservation)
    elif args.command == "run":
        for n in args.names:
            print(f"== {n} [{args.split}]", flush=True)
            run(n, args.split, [args.track] if args.track else None, args.frozen, args.seed)
    elif args.command == "compare":
        compare(args.split)
    else:
        breakdown(args.name, args.split)
