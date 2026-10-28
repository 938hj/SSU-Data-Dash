"""Named experiment configurations. Each maps a track to a model spec (see models.build_model).

Change one thing at a time between configs, and keep old configs: the results log
refers to them by name. FINAL picks what `python run.py predict` submits.
"""

BASE = ("base",)
TEXT = ("base", "text")
HIST = ("base", "text", "hist")
LA = ("list",)            # list price, added on the list-aware track only


def both(spec, la_extra=LA, la_spec=None):
    """Same spec on both tracks; the list-aware copy also gets the list price."""
    la = dict(spec, groups=tuple(spec["groups"]) + la_extra) if "groups" in spec else dict(spec)
    return {"off_market": dict(spec), "list_aware": la_spec or la}


CONFIGS = {
    # ---- baselines --------------------------------------------------------------------
    "B0_starter": {"off_market": {"model": "nearest_sales", "k": 10},
                   "list_aware": {"model": "list_price"}},
    "B0_threshold": {"off_market": {"model": "threshold"}, "list_aware": {"model": "threshold"}},

    # ---- E: ridge on log price ----------------------------------------------------------
    "E0_ridge_base": both({"model": "tabular", "engine": "ridge", "groups": BASE}),
    "E1_ridge_text": both({"model": "tabular", "engine": "ridge", "groups": TEXT}),

    # ---- A: gradient boosting on log price, adding one feature group at a time ----------
    "A0_hgb_base": both({"model": "tabular", "engine": "hgb", "groups": BASE}),
    "A1_hgb_text": both({"model": "tabular", "engine": "hgb", "groups": TEXT}),
    "A2_hgb_hist": both({"model": "tabular", "engine": "hgb", "groups": HIST}),
    "A0_cat_base": both({"model": "tabular", "engine": "catboost", "groups": BASE}),
    "A1_cat_text": both({"model": "tabular", "engine": "catboost", "groups": TEXT}),
    "A2_cat_hist": both({"model": "tabular", "engine": "catboost", "groups": HIST}),

    "A1_hgb_text_l1": both({"model": "tabular", "engine": "hgb", "groups": TEXT,
                            "params": {"loss": "absolute_error"}}),
    "A2_hgb_hist_l1": both({"model": "tabular", "engine": "hgb", "groups": HIST,
                            "params": {"loss": "absolute_error"}}),
    "A1_cat_text_mae": both({"model": "tabular", "engine": "catboost", "groups": TEXT,
                             "params": {"loss_function": "MAE"}}),

    # ---- market index: learn prices relative to the month's market, forecast the market --
    "M1_hgb_text_mkt": both({"model": "tabular", "engine": "hgb", "groups": TEXT,
                             "market": {"level_months": 3}}),
    "M1_hgb_text_mkt1": both({"model": "tabular", "engine": "hgb", "groups": TEXT,
                              "market": {"level_months": 1}}),
    "M1_hgb_text_mkt_notime": both({"model": "tabular", "engine": "hgb", "groups": TEXT,
                                    "market": {"level_months": 3},
                                    "drop": ("listed_months",)}),

    # best features + L1 + market adjustment (the summer backtests favour the adjustment)
    "M2_hgb_hist_l1_mkt": both({"model": "tabular", "engine": "hgb", "groups": HIST,
                                "params": {"loss": "absolute_error"}, "market": {"level_months": 3}}),

    # ---- B: list-aware, learn sold / list ------------------------------------------------
    "B1_cat_listratio": {"list_aware": {"model": "tabular", "engine": "catboost",
                                        "target": "log_list_ratio", "groups": HIST + LA}},
    "B1_hgb_listratio": {"list_aware": {"model": "tabular", "engine": "hgb",
                                        "target": "log_list_ratio", "groups": HIST + LA}},
    "B1_hgb_listratio_l1": {"list_aware": {"model": "tabular", "engine": "hgb", "target": "log_list_ratio",
                                           "groups": HIST + LA, "params": {"loss": "absolute_error"}}},

    # ---- F: blends (weights chosen on the summer backtests, checked on dev; see memory.md) -----
    "F1_la_blend50": {"list_aware": {"model": "blend", "mean": "geometric", "weights": [0.5, 0.5], "members": [
        {"model": "tabular", "engine": "hgb", "groups": HIST + LA, "params": {"loss": "absolute_error"},
         "market": {"level_months": 3}},
        {"model": "tabular", "engine": "hgb", "target": "log_list_ratio", "groups": HIST + LA,
         "params": {"loss": "absolute_error"}}]}},

    "F2_om_hgb_lgb": {"off_market": {"model": "blend", "mean": "geometric", "weights": [0.5, 0.5], "members": [
        {"model": "tabular", "engine": "hgb", "groups": HIST, "params": {"loss": "absolute_error"},
         "market": {"level_months": 3}},
        {"model": "tabular", "engine": "lightgbm", "groups": HIST, "params": {"objective": "regression_l1"},
         "market": {"level_months": 3}}]}},

    # list-aware, F1 plus 20% enhanced ridge (its errors differ most from the trees')
    "F4_la_3way": {"list_aware": {"model": "blend", "mean": "geometric", "weights": [0.4, 0.4, 0.2], "members": [
        {"model": "tabular", "engine": "hgb", "groups": HIST + LA, "params": {"loss": "absolute_error"},
         "market": {"level_months": 3}},
        {"model": "tabular", "engine": "hgb", "target": "log_list_ratio", "groups": HIST + LA,
         "params": {"loss": "absolute_error"}},
        {"model": "tabular", "engine": "ridge", "groups": HIST + LA, "params": {"plus": True, "alpha": 3.0},
         "market": {"level_months": 3}}]}},

    # off-market, three engines: HGB with the text model, LightGBM and CatBoost (GPU), all with
    # absolute-error loss and the market adjustment. Best on both summer backtests and on dev.
    "F3_om_3way": {"off_market": {"model": "blend", "mean": "geometric", "weights": [1, 1, 1], "members": [
        {"model": "tabular", "engine": "hgb", "groups": HIST + ("tfidf",), "params": {"loss": "absolute_error"},
         "market": {"level_months": 3}},
        {"model": "tabular", "engine": "lightgbm", "groups": HIST, "params": {"objective": "regression_l1"},
         "market": {"level_months": 3}},
        {"model": "tabular", "engine": "catboost", "groups": HIST, "market": {"level_months": 3},
         "params": {"task_type": "GPU", "devices": "0", "iterations": 3000, "learning_rate": 0.05, "depth": 8,
                    "border_count": 254, "max_ctr_complexity": 4, "loss_function": "MAE"}}]}},

    # ---- C: comparable-sales reference price + residual model ----------------------------
    "C1_cat_refratio": {"off_market": {"model": "tabular", "engine": "catboost",
                                       "target": "log_ref_ratio", "groups": HIST}},
    "C1_hgb_refratio": {"off_market": {"model": "tabular", "engine": "hgb",
                                       "target": "log_ref_ratio", "groups": HIST}},

    # ---- GPU runs (CatBoost on the user's RTX 3060; task_type="GPU"). Full-strength settings. ----
    "A2_cat_hist_gpu": both({"model": "tabular", "engine": "catboost", "groups": HIST,
                             "params": {"task_type": "GPU", "devices": "0", "iterations": 3000,
                                        "learning_rate": 0.05, "depth": 8, "border_count": 254,
                                        "max_ctr_complexity": 4}}),
    "M2_cat_hist_mkt_gpu": both({"model": "tabular", "engine": "catboost", "groups": HIST,
                                 "market": {"level_months": 3},
                                 "params": {"task_type": "GPU", "devices": "0", "iterations": 3000,
                                            "learning_rate": 0.05, "depth": 8, "border_count": 254,
                                            "max_ctr_complexity": 4}}),

    "M2_cat_hist_mae_mkt_gpu": both({"model": "tabular", "engine": "catboost", "groups": HIST,
                                     "market": {"level_months": 3},
                                     "params": {"task_type": "GPU", "devices": "0", "iterations": 3000,
                                                "learning_rate": 0.05, "depth": 8, "border_count": 254,
                                                "max_ctr_complexity": 4, "loss_function": "MAE"}}),

    # ---- D: LightGBM ---------------------------------------------------------------------
    "D2_lgb_hist": both({"model": "tabular", "engine": "lightgbm", "groups": HIST}),
    "D2_lgb_hist_l1_mkt": both({"model": "tabular", "engine": "lightgbm", "groups": HIST,
                                "params": {"objective": "regression_l1"}, "market": {"level_months": 3}}),

    # ---- E, enhanced: splines + community x type + per-type size slope (+ market adjustment) ----
    "E1_ridge_text": both({"model": "tabular", "engine": "ridge", "groups": TEXT}),
    "E2_ridge_plus_hist_mkt": both({"model": "tabular", "engine": "ridge", "groups": HIST,
                                    "params": {"plus": True, "alpha": 3.0}, "market": {"level_months": 3}}),
    "E3_ridge_plus_tfidf_mkt": both({"model": "tabular", "engine": "ridge", "groups": HIST + ("tfidf",),
                                     "params": {"plus": True, "alpha": 3.0}, "market": {"level_months": 3}}),

    # ---- T: learned text model (TF-IDF, text.py) added to the best tree models ----------------
    "T2_hgb_tfidf_l1_mkt": both({"model": "tabular", "engine": "hgb", "groups": HIST + ("tfidf",),
                                 "params": {"loss": "absolute_error"}, "market": {"level_months": 3}}),
    "T2_lgb_tfidf_l1_mkt": both({"model": "tabular", "engine": "lightgbm", "groups": HIST + ("tfidf",),
                                 "params": {"objective": "regression_l1"}, "market": {"level_months": 3}}),
    "T1_hgb_listratio_tfidf_l1": {"list_aware": {"model": "tabular", "engine": "hgb", "target": "log_list_ratio",
                                                 "groups": HIST + LA + ("tfidf",),
                                                 "params": {"loss": "absolute_error"}}},
}

# ---- U: tuning candidates (tune.py). Each F3/F4 member, then one change at a time. --------------
# Name: U_<member>_<variant>. "base" is the member exactly as it is in F3/F4, rerun on the same
# machine so the comparison is fair. Variants only change the settings listed.
TUNE_MEMBERS = {
    # member: (track, spec as in the final blend)
    "om_hgb": ("off_market", CONFIGS["F3_om_3way"]["off_market"]["members"][0]),
    "om_lgb": ("off_market", CONFIGS["F3_om_3way"]["off_market"]["members"][1]),
    "om_cat": ("off_market", CONFIGS["F3_om_3way"]["off_market"]["members"][2]),
    "la_hgb": ("list_aware", CONFIGS["F4_la_3way"]["list_aware"]["members"][0]),
    "la_ratio": ("list_aware", CONFIGS["F4_la_3way"]["list_aware"]["members"][1]),
    "la_ridge": ("list_aware", CONFIGS["F4_la_3way"]["list_aware"]["members"][2]),
}
_HGB = {"leaves31": {"max_leaf_nodes": 31}, "leaves127": {"max_leaf_nodes": 127},
        "minleaf20": {"min_samples_leaf": 20}, "minleaf100": {"min_samples_leaf": 100},
        "l2_0": {"l2_regularization": 0.0}, "l2_5": {"l2_regularization": 5.0},
        "lr03": {"learning_rate": 0.03, "max_iter": 3000}, "feat07": {"max_features": 0.7}}
_LGB = {"leaves31": {"num_leaves": 31}, "leaves127": {"num_leaves": 127},
        "mcs20": {"min_child_samples": 20}, "mcs100": {"min_child_samples": 100},
        "col06": {"colsample_bytree": 0.6}, "lambda5": {"reg_lambda": 5.0},
        "lr02": {"learning_rate": 0.02, "n_estimators": 5000}}
_CAT = {"depth6": {"depth": 6}, "depth10": {"depth": 10}, "l2_2": {"l2_leaf_reg": 2},
        "l2_10": {"l2_leaf_reg": 10}, "lr03": {"learning_rate": 0.03, "iterations": 5000}}
_RIDGE = {"alpha1": {"alpha": 1.0}, "alpha10": {"alpha": 10.0}, "alpha30": {"alpha": 30.0}}
_MARKET = {"mkt1": {"level_months": 1}, "mkt6": {"level_months": 6}}


def _tune_variants(spec):
    params = {"hgb": _HGB, "lightgbm": _LGB, "catboost": _CAT, "ridge": _RIDGE}[spec["engine"]]
    out = {"base": spec}
    for v, p in params.items():
        out[v] = {**spec, "params": {**(spec.get("params") or {}), **p}}
    if spec.get("market") is not None:
        for v, m in _MARKET.items():
            out[v] = {**spec, "market": {**spec["market"], **m}}
    return out


TUNE = {member: {f"U_{member}_{v}": s for v, s in _tune_variants(spec).items()}
        for member, (track, spec) in TUNE_MEMBERS.items()}
for _member, (_track, _spec) in TUNE_MEMBERS.items():
    for _name, _s in TUNE[_member].items():
        CONFIGS[_name] = {_track: _s}

# ---- H: market hedge (decided 2026-10-27 evening, see memory.md). Each market-adjusted member twice:
# once with a 3-month and once with a 6-month market level. Did not pass the strict "all three summers
# by >= 0.05" rule (2024/2025 within +0.04), but better on the 3-summer mean, the worst summer, and the
# rising-market check (rise21); kept as a robustness choice. Members are exactly the U_* runs.
def _m6(member):
    return {**member, "market": {**member["market"], "level_months": 6}}


_F3 = CONFIGS["F3_om_3way"]["off_market"]["members"]
_F4 = CONFIGS["F4_la_3way"]["list_aware"]["members"]
CONFIGS["H1_om_hedge"] = {"off_market": {"model": "blend", "mean": "geometric", "weights": [1] * 6,
                                         "members": [*_F3, *[_m6(m) for m in _F3]]}}
CONFIGS["H2_la_hedge"] = {"list_aware": {"model": "blend", "mean": "geometric", "weights": [0.2, 0.2, 0.4, 0.1, 0.1],
                                         "members": [_F4[0], _m6(_F4[0]), _F4[1], _F4[2], _m6(_F4[2])]}}

# What run.py validate / predict use. Change only after comparing on the dev months.
# Chosen 2026-10-27 from the summer backtests (same season and horizon as the hidden test);
# see memory.md and the day-1 summary doc.
# 2026-10-27 21:15: switched to the market hedge after the real H1/H2 runs matched the recombined
# numbers (memory.md). Previous: {"off_market": "F3_om_3way", "list_aware": "F4_la_3way"}.
FINAL = {"off_market": "H1_om_hedge", "list_aware": "H2_la_hedge"}
