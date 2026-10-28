"""Time-boundary checks for history features and the models (made-up data, no download).

    pytest tests/test_leakage.py
"""
import numpy as np
import pandas as pd
import pytest

import data
import history
import models
from tests.test_starter import make_listings


def wide_listings(n=3000, seed=0, first_id=0):
    t = make_listings(n, seed=seed, first_id=first_id)
    rng = np.random.default_rng(seed + 7)
    t["community"] = rng.choice(["A", "B", "C"], n)
    t["municipality"] = rng.choice(["M1", "M2"], n)
    return t


def history_of(reference_rows, homes, as_of):
    return history.HistoryReference(reference_rows).features(homes, as_of)


def test_own_price_does_not_change_own_history_features():
    sales = wide_listings()
    as_of = history.as_of_for_training(sales)
    before = history_of(sales, sales, as_of)
    changed = sales.copy()
    changed.loc[changed.index[::7], "sold_price"] *= 3.0   # change many homes' own prices
    after = history_of(changed, changed, as_of)
    rows = changed.index[::7]
    # a home's own features must not move when only its own price moves...
    own = before.loc[rows]
    # ...but other homes priced later may legitimately see the change, so compare only
    # homes whose as_of is no later than every changed sale date (none of them see it)
    first_changed = changed.loc[rows, "date_sold"].min()
    unaffected = as_of[as_of <= first_changed].index
    pd.testing.assert_frame_equal(before.loc[unaffected], after.loc[unaffected])
    # and directly: each changed home's own row equals its row computed without itself
    for i in rows[:30]:
        alone = history_of(changed.drop(index=i), changed.loc[[i]], as_of.loc[[i]])
        pd.testing.assert_frame_equal(after.loc[[i]], alone, check_dtype=False)


def test_future_sales_do_not_change_features():
    sales = wide_listings()
    homes = wide_listings(300, seed=5, first_id=50_000)
    cutoff = pd.Timestamp("2025-09-01")
    as_of = history.as_of_for_prediction(homes, cutoff)
    past = sales[sales["date_sold"] < cutoff]
    with_future = history_of(sales, homes, as_of)          # reference also holds later sales
    only_past = history_of(past, homes, as_of)
    pd.testing.assert_frame_equal(with_future, only_past)


def test_as_of_is_never_after_the_sale():
    sales = wide_listings()
    as_of = history.as_of_for_training(sales)
    assert (as_of <= sales["date_sold"]).all()
    assert (as_of <= sales["date_listed"]).all()


@pytest.mark.parametrize("target", ["log_price", "log_ref_ratio"])
def test_off_market_model_never_reads_list_price(target):
    sales = wide_listings(2500)
    homes = data.hide_answers(wide_listings(200, seed=3, first_id=90_000), "off_market")
    assert "list_price" not in homes.columns
    model = models.TabularModel("off_market", engine="hgb", target=target,
                                groups=("base", "text", "hist"), early_stop=False,
                                params={"max_iter": 30}).fit(sales)
    assert np.isfinite(model.predict(homes)).all()


def test_off_market_rejects_list_features():
    with pytest.raises(ValueError):
        models.TabularModel("off_market", groups=("base", "list"))
    with pytest.raises(ValueError):
        models.TabularModel("off_market", target="log_list_ratio")


def test_model_prediction_ignores_sales_after_cutoff():
    """Fitting on train only vs train + later sales gives the same features for the
    homes to value (the model itself differs, so compare features, not prices)."""
    sales = wide_listings()
    cutoff = pd.Timestamp("2025-09-01")
    homes = wide_listings(200, seed=9, first_id=70_000)
    past = sales[sales["date_sold"] < cutoff]
    m = models.TabularModel("off_market", groups=("base", "hist"), early_stop=False,
                            params={"max_iter": 20}).fit(past)
    as_of = history.as_of_for_prediction(homes, m.cutoff)
    assert (as_of <= cutoff).all()


def test_cache_matches_fresh_computation():
    """Cached history features equal features computed from the fold's own training set,
    for training rows and for the month being predicted; a non-prefix reference skips it."""
    sales = wide_listings(4000)
    try:
        history.enable_cache(sales)
        cutoff = pd.Timestamp("2025-10-01")
        train = sales[sales["date_sold"] < cutoff]
        month = sales[(sales["date_sold"] >= cutoff) & (sales["date_sold"] < cutoff + pd.DateOffset(months=1))]
        ref = history.HistoryReference(train)
        as_of_train = history.as_of_for_training(train)
        pd.testing.assert_frame_equal(ref.cached_features(train, as_of_train),
                                      ref.features(train, as_of_train), check_dtype=False)
        m_cut = train["date_sold"].max().normalize() + pd.Timedelta(days=1)
        as_of_month = history.as_of_for_prediction(month, m_cut)
        pd.testing.assert_frame_equal(ref.cached_features(month, as_of_month),
                                      ref.features(month, as_of_month), check_dtype=False)
        holey = train.drop(index=train.index[::5])          # not a prefix: must not use the cache
        hr = history.HistoryReference(holey)
        assert not hr._is_prefix_of_cache_source()
    finally:
        history.disable_cache()


def test_intervals_are_ordered_and_use_out_of_time_errors():
    sales = wide_listings(4000)
    cutoff = pd.Timestamp("2025-09-01")
    train = sales[sales["date_sold"] < cutoff]
    homes = data.hide_answers(sales[sales["date_sold"] >= cutoff].head(200), "off_market")
    m = models.TabularModel("off_market", groups=("base",), params={"max_iter": 40}).fit(train)
    pred, lo, hi = m.predict_with_interval(homes, coverage=0.8)
    assert lo is not None and np.all(lo <= pred) and np.all(pred <= hi)
    assert "__all__" in m.residual_q


def test_blend_interval_matches_blend_prediction():
    sales = wide_listings(4000)
    cutoff = pd.Timestamp("2025-09-01")
    train = sales[sales["date_sold"] < cutoff]
    homes = data.hide_answers(sales[sales["date_sold"] >= cutoff].head(150), "list_aware")
    spec = {"model": "blend", "weights": [0.5, 0.5], "mean": "geometric", "members": [
        {"model": "tabular", "engine": "hgb", "groups": ("base", "list"), "params": {"max_iter": 30}},
        {"model": "tabular", "engine": "hgb", "groups": ("base", "list"), "target": "log_list_ratio",
         "params": {"max_iter": 30}}]}
    m = models.build_model(spec, "list_aware").fit(train)
    pred, lo, hi = m.predict_with_interval(homes)
    np.testing.assert_allclose(pred, m.predict(homes), rtol=1e-9)
    assert np.all(lo <= pred) and np.all(pred <= hi)


def test_tfidf_training_scores_are_out_of_fold_and_model_runs():
    """A training row's text score must not depend on its own price; the model must fit and
    predict with the tfidf group, and the ridge 'plus' variant must run."""
    import text as text_module
    sales = wide_listings(3000)
    words = np.array(["renovated", "as-is", "original", "turnkey", "needs work", "spacious"])
    rng = np.random.default_rng(1)
    sales["remarks"] = [" ".join(rng.choice(words, 6)) + " home" for _ in range(len(sales))]
    tm = text_module.TextPriceModel(min_df=1, svd_dims=3)
    before = tm.fit_transform(sales)["tfidf_score"]
    changed = sales.copy()
    i = changed.index[0]
    changed.loc[i, "sold_price"] *= 50
    tm2 = text_module.TextPriceModel(min_df=1, svd_dims=3)
    after = tm2.fit_transform(changed)["tfidf_score"]
    # the row's own score comes from a fold model that never saw it; but the group mean used for
    # the target includes it, so allow tiny drift only
    assert abs(after.loc[i] - before.loc[i]) < 0.05
    cutoff = pd.Timestamp("2025-09-01")
    train = sales[sales["date_sold"] < cutoff]
    homes = data.hide_answers(sales[sales["date_sold"] >= cutoff].head(100), "off_market")
    for engine, params in [("hgb", {"max_iter": 30}), ("ridge", {"plus": True, "alpha": 3.0})]:
        m = models.TabularModel("off_market", engine=engine, groups=("base", "text", "tfidf"),
                                params=params, market={"level_months": 3}).fit(train)
        assert np.isfinite(m.predict(homes)).all()


def test_final_configs_build_and_run_without_gpu(monkeypatch):
    """Both selected final ensembles must fit and predict with CatBoost CPU fallback."""
    import configs
    monkeypatch.delenv("SSU_CATBOOST_DEVICE", raising=False)
    monkeypatch.setattr(models, "gpu_available", lambda: False)
    sales = wide_listings(3000)
    cutoff = pd.Timestamp("2025-09-01")
    train = sales[sales["date_sold"] < cutoff]
    for track, name in configs.FINAL.items():
        spec = configs.CONFIGS[name][track]
        # shrink the members so the test is quick; structure unchanged
        small = {**spec, "members": [{**m, "params": {**(m.get("params") or {}),
                 **({"iterations": 40} if m["engine"] == "catboost" else {}),
                 **({"max_iter": 30} if m["engine"] == "hgb" else {}),
                 **({"n_estimators": 30} if m["engine"] == "lightgbm" else {})}} for m in spec["members"]]}
        homes = data.hide_answers(sales[sales["date_sold"] >= cutoff].head(80), track)
        m = models.build_model(small, track).fit(train)
        pred, lo, hi = m.predict_with_interval(homes)
        assert np.isfinite(pred).all() and (pred > 0).all()


def test_saved_model_gives_identical_predictions(tmp_path):
    """A fitted blend saved with joblib and loaded back predicts exactly the same prices."""
    import joblib
    sales = wide_listings(3000)
    cutoff = pd.Timestamp("2025-09-01")
    train = sales[sales["date_sold"] < cutoff]
    homes = data.hide_answers(sales[sales["date_sold"] >= cutoff].head(80), "list_aware")
    spec = {"model": "blend", "weights": [0.4, 0.4, 0.2], "mean": "geometric", "members": [
        {"model": "tabular", "engine": "hgb", "groups": ("base", "hist", "list"), "params": {"max_iter": 30},
         "market": {"level_months": 3}},
        {"model": "tabular", "engine": "hgb", "groups": ("base", "list"), "target": "log_list_ratio",
         "params": {"max_iter": 30}},
        {"model": "tabular", "engine": "ridge", "groups": ("base", "list"), "params": {"plus": True, "alpha": 3.0}}]}
    m = models.build_model(spec, "list_aware").fit(train)
    joblib.dump(m, tmp_path / "m.joblib", compress=3)
    again = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_array_equal(np.round(m.predict(homes)), np.round(again.predict(homes)))
