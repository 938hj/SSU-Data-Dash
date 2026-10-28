"""The models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model(track) returns the model chosen for the final submission (FINAL in
configs.py), so `python run.py validate` and `python run.py predict` use it.
experiments.py builds any other configuration with build_model(spec).
"""
import numpy as np
import pandas as pd

import features
import history
import market
import text


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    import configs
    return build_model(configs.CONFIGS[configs.FINAL[track]][track], track)


def build_model(spec, track):
    kind = spec.get("model", "tabular")
    if kind == "list_price":
        return ListPriceBaseline()
    if kind == "nearest_sales":
        return NearestSalesBaseline(k=spec.get("k", 10))
    if kind == "threshold":
        return ThresholdBaseline()
    if kind == "tabular":
        return TabularModel(track, **{k: v for k, v in spec.items() if k != "model"})
    if kind == "blend":
        return BlendModel([build_model(s, track) for s in spec["members"]], spec["weights"],
                          spec.get("mean", "geometric"))
    raise ValueError(f"unknown model {kind}")


# ============================================================ baselines =======

class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market starter baseline: median sold price of the k nearest earlier sales
    of the same type (earlier = sold before the home was listed)."""

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        nearest = np.argsort(np.sqrt(dx ** 2 + dy ** 2), kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class ThresholdBaseline:
    """The handout's 'maximum error' bar: the median sale of the same type in the
    same neighbourhood over the 12 months before the training cutoff. Falls back to
    the same type in the municipality, then the type overall."""

    def fit(self, train):
        cutoff = train["date_sold"].max() + pd.Timedelta(days=1)
        recent = train[train["date_sold"] >= cutoff - pd.DateOffset(months=12)]
        self.levels = [recent.groupby(keys)["sold_price"].median()
                       for keys in (["community", "house_type_name"],
                                    ["municipality", "house_type_name"], ["house_type_name"])]
        self.overall = float(recent["sold_price"].median())
        return self

    def predict(self, homes):
        out = pd.Series(np.nan, index=homes.index)
        for keys, level in zip(([["community", "house_type_name"], ["municipality", "house_type_name"],
                                 ["house_type_name"]]), self.levels):
            looked_up = homes[keys].merge(level.rename("p"), left_on=keys, right_index=True, how="left")["p"]
            out = out.fillna(looked_up)
        return out.fillna(self.overall).to_numpy(dtype=float)


# ============================================================ learners ========

DEFAULT_PARAMS = {
    "hgb": dict(learning_rate=0.05, max_iter=1500, max_leaf_nodes=63, min_samples_leaf=40,
                l2_regularization=1.0, loss="squared_error"),
    # lighter than CatBoost's usual settings: on a 2-core machine depth 8 x 3000 rounds took
    # over 30 minutes per fold
    "catboost": dict(iterations=1500, learning_rate=0.08, depth=6, l2_leaf_reg=5, loss_function="RMSE",
                     border_count=128, max_ctr_complexity=1),
    "lightgbm": dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                     reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     objective="regression", verbose=-1),
    "ridge": dict(alpha=3.0),
}


def gpu_available():
    try:
        from catboost.utils import get_gpu_device_count
        return get_gpu_device_count() > 0
    except Exception:
        return False


def catboost_device(params):
    """CatBoost on the GPU when the config asks for it and a GPU is there; otherwise the CPU.

    Set SSU_CATBOOST_DEVICE=CPU to force the CPU (or GPU to insist on the GPU). The CPU gives
    the same kind of model but not bit-identical numbers, and is much slower at depth 8.
    """
    import os
    import warnings
    wanted = os.environ.get("SSU_CATBOOST_DEVICE", params.get("task_type", "CPU")).upper()
    if wanted == "GPU" and not gpu_available():
        if os.environ.get("SSU_CATBOOST_DEVICE", "").upper() == "GPU":
            raise RuntimeError("SSU_CATBOOST_DEVICE=GPU but CatBoost finds no GPU")
        warnings.warn("CatBoost: no GPU found, training on the CPU instead (slower; numbers differ "
                      "slightly from the GPU run)")
        wanted = "CPU"
    if wanted == "CPU":
        params.pop("devices", None)
        params["task_type"] = "CPU"
    else:
        params["task_type"] = "GPU"
    return params


class Learner:
    """One regressor behind a common interface: fit(X_num, X_cat, y, val=None) / predict.

    val = (X_num, X_cat, y) is a later stretch of the *training* period, used only to
    choose the number of boosting rounds. The final model is then refit on everything.
    """

    def __init__(self, engine, params=None, seed=0, threads=None):
        self.engine = engine
        self.params = {**DEFAULT_PARAMS[engine], **(params or {})}
        self.seed = seed
        self.threads = threads
        self.best_rounds = None

    # categories as pandas category with the training categories, unknown -> NaN
    def _cats(self, X_cat, fit):
        if fit:
            self.categories = {c: sorted(X_cat[c].unique()) for c in X_cat.columns}
        out = pd.DataFrame(index=X_cat.index)
        for c in X_cat.columns:
            out[c] = pd.Categorical(X_cat[c], categories=self.categories[c])
        return out

    def _frame(self, X_num, X_cat, fit):
        return pd.concat([X_num.astype(float), self._cats(X_cat, fit)], axis=1)

    def fit(self, X_num, X_cat, y, val=None, weight=None):
        engine = self.engine
        # drop numeric columns that are empty or constant in this training set
        self.num_cols = [c for c in X_num.columns if X_num[c].nunique(dropna=True) > 1]
        X_num = X_num[self.num_cols]
        if val is not None:
            val = (val[0][self.num_cols], val[1], val[2])
        if engine == "ridge":
            return self._fit_ridge(X_num, X_cat, y, weight)
        X = self._frame(X_num, X_cat, fit=True)
        Xv = None if val is None else self._frame(val[0], val[1], fit=False)
        yv = None if val is None else val[2]
        if engine == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            p = dict(self.params)
            if Xv is not None:
                probe = HistGradientBoostingRegressor(**p, categorical_features="from_dtype",
                                                      early_stopping=True, n_iter_no_change=50,
                                                      random_state=self.seed)
                probe.fit(X, y, X_val=Xv, y_val=yv)
                self.best_rounds = int(probe.n_iter_)
                p["max_iter"] = max(50, self.best_rounds)
            self.model = HistGradientBoostingRegressor(**p, categorical_features="from_dtype",
                                                       early_stopping=False, random_state=self.seed)
            self.model.fit(X, y, sample_weight=weight)
        elif engine == "catboost":
            from catboost import CatBoostRegressor, Pool
            cat_cols = list(X_cat.columns)
            Xc = pd.concat([X_num.astype(float), X_cat], axis=1)
            p = catboost_device(dict(self.params))
            self.catboost_device = p.get("task_type", "CPU")
            common = dict(random_seed=self.seed, allow_writing_files=False, verbose=False,
                          thread_count=self.threads or -1)
            if val is not None:
                Xvc = pd.concat([val[0].astype(float), val[1]], axis=1)
                probe = CatBoostRegressor(**p, **common, od_type="Iter", od_wait=150)
                probe.fit(Pool(Xc, y, cat_features=cat_cols), eval_set=Pool(Xvc, yv, cat_features=cat_cols),
                          use_best_model=True)
                self.best_rounds = int(probe.get_best_iteration()) + 1
                p["iterations"] = max(50, self.best_rounds)
            self.model = CatBoostRegressor(**p, **common)
            self.model.fit(Pool(Xc, y, cat_features=cat_cols, weight=weight))
            self.cat_cols = cat_cols
        elif engine == "lightgbm":
            import lightgbm as lgb
            p = dict(self.params)
            common = dict(random_state=self.seed, n_jobs=self.threads or -1)
            if Xv is not None:
                probe = lgb.LGBMRegressor(**p, **common)
                probe.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(150, verbose=False)])
                self.best_rounds = int(probe.best_iteration_ or p["n_estimators"])
                p["n_estimators"] = max(50, self.best_rounds)
            self.model = lgb.LGBMRegressor(**p, **common)
            self.model.fit(X, y, sample_weight=weight)
        else:
            raise ValueError(engine)
        return self

    # scheme E, enhanced: splines on a few smooth numeric columns, a community x type category,
    # and one log-area slope per property type. Set params={"plus": True} with engine "ridge".
    SPLINE_COLS = ("log_area", "listed_months", "latitude", "longitude", "log_lot_area",
                   "h_ref_logp", "knn10_implied_logp", "tfidf_score")

    def _ridge_frames(self, X_num, X_cat):
        X_num = X_num.astype(float).copy()
        X_cat = X_cat.copy()
        if self.params.get("plus"):
            kind = X_cat["house_type_name"]
            for k in self.ridge_types:
                X_num["ix_logarea_" + k] = np.where(kind == k, X_num["log_area"].fillna(0.0), 0.0)
            X_cat["community_type"] = X_cat["community"] + "|" + kind
        return X_num, X_cat

    def _fit_ridge(self, X_num, X_cat, y, weight=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.preprocessing import SplineTransformer
        self.ridge_types = sorted(X_cat["house_type_name"].unique())
        X_num, X_cat = self._ridge_frames(X_num, X_cat)
        num_cols, cat_cols = list(X_num.columns), list(X_cat.columns)
        parts = [("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                  num_cols),
                 ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat_cols)]
        if self.params.get("plus"):
            spline_cols = [c for c in self.SPLINE_COLS if c in num_cols]
            parts.append(("spline", make_pipeline(SimpleImputer(strategy="median"),
                                                  SplineTransformer(n_knots=8, degree=3)), spline_cols))
        self.model = make_pipeline(ColumnTransformer(parts), Ridge(alpha=self.params["alpha"]))
        self.model.fit(pd.concat([X_num, X_cat], axis=1), y, ridge__sample_weight=weight)
        return self

    def predict(self, X_num, X_cat):
        X_num = X_num[self.num_cols]
        if self.engine == "ridge":
            X_num, X_cat = self._ridge_frames(X_num, X_cat)
            return self.model.predict(pd.concat([X_num, X_cat], axis=1))
        if self.engine == "catboost":
            from catboost import Pool
            return self.model.predict(Pool(pd.concat([X_num.astype(float), X_cat], axis=1),
                                           cat_features=self.cat_cols))
        return self.model.predict(self._frame(X_num, X_cat, fit=False))


# ============================================================ main model ======

TARGETS = ("log_price", "log_list_ratio", "log_ref_ratio")


class TabularModel:
    """Features (features.py + history.py) -> one learner -> price.

    target
        log_price       learn log(sold)                       predict exp(f)
        log_list_ratio  learn log(sold / list)  [list-aware]  predict list * exp(f)
        log_ref_ratio   learn log(sold / ref)                 predict ref * exp(f)
                        ref = comparable-sales reference price from history.py (scheme C)
    groups      feature groups: base, text, list, hist, tfidf (learned text model, text.py)
    early_stop  pick the number of rounds on the last `stop_days` of the training period,
                then refit on all of it
    """

    def __init__(self, track, engine="hgb", target="log_price", groups=("base", "text"),
                 params=None, seed=0, early_stop=True, stop_days=61, drop=(), threads=None,
                 sample_weight_halflife=None, market=None):
        if target not in TARGETS:
            raise ValueError(target)
        if target == "log_list_ratio" and track != "list_aware":
            raise ValueError("log_list_ratio needs the list price: list-aware only")
        if track == "off_market" and "list" in groups:
            raise ValueError("the off-market model may not use the list price")
        self.track, self.engine, self.target = track, engine, target
        self.groups = tuple(groups)
        self.params, self.seed, self.early_stop, self.stop_days = params, seed, early_stop, stop_days
        self.drop = tuple(drop)
        self.threads = threads
        self.halflife = sample_weight_halflife
        # market: None, or MarketIndex options, e.g. {"level_months": 3}. Learn prices relative
        # to the market index of the sale month; add back a forecast for the expected sale month.
        if market is not None and target != "log_price":
            raise ValueError("market adjustment is implemented for the log_price target only")
        self.market_options = market

    # ------------------------------------------------------------ features
    def _features(self, homes, as_of, training=False):
        X_num, X_cat = features.build(homes, [g for g in self.groups if g not in ("hist", "tfidf")],
                                      self.track)
        if "tfidf" in self.groups:
            # training rows: out-of-fold text scores; homes to value: the full text model
            T = self._text_train.loc[homes.index] if training else self.text_model.transform(homes)
            X_num = pd.concat([X_num, T], axis=1)
        ref = None
        if "hist" in self.groups or self.target == "log_ref_ratio":
            H = self.reference.cached_features(homes, as_of)
            ref = history.reference_log_price(H).fillna(self.fallback_logp)
            if "hist" in self.groups:
                X_num = pd.concat([X_num, H], axis=1)
                X_num["h_ref_logp"] = ref
            if self.track == "list_aware" and "list" in self.groups:
                X_num["list_vs_ref"] = np.log(homes["list_price"].astype(float)) - ref
        X_num = X_num.drop(columns=[c for c in self.drop if c in X_num.columns])
        return X_num, X_cat, ref

    def _base(self, homes, ref):
        """log of the price the target is measured against."""
        if self.target == "log_price":
            return np.zeros(len(homes))
        if self.target == "log_list_ratio":
            return np.log(homes["list_price"].to_numpy(dtype=float))
        return ref.to_numpy(dtype=float)

    # ------------------------------------------------------------ fit / predict
    def fit(self, train):
        train = train[np.isfinite(train["sold_price"]) & (train["sold_price"] > 0)]
        if self.target == "log_list_ratio":
            train = train[train["list_price"].astype(float) > 0]
        self.cutoff = train["date_sold"].max().normalize() + pd.Timedelta(days=1)
        self.reference = history.HistoryReference(train) \
            if ("hist" in self.groups or self.target == "log_ref_ratio") else None
        self.fallback_logp = float(np.log(train["sold_price"]).median())
        if "tfidf" in self.groups:
            self.text_model = text.TextPriceModel(seed=self.seed)
            self._text_train = self.text_model.fit_transform(train)
        as_of = history.as_of_for_training(train)
        X_num, X_cat, ref = self._features(train, as_of, training=True)
        if "tfidf" in self.groups:
            del self._text_train          # not needed after fitting; keeps saved models small
        base = self._base(train, ref)
        self.market = None
        if self.market_options is not None:
            self.market = market.MarketIndex(**self.market_options).fit(train)
            base = base + self.market.of_months(market.month_number(train["date_sold"]))
        y = np.log(train["sold_price"].to_numpy(dtype=float)) - base
        keep = np.isfinite(y)
        X_num, X_cat, y = X_num[keep], X_cat[keep], y[keep]
        sold = train["date_sold"][keep]
        weight = None
        if self.halflife:
            age = (self.cutoff - sold).dt.days.to_numpy()
            weight = 0.5 ** (age / self.halflife)

        params = self.params
        self.best_rounds = None
        if self.early_stop and self.engine != "ridge":
            split = self.cutoff - pd.Timedelta(days=self.stop_days)
            inner = (sold < split).to_numpy()
            if inner.sum() > 1000 and (~inner).sum() > 200:
                probe = Learner(self.engine, params, self.seed, self.threads)
                probe.fit(X_num[inner], X_cat[inner], y[inner],
                          val=(X_num[~inner], X_cat[~inner], y[~inner]),
                          weight=None if weight is None else weight[inner])
                self.best_rounds = probe.best_rounds
                # Out-of-time errors of the probe on the last stop_days of training: used for
                # prediction intervals (split conformal, by property type).
                residual = y[~inner] - probe.predict(X_num[~inner], X_cat[~inner])
                kinds = train["house_type_name"][keep][~inner].astype(str).to_numpy()
                self._fit_intervals(residual, kinds)
                key = {"hgb": "max_iter", "catboost": "iterations", "lightgbm": "n_estimators"}[self.engine]
                # the full fit has more data than the probe: scale rounds up a little
                rounds = int(round(self.best_rounds * len(y) / inner.sum()))
                params = {**(params or {}), key: max(50, rounds)}
        self.learner = Learner(self.engine, params, self.seed, self.threads)
        self.learner.fit(X_num, X_cat, y, weight=weight)
        self.feature_names = list(X_num.columns) + list(X_cat.columns)
        return self

    INTERVAL_QS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)

    def _fit_intervals(self, residual, kinds, min_n=50):
        qs = self.INTERVAL_QS
        self.residual_q = {"__all__": np.quantile(residual, qs)}
        for kind in np.unique(kinds):
            r = residual[kinds == kind]
            if len(r) >= min_n:
                self.residual_q[kind] = np.quantile(r, qs)

    def predict_with_interval(self, homes, coverage=0.8):
        """(prediction, lower, upper). The interval adds the out-of-time residual quantiles of
        the home's property type to the log prediction. None if no intervals were fitted."""
        log_pred = self.predict_log(homes)
        q = getattr(self, "residual_q", None)
        if not q:
            return np.exp(log_pred), None, None
        lo_q, hi_q = round((1 - coverage) / 2, 3), round(1 - (1 - coverage) / 2, 3)
        i_lo, i_hi = self.INTERVAL_QS.index(lo_q), self.INTERVAL_QS.index(hi_q)
        table = np.array([q.get(k, q["__all__"]) for k in homes["house_type_name"].astype(str)])
        return np.exp(log_pred), np.exp(log_pred + table[:, i_lo]), np.exp(log_pred + table[:, i_hi])

    def predict_log(self, homes):
        as_of = history.as_of_for_prediction(homes, self.cutoff)
        X_num, X_cat, ref = self._features(homes, as_of)
        base = self._base(homes, ref)
        if self.market is not None:
            base = base + self.market.forecast(self.market.expected_sale_month(homes, self.cutoff))
        f = self.learner.predict(X_num, X_cat)
        out = base + f
        if not np.all(np.isfinite(out)):
            # no usable reference: fall back to the comps' own log price, then the type level
            bad = ~np.isfinite(out)
            raise ValueError(f"{bad.sum()} predictions are not finite")
        return out

    def predict(self, homes):
        return np.exp(self.predict_log(homes))


class BlendModel:
    """Weighted blend of fitted members. mean="geometric" averages log prices."""

    def __init__(self, members, weights, mean="geometric"):
        self.members, self.weights, self.mean = members, np.asarray(weights, float), mean

    def fit(self, train):
        for m in self.members:
            m.fit(train)
        return self

    def predict_with_interval(self, homes, coverage=0.8):
        """Blend the members' predictions and their interval bounds the same way (weighted
        geometric mean). None for the bounds if any member has no interval."""
        outs = [m.predict_with_interval(homes, coverage) if hasattr(m, "predict_with_interval")
                else (m.predict(homes), None, None) for m in self.members]
        w = self.weights / self.weights.sum()

        def mix(arrays):
            arrays = np.array(arrays)
            if self.mean == "geometric":
                return np.exp((w[:, None] * np.log(arrays)).sum(axis=0))
            return (w[:, None] * arrays).sum(axis=0)

        pred = mix([o[0] for o in outs])
        if any(o[1] is None for o in outs):
            return pred, None, None
        return pred, mix([o[1] for o in outs]), mix([o[2] for o in outs])

    def predict(self, homes):
        preds = np.array([m.predict(homes) for m in self.members])
        w = self.weights[:, None] / self.weights.sum()
        if self.mean == "geometric":
            return np.exp((w * np.log(preds)).sum(axis=0))
        return (w * preds).sum(axis=0)


# kept for the starter tests
class FeatureModel:
    def __init__(self, make_features):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))
