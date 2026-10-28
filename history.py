"""Features built from other homes' earlier sales, with a strict time boundary.

The one rule: a home's history features use only sales with
    date_sold < as_of(home)
and as_of is never later than the moment the home could be valued:

    training rows   as_of = min(date_listed, first day of the month it sold)
    homes to value  as_of = min(date_listed, cutoff)    cutoff = day after the last training sale

So a home never sees its own sale (it sold on or after as_of) or anything after the
validation month starts. Because the answer depends only on sales before as_of, the
same home gets the same features whichever fold it appears in, and the reference
table can safely be the whole training table: later rows are filtered out by date.
tests/test_leakage.py checks both properties.

Two kinds of features:
    group windows   mean log price, mean log price per sq ft and count of earlier sales in
                    the same community and type (3, 6, 12 months), municipality and type
                    (12 months) and type (3 months); vectorised with cumulative sums.
    nearest comps   the k nearest earlier sales of the same type in the 24 months before
                    the as_of month, by location, each comp's price moved to the as_of
                    month with a type-level monthly price index (KD-tree per type and month).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import features

DAY0 = np.datetime64("2000-01-01", "D")
KEY_SHIFT = 1_000_000          # days fit well below this
KM_PER_DEG_LAT = 111.0
LON_SCALE = 0.72               # cos(43.7 degrees)

GROUP_WINDOWS = [
    (("community", "house_type_name"), [91, 182, 365]),
    (("municipality", "house_type_name"), [365]),
    (("house_type_name",), [91]),
    (("building",), [730, 3650]),
]


def day_number(dates):
    return ((dates.to_numpy().astype("datetime64[D]") - DAY0).astype(np.int64))


def month_start(dates):
    return dates.dt.to_period("M").dt.start_time


def as_of_for_training(train):
    """min(date_listed, first day of the sale month)."""
    return np.minimum(train["date_listed"], month_start(train["date_sold"]))


def as_of_for_prediction(homes, cutoff):
    return np.minimum(homes["date_listed"], pd.Series(cutoff, index=homes.index))


def _sales_table(train):
    sales = pd.DataFrame(index=train.index)
    sales["day"] = day_number(train["date_sold"])
    sales["month"] = train["date_sold"].dt.to_period("M").astype("int64")   # months since 1970
    sales["log_price"] = np.log(train["sold_price"].astype(float))
    area = features.square_feet(train["house_area"])
    sales["log_area"] = np.log(area)
    sales["log_ppsf"] = sales["log_price"] - sales["log_area"]
    for column in ["community", "municipality", "house_type_name"]:
        sales[column] = train[column].astype("string").fillna(features.MISSING).astype(str)
    sales["building"] = building_key(train)
    sales["x"] = train["longitude"].astype(float) * LON_SCALE
    sales["y"] = train["latitude"].astype(float)
    sales["ml_num"] = train["ml_num"].astype(str).to_numpy()
    return sales


def building_key(homes):
    """Homes at exactly the same coordinates and of the same type: one building
    (condos) or one address. Synthetic coordinates, but shared within the data."""
    return (homes["latitude"].round(6).astype(str) + "," + homes["longitude"].round(6).astype(str)
            + "|" + homes["house_type_name"].astype("string").fillna(features.MISSING).astype(str))


class HistoryReference:
    """Holds earlier sales and computes time-safe history features for any homes."""

    def __init__(self, train, knn_k=(10, 30), knn_months=24):
        self.sales = _sales_table(train)
        self.knn_k = tuple(knn_k)
        self.knn_months = knn_months
        self._index = self._monthly_index()

    # ---------------------------------------------------------- group windows
    def _group_stats(self, homes_keys, as_of_day, keys, windows):
        sales = self.sales
        sale_key = sales[list(keys)].astype(str).agg("|".join, axis=1)
        home_key = homes_keys[list(keys)].astype(str).agg("|".join, axis=1)
        codes, uniques = pd.factorize(pd.concat([sale_key, home_key]), sort=True)
        s_code = codes[: len(sales)].astype(np.int64)
        h_code = codes[len(sales):].astype(np.int64)

        composite = s_code * KEY_SHIFT + sales["day"].to_numpy()
        order = np.argsort(composite, kind="stable")
        composite = composite[order]
        lp = sales["log_price"].to_numpy()[order]
        lpp = sales["log_ppsf"].to_numpy()[order]
        ok = np.isfinite(lpp)
        c_lp = np.concatenate([[0.0], np.cumsum(lp)])
        c_lpp = np.concatenate([[0.0], np.cumsum(np.where(ok, lpp, 0.0))])
        c_ok = np.concatenate([[0], np.cumsum(ok)])

        out = {}
        name = "_".join(k.replace("house_type_name", "type") for k in keys)
        for window in windows:
            hi = np.searchsorted(composite, h_code * KEY_SHIFT + as_of_day, side="left")       # day < as_of
            lo = np.searchsorted(composite, h_code * KEY_SHIFT + as_of_day - window, side="left")
            n = hi - lo
            n_ok = c_ok[hi] - c_ok[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"h_{name}_{window}d_n"] = n.astype(float)
                out[f"h_{name}_{window}d_logp"] = np.where(n > 0, (c_lp[hi] - c_lp[lo]) / n, np.nan)
                out[f"h_{name}_{window}d_logppsf"] = np.where(n_ok > 0, (c_lpp[hi] - c_lpp[lo]) / n_ok, np.nan)
        return out

    # ---------------------------------------------------------- price index
    def _monthly_index(self):
        """Type-level monthly mean log price per sq ft, smoothed over 3 months.
        index[type][month] uses only sales in that month and the two before."""
        table = {}
        sales = self.sales[np.isfinite(self.sales["log_ppsf"])]
        for kind, group in sales.groupby("house_type_name"):
            by_month = group.groupby("month")["log_ppsf"].agg(["sum", "count"])
            months = np.arange(by_month.index.min(), by_month.index.max() + 1)
            by_month = by_month.reindex(months, fill_value=0)
            s = by_month["sum"].rolling(3, min_periods=1).sum()
            c = by_month["count"].rolling(3, min_periods=1).sum()
            table[kind] = (s / c.replace(0, np.nan)).ffill()
        return table

    def index_at(self, kind, months):
        series = self._index.get(kind)
        if series is None or len(series) == 0:
            return np.full(len(months), np.nan)
        clipped = np.clip(months, series.index.min(), series.index.max())
        values = series.reindex(clipped).to_numpy().copy()
        values[months < series.index.min()] = np.nan
        return values

    # ---------------------------------------------------------- nearest comps
    def _knn(self, homes, as_of):
        sales = self.sales
        n_homes = len(homes)
        kmax = max(self.knn_k)
        out = {f"knn{k}_{s}": np.full(n_homes, np.nan) for k in self.knn_k
               for s in ["logp", "logppsf", "adj_logp", "adj_logppsf", "dist_km", "age_months"]}
        out["knn_area_ratio"] = np.full(n_homes, np.nan)
        out["idx_mom_3m"] = np.full(n_homes, np.nan)
        out["idx_mom_12m"] = np.full(n_homes, np.nan)

        bucket = as_of.dt.to_period("M").astype("int64").to_numpy()   # comps sold before this month
        kinds = homes["house_type_name"].astype("string").fillna(features.MISSING).astype(str).to_numpy()
        hx = homes["longitude"].to_numpy(dtype=float) * LON_SCALE
        hy = homes["latitude"].to_numpy(dtype=float)
        h_log_area = np.log(features.square_feet(homes["house_area"]).to_numpy(dtype=float))

        for kind in np.unique(kinds):
            group = sales[sales["house_type_name"] == kind].sort_values("month", kind="stable")
            g_month = group["month"].to_numpy()
            g_xy = group[["x", "y"]].to_numpy()
            g_lp = group["log_price"].to_numpy()
            g_lpp = group["log_ppsf"].to_numpy()
            g_la = group["log_area"].to_numpy()
            g_idx = self.index_at(kind, g_month)
            home_rows = np.flatnonzero(kinds == kind)
            for b in np.unique(bucket[home_rows]):
                rows = home_rows[bucket[home_rows] == b]
                lo = np.searchsorted(g_month, b - self.knn_months, side="left")
                hi = np.searchsorted(g_month, b, side="left")          # month < as_of month
                idx_now = self.index_at(kind, np.array([b - 1]))[0]
                idx_3 = self.index_at(kind, np.array([b - 4]))[0]
                idx_12 = self.index_at(kind, np.array([b - 13]))[0]
                out["idx_mom_3m"][rows] = idx_now - idx_3
                out["idx_mom_12m"][rows] = idx_now - idx_12
                if hi - lo == 0:
                    continue
                tree = cKDTree(g_xy[lo:hi])
                k = min(kmax, hi - lo)
                q = np.column_stack([hx[rows], hy[rows]])
                valid = np.isfinite(q).all(axis=1)
                if not valid.any():
                    continue
                dist, nn = tree.query(q[valid], k=k)
                dist = dist.reshape(len(nn), -1)
                nn = nn.reshape(len(nn), -1) + lo
                vrows = rows[valid]
                adj = idx_now - g_idx[nn]
                adj = np.where(np.isfinite(adj), adj, 0.0)
                for kk in self.knn_k:
                    kk_ = min(kk, k)
                    sl = nn[:, :kk_]
                    out[f"knn{kk}_logp"][vrows] = np.median(g_lp[sl], axis=1)
                    out[f"knn{kk}_logppsf"][vrows] = np.nanmedian(g_lpp[sl], axis=1) \
                        if np.isfinite(g_lpp[sl]).any() else np.nan
                    out[f"knn{kk}_adj_logp"][vrows] = np.median(g_lp[sl] + adj[:, :kk_], axis=1)
                    with np.errstate(all="ignore"):
                        out[f"knn{kk}_adj_logppsf"][vrows] = np.nanmedian(g_lpp[sl] + adj[:, :kk_], axis=1)
                    out[f"knn{kk}_dist_km"][vrows] = dist[:, :kk_].mean(axis=1) * KM_PER_DEG_LAT
                    out[f"knn{kk}_age_months"][vrows] = (b - g_month[sl]).mean(axis=1)
                with np.errstate(all="ignore"):
                    out["knn_area_ratio"][vrows] = h_log_area[vrows] - np.nanmedian(g_la[nn[:, :min(10, k)]], axis=1)
        return out

    # ---------------------------------------------------------- public
    def features(self, homes, as_of):
        """History features for homes, using only sales before each home's as_of."""
        as_of = pd.Series(pd.to_datetime(as_of), index=homes.index)
        as_of_day = day_number(as_of)
        keys = pd.DataFrame(index=homes.index)
        for column in ["community", "municipality", "house_type_name"]:
            keys[column] = homes[column].astype("string").fillna(features.MISSING).astype(str)
        keys["building"] = building_key(homes)
        out = {}
        for group_keys, windows in GROUP_WINDOWS:
            out.update(self._group_stats(keys, as_of_day, group_keys, windows))
        out.update(self._knn(homes, as_of))
        X = pd.DataFrame(out, index=homes.index)
        # Prices relative to this home's size: what the comps imply for it.
        log_area = np.log(features.square_feet(homes["house_area"]).astype(float))
        for k in self.knn_k:
            X[f"knn{k}_implied_logp"] = X[f"knn{k}_adj_logppsf"] + log_area
        X["h_comm_implied_logp"] = X["h_community_type_365d_logppsf"] + log_area
        X["bldg_implied_logp"] = X["h_building_3650d_logppsf"] + log_area
        X["as_of_lag_days"] = (homes["date_listed"] - as_of).dt.days.astype(float)
        return X.replace([np.inf, -np.inf], np.nan)

    def cached_features(self, homes, as_of):
        """Same result as features(), but rows whose (ml_num, as_of) is in the cache are
        looked up instead of recomputed. Only used when the cache is valid for this
        reference: the reference must hold *every* source sale before its own end
        (a prefix of the source table), so 'sales before as_of' is the same set."""
        as_of = pd.Series(pd.to_datetime(as_of), index=homes.index)
        cache = _CACHE.get("frame")
        if cache is None or not self._is_prefix_of_cache_source():
            return self.features(homes, as_of)
        end = self.sales["day"].max() + 1
        key = pd.MultiIndex.from_arrays([homes["ml_num"].astype(str).to_numpy(), day_number(as_of)])
        hit = key.isin(cache.index) & (day_number(as_of) <= end)
        parts = []
        if hit.any():
            got = cache.loc[key[hit]]
            got.index = homes.index[hit]
            parts.append(got)
        if (~hit).any():
            parts.append(self.features(homes[~hit], as_of[~hit]))
        return pd.concat(parts).loc[homes.index]

    def _is_prefix_of_cache_source(self):
        src = _CACHE["source_days"]
        end = self.sales["day"].max() + 1
        return int((src < end).sum()) == len(self.sales) and \
            set(self.sales["ml_num"]).issubset(_CACHE["source_ids"])


_CACHE = {}


def enable_cache(listings):
    """Compute training-row history features once for the whole listings table.
    Valid for any fold whose training set is listings[date_sold < cutoff]."""
    as_of = as_of_for_training(listings)
    frame = HistoryReference(listings).features(listings, as_of)
    frame.index = pd.MultiIndex.from_arrays([listings["ml_num"].astype(str).to_numpy(), day_number(as_of)])
    _CACHE.update(frame=frame, source_days=day_number(listings["date_sold"]),
                  source_ids=set(listings["ml_num"].astype(str)))


def disable_cache():
    _CACHE.clear()


def reference_log_price(H):
    """A single reference log price per home (scheme C), with a fallback chain:
    nearest-comps implied price -> community/type 12m -> municipality/type -> type 3m."""
    ref = H["knn10_implied_logp"].copy()
    for column in ["h_comm_implied_logp", "bldg_implied_logp", "knn10_adj_logp", "h_community_type_365d_logp",
                   "h_municipality_type_365d_logp", "h_type_91d_logp"]:
        ref = ref.fillna(H[column])
    return ref
