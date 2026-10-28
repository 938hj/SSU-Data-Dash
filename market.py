"""A monthly market price index, its seasonal pattern, and a forecast for future months.

Why: tree models cannot extrapolate in time. Trained through May, they predict roughly
May's price level for June to September, but prices here follow a strong yearly cycle
(peak in April, trough in November-December, about 8% apart). So the model learns
prices *relative to the market of their sale month*, and we add back a forecast of the
market for the month each home is expected to sell.

Index      two-way fixed effects on log price per sq ft: month effect + community x type
           effect, fitted by alternating means on the training sales only.
Seasonal   index minus its centred 2x12 moving average, averaged by calendar month.
Level      mean of (index - seasonal) over the last `level_months` training months.
Forecast   level + seasonal(calendar month) [+ damped recent trend, optional].
"""
import numpy as np
import pandas as pd

import features


def month_number(dates):
    return dates.dt.to_period("M").astype("int64").to_numpy()     # months since 1970-01


class MarketIndex:
    def __init__(self, level_months=3, seasonal=True, trend_months=0, damping=0.5,
                 sale_lag_days=12, iterations=6):
        self.level_months = level_months
        self.seasonal = seasonal
        self.trend_months = trend_months
        self.damping = damping
        self.sale_lag_days = sale_lag_days
        self.iterations = iterations

    def fit(self, train):
        y = np.log(train["sold_price"].to_numpy(float)) - \
            np.log(features.square_feet(train["house_area"]).to_numpy(float))
        month = month_number(train["date_sold"])
        group = (train["community"].astype(str) + "|" + train["house_type_name"].astype(str)).to_numpy()
        ok = np.isfinite(y)
        y, month, group = y[ok], month[ok], group[ok]
        m_codes, m_values = pd.factorize(month, sort=True)
        g_codes, _ = pd.factorize(group)
        m_eff = np.zeros(len(m_values))
        for _ in range(self.iterations):
            g_eff = np.bincount(g_codes, y - m_eff[m_codes]) / np.bincount(g_codes)
            m_eff = np.bincount(m_codes, y - g_eff[g_codes]) / np.bincount(m_codes)
        index = pd.Series(m_eff - m_eff.mean(), index=m_values)
        index = index.reindex(np.arange(index.index.min(), index.index.max() + 1)).interpolate()
        self.index = index
        moy = (index.index % 12) + 1                                   # calendar month 1..12
        if self.seasonal and len(index) >= 30:
            trend = index.rolling(12, center=True).mean().rolling(2).mean().shift(-1)   # 2x12 MA
            detrended = (index - trend).dropna()
            s = detrended.groupby((detrended.index % 12) + 1).mean()
            self.season = (s - s.mean()).reindex(range(1, 13)).fillna(0.0)
        else:
            self.season = pd.Series(0.0, index=range(1, 13))
        deseason = index - self.season.reindex(moy).to_numpy()
        self.deseason = deseason
        self.last_month = int(index.index.max())
        self.level = float(deseason.iloc[-self.level_months:].mean())
        self.slope = 0.0
        if self.trend_months:
            tail = deseason.iloc[-self.trend_months:]
            self.slope = float(np.polyfit(np.arange(len(tail)), tail.to_numpy(), 1)[0])
        return self

    def of_months(self, months):
        """Fitted index for months inside the training period (used to deflate training prices)."""
        return self.index.reindex(months).to_numpy()

    def forecast(self, months):
        months = np.asarray(months)
        ahead = np.clip(months - self.last_month, 0, None)
        # damped trend: slope * (d + d^2 + ... + d^h)
        if self.slope and self.damping < 1:
            d = self.damping
            trend = self.slope * d * (1 - d ** ahead) / (1 - d)
        else:
            trend = self.slope * ahead
        return self.level + trend + self.season.reindex((months % 12) + 1).to_numpy()

    def expected_sale_month(self, homes, cutoff):
        """The month a home is expected to sell: listing date plus a typical time on market,
        but never before the end of the training data."""
        expected = homes["date_listed"] + pd.Timedelta(days=self.sale_lag_days)
        expected = expected.where(expected >= cutoff, pd.Timestamp(cutoff))
        return month_number(expected)
