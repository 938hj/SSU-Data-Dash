"""Learned text features from the listing remarks (feature group "tfidf").

The keyword counts in features.py only see phrases we thought of. Here a model learns,
from the training sales, which words and word pairs go with homes that sell above or
below what their neighbourhood and type would suggest.

Target      r = log(price per sq ft) - mean of the same for its community x type
            (means from the training sales), so the text explains what location and
            type do not.
Model       TF-IDF on words and word pairs -> Ridge regression.
            Also a 16-dimension SVD of the TF-IDF matrix ("topics") as plain features.
No leakage  Training rows get out-of-fold scores: the rows are split into K folds and
            each fold is scored by a model fitted on the other folds, so a row's own
            price never shapes its own feature. Homes to value are scored by a model
            fitted on all training rows. Only training-period labels are ever used.
"""
import numpy as np
import pandas as pd

import features


class TextPriceModel:
    def __init__(self, folds=5, alpha=3.0, svd_dims=16, min_df=20, seed=0):
        self.folds, self.alpha, self.svd_dims, self.min_df, self.seed = folds, alpha, svd_dims, min_df, seed

    @staticmethod
    def _texts(homes):
        return homes["remarks"].astype("string").fillna("").str.lower().to_numpy()

    def _target(self, train):
        log_ppsf = np.log(train["sold_price"].to_numpy(float)) - \
            np.log(features.square_feet(train["house_area"]).to_numpy(float))
        group = (train["community"].astype(str) + "|" + train["house_type_name"].astype(str)).to_numpy()
        frame = pd.DataFrame({"y": log_ppsf, "g": group})
        ok = np.isfinite(frame["y"])
        self.group_mean = frame[ok].groupby("g")["y"].mean()
        r = frame["y"] - frame["g"].map(self.group_mean)
        return r.to_numpy(), ok.to_numpy() & np.isfinite(r.to_numpy())

    def fit_transform(self, train):
        """Fit on the training sales; return out-of-fold features for those same rows."""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        texts = self._texts(train)
        y, ok = self._target(train)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, sublinear_tf=True,
                                          token_pattern=r"(?u)\b[a-z][a-z\-']+\b", dtype=np.float32)
        X = self.vectorizer.fit_transform(texts)
        # topics need at least 2 words in the vocabulary (only tiny toy data has fewer)
        dims = min(self.svd_dims, X.shape[1] - 1)
        self.svd = TruncatedSVD(dims, random_state=self.seed).fit(X[ok]) if dims >= 1 else None

        oof = np.full(len(train), np.nan)
        rng = np.random.default_rng(self.seed)
        fold = rng.integers(0, self.folds, len(train))
        for k in range(self.folds):
            fit_rows = ok & (fold != k)
            model = Ridge(alpha=self.alpha).fit(X[fit_rows], y[fit_rows])
            oof[fold == k] = model.predict(X[fold == k])
        self.ridge = Ridge(alpha=self.alpha).fit(X[ok], y[ok])
        return self._frame(train.index, oof, self._topics(X))

    def transform(self, homes):
        X = self.vectorizer.transform(self._texts(homes))
        return self._frame(homes.index, self.ridge.predict(X), self._topics(X))

    def _topics(self, X):
        return self.svd.transform(X) if self.svd is not None else np.zeros((X.shape[0], 0))

    def _frame(self, index, score, topics):
        out = pd.DataFrame({"tfidf_score": score}, index=index)
        for j in range(topics.shape[1]):
            out[f"tfidf_topic{j}"] = topics[:, j]
        return out

    def top_words(self, n=15):
        """The words and pairs the text model reads as most above / below par (for the report)."""
        names = np.array(self.vectorizer.get_feature_names_out())
        order = np.argsort(self.ridge.coef_)
        return list(names[order[::-1][:n]]), list(names[order[:n]])
