"""Features: turn a table of homes into numbers a model can learn from.

Every function here only reads columns that the test files have, so it can never
see sold_price, date_sold, days_on_market or the external estimate of the home it
describes. Features built from *other* homes' past sales live in history.py.

Groups (switch them on and off in configs.py):
    base   size, rooms, lot, location, listing date
    text   condition and amenity words in the listing remarks
    list   list price (list-aware track only; never off-market)
"""
import re

import numpy as np
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]
CATEGORICAL = ["house_type_name", "house_style", "municipality", "community"]
MISSING = "__MISSING__"
EPOCH = pd.Timestamp("2021-01-01")


# ---------------------------------------------------------------- size --------

def area_bounds(house_area):
    """house_area is text: one number ("622") or a range ("900-1099").
    Return (low, high) as floats; both NaN when the text is not a number."""
    parts = house_area.astype("string").str.strip().str.split("-", expand=True)
    parts = parts.reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce").astype(float)
    high = pd.to_numeric(parts[1], errors="coerce").astype(float).fillna(low)
    return low, high


def square_feet(house_area):
    """The number itself, or the middle of the range."""
    low, high = area_bounds(house_area)
    return (low + high) / 2


def months_since_epoch(dates):
    return (dates - EPOCH).dt.days / 30.44


# ---------------------------------------------------------------- text --------
# Phrase groups from reading listing remarks. Counts, not a hand-made score, so the
# model decides how much each group is worth.
TEXT_GROUPS = {
    "reno_full": [r"fully (?:redone|renovated|updated)", r"redone (?:from )?top to bottom",
                  r"completely (?:redone|renovated)", r"turn-?key", r"nothing (?:left )?to do but move in",
                  r"gut renovat", r"renovated from top"],
    "reno_part": [r"renovated", r"updated", r"refreshed", r"upgraded", r"\bnewer\b",
                  r"\bnew (?:windows|roof|stainless|countertops|flooring|vinyl|furnace|doors|kitchen|"
                  r"appliances|electrical|plumbing)", r"freshly painted", r"fresh paint"],
    "premium": [r"custom kitchen", r"quartz", r"stone counters", r"spa-like", r"heated (?:bathroom )?floors",
                r"pot lights", r"designer light", r"wide-plank", r"smooth ceilings", r"built-in appliances",
                r"frameless glass", r"high-efficiency", r"large island"],
    "maintained": [r"clean and cared for", r"well[- ]maintained", r"cared[- ]for", r"move-in ready",
                   r"move-in condition", r"well-kept", r"in good (?:shape|condition)", r"in fine shape"],
    "original": [r"\boriginal\b", r"\bdated\b", r"\bolder\b", r"\btired\b", r"vintage", r"parquet",
                 r"broadloom"],
    "needs_work": [r"as-is", r"\bas is\b", r"\bneeds?\b", r"fixer", r"\bworn\b", r"overhaul", r"repair",
                   r"\bproject\b", r"unfinished", r"end of (?:their|its) life", r"overgrown", r"waterproofing",
                   r"renovator", r"visionary", r"contractor", r"diamond in the rough", r"sweat equity",
                   r"bring your", r"awaits your", r"ready for your (?:vision|ideas|personal)"],
    "potential": [r"potential", r"great bones", r"solid bones", r"blank canvas", r"\bcanvas\b",
                  r"opportunity", r"make (?:it|this) your own"],
    "basement_fin": [r"finished basement", r"renovated (?:basement|bsmt)", r"rec (?:room|rm)"],
    "income": [r"separate (?:side )?entrance", r"side entrance", r"in-law", r"income"],
    "outdoor": [r"south-facing", r"ravine", r"backyard", r"fenced yard", r"\bdeck\b", r"patio", r"garden",
                r"rooftop terrace", r"mature trees"],
    "building": [r"concierge", r"\bgym\b", r"party room", r"guest suite", r"rooftop"],
    "condo_extra": [r"\blocker\b", r"ensuite laundry", r"\bden\b", r"unobstructed view",
                    r"floor-to-ceiling"],
    "motivated": [r"motivated", r"flexible on closing", r"priced to sell", r"act now", r"don't (?:miss|wait)"],
}
_TEXT_RE = {group: re.compile("|".join(patterns)) for group, patterns in TEXT_GROUPS.items()}


def text_features(remarks):
    """Counts of each phrase group in the remarks, a combined condition score, length."""
    text = remarks.astype("string").fillna("").str.lower()
    X = pd.DataFrame(index=remarks.index)
    for group, pattern in _TEXT_RE.items():
        X["txt_" + group] = text.str.count(pattern).astype(float)
    X["txt_condition"] = (2 * X["txt_reno_full"] + X["txt_reno_part"] + X["txt_premium"]
                          - X["txt_original"] - 2 * X["txt_needs_work"])
    X["txt_length"] = text.str.len().astype(float)
    return X


# ---------------------------------------------------------------- tables ------

def categorical_frame(homes):
    """The category columns as plain strings, with one label for missing."""
    out = pd.DataFrame(index=homes.index)
    for column in CATEGORICAL:
        out[column] = homes[column].astype("string").fillna(MISSING).astype(str)
    return out


def base_features(homes):
    """Numbers that need nothing but the home's own row."""
    X = pd.DataFrame(index=homes.index)
    low, high = area_bounds(homes["house_area"])
    X["area_low"] = low
    X["area_high"] = high
    X["area_mid"] = (low + high) / 2
    X["area_width"] = high - low
    X["area_is_range"] = (high > low).astype(float)
    X["log_area"] = np.log(X["area_mid"])
    for column in ["bedrooms", "washrooms", "parking_total", "garage", "lot_front", "lot_depth"]:
        X[column] = pd.to_numeric(homes[column], errors="coerce").astype(float)
    X["lot_area"] = X["lot_front"] * X["lot_depth"]
    X["log_lot_area"] = np.log(X["lot_area"])
    X["has_lot"] = X["lot_area"].notna().astype(float)
    X["area_per_bedroom"] = X["area_mid"] / X["bedrooms"].clip(lower=1)
    X["latitude"] = homes["latitude"].astype(float)
    X["longitude"] = homes["longitude"].astype(float)
    # Rotated coordinates let trees cut along diagonals.
    X["lat_plus_lon"] = X["latitude"] + 0.72 * X["longitude"]
    X["lat_minus_lon"] = X["latitude"] - 0.72 * X["longitude"]
    X["listed_months"] = months_since_epoch(homes["date_listed"])
    X["listed_month_of_year"] = homes["date_listed"].dt.month.astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


def list_features(homes):
    """List price features. Only for the list-aware track."""
    X = pd.DataFrame(index=homes.index)
    X["log_list_price"] = np.log(homes["list_price"].astype(float))
    X["log_list_ppsf"] = X["log_list_price"] - np.log(square_feet(homes["house_area"]))
    return X.replace([np.inf, -np.inf], np.nan)


def build(homes, groups, track):
    """Numeric features and category columns for the chosen groups.

    Returns (numeric DataFrame, categorical DataFrame), both indexed like homes.
    """
    if "list" in groups and track != "list_aware":
        raise ValueError("list-price features are not allowed on the off-market track")
    parts = []
    if "base" in groups:
        parts.append(base_features(homes))
    if "text" in groups:
        parts.append(text_features(homes["remarks"]))
    if "list" in groups:
        parts.append(list_features(homes))
    numeric = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=homes.index)
    return numeric, categorical_frame(homes)


# ------------------------------------------------ starter functions, kept ----

def off_market_features(homes):
    X = base_features(homes)
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X
