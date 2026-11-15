"""Shared feature definitions, used by train.py, predictor.py and app.py so that
training and serving build exactly the same features.

Records are plain dicts fed to DictVectorizer:
  * numeric field missing  -> NaN   (the pipeline's imputer then fills the training median)
  * categorical missing    -> key left out (all one-hot columns are 0 = "ignored")
"""
import math

import pandas as pd

NUMERICAL = [
    "chol", "hdl", "ratio", "age", "height", "weight",
    "bp.1s", "bp.1d", "waist", "hip", "time.ppn",
]
CATEGORICAL = ["gender", "frame"]
RAW_FEATURES = NUMERICAL + CATEGORICAL
ENGINEERED = ["BMI", "waist_hip", "whtr"]

# Units of the dataset: height in inches, weight in pounds, waist/hip in inches.

NAN = float("nan")


def is_missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _num(raw, key):
    v = raw.get(key)
    if is_missing(v):
        return NAN
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number, got {v!r}")


def build_record(raw: dict) -> dict:
    """raw dict (any subset of RAW_FEATURES, None/NaN allowed) -> dict for DictVectorizer."""
    rec = {k: _num(raw, k) for k in NUMERICAL}

    h, w, waist, hip = rec["height"], rec["weight"], rec["waist"], rec["hip"]
    ok = lambda x: not math.isnan(x) and x > 0
    rec["BMI"] = (w * 703) / (h ** 2) if ok(h) and not math.isnan(w) else NAN
    rec["waist_hip"] = waist / hip if ok(hip) and not math.isnan(waist) else NAN
    rec["whtr"] = waist / (h + 1e-5) if ok(h) and not math.isnan(waist) else NAN

    for k in CATEGORICAL:
        v = raw.get(k)
        if not is_missing(v):
            rec[k] = str(v).strip().lower()
    return rec


def build_records(df: pd.DataFrame) -> list:
    cols = [c for c in RAW_FEATURES if c in df.columns]
    return [build_record(r) for r in df[cols].to_dict(orient="records")]
