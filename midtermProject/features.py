# features.py
# Feature building. train.py and the API both use this file,
# so the model sees the same features in training and in serving.

import math

import pandas as pd

NUMERICAL = [
    "chol", "hdl", "ratio", "age", "height", "weight",
    "bp.1s", "bp.1d", "waist", "hip", "time.ppn",
]
CATEGORICAL = ["gender", "frame"]
RAW_FEATURES = NUMERICAL + CATEGORICAL
ENGINEERED = ["BMI", "waist_hip", "whtr"]

# units: height, waist, hip in inches / weight in pounds

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


def _ok(x):
    return not math.isnan(x) and x > 0


def build_record(raw):
    # raw dict (can have any of the raw features) -> dict for DictVectorizer
    # missing numbers become NaN, the imputer fills them later with the train median
    rec = {k: _num(raw, k) for k in NUMERICAL}

    h, w, waist, hip = rec["height"], rec["weight"], rec["waist"], rec["hip"]

    rec["BMI"] = (w * 703) / (h ** 2) if _ok(h) and not math.isnan(w) else NAN
    rec["waist_hip"] = waist / hip if _ok(hip) and not math.isnan(waist) else NAN
    rec["whtr"] = waist / (h + 1e-5) if _ok(h) and not math.isnan(waist) else NAN

    # categorical: if it's missing we just leave the key out (all one-hot = 0)
    for k in CATEGORICAL:
        v = raw.get(k)
        if not is_missing(v):
            rec[k] = str(v).strip().lower()
    return rec


def build_records(df):
    cols = [c for c in RAW_FEATURES if c in df.columns]
    return [build_record(r) for r in df[cols].to_dict(orient="records")]
