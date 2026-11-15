# predictor.py
# Loads the saved model and makes predictions. app.py uses this.

import os

import joblib

from features import CATEGORICAL, RAW_FEATURES, build_record, is_missing


class Predictor:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run 'python train.py --data diabetes.csv' first.")

        saved = joblib.load(path)
        self.model = saved["model"]
        self.threshold = saved["threshold"]

        # categories the model saw in training, like "gender=male", "frame=small"
        self.known_categories = {
            name for name in self.model.named_steps["dv"].vocabulary_ if "=" in name}

    def predict(self, raw):
        missing = [k for k in RAW_FEATURES if is_missing(raw.get(k))]
        warnings = []

        if missing:
            warnings.append(
                f"{len(missing)} field(s) missing and ignored: {', '.join(missing)}.")
        if len(missing) > len(RAW_FEATURES) // 3:
            warnings.append("Many fields are missing, so this result is less reliable.")

        # warn if a category value was never seen in training
        for c in CATEGORICAL:
            v = raw.get(c)
            if is_missing(v):
                continue
            if f"{c}={str(v).strip().lower()}" not in self.known_categories:
                warnings.append(f"Unknown value {v!r} for '{c}' was ignored.")

        record = build_record(raw)  # raises ValueError if a number is not a number
        proba = float(self.model.predict_proba([record])[0, 1])
        pred = int(proba >= self.threshold)

        return {
            "probability_diabetic": round(proba, 4),
            "threshold": round(self.threshold, 4),
            "prediction": pred,
            "label": "Diabetic (refer for confirmatory test)" if pred else "Non-diabetic",
            "missing_fields": missing,
            "warnings": warnings,
        }
