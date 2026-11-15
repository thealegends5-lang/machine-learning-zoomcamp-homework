"""Train the best Logistic Regression for diabetes screening.

Usage:
    python train.py --data diabetes.csv

What it does
  1. Same cleaning / target / stratified split (random_state=1) as theLab.ipynb.
  2. DictVectorizer + imputation + scaling + model live inside ONE pipeline,
     so cross-validation never leaks information between folds and the API
     uses exactly the same preprocessing.
  3. GridSearchCV over C / solver / class_weight with repeated stratified CV,
     refit on F2 (recall-weighted).
  4. The decision threshold is chosen from out-of-fold probabilities on the
     TRAIN data only (highest precision with recall >= --min-recall).
  5. The test set is used ONCE at the end.
  6. Saves model.joblib, metrics.json and samples.json (2 test + 2 train rows).
"""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    fbeta_score, make_scorer, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import RAW_FEATURES, build_records

warnings.filterwarnings("ignore")


def load_data(path):
    df = pd.read_csv(path)
    # same cleaning as the notebook
    df = df.drop(columns=["bp.2s", "bp.2d", "id", "location"], errors="ignore")
    df = df.dropna(subset=["glyhb"])
    # target: diabetic if HbA1c >= 6.5 or fasting glucose >= 126
    df["target"] = ((df["glyhb"] >= 6.5) | (df["stab.glu"] >= 126)).astype(int)
    return df


def build_pipeline():
    # DictVectorizer: list of dicts -> matrix (strings are one-hot encoded,
    # unseen categories are ignored, missing numerics arrive as NaN and are
    # filled by the imputer with the TRAIN median).
    return Pipeline([
        ("dv", DictVectorizer(sparse=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ])


def choose_threshold(y, proba, min_recall):
    """Highest precision with recall >= min_recall; fallback = best F2."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= min_recall
    if ok.any():
        idx = np.where(ok)[0][np.argmax(precision[ok])]
        return float(thresholds[idx]), "max precision with recall >= %.2f" % min_recall
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-12)
    return float(thresholds[np.argmax(f2)]), "best F2 (min recall not reachable)"


def report(name, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out = {
        "threshold": round(float(threshold), 4),
        "confusion_matrix": cm.tolist(),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "f2": round(fbeta_score(y_true, pred, beta=2), 4),
    }
    print(f"\n=== {name} (threshold = {threshold:.3f}) ===")
    print("Confusion matrix  [[TN FP] [FN TP]]")
    print(cm)
    print(f"Accuracy : {out['accuracy']}")
    print(f"Precision: {out['precision']}")
    print(f"Recall   : {out['recall']}")
    print(f"F1       : {out['f1']}")
    print(f"F2       : {out['f2']}")
    print(f"Diabetic patients predicted as healthy: {cm[1, 0]} out of {cm[1].sum()}")
    return out


def pick_samples(X_raw, y, source, seed=1):
    """One diabetic + one healthy row from this split, as raw JSON-ready dicts."""
    rng = np.random.RandomState(seed)
    rows = []
    for label in (1, 0):
        idx = np.where(y == label)[0]
        i = int(rng.choice(idx))
        feats = X_raw.iloc[i][RAW_FEATURES].astype(object)
        feats = {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                 for k, v in feats.items()}
        rows.append({"source": source, "true_label": int(label), "features": feats})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="diabetes.csv")
    ap.add_argument("--min-recall", type=float, default=0.85)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    df = load_data(args.data)
    print("Rows:", len(df), "| diabetic:", int(df["target"].sum()),
          f"({df['target'].mean():.1%})")

    # train 80% / test 20%, stratified (same random_state as the notebook)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df["target"])
    y_train, y_test = df_train["target"].values, df_test["target"].values

    # glyhb / stab.glu created the target -> never used as features
    X_train = build_records(df_train)   # list of dicts for DictVectorizer
    X_test = build_records(df_test)

    # ---- hyper-parameter search ----------------------------------------
    param_grid = {
        "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
        "clf__solver": ["lbfgs", "liblinear", "newton-cg"],
        "clf__class_weight": ["balanced", None, {0: 1, 1: 2}, {0: 1, 1: 3}],
    }
    scoring = {
        "f2": make_scorer(fbeta_score, beta=2),
        "recall": "recall",
        "f1": "f1",
        "accuracy": "accuracy",
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    grid = GridSearchCV(build_pipeline(), param_grid, scoring=scoring,
                        refit="f2", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    i = grid.best_index_
    print("\nBest params:", grid.best_params_)
    print("CV (5x5) -> F2 {:.3f} | recall {:.3f} | F1 {:.3f} | accuracy {:.3f}".format(
        grid.cv_results_["mean_test_f2"][i], grid.cv_results_["mean_test_recall"][i],
        grid.cv_results_["mean_test_f1"][i], grid.cv_results_["mean_test_accuracy"][i]))
    best = grid.best_estimator_

    # ---- threshold from out-of-fold probabilities (train only) ----------
    oof = cross_val_predict(
        best, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=1),
        method="predict_proba")[:, 1]
    threshold, how = choose_threshold(y_train, oof, args.min_recall)
    print(f"\nChosen threshold: {threshold:.3f}  ({how})")

    metrics = {
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "threshold_rule": how,
        "train_out_of_fold_default_0.5": report("TRAIN out-of-fold @0.5", y_train, oof, 0.5),
        "train_out_of_fold_chosen": report("TRAIN out-of-fold @chosen", y_train, oof, threshold),
    }

    # ---- final fit + single look at the test set -------------------------
    best.fit(X_train, y_train)
    p_test = best.predict_proba(X_test)[:, 1]
    metrics["test_default_0.5"] = report("TEST @0.5", y_test, p_test, 0.5)
    metrics["test_chosen"] = report("TEST @chosen", y_test, p_test, threshold)
    metrics["test_roc_auc"] = round(roc_auc_score(y_test, p_test), 4)
    metrics["test_pr_auc"] = round(average_precision_score(y_test, p_test), 4)
    print(f"\nTest ROC-AUC {metrics['test_roc_auc']} | PR-AUC {metrics['test_pr_auc']}")
    print("\n", classification_report(
        y_test, (p_test >= threshold).astype(int),
        target_names=["Non-Diabetic (0)", "Diabetic (1)"], zero_division=0))

    # ---- save model + API samples ---------------------------------------
    joblib.dump({"model": best, "threshold": threshold,
                 "features": RAW_FEATURES}, args.out)
    samples = (pick_samples(df_test, y_test, "test")
               + pick_samples(df_train, y_train, "train"))
    with open("samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {args.out}, metrics.json, samples.json")


if __name__ == "__main__":
    main()
