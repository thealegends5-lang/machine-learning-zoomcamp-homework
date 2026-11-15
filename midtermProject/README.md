# Diabetes Screening Model (Logistic Regression) + FastAPI

A diabetes **screening** model built on simple measurements, without glyhb or glucose, served through a small API.

## 1. Train the model

Put `diabetes.csv` in the project folder, then:

```
pip install -r requirements.txt
python train.py --data diabetes.csv
```

This prints the confusion matrix, accuracy, precision, recall, F1 and F2, and creates
`model.joblib`, `metrics.json` and `samples.json` (2 test + 2 train examples).
**The API cannot start until `model.joblib` exists.**

## 2. Run the API

### Option A: locally

```
uvicorn app:app --reload          # http://localhost:8000
python test_api.py                # second terminal: calls the API on the 4 saved samples
```

### Option B: with Docker

The saved model must be loaded with the same scikit-learn version that trained it.
The safest way is to train **inside** the container, so the versions always match.
Run these commands from the project folder (with `diabetes.csv` in it):

```
# 1) build the image
docker build -t diabetes-api .

# 2) train inside the container; model.joblib, metrics.json and samples.json are written to your folder
docker run --rm -v "$(pwd)":/app diabetes-api python train.py --data diabetes.csv

# 3) rebuild so the model is baked into the image
docker build -t diabetes-api .

# 4) run the API
docker run --rm -p 8000:8000 diabetes-api
```

On Windows PowerShell use `${PWD}` instead of `$(pwd)`.

Shortcut without rebuilding: after step 2, run the API with the folder mounted:
`docker run --rm -p 8000:8000 -v "$(pwd)":/app diabetes-api`.

Check that it works (either option):

```
curl http://localhost:8000/health
python test_api.py
```

Interactive page to try the API in the browser: http://localhost:8000/docs
(open `POST /predict`, click "Try it out"; an example body is pre-filled).

## 3. Using the API

| Endpoint | Description |
|---|---|
| `GET /health` | Status, decision threshold and expected fields |
| `GET /samples` | 2 test + 2 train examples with their true labels |
| `POST /predict` | Predict for one patient (JSON body) |

Units: height, waist and hip in inches; weight in pounds.
All fields are optional. Missing fields are ignored (numeric fields use the training median,
categorical fields get no category) and the response lists them in `missing_fields` and `warnings`:

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":58,"weight":210,"height":67}'
```

Troubleshooting:
- `FileNotFoundError: model.joblib not found`: run the training step first.
- Warnings about unpickling or a scikit-learn version mismatch: retrain with the same
  environment that serves the API (this is why Option B trains inside the container).

---

## Files

| File | Purpose |
|---|---|
| `theLab.ipynb` | Exploration, Risk Ratio / Risk Difference analysis, model comparison |
| `train.py` | Trains the final Logistic Regression and computes the metrics |
| `features.py` | Feature building (shared by training and the API) |
| `predictor.py` | Prediction logic and warnings |
| `app.py` | FastAPI service |
| `test_api.py` | Calls the API on samples from the test and train sets |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | Docker image and dependencies for the API |

---

## Problems I faced and how I handled them

### 1) Small dataset (about 390 patients after cleaning)

**Problem:** The dataset is small and the diabetic class is much smaller than the healthy class (only about 17 diabetic patients in the test set). Any metric on the validation or test set is therefore unstable: one patient changes recall by about 6%, so a 1-2% difference between two models may be noise rather than a real improvement.

**What I did:**
- Stratified splits, so the diabetic ratio is the same in train and test.
- `class_weight` (including higher weights for the diabetic class) instead of dropping healthy samples.
- Evaluation with **repeated stratified CV (5x5)** instead of relying on a single split.
- A **learning curve** to check whether more data would help.
- The decision threshold is chosen from cross-validated probabilities on the training set, **not** on the test set, so the reported result is not optimistic.
- I considered generating synthetic training data (SMOTE) for the training set only, but did not apply it because of time. It adds no new information (it only balances the classes) and could create unrealistic relationships with so few positive cases.

### 2) Missing values and useless columns

- `bp.2s` and `bp.2d` are missing in more than half of the rows, so I dropped them.
- `id` and `location` have no medical value, so I dropped them.
- Rows missing `glyhb` were dropped, because it is the basis of the target definition.
- Remaining missing values are filled with the **median** (numeric) computed from training data only, inside the pipeline, so no statistics from validation or test leak into training.

### 3) Data leakage

The target is defined as `glyhb >= 6.5` or `stab.glu >= 126`. Keeping either as a feature would give a near-perfect but fake accuracy, so both are excluded from the features. The model therefore predicts diabetes from **risk factors and simple measurements**, not from the glucose test itself. This is the point of a screening tool, and it is also why the accuracy ceiling is relatively low.

I did not split Type 1 from Type 2 because the dataset has nobody under 25.

### 4) Accuracy stuck around 81% whatever I tried

**What I tried:**
- Four models with GridSearchCV: **Logistic Regression, Decision Tree, Random Forest, XGBoost**.
- Extra engineered medical features: `BMI`, `waist_hip`, `whtr`, and I also prepared pulse pressure, MAP, interaction terms (e.g. BMI x age) and log transforms for skewed variables.

**What I observed:**
- The stronger models did not beat Logistic Regression. This is expected with about 390 rows: complex models need more data and tend to overfit, while a regularized linear model is more stable.
- The new features are derived from `weight`, `height`, `waist` and `hip`, which are already in the data, so a linear model gets little new information from them.
- I therefore chose **Logistic Regression** as the final model: simpler, more stable and medically interpretable.

### 5) Medical goal: recall matters more than accuracy

Missing a diabetic patient (false negative) is worse than sending a healthy person for an extra test (false positive). So:
- Hyperparameters are selected by **F2** (weights recall more heavily) instead of F1 or accuracy.
- The decision threshold is lowered below 0.5 to raise recall at the cost of precision and overall accuracy. Rule: highest precision subject to recall >= 0.85 (change it with `--min-recall`).
- My best earlier result missed 4 of 17 diabetic patients (recall about 0.76), which is below the medical target.

---

## Risk Ratio and Risk Difference analysis

**Goal:** find out whether each feature carries real information about diabetes, or is a low-value feature that only adds noise and may push the model toward overfitting.

**Method** (in the notebook): each numeric feature is split into 4 quartile bins with `qcut`; categorical features (`gender`, `frame`) use their categories. For each bin I computed:

- **Risk Rate** = share of diabetic patients in the bin.
- **Risk Ratio** = Risk Rate / overall risk of the sample.
- **Risk Difference** = Risk Rate - overall risk.

**How to read it:**
- All bins with Risk Ratio close to 1 and Risk Difference close to 0 mean the feature does not separate diabetic from healthy patients (low information value).
- A bin with Risk Ratio clearly above 1 (e.g. the top quartile of age or waist) indicates a real risk factor.
- Small bins (few patients) give unstable ratios, so always read the `Count` column next to each ratio.

**Limits of this analysis (worth stating in the report):**
- It looks at each feature **on its own** and does not capture interactions between features.
- Risk Ratio / Difference measures the **information value** of a feature, not overfitting itself. Overfitting is detected by comparing train performance with CV performance, by the learning curve, and by the spread of results in repeated CV.
- It was computed on all the data before splitting, so it was used for understanding only. If it were used to drop features, it would have to be done **inside** the CV to avoid leakage.

The final metrics are printed by `train.py` and saved in `metrics.json`.

---

## Limitations and future work

- The test set has few diabetic patients (about 17), so results are approximate and confidence intervals are wide.
- `chol`, `hdl` and `ratio` are blood tests, not measurements a patient can easily take alone. A version without them could be built for home use and compared with the current one.
- Try SMOTE inside the pipeline, and run the same pipeline on a larger dataset such as BRFSS.
- This is an educational screening project, not a medical diagnostic tool.

---

## Notes and credits

- I chose FastAPI over Flask because I prefer it (it also gives automatic request validation and the interactive `/docs` page).
- Canva AI was used to improve the look of the website for this API, including the layout and the animations.
- AI was used to help improve this README.
