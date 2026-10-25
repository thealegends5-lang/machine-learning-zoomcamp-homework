import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# prameters
C=1.0
n_splits = 5
output_file= f'model_C={C}.bin'

# Read the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
category = list(df.dtypes[df.dtypes == 'object'].index)

for c in category:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)


# Convert churn to binary
df.churn = (df.churn == 'yes').astype(int)

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

numerical = ['tenure', 'monthlycharges', 'seniorcitizen']
category = ['gender', 'partner', 'dependents',
            'phoneservice', 'multiplelines', 'internetservice',
            'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
            'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
            'paymentmethod']

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values
del df_train['churn']
del df_val['churn']
del df_test['churn']


df_full_train = df_full_train.reset_index(drop=True)
df_full_train.churn.value_counts(normalize=True)

train_dicts = df_train[category + numerical].to_dict(orient='records')
val_dicts = df_val[category + numerical].to_dict(orient='records')
test_dicts = df_test[category + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)


# Define KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

# Initialize list to store roc_auc_scores
roc_auc_scores = []

# Train and evaluate the model using KFold
for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    model = LogisticRegression()
    model.fit(X_train_fold, y_train_fold)
    
    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
    roc_auc_scores.append(roc_auc)

# Compute average roc_auc_score
average_roc_auc = np.mean(roc_auc_scores)
print(f'Average ROC AUC Score: {average_roc_auc}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
