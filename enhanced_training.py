"""
enhanced_training.py
Enhanced ML training with RI + SIR + Policy features.

FIXES vs original:
  1. Policy feature capture includes ALL four column types:
       policy_mean_*, policy_slope_*, policy_start_*, policy_end_*
       growth_rate_mean, growth_rate_max
     (original only captured policy_mean_*, giving ~35 features instead of 100)
  2. Saves feature_list.pkl alongside the scaler so evaluate_all.py can
     reconstruct the exact same column order without guessing.
  3. Cross-validation scores printed for each model.
  4. Per-class classification report printed for each model.
     Uses manual indentation to avoid the indent= parameter that was
     added only in newer sklearn versions.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("=" * 60)
print("ENHANCED ML TRAINING  [FULL FEATURE SET]")
print("=" * 60)

os.makedirs('saved_models', exist_ok=True)
os.makedirs('result', exist_ok=True)

# ============================================================
# 1. Load dataset
# ============================================================
print("\n[1] Loading enhanced dataset...")

df = pd.read_csv('result/enhanced_with_policies.csv')
print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 2. Clean
# ============================================================
print("\n[2] Cleaning data...")

df = df.replace([np.inf, -np.inf], np.nan)
cols_with_nan = [c for c in df.columns if df[c].isna().any()]
if cols_with_nan:
    for col in cols_with_nan:
        med = df[col].median()
        df[col] = df[col].fillna(med if pd.notna(med) else 0)
    print(f"  Filled NaNs in {len(cols_with_nan)} columns with median.")
else:
    print("  No NaN values found.")

# ============================================================
# 3. Label column
# ============================================================
print("\n[3] Finding label column...")

if 'label' in df.columns:
    label_col = 'label'
elif 'Label' in df.columns:
    label_col = 'Label'
else:
    raise ValueError("No label column found in dataset.")

print(f"  Using '{label_col}'")
print(f"  Distribution:\n{df[label_col].value_counts().sort_index().to_string()}")

# ============================================================
# 4. Train / test split
# ============================================================
print("\n[4] Creating 70/30 train/test split (seed=42)...")

np.random.seed(42)
n_train = int(0.7 * len(df))
train_indices = np.random.choice(df.index, n_train, replace=False)
test_indices  = [i for i in df.index if i not in train_indices]

train = df.iloc[train_indices].copy()
test  = df.iloc[test_indices].copy()

print(f"  Train: {len(train)} windows")
print(f"  Test : {len(test)} windows")

# ============================================================
# 5. Define feature groups
#    FIX: capture ALL four policy column types + growth rate extras
# ============================================================
print("\n[5] Defining feature groups...")

ri_features  = ['Week', r'$\mu^c$', r'$\beta^c$', r'$\sigma^c$',
                r'$Delta^c$', r'$Omicron^c$', r'$Policy^c$', r'$Policy^p$']
sir_features = ['Re', 'growth_rate', 'sir_fit_r2', 'R0',
                'doubling_time', 'sir_reliable']

# FIX: collect all four policy column types, not just policy_mean_
policy_prefixes = ('policy_mean_', 'policy_slope_', 'policy_start_', 'policy_end_')
growth_extras   = ['growth_rate_mean', 'growth_rate_max']

policy_features = (
    [c for c in df.columns if c.startswith(policy_prefixes)]
    + [c for c in growth_extras if c in df.columns]
)

ri_avail     = [f for f in ri_features     if f in df.columns]
sir_avail    = [f for f in sir_features    if f in df.columns]
policy_avail = [f for f in policy_features if f in df.columns]

# Remove zero-variance features (all zeros across all 953 windows).
# policy_extraction.py now strips these at the source, but this
# defensive filter catches any that slip through from older CSV files.
zero_policy  = [c for c in policy_avail if (df[c] == 0).all()]
policy_avail = [c for c in policy_avail if c not in zero_policy]
if zero_policy:
    print(f"\n  Dropped {len(zero_policy)} all-zero policy columns.")

print(f"\n  RI features     : {len(ri_avail)}")
print(f"  SIR features    : {len(sir_avail)}")
print(f"  Policy features : {len(policy_avail)}")

for prefix in policy_prefixes:
    n = sum(1 for c in policy_avail if c.startswith(prefix))
    print(f"    {prefix:25s}: {n}")
for extra in growth_extras:
    if extra in policy_avail:
        print(f"    {extra:25s}: 1")

full_features = ri_avail + sir_avail + policy_avail
print(f"\n  TOTAL features  : {len(full_features)}")

# ============================================================
# 6. Prepare arrays
# ============================================================
X_train = np.nan_to_num(train[full_features].values.astype(float))
X_test  = np.nan_to_num(test[full_features].values.astype(float))
y_train = train[label_col].values
y_test  = test[label_col].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# FIX: save both the scaler and the ordered feature list
joblib.dump(scaler,        'saved_models/scaler_full.pkl')
joblib.dump(full_features, 'saved_models/feature_list.pkl')
print("\n  Saved scaler       -> saved_models/scaler_full.pkl")
print("  Saved feature list -> saved_models/feature_list.pkl")

# ============================================================
# 7. Hyperparameters (from original paper)
# ============================================================
params = {
    'svm': {'C': 50.0, 'gamma': 0.3, 'kernel': 'rbf'},
    'rf':  {'max_depth': 14, 'n_estimators': 85,  'random_state': 42},
    'xgb': {'max_depth': 7,  'n_estimators': 110, 'random_state': 42},
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 8. Train, evaluate and save each model
# ============================================================
print("\n[6] Training models...")

models = {}

# --- SVM ---
print("\n  [SVM]")
svm_model = SVC(**params['svm'], probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
joblib.dump(svm_model, 'saved_models/svm_full.pkl')
models['svm'] = svm_model

svm_pred = svm_model.predict(X_test_scaled)
svm_acc  = accuracy_score(y_test, svm_pred)
svm_cv   = cross_val_score(svm_model, X_train_scaled, y_train,
                            cv=cv, scoring='accuracy')
print(f"    Test accuracy : {svm_acc:.4f}")
print(f"    CV accuracy   : {svm_cv.mean():.4f} +/- {svm_cv.std():.4f}")
report = classification_report(y_test, svm_pred, target_names=['L0','L1','L2'])
print('\n'.join('      ' + line for line in report.splitlines()))

# --- Random Forest ---
print("\n  [Random Forest]")
rf_model = RandomForestClassifier(**params['rf'])
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, 'saved_models/rf_full.pkl')
models['rf'] = rf_model

rf_pred = rf_model.predict(X_test_scaled)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_cv   = cross_val_score(rf_model, X_train_scaled, y_train,
                           cv=cv, scoring='accuracy')
print(f"    Test accuracy : {rf_acc:.4f}")
print(f"    CV accuracy   : {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
report = classification_report(y_test, rf_pred, target_names=['L0','L1','L2'])
print('\n'.join('      ' + line for line in report.splitlines()))

# --- XGBoost ---
print("\n  [XGBoost]")
xgb_model = XGBClassifier(**params['xgb'])
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, 'saved_models/xgb_full.pkl')
models['xgb'] = xgb_model

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_cv   = cross_val_score(xgb_model, X_train_scaled, y_train,
                            cv=cv, scoring='accuracy')
print(f"    Test accuracy : {xgb_acc:.4f}")
print(f"    CV accuracy   : {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f}")
report = classification_report(y_test, xgb_pred, target_names=['L0','L1','L2'])
print('\n'.join('      ' + line for line in report.splitlines()))

# ============================================================
# 9. Save test predictions and probabilities
# ============================================================
print("\n[7] Saving predictions and probabilities...")

test_results = test[['data_num', label_col]].copy()
test_results['true_label'] = y_test

for name, model in models.items():
    pred  = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)
    test_results[f'{name}_pred'] = pred
    np.savetxt(f'result/enhanced_{name}_proba.csv', proba, delimiter=',')

test_results.to_csv('result/enhanced_test_predictions.csv', index=False)
print("  Saved result/enhanced_test_predictions.csv")
print("  Saved result/enhanced_*_proba.csv")

# ============================================================
# 10. Feature importance (RF + XGB side by side)
# ============================================================
print("\n[8] Feature importance...")

rf_imp  = pd.DataFrame({'feature': full_features,
                         'rf_importance': rf_model.feature_importances_})
xgb_imp = pd.DataFrame({'feature': full_features,
                         'xgb_importance': xgb_model.feature_importances_})
importance = pd.merge(rf_imp, xgb_imp, on='feature')
importance['mean_importance'] = (
    importance['rf_importance'] + importance['xgb_importance']
) / 2
importance = importance.sort_values('mean_importance', ascending=False)

print("\n  Top 15 features (mean of RF + XGB):")
print(importance[['feature', 'rf_importance', 'xgb_importance', 'mean_importance']]
      .head(15).to_string(index=False))

importance.to_csv('result/enhanced_feature_importance.csv', index=False)
print("\n  Saved result/enhanced_feature_importance.csv")

# ============================================================
# 11. Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Feature set: {len(full_features)} total "
      f"(RI={len(ri_avail)}, SIR={len(sir_avail)}, Policy={len(policy_avail)})")
print(f"\n  Test accuracy:")
print(f"    SVM : {svm_acc:.4f}  (CV: {svm_cv.mean():.4f} +/- {svm_cv.std():.4f})")
print(f"    RF  : {rf_acc:.4f}  (CV: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f})")
print(f"    XGB : {xgb_acc:.4f}  (CV: {xgb_cv.mean():.4f} +/- {xgb_cv.std():.4f})")
print("\n  Saved files:")
print("    saved_models/scaler_full.pkl")
print("    saved_models/feature_list.pkl")
print("    saved_models/svm_full.pkl")
print("    saved_models/rf_full.pkl")
print("    saved_models/xgb_full.pkl")
print("    result/enhanced_test_predictions.csv")
print("    result/enhanced_*_proba.csv")
print("    result/enhanced_feature_importance.csv")
print("=" * 60)
print("ENHANCED TRAINING COMPLETE")