"""
visualize_results.py  —  Definitive version
Run AFTER evaluate_all.py has completed.

FIXES vs previous version:
  1.  shorten() defined globally (was defined mid-script, causing
      NameError if figures ran out of order)
  2.  Fig 4  now shows AGGREGATED importance (base variable level) —
      vaccination, school closures etc. show their true combined weight
  3.  Fig 4b added — horizontal bar chart of aggregated importance,
      the single most readable feature importance figure for the paper
  4.  Table 2 now uses aggregated importance (matches Fig 4)
  5.  Scaler compatibility check before transform to avoid shape errors
  6.  imp_path and imp_df defined before any figure that needs them
  7.  model_keys / model_labels defined once at the top, used everywhere
  8.  Fig 13 variant labels use ax.get_ylim() AFTER plotting so y-coords
      are valid (previously called before data was drawn)
  9.  FIX for AttributeError when agg_imp loaded from CSV: robust column
      detection and type safety in shorten()

OUTPUTS  (all to result/figures/):
  fig01_label_distribution.png
  fig02_ri_distributions.png
  fig03_sir_distributions.png
  fig04_aggregated_importance_heatmap.png
  fig04b_aggregated_importance_bar.png       ← KEY interpretability figure
  fig05_feature_set_comparison.png
  fig06_model_accuracy_cv.png
  fig07_per_class_f1.png
  fig08_confusion_matrices.png
  fig09_timeseries_overview.png
  fig10_early_warning.png
  fig11_transferability.png
  fig12_policy_trends.png
  fig13_sir_trends.png
  fig14_correlation_heatmap.png
  table1_model_performance.png  +  .csv
  table2_aggregated_importance.png  +  .csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')

# ============================================================
# 0. Constants and theme
# ============================================================
OUT_DIR = 'result/figures'
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    'L0':    '#2E86AB',
    'L1':    '#F6AE2D',
    'L2':    '#F26419',
    'bg':    '#FAFAF8',
    'grid':  '#E8E6E0',
    'text':  '#1A1A1A',
    'accent':'#D62839',
}
CLASS_COLORS = [PALETTE['L0'], PALETTE['L1'], PALETTE['L2']]
CLASS_NAMES  = ['L0 (Decrease)', 'L1 (Maintain)', 'L2 (Increase)']
MODEL_KEYS   = ['svm', 'rf', 'xgb']
MODEL_LABELS = ['SVM', 'Random Forest', 'XGBoost']

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['bg'],
    'axes.edgecolor':    PALETTE['grid'],
    'axes.labelcolor':   PALETTE['text'],
    'xtick.color':       PALETTE['text'],
    'ytick.color':       PALETTE['text'],
    'text.color':        PALETTE['text'],
    'grid.color':        PALETTE['grid'],
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'legend.framealpha': 0.85,
    'legend.edgecolor':  PALETTE['grid'],
    'savefig.bbox':      'tight',
    'savefig.dpi':       150,
    'savefig.facecolor': PALETTE['bg'],
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

# ---- FIX 1: shorten() defined globally with type safety ----
def shorten(name):
    """Shorten long feature names for axis labels."""
    if not isinstance(name, str):
        name = str(name)
    name = (name
            .replace('policy_mean_',  'mean: ')
            .replace('policy_slope_', 'slope: ')
            .replace('policy_start_', 'start: ')
            .replace('policy_end_',   'end: ')
            .replace('Restrictions on ', '')
            .replace('StringencyIndex_Average', 'Stringency Index')
            .replace('_Average', ''))
    if len(name) > 40:
        name = name[:38] + '..'
    return name

def get_base_variable(name):
    for prefix in ('policy_mean_', 'policy_slope_',
                   'policy_start_', 'policy_end_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

# ============================================================
# 1. Load data
# ============================================================
print("=" * 60)
print("VISUALIZE RESULTS  —  LOADING DATA AND MODELS")
print("=" * 60)

df = pd.read_csv('result/enhanced_with_policies.csv')
df = df.replace([np.inf, -np.inf], np.nan)
for col in df.columns:
    if df[col].isna().any():
        med = df[col].median()
        df[col] = df[col].fillna(med if pd.notna(med) else 0)

label_col = 'label' if 'label' in df.columns else 'Label'
print(f"  Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

ri_features  = ['Week', r'$\mu^c$', r'$\beta^c$', r'$\sigma^c$',
                r'$Delta^c$', r'$Omicron^c$', r'$Policy^c$', r'$Policy^p$']
sir_features = ['Re', 'growth_rate', 'sir_fit_r2', 'R0',
                'doubling_time', 'sir_reliable']
policy_prefixes = ('policy_mean_', 'policy_slope_',
                   'policy_start_', 'policy_end_')
growth_extras   = ['growth_rate_mean', 'growth_rate_max']

ri_avail     = [f for f in ri_features  if f in df.columns]
sir_avail    = [f for f in sir_features if f in df.columns]
policy_avail = ([c for c in df.columns if c.startswith(policy_prefixes)]
                + [c for c in growth_extras if c in df.columns])

feat_path = 'saved_models/feature_list.pkl'
full_feats = (
    [f for f in joblib.load(feat_path) if f in df.columns]
    if os.path.exists(feat_path)
    else ri_avail + sir_avail + policy_avail
)
print(f"  Features: {len(full_feats)} total")

# ---- Load importance files BEFORE any figure needs them ----
raw_imp_path = 'result/full_model_feature_importance.csv'
agg_imp_path = 'result/aggregated_feature_importance.csv'

if os.path.exists(raw_imp_path):
    raw_imp = pd.read_csv(raw_imp_path)
else:
    raw_imp = None

if os.path.exists(agg_imp_path):
    agg_imp = pd.read_csv(agg_imp_path)
else:
    agg_imp = None

# ============================================================
# 2. Train/test split and model loading
# ============================================================
np.random.seed(42)
n_train   = int(0.7 * len(df))
train_idx = np.random.choice(df.index, n_train, replace=False)
test_idx  = [i for i in df.index if i not in train_idx]
train = df.iloc[train_idx].copy()
test  = df.iloc[test_idx].copy()

X_train_raw = np.nan_to_num(train[full_feats].values.astype(float))
X_test_raw  = np.nan_to_num(test[full_feats].values.astype(float))
y_train     = train[label_col].values
y_test      = test[label_col].values

# ---- Check scaler feature count before transforming ----
scaler_path = 'saved_models/scaler_full.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    if scaler.n_features_in_ == len(full_feats):
        X_train_sc = scaler.transform(X_train_raw)
        X_test_sc  = scaler.transform(X_test_raw)
    else:
        print(f"  WARNING: saved scaler expects {scaler.n_features_in_} "
              f"features, data has {len(full_feats)}. Refitting.")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_test_sc  = scaler.transform(X_test_raw)
else:
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_test_sc  = scaler.transform(X_test_raw)

model_params = {
    'svm': {'C': 50.0, 'gamma': 0.3, 'kernel': 'rbf'},
    'rf':  {'max_depth': 14, 'n_estimators': 85,  'random_state': 42},
    'xgb': {'max_depth': 7,  'n_estimators': 110, 'random_state': 42},
}
models = {}
for name in MODEL_KEYS:
    path = f'saved_models/{name}_full.pkl'
    if os.path.exists(path):
        m = joblib.load(path)
        # Verify feature count matches
        try:
            m.predict(X_test_sc[:1])
            models[name] = m
            print(f"  Loaded {name}")
        except Exception:
            print(f"  Retraining {name} (feature mismatch)...")
            if name == 'svm':
                m = SVC(**model_params['svm'], probability=True,
                         random_state=42)
            elif name == 'rf':
                m = RandomForestClassifier(**model_params['rf'])
            else:
                m = XGBClassifier(**model_params['xgb'])
            m.fit(X_train_sc, y_train)
            models[name] = m
    else:
        print(f"  Retraining {name}...")
        if name == 'svm':
            m = SVC(**model_params['svm'], probability=True, random_state=42)
        elif name == 'rf':
            m = RandomForestClassifier(**model_params['rf'])
        else:
            m = XGBClassifier(**model_params['xgb'])
        m.fit(X_train_sc, y_train)
        models[name] = m

preds  = {n: m.predict(X_test_sc)       for n, m in models.items()}
probas = {n: m.predict_proba(X_test_sc) for n, m in models.items()}
accs   = {n: accuracy_score(y_test, p)  for n, p in preds.items()}

cv_strat  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {n: cross_val_score(m, X_train_sc, y_train,
                                 cv=cv_strat, scoring='accuracy')
             for n, m in models.items()}

print(f"\n  Accuracies: "
      + "  ".join(f"{k.upper()}: {v:.4f}" for k, v in accs.items()))

# Build raw importance if CSV not available
if raw_imp is None:
    X_all = np.nan_to_num(df[full_feats].values.astype(float))
    sc_a  = StandardScaler()
    xgb_a = XGBClassifier(**model_params['xgb'])
    xgb_a.fit(sc_a.fit_transform(X_all), df[label_col].values)
    raw_imp = pd.DataFrame({'feature': full_feats,
                             'importance': xgb_a.feature_importances_,
                             'rf_importance': models['rf'].feature_importances_,
                             'xgb_importance': xgb_a.feature_importances_})
    raw_imp['mean_importance'] = (raw_imp['rf_importance'] +
                                   raw_imp['xgb_importance']) / 2
    raw_imp = raw_imp.sort_values('mean_importance', ascending=False)

# Ensure mean_importance column exists
if 'mean_importance' not in raw_imp.columns:
    if 'rf_importance' in raw_imp.columns and 'xgb_importance' in raw_imp.columns:
        raw_imp['mean_importance'] = (raw_imp['rf_importance'] +
                                       raw_imp['xgb_importance']) / 2
    else:
        raw_imp['mean_importance'] = raw_imp['importance']

# Build aggregated importance if CSV not available
if agg_imp is None:
    raw_imp['base_variable'] = raw_imp['feature'].apply(get_base_variable)
    agg_imp = (raw_imp.groupby('base_variable')['mean_importance']
               .sum().sort_values(ascending=False).reset_index())
    agg_imp.columns = ['Variable', 'Aggregated Importance']

# Attach dates
try:
    case_data = pd.read_excel('temp/number1.xlsx')
    case_data['date'] = pd.to_datetime(case_data['date'])
    date_map = {i: case_data['date'].iloc[i]
                for i in range(min(len(df), len(case_data)))}
    df['date'] = df['data_num'].map(date_map)
except Exception:
    df['date'] = pd.date_range('2020-01-01', periods=len(df), freq='D')
df['date'] = pd.to_datetime(df['date'])
df_sorted  = df.sort_values('date').reset_index(drop=True)

print("\nGenerating figures...")

# ============================================================
# Fig 1 – Label distribution
# ============================================================
print("[Fig 1] Label distribution...")

fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df[label_col].value_counts().sort_index()
bars   = ax.bar([0,1,2], counts.values, color=CLASS_COLORS,
                width=0.55, edgecolor='white', linewidth=1.5)
total  = counts.sum()
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
            str(val), ha='center', va='bottom',
            fontweight='bold', fontsize=12)
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
            f'{val/total*100:.1f}%', ha='center', va='center',
            color='white', fontweight='bold', fontsize=13)
ax.set_xticks([0,1,2])
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel('Number of Windows')
ax.set_title('Label Distribution Across 953 Sliding Windows')
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig01_label_distribution.png')

# ============================================================
# Fig 2 – RI feature distributions by label
# ============================================================
print("[Fig 2] RI feature distributions...")

ri_plot = [r'$\mu^c$', r'$\beta^c$', r'$\sigma^c$', r'$Policy^c$']
ri_plot = [f for f in ri_plot if f in df.columns]

if ri_plot:
    fig, axes = plt.subplots(1, len(ri_plot),
                              figsize=(4*len(ri_plot), 5))
    if len(ri_plot) == 1:
        axes = [axes]
    for ax, feat in zip(axes, ri_plot):
        data_by_class = [df[df[label_col]==c][feat].dropna().values
                         for c in [0,1,2]]
        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, color in zip(bp['boxes'], CLASS_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.85)
        ax.set_xticklabels(['L0','L1','L2'])
        ax.set_title(feat)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    legend_elems = [Patch(facecolor=c, label=l)
                    for c, l in zip(CLASS_COLORS, ['L0','L1','L2'])]
    fig.legend(handles=legend_elems, loc='upper right')
    fig.suptitle('RI Feature Distributions by Outbreak Class',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig02_ri_distributions.png')

# ============================================================
# Fig 3 – SIR parameter distributions by label
# ============================================================
print("[Fig 3] SIR parameter distributions...")

sir_plot = ['Re', 'R0', 'growth_rate', 'doubling_time', 'sir_fit_r2']
sir_plot = [f for f in sir_plot if f in df.columns]

if sir_plot:
    fig, axes = plt.subplots(1, len(sir_plot),
                              figsize=(4*len(sir_plot), 5))
    if len(sir_plot) == 1:
        axes = [axes]
    for ax, feat in zip(axes, sir_plot):
        data_by_class = [df[df[label_col]==c][feat].dropna().values
                         for c in [0,1,2]]
        if feat == 'doubling_time':
            data_by_class = [np.clip(d, -200, 200) for d in data_by_class]
        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, color in zip(bp['boxes'], CLASS_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.85)
        ax.set_xticklabels(['L0','L1','L2'])
        ax.set_title(feat)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    legend_elems = [Patch(facecolor=c, label=l)
                    for c, l in zip(CLASS_COLORS, ['L0','L1','L2'])]
    fig.legend(handles=legend_elems, loc='upper right')
    fig.suptitle('SIR Parameter Distributions by Outbreak Class',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig03_sir_distributions.png')

# ============================================================
# Fig 4 – Aggregated importance heatmap (top 20 base variables)
#    FIX: robust column detection for agg_imp loaded from CSV
# ============================================================
print("[Fig 4] Aggregated importance heatmap...")

# ---- Robustly identify variable and importance columns ----
if 'Variable' in agg_imp.columns:
    var_col = 'Variable'
    imp_col = 'Aggregated Importance'
else:
    # If columns are unnamed or in different order, pick the non-numeric column as variable
    numeric_cols = agg_imp.select_dtypes(include=['number']).columns
    non_numeric_cols = agg_imp.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        var_col = non_numeric_cols[0]
    else:
        # Fallback: assume first column is variable (may still fail if numeric)
        var_col = agg_imp.columns[0]
    # The importance column is the first numeric column
    if len(numeric_cols) > 0:
        imp_col = numeric_cols[0]
    else:
        imp_col = agg_imp.columns[1]   # fallback

top20_agg = agg_imp.head(20).copy()
# Ensure the variable column is treated as string
top20_agg[var_col] = top20_agg[var_col].astype(str)

short_vars = [shorten(v) for v in top20_agg[var_col].values]

# For heatmap we need two rows — use raw_imp to get RF vs XGB per base var
raw_imp['base_variable'] = raw_imp['feature'].apply(get_base_variable)

def group_by_base(col_name):
    return (raw_imp.groupby('base_variable')[col_name]
            .sum().reindex(top20_agg[var_col].values, fill_value=0).values)

rf_col  = 'rf_importance'  if 'rf_importance'  in raw_imp.columns else 'mean_importance'
xgb_col = 'xgb_importance' if 'xgb_importance' in raw_imp.columns else 'mean_importance'

heat_data = np.vstack([group_by_base(rf_col), group_by_base(xgb_col)])
cmap_heat = LinearSegmentedColormap.from_list(
    'heat', ['#FAFAF8', '#F6AE2D', '#F26419', '#D62839'])

fig, ax = plt.subplots(figsize=(13, 4.5))
im = ax.imshow(heat_data, aspect='auto', cmap=cmap_heat)
ax.set_xticks(range(20))
ax.set_xticklabels(short_vars, rotation=40, ha='right', fontsize=8.5)
ax.set_yticks([0,1])
ax.set_yticklabels(['Random Forest', 'XGBoost'], fontsize=11)
plt.colorbar(im, ax=ax, label='Aggregated Importance', shrink=0.8)
ax.set_title('Feature Importance Heatmap — Top 20 Variables '
             '(RF vs XGBoost, aggregated across feature types)')
for i in range(2):
    for j in range(20):
        val = heat_data[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=7.5,
                color='black' if val < 0.05 else 'white')
fig.tight_layout()
save(fig, 'fig04_aggregated_importance_heatmap.png')

# ============================================================
# Fig 4b – Aggregated importance horizontal bar chart
#          THE key interpretability figure for the paper
# ============================================================
print("[Fig 4b] Aggregated importance bar chart...")

top15_agg = agg_imp.head(15).copy()
top15_agg[var_col] = top15_agg[var_col].astype(str)

def tag_color(name):
    ri_set  = {'Week', r'$\mu^c$', r'$\beta^c$', r'$\sigma^c$',
               r'$Delta^c$', r'$Omicron^c$', r'$Policy^c$', r'$Policy^p$'}
    sir_set = {'Re', 'R0', 'growth_rate', 'doubling_time',
               'sir_fit_r2', 'sir_reliable',
               'growth_rate_mean', 'growth_rate_max'}
    if name in ri_set:  return PALETTE['L0']
    if name in sir_set: return PALETTE['L1']
    return PALETTE['L2']

bar_colors = [tag_color(v) for v in top15_agg[var_col].values]
bar_labels  = [shorten(v) for v in top15_agg[var_col].values]
bar_vals    = top15_agg[imp_col].values

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(bar_vals)), bar_vals[::-1],
               color=bar_colors[::-1], height=0.65,
               edgecolor='white', linewidth=1.2)
ax.set_yticks(range(len(bar_vals)))
ax.set_yticklabels(bar_labels[::-1], fontsize=10)
for bar, val in zip(bars, bar_vals[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9.5, fontweight='bold')
ax.set_xlabel('Aggregated Importance (sum across mean/slope/start/end)')
ax.set_title('Top 15 Variables by Aggregated Feature Importance\n'
             'Aggregation reveals true contribution of each real-world factor')
legend_elems = [
    Patch(facecolor=PALETTE['L0'], label='RI features'),
    Patch(facecolor=PALETTE['L1'], label='SIR / epidemiological'),
    Patch(facecolor=PALETTE['L2'], label='Policy (OxCGRT)'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=10)
ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig04b_aggregated_importance_bar.png')


####FIGURE 5


comp_path = 'result/comparison_random_split.csv'
if os.path.exists(comp_path):
    comp_df = pd.read_csv(comp_path).sort_values('Mean', ascending=True)
    
else:
    policy_mean_only = [c for c in policy_avail
                        if c.startswith('policy_mean_')]
    experiments = {
        'RI Only':      ri_avail,
        'SIR Only':     sir_avail,
        'Policy Only':  policy_mean_only,
        'RI + SIR':     ri_avail + sir_avail,
        'RI + Policy':  ri_avail + policy_avail,
        'SIR + Policy': sir_avail + policy_avail,
        'ALL Features': full_feats,
    }
    rows = []
    for name, feats in experiments.items():
        if not feats: continue
        Xtr = np.nan_to_num(train[feats].values.astype(float))
        Xte = np.nan_to_num(test[feats].values.astype(float))
        sc  = StandardScaler()
        m   = XGBClassifier(**model_params['xgb'])
        m.fit(sc.fit_transform(Xtr), y_train)
        acc = accuracy_score(y_test, m.predict(sc.transform(Xte)))
        rows.append({'Experiment': name, 'Mean': acc})  # <-- changed here
    comp_df = pd.DataFrame(rows).sort_values('Mean', ascending=True)

bar_clrs = [PALETTE['accent'] if 'ALL' in r else PALETTE['L0']
            for r in comp_df['Experiment']]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(comp_df['Experiment'], comp_df['Mean'],
               color=bar_clrs, height=0.55, edgecolor='white')

for bar, val in zip(bars, comp_df['Mean']):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0.5, 1.05)
ax.axvline(comp_df['Mean'].max(), color=PALETTE['accent'],
           linestyle='--', linewidth=1.2, alpha=0.7)

ax.set_xlabel('Mean Accuracy (across models)')
ax.set_title('Feature Set Ablation — Which combination works best?')
ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)

fig.tight_layout()
save(fig, 'fig05_feature_set_comparison.png')
# ============================================================
# Fig 6 – Model accuracy + CV error bars
# ============================================================
print("[Fig 6] Model accuracy with CV error bars...")

test_accs = [accs[k]              for k in MODEL_KEYS]
cv_means  = [cv_scores[k].mean()  for k in MODEL_KEYS]
cv_stds   = [cv_scores[k].std()   for k in MODEL_KEYS]
x = np.arange(3); w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x-w/2, test_accs, w, label='Test Accuracy',
            color='#2E86AB', edgecolor='white', linewidth=1.2)
b2 = ax.bar(x+w/2, cv_means,  w, label='CV Mean (5-fold)',
            color='#F26419', edgecolor='white', linewidth=1.2,
            yerr=cv_stds, capsize=5,
            error_kw=dict(elinewidth=1.5, ecolor=PALETTE['text'],
                          capthick=1.5))
for bar, val in zip(b1, test_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9.5)
for bar, val, std in zip(b2, cv_means, cv_stds):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+std+0.006,
            f'{val:.3f}±{std:.3f}',
            ha='center', va='bottom', fontsize=8.5)
ax.set_ylim(0.6, 1.08)
ax.set_xticks(x); ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance — Test Accuracy vs 5-Fold Cross-Validation')
ax.legend(loc='lower right')
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig06_model_accuracy_cv.png')

# ============================================================
# Fig 7 – Per-class F1 scores
# ============================================================
print("[Fig 7] Per-class F1 scores...")

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(3); w = 0.25
bar_model_colors = ['#2E86AB', '#F6AE2D', '#D62839']
for i, (key, label) in enumerate(zip(MODEL_KEYS, MODEL_LABELS)):
    f1s    = f1_score(y_test, preds[key], average=None, labels=[0,1,2])
    offset = (i-1)*w
    bars   = ax.bar(x+offset, f1s, w, label=label,
                    color=bar_model_colors[i],
                    edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8.5)
ax.set_ylim(0.6, 1.08)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Scores — L1 (Maintain) is the hardest to predict')
ax.legend()
ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, 'fig07_per_class_f1.png')

# ============================================================
# Fig 8 – Confusion matrices (3-panel)
# ============================================================
print("[Fig 8] Confusion matrices...")

cmap_cm = LinearSegmentedColormap.from_list('cm', ['#FAFAF8', '#2E86AB'])
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, key, label in zip(axes, MODEL_KEYS, MODEL_LABELS):
    cm     = confusion_matrix(y_test, preds[key])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im     = ax.imshow(cm_pct, cmap=cmap_cm, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i,
                    f'{cm[i,j]}\n({cm_pct[i,j]:.0%})',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold',
                    color='white' if cm_pct[i,j] > 0.5
                    else PALETTE['text'])
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(['L0','L1','L2'])
    ax.set_yticklabels(['L0','L1','L2'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{label}  |  Accuracy: {accs[key]:.4f}',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='% of Actual Class')
fig.suptitle('Confusion Matrices — Test Set (286 windows)',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'fig08_confusion_matrices.png')

# ============================================================
# Fig 9 – Full time-series overview
# ============================================================
print("[Fig 9] Time-series overview...")

X_all    = np.nan_to_num(df_sorted[full_feats].values.astype(float))
sc_a     = StandardScaler()
xgb_a    = XGBClassifier(**model_params['xgb'])
xgb_a.fit(sc_a.fit_transform(X_all), df_sorted[label_col].values)
proba_all = xgb_a.predict_proba(sc_a.transform(X_all))

label_vals = df_sorted[label_col].values
dates      = df_sorted['date'].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

for i in range(len(dates)-1):
    ax1.axvspan(dates[i], dates[i+1], alpha=0.35,
                color=CLASS_COLORS[label_vals[i]], linewidth=0)
ax1.plot(dates, label_vals, color=PALETTE['text'],
         linewidth=0.7, alpha=0.6)
ax1.set_ylabel('Outbreak Class')
ax1.set_yticks([0,1,2]); ax1.set_yticklabels(['L0','L1','L2'])
ax1.set_title('True Outbreak Labels Over Time')
legend_elems = [Patch(facecolor=c, alpha=0.5, label=l)
                for c, l in zip(CLASS_COLORS,
                                ['L0 Decrease','L1 Maintain',
                                 'L2 Increase'])]
ax1.legend(handles=legend_elems, loc='upper left', fontsize=9)

ax2.fill_between(dates, proba_all[:,0], alpha=0.35,
                 color=PALETTE['L0'], label='P(L0 – Decrease)')
ax2.fill_between(dates, proba_all[:,1], alpha=0.35,
                 color=PALETTE['L1'], label='P(L1 – Maintain)')
ax2.fill_between(dates, proba_all[:,2], alpha=0.5,
                 color=PALETTE['L2'], label='P(L2 – Increase)')
ax2.set_ylabel('Predicted Probability')
ax2.set_xlabel('Date')
ax2.set_title('XGBoost Predicted Class Probabilities Over Time')
ax2.legend(loc='upper left', fontsize=9)
ax2.yaxis.grid(True); ax2.set_axisbelow(True)

fig.suptitle('Outbreak Classification — Full Time Series (2020–2022)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig09_timeseries_overview.png')

# ============================================================
# Fig 10 – Early warning  (time-split model)
# ============================================================
print("[Fig 10] Early warning plot...")

ALERT_THRESHOLD = 0.3
SUSTAINED_DAYS  = 7

train_time = df_sorted[df_sorted['date'] < '2022-01-01'].copy()
test_time  = (df_sorted[df_sorted['date'] >= '2022-01-01']
              .copy().reset_index(drop=True))

X_tr_t = np.nan_to_num(train_time[full_feats].values.astype(float))
X_te_t = np.nan_to_num(test_time[full_feats].values.astype(float))
y_te_t = test_time[label_col].values

sc_t = StandardScaler()
m_t  = XGBClassifier(**model_params['xgb'])
m_t.fit(sc_t.fit_transform(X_tr_t), train_time[label_col].values)
prob_l2    = m_t.predict_proba(sc_t.transform(X_te_t))[:,2]
test_dates = test_time['date'].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

for i in range(len(test_dates)-1):
    ax1.axvspan(test_dates[i], test_dates[i+1], alpha=0.35,
                color=CLASS_COLORS[y_te_t[i]], linewidth=0)
ax1.plot(test_dates, y_te_t, color=PALETTE['text'],
         linewidth=0.7, alpha=0.7)
ax1.set_ylabel('True Class')
ax1.set_yticks([0,1,2]); ax1.set_yticklabels(['L0','L1','L2'])
ax1.set_title('True Labels (Test Period: 2022 onward)')
legend_elems = [Patch(facecolor=c, alpha=0.5, label=l)
                for c, l in zip(CLASS_COLORS,
                                ['L0 Decrease','L1 Maintain',
                                 'L2 Increase'])]
ax1.legend(handles=legend_elems, loc='upper right', fontsize=9)

ax2.fill_between(test_dates, prob_l2, alpha=0.4, color=PALETTE['L2'],
                 label='P(L2 – Increasing)')
ax2.plot(test_dates, prob_l2, color=PALETTE['L2'], linewidth=1.2)
ax2.axhline(ALERT_THRESHOLD, color=PALETTE['accent'],
            linestyle='--', linewidth=1.8,
            label=f'Alert threshold  ({ALERT_THRESHOLD})')
alert_on = prob_l2 >= ALERT_THRESHOLD
ax2.fill_between(test_dates, 0, prob_l2, where=alert_on,
                 alpha=0.22, color=PALETTE['accent'],
                 label='Alert active')
ax2.set_ylabel('P(L2 — Increasing Outbreak)')
ax2.set_xlabel('Date')
ax2.set_ylim(0, 1.05)
ax2.set_title(f'Early Warning Signal — P(L2) vs Alert Threshold '
              f'({ALERT_THRESHOLD})')
ax2.legend(loc='upper right', fontsize=9)
ax2.yaxis.grid(True); ax2.set_axisbelow(True)

fig.suptitle('Early Warning Analysis (Time-Split Model, 2022 Test Period)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'fig10_early_warning.png')

# ============================================================
# Fig 11 – Transferability bar chart
# ============================================================
# ============================================================
# Fig 11 – Transferability bar chart (FIXED)
# ============================================================
print("[Fig 11] Transferability chart...")

comp_file = 'result/comparison_random_split.csv'

if os.path.exists(comp_file):
    comp_df = pd.read_csv(comp_file)

    # FIX: ensure correct column exists
    if 'Mean' not in comp_df.columns:
        raise ValueError(f"Expected 'Mean' column in {comp_file}, found {comp_df.columns}")

    rand_row = comp_df.query("Experiment == 'ALL Features'")

    if len(rand_row) > 0:
        rand_acc = rand_row['Mean'].values[0]
    else:
        rand_acc = comp_df['Mean'].max()

else:
    rand_acc = accs['xgb']  # fallback if CSV missing

time_acc = accuracy_score(y_te_t, m_t.predict(sc_t.transform(X_te_t)))

fig, ax = plt.subplots(figsize=(7, 4.5))

labels_t = ['Random Split\n(70/30)', 'Time Split\n(pre-2022 / 2022+)']
vals_t   = [rand_acc, time_acc]

bars = ax.bar(
    labels_t,
    vals_t,
    color=[PALETTE['L0'], PALETTE['L2']],
    width=0.45,
    edgecolor='white',
    linewidth=1.5
)

for bar, val in zip(bars, vals_t):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold'
    )

ax.annotate(
    '',
    xy=(1, time_acc),
    xytext=(0, rand_acc),
    arrowprops=dict(
        arrowstyle='->',
        color=PALETTE['accent'],
        lw=2,
        connectionstyle='arc3,rad=0.2'
    )
)

ax.text(
    0.5,
    (rand_acc + time_acc) / 2 + 0.02,
    f'Drop: {rand_acc - time_acc:.4f}',
    ha='center',
    color=PALETTE['accent'],
    fontweight='bold',
    fontsize=11
)

ax.set_ylim(0.5, 1.10)
ax.set_ylabel('XGBoost Test Accuracy')
ax.set_title(
    'Transferability — Random vs Time-Based Split\n'
    'Drop reflects temporal shift, not model failure'
)

ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
save(fig, 'fig11_transferability.png')

# ============================================================
# Fig 12 – Key policy trends over time
# ============================================================
print("[Fig 12] Policy trends over time...")

policy_plot_map = {
    'policy_mean_StringencyIndex_Average':  'Stringency Index',
    'policy_mean_H7_Vaccination policy':    'Vaccination Policy',
    'policy_mean_C1M_School closing':       'School Closures',
    'policy_mean_C6M_Stay at home requirements': 'Stay-at-Home',
}
avail_plt = {k: v for k, v in policy_plot_map.items()
             if k in df_sorted.columns}

if avail_plt:
    fig, ax = plt.subplots(figsize=(13, 5))
    line_colors = ['#2E86AB', '#F26419', '#D62839', '#6A0572']
    for (col, lbl), color in zip(avail_plt.items(), line_colors):
        vals = df_sorted[col].values
        vmin, vmax = vals.min(), vals.max()
        norm = (vals-vmin)/(vmax-vmin) if vmax > vmin else vals
        ax.plot(df_sorted['date'].values, norm,
                label=lbl, color=color, linewidth=1.8, alpha=0.85)
    for vd, vl in [('2021-07-01','Delta'), ('2022-02-01','Omicron')]:
        ax.axvline(pd.Timestamp(vd), color='gray',
                   linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(pd.Timestamp(vd), 1.03, vl,
                fontsize=9, color='gray', ha='center')
    ax.set_ylabel('Normalised Policy Level (0–1)')
    ax.set_xlabel('Date')
    ax.set_title('Key Policy Measures Over Time (Normalised to 0–1)')
    ax.legend(loc='upper right', fontsize=9)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'fig12_policy_trends.png')

# ============================================================
# Fig 13 – SIR parameters over time
#    FIX: get y-limits AFTER plotting so variant labels sit inside the axes
# ============================================================
print("[Fig 13] SIR parameter trends...")

sir_time_avail = [c for c in ['Re','R0','growth_rate']
                  if c in df_sorted.columns]

if sir_time_avail:
    fig, axes = plt.subplots(len(sir_time_avail), 1,
                             figsize=(13, 3.5*len(sir_time_avail)),
                             sharex=True)
    if len(sir_time_avail) == 1:
        axes = [axes]
    sir_colors = ['#2E86AB', '#F26419', '#D62839']
    for ax, col, color in zip(axes, sir_time_avail, sir_colors):
        vals  = df_sorted[col].values
        dates = df_sorted['date'].values
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        p1, p99 = np.percentile(valid, [1, 99])
        vc = np.clip(vals, p1, p99)
        ax.fill_between(dates, vc, alpha=0.25, color=color)
        ax.plot(dates, vc, color=color, linewidth=1.5, label=col)
        threshold = 1.0 if col in ['Re','R0'] else 0.0
        thresh_lbl = 'Threshold = 1' if col in ['Re','R0'] else 'No growth'
        ax.axhline(threshold, color=PALETTE['accent'], linestyle='--',
                   linewidth=1.5, alpha=0.8, label=thresh_lbl)
        ax.set_ylabel(col)
        ax.legend(loc='upper right', fontsize=9)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        # FIX: read y-limits after data is drawn
        ylo, yhi = ax.get_ylim()
        for vd, vl in [('2021-07-01','Delta'), ('2022-02-01','Omicron')]:
            ax.axvline(pd.Timestamp(vd), color='gray',
                       linestyle=':', linewidth=1.2, alpha=0.7)
            ax.text(pd.Timestamp(vd), ylo + (yhi-ylo)*0.88,
                    vl, fontsize=8.5, color='gray', ha='center')
    axes[-1].set_xlabel('Date')
    fig.suptitle('SIR Parameters Over Time — Re > 1 signals an expanding outbreak',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig13_sir_trends.png')

# ============================================================
# Fig 14 – Correlation heatmap
# ============================================================
print("[Fig 14] Correlation heatmap...")

corr_feats = (ri_avail + sir_avail
              + [c for c in policy_avail if c.startswith('policy_mean_')])
corr_feats = [f for f in corr_feats if f in df.columns]
corr_data  = df[corr_feats + [label_col]].corr()

cmap_corr = LinearSegmentedColormap.from_list(
    'corr', ['#D62839', '#FAFAF8', '#2E86AB'])
n = len(corr_feats) + 1
fig, ax = plt.subplots(figsize=(max(10, n*0.55), max(8, n*0.55)))
sns.heatmap(corr_data, ax=ax, cmap=cmap_corr, vmin=-1, vmax=1,
            center=0, linewidths=0.3, linecolor=PALETTE['grid'],
            annot=(n <= 22), fmt='.2f', annot_kws={'size': 7.5},
            cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
short_lbs = [shorten(f).replace(r'$','').replace('\\','')
             for f in corr_feats + [label_col]]
ax.set_xticklabels(short_lbs, rotation=45, ha='right', fontsize=8.5)
ax.set_yticklabels(short_lbs, rotation=0, fontsize=8.5)
ax.set_title('Feature Correlation Matrix — RI + SIR + Policy Means',
             fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
save(fig, 'fig14_correlation_heatmap.png')

# ============================================================
# Table 1 – Model performance summary
# ============================================================
print("[Table 1] Model performance summary...")

rows = []
for key, lbl in zip(MODEL_KEYS, MODEL_LABELS):
    rpt = classification_report(y_test, preds[key],
                                target_names=['L0','L1','L2'],
                                output_dict=True)
    rows.append({
        'Model':       lbl,
        'Test Acc':    f"{accs[key]:.4f}",
        'CV Mean':     f"{cv_scores[key].mean():.4f}",
        'CV Std':      f"{cv_scores[key].std():.4f}",
        'F1-L0':       f"{rpt['L0']['f1-score']:.4f}",
        'F1-L1':       f"{rpt['L1']['f1-score']:.4f}",
        'F1-L2':       f"{rpt['L2']['f1-score']:.4f}",
        'Macro F1':    f"{rpt['macro avg']['f1-score']:.4f}",
        'Weighted F1': f"{rpt['weighted avg']['f1-score']:.4f}",
    })
table1 = pd.DataFrame(rows)
table1.to_csv(os.path.join(OUT_DIR, 'table1_model_performance.csv'),
              index=False)

fig, ax = plt.subplots(figsize=(14, 2.2))
ax.axis('off')
tbl = ax.table(cellText=table1.values, colLabels=table1.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE['grid'])
    if r == 0:
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EEF4F8')
    else:
        cell.set_facecolor(PALETTE['bg'])
ax.set_title('Table 1 — Model Performance Summary (Test Set: 286 windows)',
             fontsize=11, fontweight='bold', pad=8)
fig.tight_layout()
save(fig, 'table1_model_performance.png')
print("  Saved table1_model_performance.csv")

# ============================================================
# Table 2 – Aggregated feature importance
#    FIX: uses aggregated importance so vaccination shows correctly
# ============================================================
print("[Table 2] Aggregated feature importance table...")

# Use the same var_col and imp_col detected earlier
t2 = agg_imp.head(20).copy().reset_index(drop=True)
t2.index += 1
# Ensure string conversion
t2[var_col] = t2[var_col].astype(str)
t2[var_col]   = t2[var_col].apply(shorten)
t2[imp_col]   = t2[imp_col].apply(lambda x: f'{x:.4f}')
t2.to_csv(os.path.join(OUT_DIR, 'table2_aggregated_importance.csv'))

display_cols = [var_col, imp_col]
if 'Group' in t2.columns:
    display_cols.append('Group')

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
tbl = ax.table(
    cellText=[[str(i+1)] + [str(t2.iloc[i][c]) for c in display_cols]
              for i in range(len(t2))],
    colLabels=['Rank'] + display_cols,
    cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE['grid'])
    if r == 0:
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EEF4F8')
    else:
        cell.set_facecolor(PALETTE['bg'])
    if c == 0 and r > 0:
        cell.set_facecolor(
            '#F6AE2D' if r <= 3 else
            PALETTE['bg'] if r % 2 != 0 else '#EEF4F8')
ax.set_title('Table 2 — Top 20 Variables by Aggregated Feature Importance',
             fontsize=11, fontweight='bold', pad=8)
fig.tight_layout()
save(fig, 'table2_aggregated_importance.png')
print("  Saved table2_aggregated_importance.csv")

# ============================================================
# Done
# ============================================================
print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETE")
print(f"Output folder: {OUT_DIR}/")
print("=" * 60)
print("\nFiles generated:")
for fname in sorted(os.listdir(OUT_DIR)):
    kb = os.path.getsize(os.path.join(OUT_DIR, fname)) // 1024
    print(f"  {fname:55s} {kb:5d} KB")