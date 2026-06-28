"""
policy_extraction.py
Extract government policy features for COVID-19 ML model.
Uses OxCGRT_compact_national_v1.csv (official Oxford dataset).

CHANGES FROM PREVIOUS VERSION:
  - Source: data/OxCGRT_compact_national_v1.csv  (was owid-covid-data.csv)
  - Filter: CountryCode == 'KOR' and Jurisdiction == 'NAT_TOTAL'
  - Date format: YYYYMMDD integer  (was YYYY-MM-DD string)
  - Column names: C1M_School closing, H7_Vaccination policy, etc.
    (were short OWID aliases that did not exist in that file)
  - Case growth rate derived from ConfirmedCases diff  (was new_cases column)
  - Window alignment: date-based lookup into OxCGRT  (was fragile iloc alignment)
  - Mobility placeholders kept as zeros (Google mobility not in OxCGRT compact)
  - _missing flag columns removed (they added noise, not signal)
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("POLICY FEATURE EXTRACTION  [OxCGRT compact v1]")
print("=" * 60)

# ============================================================
# 1. Load OxCGRT compact dataset and filter to South Korea
# ============================================================
print("\n[1] Loading OxCGRT compact dataset...")

raw = pd.read_csv('data/OxCGRT_compact_national_v1.csv',
                  dtype={'Date': str},   # keep as string for safe parsing
                  low_memory=False)
print(f"  Loaded {len(raw)} rows, {raw.shape[1]} columns")

# National-level South Korea only
korea_df = raw[
    (raw['CountryCode'] == 'KOR') &
    (raw['Jurisdiction'] == 'NAT_TOTAL')
].copy()

# Parse YYYYMMDD date format
korea_df['date'] = pd.to_datetime(korea_df['Date'], format='%Y%m%d')
korea_df = korea_df.sort_values('date').reset_index(drop=True)

print(f"  South Korea rows: {len(korea_df)}")
print(f"  Date range: {korea_df['date'].min().date()} to {korea_df['date'].max().date()}")

# ============================================================
# 2. Column mapping: paper policy -> OxCGRT compact column
# ============================================================
print("\n[2] Mapping policy columns...")

paper_to_oxcgrt = {
    'P5_testing_policy':        'H2_Testing policy',
    'P6_contact_tracing':       'H3_Contact tracing',
    'P7_stringency_index':      'StringencyIndex_Average',
    'P8_debt_relief':           'E2_Debt/contract relief',
    'P9_income_support':        'E1_Income support',
    'P10_internal_movement':    'C7M_Restrictions on internal movement',
    'P11_international_travel': 'C8EV_International travel controls',
    'P12_public_info':          'H1_Public information campaigns',
    'P13_cancel_events':        'C3M_Cancel public events',
    'P14_gatherings':           'C4M_Restrictions on gatherings',
    'P16_school_closures':      'C1M_School closing',
    'P17_stay_home':            'C6M_Stay at home requirements',
    'P18_public_transport':     'C5M_Close public transport',
    'P20_workplace_closures':   'C2M_Workplace closing',
    'P21_vaccination':          'H7_Vaccination policy',
}

# Reverse map for display and correlation lookup
oxcgrt_to_paper = {v: k for k, v in paper_to_oxcgrt.items()}

# Keep only columns that actually exist in the file
available_policies = [
    (paper, oxcgrt)
    for paper, oxcgrt in paper_to_oxcgrt.items()
    if oxcgrt in korea_df.columns
]
available_oxcgrt_vars = [oxcgrt for _, oxcgrt in available_policies]

print(f"\n  Available OxCGRT policy columns: {len(available_policies)}")
for paper, oxcgrt in available_policies:
    non_zero = (korea_df[oxcgrt].fillna(0) > 0).sum()
    print(f"    {paper:30s} -> {oxcgrt}  ({non_zero} non-zero days)")

# ============================================================
# 3. Mobility placeholders (Google mobility not in OxCGRT compact)
# ============================================================
print("\n[3] Creating mobility placeholders (zeros)...")

mobility_policies = [
    'P1_grocery_pharmacy',
    'P2_parks',
    'P3_retail_recreation',
    'P4_time_at_home',
    'P15_public_transport_stations',
    'P19_workplace_visitors',
]
for policy in mobility_policies:
    korea_df[policy] = 0.0

print(f"  {len(mobility_policies)} mobility columns set to 0 (not in OxCGRT compact)")

# ============================================================
# 4. Forward-fill missing values in all policy columns
# ============================================================
print("\n[4] Forward-filling missing policy values...")

all_policy_cols = available_oxcgrt_vars + mobility_policies
for col in available_oxcgrt_vars:
    korea_df[col] = korea_df[col].ffill().fillna(0)

print(f"  Done. {len(all_policy_cols)} columns cleaned.")

# ============================================================
# 5. Derive daily new cases and smoothed growth rate
#    OxCGRT has ConfirmedCases (cumulative) – take daily diff
# ============================================================
print("\n[5] Deriving case growth rate from ConfirmedCases...")

if 'ConfirmedCases' in korea_df.columns:
    korea_df['new_cases'] = (
        korea_df['ConfirmedCases']
        .ffill()
        .diff()
        .clip(lower=0)
        .fillna(0)
    )
else:
    print("  WARNING: ConfirmedCases not found – growth rate will be 0")
    korea_df['new_cases'] = 0.0

cases = korea_df['new_cases'].values
growth_rate = np.zeros(len(cases), dtype=float)
for i in range(1, len(cases)):
    if cases[i - 1] > 0:
        growth_rate[i] = cases[i] / cases[i - 1] - 1

smoothed_growth = gaussian_filter1d(growth_rate, sigma=2, mode='reflect')
korea_df['growth_rate_7d_smoothed'] = smoothed_growth
print(f"  Growth rate computed for {len(korea_df)} days.")

# ============================================================
# 6. Apply 14-day policy delay
# ============================================================
print("\n[6] Applying 14-day policy delay...")

DELAY = 14
for col in all_policy_cols:
    korea_df[f'{col}_delayed'] = korea_df[col].shift(DELAY)

print(f"  Applied {DELAY}-day delay to {len(all_policy_cols)} columns.")

# ============================================================
# 7. Load case data to get window start dates
#    (needed for date-based alignment with OxCGRT)
# ============================================================
print("\n[7] Loading case data for window date alignment...")

case_data = pd.read_excel('temp/number1.xlsx')
case_data['date'] = pd.to_datetime(case_data['date'])
print(f"  Case data: {len(case_data)} days, "
      f"{case_data['date'].min().date()} to {case_data['date'].max().date()}")

# Build a date-indexed version of korea_df for fast lookup
korea_indexed = korea_df.set_index('date')

# ============================================================
# 8. Load existing RI+SIR windows to align on data_num
# ============================================================
print("\n[8] Loading enhanced_pre_data.csv for window indices...")

try:
    enhanced_df = pd.read_csv('result/enhanced_pre_data.csv')
    print(f"  Loaded {len(enhanced_df)} windows from enhanced_pre_data.csv")
    window_starts = enhanced_df['data_num'].values
except Exception as e:
    print(f"  WARNING: {e} – using default range 0-952")
    window_starts = np.arange(953)

# ============================================================
# 9. Build per-window policy features using date-based lookup
# ============================================================
print("\n[9] Building per-window policy features...")

window_policy_features = []
skipped = 0

for window_idx in window_starts:
    # Get the calendar date this window starts on
    if window_idx >= len(case_data):
        skipped += 1
        continue

    win_start_date = case_data['date'].iloc[window_idx]

    # Calibration window: days 0-20 (21 days)
    calib_start = win_start_date
    calib_end   = win_start_date + pd.Timedelta(days=20)

    # Delayed policy window: shift back 14 days
    pol_start = calib_start - pd.Timedelta(days=DELAY)
    pol_end   = calib_end   - pd.Timedelta(days=DELAY)

    # Slice OxCGRT by date range
    mask = (korea_df['date'] >= pol_start) & (korea_df['date'] <= pol_end)
    window_data = korea_df[mask]

    if len(window_data) == 0:
        skipped += 1
        continue

    features = {'data_num': window_idx}

    # --- OxCGRT policy features ---
    for col in available_oxcgrt_vars:
        vals = window_data[col].values.astype(float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            vals = np.array([0.0])

        features[f'policy_mean_{col}']  = float(np.mean(vals))
        features[f'policy_start_{col}'] = float(vals[0])
        features[f'policy_end_{col}']   = float(vals[-1])

        if len(vals) > 1 and np.std(vals) > 0:
            x = np.arange(len(vals))
            features[f'policy_slope_{col}'] = float(np.polyfit(x, vals, 1)[0])
        else:
            features[f'policy_slope_{col}'] = 0.0

    # --- Mobility placeholders (constant 0) ---
    for col in mobility_policies:
        features[f'policy_mean_{col}']  = 0.0
        features[f'policy_slope_{col}'] = 0.0
        features[f'policy_start_{col}'] = 0.0
        features[f'policy_end_{col}']   = 0.0

    # --- Case growth rate over calibration window ---
    gr_mask = (korea_df['date'] >= calib_start) & (korea_df['date'] <= calib_end)
    gr_vals = korea_df[gr_mask]['growth_rate_7d_smoothed'].values
    features['growth_rate_mean'] = float(np.mean(gr_vals)) if len(gr_vals) > 0 else 0.0
    features['growth_rate_max']  = float(np.max(gr_vals))  if len(gr_vals) > 0 else 0.0

    window_policy_features.append(features)

policy_window_df = pd.DataFrame(window_policy_features)
print(f"  Built features for {len(policy_window_df)} windows  ({skipped} skipped).")

# ============================================================
# 10. Select top policies by expected correlation strength
#     Keys now match the OxCGRT column names used above
# ============================================================
print("\n[10] Ranking policies by expected correlation...")

policy_correlations = {
    'StringencyIndex_Average':              -0.521,
    'H7_Vaccination policy':               -0.482,
    'H1_Public information campaigns':     -0.424,
    'C5M_Close public transport':          -0.431,
    'C2M_Workplace closing':               -0.383,
    'C1M_School closing':                  -0.382,
    'C3M_Cancel public events':            -0.381,
    'C4M_Restrictions on gatherings':      -0.393,
    'E1_Income support':                   -0.345,
    'C7M_Restrictions on internal movement': -0.322,
    'C8EV_International travel controls':  -0.337,
    'C6M_Stay at home requirements':       -0.321,
    'E2_Debt/contract relief':             -0.311,
    'H2_Testing policy':                   -0.235,
    'H3_Contact tracing':                  -0.111,
}

available_corr = {
    k: v for k, v in policy_correlations.items()
    if k in available_oxcgrt_vars
}

top_n = 10
sorted_policies = sorted(available_corr.items(), key=lambda x: abs(x[1]), reverse=True)
top_oxcgrt_cols = [p[0] for p in sorted_policies[:top_n]]

print(f"\n  Top {top_n} policies by |correlation|:")
for i, (col, corr) in enumerate(sorted_policies[:top_n]):
    paper = oxcgrt_to_paper.get(col, col)
    print(f"    {i+1:2d}. {paper:30s}  ({col}): {corr:.3f}")

# ============================================================
# 11. Merge with RI + SIR features and save
# ============================================================
print("\n[11] Merging with RI+SIR features and saving...")

try:
    enhanced_df = pd.read_csv('result/enhanced_pre_data.csv')
    final_df = pd.merge(enhanced_df, policy_window_df, on='data_num', how='left')

    # ---- Remove all-zero columns at the source ----
    # Covers: 6 mobility placeholder variables × 4 feature types (24 cols),
    # C5M_Close public transport × 4 (4 cols), policy_slope_H1 (1 col) = 29 cols.
    # Keeping these in the CSV just adds noise for every downstream script.
    protected = {'data_num', 'RI', 'label', 'date', 'Week'}
    zero_cols = [c for c in final_df.columns
                 if c not in protected and final_df[c].eq(0).all()]
    if zero_cols:
        final_df = final_df.drop(columns=zero_cols)
        print(f"  Dropped {len(zero_cols)} all-zero columns (dead weight).")
        print(f"  Remaining shape: {final_df.shape}")
    else:
        print("  No all-zero columns found.")

    final_df.to_csv('result/enhanced_with_policies.csv', index=False)
    print(f"  Saved enhanced_with_policies.csv:  {final_df.shape}")

    # Top-policy slim version
    keep_cols = ['data_num']
    for col in top_oxcgrt_cols:
        for prefix in ['policy_mean_', 'policy_slope_', 'policy_start_', 'policy_end_']:
            c = f'{prefix}{col}'
            if c in final_df.columns:
                keep_cols.append(c)
    for extra in ['growth_rate_mean', 'growth_rate_max']:
        if extra in final_df.columns:
            keep_cols.append(extra)

    top_df = final_df[keep_cols].copy()
    top_df.to_csv('result/enhanced_top_policies.csv', index=False)
    print(f"  Saved enhanced_top_policies.csv:   {top_df.shape}")

except Exception as e:
    print(f"  ERROR during merge: {e}")
    policy_window_df.to_csv('result/policy_features.csv', index=False)
    print(f"  Saved policy_features.csv as fallback.")

# ============================================================
# 12. Summary
# ============================================================
print("\n" + "=" * 60)
print("POLICY FEATURE EXTRACTION SUMMARY")
print("=" * 60)
policy_feat_count = len(policy_window_df.columns) - 1
print(f"\n  Policy features per window : {policy_feat_count}")
print(f"  Windows with policy data   : {len(policy_window_df)}")
print(f"  Windows skipped            : {skipped}")

print("\n  Feature types created per policy column:")
print("    policy_mean_*   – mean level over delayed calibration window")
print("    policy_slope_*  – linear trend (positive = tightening)")
print("    policy_start_*  – value at window start")
print("    policy_end_*    – value at window end")
print("    growth_rate_mean/max – smoothed case growth over calib window")

print("\n  Non-zero day counts for key columns:")
for col in available_oxcgrt_vars:
    nz = (korea_df[col] > 0).sum()
    print(f"    {col:45s}: {nz:4d} days")

print("\n" + "=" * 60)
print("POLICY FEATURE EXTRACTION COMPLETE")
print("=" * 60)