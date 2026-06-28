"""
sir_feature_extraction.py

CHANGES IN THIS VERSION:
  - Policy^c / Policy^p derived from OxCGRT_compact_national_v1.csv
    (was owid-covid-data.csv, which lacked the granular policy columns).
  - Stringency column now 'StringencyIndex_Average'  (was 'stringency_index').
  - Filter: CountryCode == 'KOR' and Jurisdiction == 'NAT_TOTAL'.
  - Date format: YYYYMMDD  (was YYYY-MM-DD).
  - Label assignment: merged on data_num key, not positional alignment.
  - Delta^c / Omicron^c reconstructed from known variant date ranges.
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SIR FEATURE EXTRACTION FOR COVID-19 OUTBREAK")
print("=" * 60)

# ============================================================
# 1. Load case data
# ============================================================
print("\n[1] Loading case data...")

case_data = pd.read_excel('temp/number1.xlsx')
case_data['date'] = pd.to_datetime(case_data['date'])
print(f"  Loaded {len(case_data)} days")
print(f"  Date range: {case_data['date'].min().date()} to {case_data['date'].max().date()}")

# ============================================================
# 2. Robust SIR estimation (unchanged)
# ============================================================
print("\n[2] Defining robust SIR estimation...")

def estimate_sir_robust(window_cases, population=52000000):
    raw_cases = window_cases[:21].values
    if len(raw_cases) >= 7:
        try:
            wl = 7
            if wl > len(raw_cases): wl = len(raw_cases)
            if wl % 2 == 0: wl -= 1
            cases = savgol_filter(raw_cases, wl, polyorder=2) if wl >= 5 else raw_cases
        except:
            cases = raw_cases
    else:
        cases = raw_cases
    cases = np.maximum(cases, 1)
    t = np.arange(len(cases))
    I0 = max(cases[0], 1)
    y0 = [population - I0, I0, 0.0]

    def sir_model(y, t, beta, gamma):
        S, I, R = y
        dS = -beta * S * I / population
        dI =  beta * S * I / population - gamma * I
        dR =  gamma * I
        return [dS, dI, dR]

    def objective(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e10
        try:
            sol  = odeint(sir_model, y0, t, args=(beta, gamma))
            pred = sol[:, 1]
            w    = np.linspace(0.5, 1.0, len(t))
            mse  = np.average((pred - cases) ** 2, weights=w)
            R0   = beta / gamma
            pen  = 0
            if R0 < 0.5:  pen += 1000 * (0.5 - R0) ** 2
            elif R0 > 8.: pen += 1000 * (R0 - 8.0) ** 2
            gr = beta - gamma
            if gr >  0.4: pen += 1000 * (gr - 0.4) ** 2
            if gr < -0.3: pen += 1000 * (-0.3 - gr) ** 2
            if gamma < 0.05: pen += 1000 * (0.05 - gamma) ** 2
            if gamma > 0.33: pen += 1000 * (gamma - 0.33) ** 2
            return mse + pen
        except:
            return 1e10

    starts = [(0.15,0.07),(0.25,0.10),(0.35,0.12),(0.20,0.05),
              (0.40,0.15),(0.30,0.08),(0.18,0.09)]
    bounds = [(0.01,0.8),(0.03,0.2)]
    best, best_err = None, 1e10

    for b0, g0 in starts:
        try:
            res = minimize(objective, [b0,g0], bounds=bounds,
                           method='L-BFGS-B', options={'maxiter':500})
            if res.success and res.fun < best_err:
                b, g = res.x
                if 0.3 <= b/g <= 8.5:
                    best_err = res.fun
                    best = res
        except:
            continue

    beta, gamma = best.x if best else (0.25, 0.10)

    R0 = beta / gamma
    S_cur = max(population - np.sum(cases), population * 0.01)
    Re = R0 * (S_cur / population)
    gr = beta - gamma
    dt = np.log(2) / gr if gr > 0.001 else (-np.log(2)/gr if gr < -0.001 else np.inf)

    sol    = odeint(sir_model, y0, t, args=(beta, gamma))
    fitted = sol[:, 1]
    ss_res = np.sum((cases - fitted) ** 2)
    ss_tot = np.sum((cases - np.mean(cases)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1

    residual = cases[-1] - fitted[-1]
    half = len(cases) // 2
    if half >= 5:
        Re_first  = np.mean(cases[5:half]) / (np.mean(cases[:5]) + 1)
        Re_second = (np.mean(cases[half+5:]) / (np.mean(cases[half:half+5]) + 1)
                     if half+5 < len(cases) else Re_first)
        Re_trend  = (Re_second - Re_first) / max(half, 1)
    else:
        Re_trend = 0

    return {
        'window_id':        None,
        'beta':             round(beta, 4),
        'gamma':            round(gamma, 4),
        'R0':               round(R0, 3),
        'Re':               round(Re, 3),
        'growth_rate':      round(gr, 4),
        'doubling_time':    round(dt, 1) if np.isfinite(dt) else np.inf,
        'sir_fit_r2':       round(r2, 4),
        'sir_residual':     round(residual, 1),
        'sir_rel_residual': round(residual / (cases[-1]+1), 4),
        'Re_trend':         round(Re_trend, 4),
        'peak_estimate':    round(max(cases)*(1+gr*7) if gr>0 else max(cases), 1),
        'initial_cases':    int(cases[0]),
        'final_cases':      int(cases[-1]),
        'case_ratio':       round(cases[-1]/(cases[0]+1), 3),
        'sir_reliable':     1 if r2>0 and 0.3<R0<8.0 else 0,
    }

# ============================================================
# 3. Extract SIR features for all 953 windows
# ============================================================
print("\n[3] Extracting SIR features for 953 windows...")

all_sir = []
for ws in range(953):
    wc = case_data['number'].iloc[ws:ws+35]
    try:
        f = estimate_sir_robust(wc)
        f['window_id'] = ws
    except Exception as e:
        print(f"  Warning: window {ws} failed – {e}")
        f = {k: np.nan for k in ['beta','gamma','R0','Re','growth_rate','doubling_time',
                                  'sir_fit_r2','sir_residual','sir_rel_residual','Re_trend',
                                  'peak_estimate','initial_cases','final_cases','case_ratio']}
        f.update({'window_id': ws, 'sir_reliable': 0, 'sir_fit_r2': -999})
    all_sir.append(f)
    if (ws+1) % 100 == 0:
        print(f"  Processed {ws+1} windows...")

sir_df = pd.DataFrame(all_sir)
print(f"\n  SIR features extracted: {sir_df.shape}")

# ============================================================
# 4. Build RI features
#    CHANGE: load OxCGRT_compact_national_v1.csv for Policy^c/Policy^p
#            and use StringencyIndex_Average (not stringency_index)
# ============================================================
print("\n[4] Building RI features (with variant & policy flags)...")

# --- Load OxCGRT for stringency-based policy indicators ---
try:
    oxcgrt_raw = pd.read_csv('data/OxCGRT_compact_national_v1.csv',
                             dtype={'Date': str}, low_memory=False)
    kor = oxcgrt_raw[
        (oxcgrt_raw['CountryCode'] == 'KOR') &
        (oxcgrt_raw['Jurisdiction'] == 'NAT_TOTAL')
    ][['Date', 'StringencyIndex_Average']].copy()

    kor['date'] = pd.to_datetime(kor['Date'], format='%Y%m%d')
    kor = kor.sort_values('date').reset_index(drop=True)
    kor['StringencyIndex_Average'] = (
        kor['StringencyIndex_Average'].ffill().fillna(0)
    )
    policy_available = True
    print("  OxCGRT loaded – Policy^c / Policy^p will use StringencyIndex_Average.")
    print(f"  OxCGRT rows for KOR: {len(kor)}  "
          f"({kor['date'].min().date()} to {kor['date'].max().date()})")
except Exception as e:
    print(f"  WARNING: Could not load OxCGRT ({e}). Policy^c/Policy^p set to 0.")
    policy_available = False
    kor = None

# Variant date ranges (South Korea)
DELTA_START   = pd.Timestamp('2021-07-01')
DELTA_END     = pd.Timestamp('2022-02-01')
OMICRON_START = pd.Timestamp('2022-02-01')
DELAY_DAYS    = 14

def get_policy_signal(win_start_date, kor_df, calib_days=21, delay=DELAY_DAYS):
    """
    Returns (Policy^c, Policy^p) as normalised [0,1] stringency means
    over the delayed calibration window and delayed prediction window.
    """
    pol_calib_start = win_start_date + pd.Timedelta(days=delay)
    pol_calib_end   = pol_calib_start + pd.Timedelta(days=calib_days - 1)
    pol_pred_start  = pol_calib_end   + pd.Timedelta(days=1)
    pol_pred_end    = pol_pred_start  + pd.Timedelta(days=13)   # 14-day pred window

    def mean_in_range(start, end):
        mask = (kor_df['date'] >= start) & (kor_df['date'] <= end)
        vals = kor_df[mask]['StringencyIndex_Average'].values
        return float(np.mean(vals)) / 100.0 if len(vals) > 0 else 0.0

    return round(mean_in_range(pol_calib_start, pol_calib_end), 4), \
           round(mean_in_range(pol_pred_start,  pol_pred_end),   4)

def calculate_ri_features(case_data_df, window_start, kor_df=None):
    window = case_data_df.iloc[window_start:window_start+35].copy()
    window['idx'] = range(len(window))

    total_min = window['number'].min()
    total_max = window['number'].max()
    if total_max > total_min:
        window['N_total'] = (window['number'] - total_min) / (total_max - total_min)
    else:
        window['N_total'] = 0.0

    calib = window.iloc[:21].copy()

    lr = LinearRegression()
    lr.fit(calib['idx'].values.reshape(-1,1), calib['N_total'].values)

    win_date = pd.Timestamp(window['date'].iloc[0])

    delta_flag   = 1 if DELTA_START   <= win_date < DELTA_END   else 0
    omicron_flag = 1 if win_date >= OMICRON_START                else 0

    if kor_df is not None:
        policy_c, policy_p = get_policy_signal(win_date, kor_df)
    else:
        policy_c, policy_p = 0.0, 0.0

    return {
        'data_num':       window_start,
        'Week':           win_date.weekday(),
        r'$\mu^c$':       round(calib['N_total'].mean(),  4),
        r'$\beta^c$':     round(lr.coef_[0],              4),
        r'$\sigma^c$':    round(calib['N_total'].std(),   4),
        r'$Delta^c$':     delta_flag,
        r'$Omicron^c$':   omicron_flag,
        r'$Policy^c$':    policy_c,
        r'$Policy^p$':    policy_p,
    }

ri_list = []
for ws in range(953):
    ri_list.append(calculate_ri_features(case_data, ws, kor_df=kor))
    if (ws+1) % 100 == 0:
        print(f"  Processed {ws+1} RI windows...")

ri_df = pd.DataFrame(ri_list)
print(f"\n  RI features: {ri_df.shape}")
print(f"  Delta windows:   {ri_df[r'$Delta^c$'].sum()}")
print(f"  Omicron windows: {ri_df[r'$Omicron^c$'].sum()}")
print(f"  Policy^c range:  {ri_df[r'$Policy^c$'].min():.3f} – {ri_df[r'$Policy^c$'].max():.3f}")

# ============================================================
# 5. Load labels from number2.xlsx
#    FIX: merge on data_num, not positional alignment
# ============================================================
print("\n[5] Loading labels from number2.xlsx...")

ri_values = pd.read_excel('temp/number2.xlsx')
ri_values = ri_values.reset_index(drop=True)
ri_values['data_num'] = ri_values.index   # window 0 = row 0

# Assign labels using RI tercile thresholds on the original order
t_low  = ri_values['RI'].quantile(1/3)
t_high = ri_values['RI'].quantile(2/3)
print(f"  Tercile thresholds: L0 < {t_low:.4f}  ≤ L1 < {t_high:.4f}  ≤ L2")

ri_values['label'] = ri_values['RI'].apply(
    lambda x: 0 if x < t_low else (1 if x < t_high else 2)
)
print(f"  Label distribution: {ri_values['label'].value_counts().sort_index().to_dict()}")

# ============================================================
# 6. Combine RI + labels + SIR
# ============================================================
print("\n[6] Combining features...")

final = pd.merge(ri_df, ri_values[['data_num','RI','label']], on='data_num', how='left')
final = pd.merge(final, sir_df, left_on='data_num', right_on='window_id', how='left')
final = final.drop(columns=['window_id'])

print(f"  Final shape: {final.shape}")
print(f"  Reliable SIR windows: {final['sir_reliable'].sum()} / {len(final)}")

# ============================================================
# 7. Save
# ============================================================
print("\n[7] Saving...")

final.to_csv('result/enhanced_pre_data.csv', index=False)
print("  Saved result/enhanced_pre_data.csv")

np.random.seed(42)
n_tr = int(0.7 * len(final))
tr_idx = np.random.choice(final.index, n_tr, replace=False)
te_idx = [i for i in final.index if i not in tr_idx]
final.iloc[tr_idx].to_csv('data/train_enhanced.csv', index=False)
final.iloc[te_idx].to_csv('data/test_enhanced.csv',  index=False)
print(f"  Train: {len(tr_idx)} | Test: {len(te_idx)}")

# ============================================================
# 8. Summary
# ============================================================
print("\n" + "=" * 60)
print("SIR FEATURES SUMMARY")
print("=" * 60)
rel = final[final['sir_reliable'] == 1]
print(f"\nReliable windows: {len(rel)}")
for col, lbl in [('beta','β'),('gamma','γ'),('R0','R₀'),('Re','Re'),('growth_rate','growth_rate')]:
    if col in rel.columns:
        print(f"  {lbl:12s}: {rel[col].min():.3f} – {rel[col].max():.3f}  "
              f"(mean {rel[col].mean():.3f})")

print("\nCorrelation with RI:")
print(rel[['RI','Re','growth_rate','sir_fit_r2','R0']].corr()['RI'].round(3))

print("\nFirst 5 rows:")
print(final[['data_num','RI','label','Re','growth_rate',
             r'$Delta^c$',r'$Omicron^c$',r'$Policy^c$']].head())

print("\nSIR FEATURE EXTRACTION COMPLETE")