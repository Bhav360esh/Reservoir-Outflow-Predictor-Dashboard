"""Leak-free reservoir outflow model.  Usage: python reservoir_fixed.py dataset.xlsx"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

d = pd.read_excel(sys.argv[1], sheet_name="Dataset")
d.columns = d.columns.str.strip()
y_col = "RELEASE (OUTFLOW)"
now = ["INFLOW", "EVAPORATION", "RESERVOIR LEVEL", "RESERVOIR STORAGE"]

def run(label, X, y):
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    k = int(len(y) * 0.8)                      # first 80 % trains, last 20 % tests
    print(f"\n{label}  (train {k}, test {len(y) - k})")
    for name, m in [("linear", LinearRegression()),
                    ("random forest", RandomForestRegressor(n_estimators=300, random_state=42))]:
        p = m.fit(X.iloc[:k], y.iloc[:k]).predict(X.iloc[k:])
        print(f"  {name:14s} R2={r2_score(y.iloc[k:], p):.3f} RMSE={mean_squared_error(y.iloc[k:], p) ** .5:.0f}")

run("A. same-month inputs (assumes level and storage are known before the release)", d[now], d[y_col])

# B. Only what is known before the month starts, plus this month's inflow forecast taken as observed.
lag = pd.DataFrame({"INFLOW": d["INFLOW"],
                    "prev level": d["RESERVOIR LEVEL"].shift(1),
                    "prev storage": d["RESERVOIR STORAGE"].shift(1),
                    "prev release": d[y_col].shift(1)})
run("B. lagged state (safe if level/storage are month-end values)", lag, d[y_col])

# C. Baseline every model must beat: the same month one year earlier.
base = d[y_col].shift(12)
k = int((len(d) - 12) * 0.8) + 12
print(f"\nC. same month last year  R2={r2_score(d[y_col].iloc[k:], base.iloc[k:]):.3f}")

# D. Add the season: month number and the release in the same month last year.
m = np.arange(len(d)) % 12                    # rows are consecutive months
season = lag.assign(month_sin=np.sin(2*np.pi*m/12), month_cos=np.cos(2*np.pi*m/12),
                    release_last_year=d[y_col].shift(12))
run("D. lagged state + season", season, d[y_col])
last = season.notna().all(axis=1)
idx = last[last].index[int(last.sum() * 0.8):]
print(f"  same month last year on D's test months: R2={r2_score(d[y_col][idx], base[idx]):.3f} RMSE={mean_squared_error(d[y_col][idx], base[idx])**.5:.0f}")
