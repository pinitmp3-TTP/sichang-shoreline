"""
02_fit_tides.py  —  harmonic analysis of the Ko Si Chang hourly series.

Fits tidal constituents with UTide on the 1-year (2026) record so the tide can
later be back-predicted at any satellite overpass time, 1988-present.

CRITICAL: trend=False.
    UTide's default (trend=True) fits a secular linear trend. A tiny slope from
    the 1-year record extrapolates to METRES of spurious offset over decades
    (e.g. +3.7 m at 2001), silently corrupting every pre-2026 tide value.
    We disable it and handle real sea-level rise separately, if at all.

Datum: heights are already ABOVE MSL, so predictions are in the MSL frame we use
as the shoreline correction target (z_target = 0).

Usage:  python 02_fit_tides.py ko_sichang_2026_hourly.csv sichang_tide_coef.pkl
"""
import sys, pickle
import numpy as np, pandas as pd, utide

LAT = 13.1617  # Ko Si Chang tide station (13 deg 09' 42" N)


def fit(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["datetime_utc"])
    t, h = df.datetime_utc.to_numpy(), df.height_m.to_numpy()
    coef = utide.solve(t, h, lat=LAT, method="ols",
                       trend=False, conf_int="none", verbose=False)
    rec = utide.reconstruct(t, coef, verbose=False)
    rmse = float(np.sqrt(np.mean((rec.h - h) ** 2)))
    top = np.argsort(coef.A)[::-1][:6]
    print(f"{len(coef.name)} constituents | self-RMSE vs table {rmse:.3f} m")
    print("top:", [(coef.name[i], round(float(coef.A[i]), 3)) for i in top])
    print("regime: diurnal (K1+O1+P1) > semidiurnal (M2+S2) -> mixed, mainly diurnal")
    return coef


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "ko_sichang_2026_hourly.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "sichang_tide_coef.pkl"
    coef = fit(csv)
    pickle.dump(coef, open(out, "wb"))
    print("wrote", out)
