"""
03_predict_at_overpass.py  —  tide level + horizontal correction at image times.

Given satellite acquisition datetimes (UTC) and the fitted coefficients, predict
the tide level (m, MSL) at each image, then convert to the horizontal shoreline
offset that moves the instantaneous waterline onto the MSL datum.

Horizontal correction (feeds the DSAS/CoastSat step):
    dx = (z_tide - z_target) / tan(beta)        z_target = MSL = 0
    positive dx -> waterline is seaward of MSL; subtract dx along the transect.

beta = beach slope (rad). Estimate per beach with CoastSat.slope; a placeholder
default is provided so this runs standalone. Rock-coast / vertical faces: slope
correction is not meaningful — leave those uncorrected and treat as the control.

Input CSV needs a `datetime_utc` column (ISO 8601). GEE gives acquisition time
as system:time_start; convert ms->UTC before writing it here.

Usage:
    python 03_predict_at_overpass.py image_times.csv sichang_tide_coef.pkl \
           tides_at_overpass.csv  [beach_slope]
"""
import sys, pickle
import numpy as np, pandas as pd, utide

DEFAULT_SLOPE = 0.06   # ~1:16 placeholder; replace with CoastSat.slope per beach


def tide_at(times_utc, coef):
    """Predicted tide level (m, MSL) at each UTC datetime."""
    t = pd.to_datetime(times_utc).to_numpy()
    return utide.reconstruct(t, coef, verbose=False).h


def horizontal_offset(z_tide, slope, z_target=0.0):
    """Cross-shore distance (m) to move the waterline onto z_target (MSL)."""
    return (np.asarray(z_tide) - z_target) / np.tan(slope)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "image_times.csv"
    coefp = sys.argv[2] if len(sys.argv) > 2 else "sichang_tide_coef.pkl"
    out = sys.argv[3] if len(sys.argv) > 3 else "tides_at_overpass.csv"
    slope = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_SLOPE

    coef = pickle.load(open(coefp, "rb"))
    df = pd.read_csv(img, parse_dates=["datetime_utc"])
    df["tide_m_msl"] = np.round(tide_at(df.datetime_utc, coef), 3)
    df["dx_m"] = np.round(horizontal_offset(df.tide_m_msl, slope), 2)
    df.to_csv(out, index=False)
    print(f"slope={slope} (1:{1/np.tan(slope):.0f}) | {len(df)} images")
    print(df.head(10).to_string(index=False) if len(df) > 20 else df.to_string(index=False))
    if len(df) > 20:
        print(f"... ({len(df) - 10} more rows) | tide at images {df.tide_m_msl.min():.2f} to "
              f"{df.tide_m_msl.max():.2f} m MSL | |dx| up to {df.dx_m.abs().max():.0f} m")
    print("wrote", out)
