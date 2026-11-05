"""
09_timing_check.py  —  does backshore development line up with shoreline change?

Joins the annual backshore series from 08 (built_frac, ndvi_mean per site-year)
with the shoreline results from 05 and reports, per site:

  * change in built-up fraction and NDVI, first 5 vs last 5 valid years
  * the year of the largest 3-year rise in built_frac (development pulse), only
    when that rise is at least --min-rise (default 0.10 = 10 % of the backshore);
    smaller rises are reported as no development detected
  * for sites with a shoreline breakpoint: whether the development pulse falls
    inside the break year's 95% range (+/- --tolerance years) -> "coincident"

Coincidence is circumstantial evidence for human-driven change, not proof:
confirm with historical imagery and the structure's construction record.

    python 09_timing_check.py downloads/backshore_annual.csv results
"""
import argparse, os, glob
import numpy as np, pandas as pd


def pulse_year(s):
    """Year ending the largest 3-yr rise of a (year-indexed) series."""
    s = s.dropna()
    if len(s) < 8:
        return None, None
    sm = s.rolling(3, center=True, min_periods=2).mean()
    rise = sm.diff(3)
    if rise.dropna().empty:
        return None, None
    y = int(rise.idxmax())
    return y, float(rise.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backshore_csv"); ap.add_argument("results_dir")
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--min-images", type=int, default=1)
    ap.add_argument("--min-rise", type=float, default=0.10,
                    help="smallest 3-yr rise in built-up fraction counted as a development pulse")
    a = ap.parse_args()

    b = pd.read_csv(a.backshore_csv)
    b = b[b.n_images >= a.min_images]
    out = []
    for site, g in b.groupby("site"):
        g = g.set_index("year").sort_index()
        v = g.dropna(subset=["built_frac"])
        if len(v) < 10:
            out.append({"site": site, "note": "too few valid years"}); continue
        first, last = v.head(5), v.tail(5)
        py, rise = pulse_year(v.built_frac)
        detected = rise is not None and rise >= a.min_rise
        if not detected:
            py = None                      # rise within noise: no development pulse
        row = {"site": site, "valid_years": len(v),
               "built_first5": round(first.built_frac.mean(), 3),
               "built_last5": round(last.built_frac.mean(), 3),
               "built_change": round(last.built_frac.mean() - first.built_frac.mean(), 3),
               "ndvi_change": round(last.ndvi_mean.mean() - first.ndvi_mean.mean(), 3),
               "development_detected": detected,
               "dev_pulse_year": py, "dev_pulse_rise": None if rise is None else round(rise, 3)}
        rates = os.path.join(a.results_dir, site, f"{site}_rates.csv")
        if os.path.exists(rates):
            r = pd.read_csv(rates)
            row["shore_lrr_median"] = round(r.lrr.median(), 3)
            if "break_year" in r.columns and r.break_year.notna().any():
                rb = r.dropna(subset=["break_year"])
                by, lo, hi = int(rb.break_year.median()), int(rb.break_lo.median()), int(rb.break_hi.median())
                row.update(shore_break_year=by, shore_break_range=f"{lo}-{hi}",
                           shore_post_minus_pre=round(rb.lrr_post.median() - rb.lrr_pre.median(), 3))
                if py is not None:
                    row["coincident"] = (lo - a.tolerance) <= py <= (hi + a.tolerance)
        out.append(row)

    t = pd.DataFrame(out)
    path = os.path.join(a.results_dir, "driver_timing.csv")
    t.to_csv(path, index=False)
    print(t.to_string(index=False))
    print("\ncoincident = development pulse inside the shoreline break range "
          f"(+/- {a.tolerance} yr): circumstantial support for human-driven change; "
          "confirm with historical imagery.")
    print("wrote", path)


if __name__ == "__main__":
    main()
