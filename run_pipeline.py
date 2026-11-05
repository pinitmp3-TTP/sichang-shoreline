"""
run_pipeline.py  —  one command from downloaded GEE shorelines to rates + drivers.

Run AFTER 04_waterline_gee.py's export tasks finish and the Drive folder has been
downloaded into ./downloads/. For every site in sites.json it runs:

    07 prepare (merge + QC + auto-baseline)  ->  03 tide at each image
    ->  05 transect rates (tide-corrected; rock sites uncorrected)
    ->  06 driver attribution (WorldCover 2021 backshore covariate, default on)
then 09 timing check if downloads/backshore_annual.csv (from 08) is present

and writes a combined summary table results/site_summary.csv.

    python run_pipeline.py                         # rates only
    python run_pipeline.py --lulc lulc.tif --classmap classmap.csv \
                           --structures piers.geojson

Tide prerequisites (01/02) are run automatically if their outputs are missing.
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(args, label):
    print(f"\n--- {label} ---", flush=True)
    r = subprocess.run([PY, *args], cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"FAILED at: {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default="downloads")
    ap.add_argument("--results", default="results")
    ap.add_argument("--sites", nargs="*", help="subset of site names (default: all)")
    ap.add_argument("--lulc", default="Sichang_WorldCover2021_10m.tif",
                    help="static land cover for 06 (default: the WorldCover 2021 tif in this folder)")
    ap.add_argument("--classmap", default="classmap_worldcover.csv")
    ap.add_argument("--structures")
    ap.add_argument("--backshore", help="08 output; default: downloads/backshore_annual*.csv if present")
    a = ap.parse_args()

    os.chdir(HERE)
    if not os.path.exists("sichang_tide_coef.pkl"):
        if not os.path.exists("ko_sichang_2026_hourly.csv"):
            run(["01_parse_tides.py", "SC2026msl.pdf", "ko_sichang_2026_hourly.csv"], "01 parse tide table")
        run(["02_fit_tides.py", "ko_sichang_2026_hourly.csv", "sichang_tide_coef.pkl"], "02 fit tides")

    cfg = json.load(open("sites.json"))
    names = a.sites or list(cfg["sites"])
    summary = []
    for site in names:
        files = sorted(glob.glob(os.path.join(a.downloads, f"shoreline_*_{site}_*.geojson"))
                       + glob.glob(os.path.join(a.downloads, f"shoreline_*_{site}.geojson")))
        if not files:
            print(f"\n[skip] {site}: no files matching shoreline_*_{site}_*.geojson in {a.downloads}/")
            continue
        out = os.path.join(a.results, site)
        run(["07_prepare_site.py", "--site", site, "--inputs", *files, "--out", out],
            f"07 prepare {site} ({len(files)} files)")
        sc = json.load(open(os.path.join(out, "site_config.json")))

        rock = sc["type"] == "rock" or sc.get("slope") is None
        slope = str(sc["slope"] or 0.06)
        run(["03_predict_at_overpass.py", os.path.join(out, "image_times.csv"),
             "sichang_tide_coef.pkl", os.path.join(out, "tides_at_overpass.csv"), slope],
            f"03 tide at overpass {site}")

        prefix = os.path.join(out, site)
        brk = cfg["sites"][site].get("break_year")
        args05 = ["05_transect_rates.py", os.path.join(out, "shorelines.geojson"),
                  os.path.join(out, "baseline.geojson"),
                  "none" if rock else "sichang_tide_coef.pkl", prefix,
                  "--spacing", str(sc["spacing_m"]), "--length", str(sc["length_m"]),
                  "--slope", slope, "--sign", str(sc["sign"]), "--epsg", str(sc["epsg"])]
        if brk:
            args05 += ["--break-year", str(brk)]
        run(args05, f"05 rates {site}{' (rock: no tide correction)' if rock else ''}"
                    f"{f' (breakpoint: {brk})' if brk else ''}")

        if a.lulc and os.path.exists(a.lulc):
            args = ["06_driver_attribution.py", prefix + "_transects.geojson", a.lulc, prefix]
            if a.classmap: args += ["--classmap", a.classmap]
            if a.structures: args += ["--structures", a.structures]
            run(args, f"06 drivers {site}")

        r = pd.read_csv(prefix + "_rates.csv")
        v = r.dropna(subset=["lrr"])
        row = {"site": site, "type": sc["type"], "role": cfg["sites"][site].get("role", "target"),
               "scenes": sc["n_kept"],
               "qc_dropped": sc["n_dropped_qc"], "transects": len(r),
               "median_lrr_m_yr": round(v.lrr.median(), 3) if len(v) else None,
               "mean_ci95_m_yr": round(v.ci95.mean(), 3) if len(v) else None,
               "pct_significant": round(100 * v.significant.mean(), 0) if len(v) else None,
               "median_lrr_landsat": round(v.lrr_landsat.median(), 3) if "lrr_landsat" in v and v.lrr_landsat.notna().any() else None,
               "offset_s2_m": round(v.offset_s2_m.median(), 1) if "offset_s2_m" in v and v.offset_s2_m.notna().any() else None,
               "offset_s1_m": round(v.offset_s1_m.median(), 1) if "offset_s1_m" in v and v.offset_s1_m.notna().any() else None}
        if "step_flag" in r.columns:
            f = r[r.step_flag.fillna(False).astype(bool)]
            row.update(step_transects=int(len(f)),
                       step_year=int(f.step_year.median()) if len(f) else None,
                       step_shift_m=round(f.step_shift_m.median(), 1) if len(f) else None)
        if "break_year" in r.columns and r.break_year.notna().any():
            b = r.dropna(subset=["break_year"])
            row.update(break_year=int(b.break_year.median()),
                       break_95range=f"{int(b.break_lo.median())}-{int(b.break_hi.median())}",
                       lrr_pre=round(b.lrr_pre.median(), 3),
                       lrr_post=round(b.lrr_post.median(), 3),
                       pct_break_supported=round(100 * b.break_supported.mean(), 0),
                       break_step_m=round(b.break_step_m.median(), 1) if "break_step_m" in b else None)
        summary.append(row)

    if not summary:
        raise SystemExit(f"\nNo site was processed: put the GEE exports (shoreline_*_<site>_*.geojson) "
                         f"in {a.downloads}/ first.")
    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(os.path.join(a.results, "site_summary.csv"), index=False)
        print("\n=== SITE SUMMARY ===\n" + s.to_string(index=False))
        # Empirical precision floor from the control: regression CIs assume independent
        # errors and come out too narrow with ~1000 scenes, so a site rate is only
        # counted as real change if it exceeds what the undeveloped cliff shows.
        ctrl = [r for r in summary if r["role"] == "control"]
        if ctrl:
            cr = pd.read_csv(os.path.join(a.results, ctrl[0]["site"], f"{ctrl[0]['site']}_rates.csv")).dropna(subset=["lrr"])
            # floor = the larger of (95th pct of |control rates|, mean control CI):
            # never narrower than the regression CI, wider where the cliff scatters more
            floor = float(max(np.percentile(np.abs(cr.lrr), 95), cr.ci95.mean()))
            s["control_floor_m_yr"] = round(floor, 3)
            beyond = []
            for r in s.itertuples():
                rr = pd.read_csv(os.path.join(a.results, r.site, f"{r.site}_rates.csv")).dropna(subset=["lrr"])
                beyond.append(round(100 * (np.abs(rr.lrr) > floor).mean(), 0) if len(rr) else None)
            s["pct_beyond_floor"] = beyond
            s.to_csv(os.path.join(a.results, "site_summary.csv"), index=False)
            c = s[s.role == "control"].iloc[0]
            print("\n=== SITE SUMMARY (with control floor) ===\n" + s.to_string(index=False))
            print(f"\nControl [{c.site}]: median LRR {c.median_lrr_m_yr} m/yr; precision floor "
                  f"+/-{floor:.3f} m/yr (95th pct of |control rates| or control CI, whichever is larger). pct_beyond_floor = share of a site's transects "
                  f"moving faster than any undeveloped cliff does. The regression CI ({c.mean_ci95_m_yr}) "
                  f"is too narrow on its own ({c.pct_significant}% of control transects 'significant').")

    bs = a.backshore or next(iter(sorted(glob.glob(os.path.join(a.downloads, "backshore_annual*.csv")))), None)
    if bs and os.path.exists(bs):
        run(["09_timing_check.py", bs, a.results], "09 driver timing (backshore development vs shoreline break)")
    else:
        print("\n[skip] 09: no backshore_annual.csv in downloads/ (run 08_backshore_change_gee.py)")
    print(f"\nAll outputs in {a.results}/")


if __name__ == "__main__":
    main()
