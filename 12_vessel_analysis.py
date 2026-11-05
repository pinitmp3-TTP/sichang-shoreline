"""
12_vessel_analysis.py  —  vessel pressure per site vs shoreline change.

Input : vessels_s1.csv from 11_vessels_s1_gee.py (Drive) + results/ from run_pipeline.py
Output: results/vessels_site_summary.csv, results/vessels_annual.csv, results/fig_vessels.png

    python 12_vessel_analysis.py --csv "/content/drive/MyDrive/GEE_Sichang/vessels_s1.csv"

Two tests, both descriptive (eight sites cannot support a causal claim):
  1) across sites: mean vessel area per scene vs long-term LRR and % transects beyond floor
  2) within a site (default tha_yai_thim): annual vessel area vs annual shoreline
     anomaly (optical waterlines, de-trended per transect, site median), lag 0 and 1 yr
"""
import argparse, json, os, importlib.util, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"hat_tham_phang": "Hat Tham Phang", "tha_yai_thim": "Tha Yai Thim", "hin_muat_sila": "Hin Muat Sila",
         "tha_wang": "Tha Wang", "laem_ngu": "Laem Ngu", "town_harbour": "Town harbour",
         "chong_khao_khat": "Chong Khao Khat", "rock_control": "Cliff control", "island": "Whole island"}


def load(path):
    v = pd.read_csv(path)
    v["t"] = pd.to_datetime(v.datetime_utc)
    for c in ["vessel_m2", "n_targets"]:
        v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0)
    v["sea_m2"] = pd.to_numeric(v.sea_m2, errors="coerce")
    # drop scenes whose sea coverage in a zone is < 80% of that zone's usual coverage
    v = v[v.sea_m2 >= 0.8 * v.groupby("zone").sea_m2.transform("median")]
    v["vessel_ppm"] = 1e6 * v.vessel_m2 / v.sea_m2          # m2 of vessel per km2 of sea
    v["occupied"] = v.n_targets > 0
    return v


def shoreline_anomaly(site, results):
    """Annual median of de-trended optical waterline position (m), site median of transects."""
    spec = importlib.util.spec_from_file_location("m05", os.path.join(HERE, "05_transect_rates.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sc = json.load(open(os.path.join(results, site, "site_config.json")))
    tr = m.build_transects(os.path.join(results, site, "baseline.geojson"), sc["spacing_m"], sc["length_m"], sc["sign"], sc["epsg"])
    sl = m.load_shorelines(os.path.join(results, site, "shorelines.geojson"), sc["epsg"])
    sl = sl[(sl.sensor != "S1") & (sl.t >= "2015-01-01")]
    d = m.measure(tr, sl)
    coef = None if sc.get("slope") is None else pickle.load(open(os.path.join(HERE, "sichang_tide_coef.pkl"), "rb"))
    d = m.apply_tide(d, coef, sc.get("slope") or 0.06)
    d["yr"] = d.t.dt.year + d.t.dt.day_of_year / 365.25
    out = []
    for tid, q in d.groupby("transect_id"):
        if len(q) < 20: continue
        b = np.polyfit(q.yr, q.dist, 1)
        q = q.assign(res=q.dist - np.polyval(b, q.yr)); out.append(q)
    a = pd.concat(out)
    return a.groupby([a.t.dt.year, "transect_id"]).res.median().groupby(level=0).median().rename("shore_anom_m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--site", default="tha_yai_thim", help="site for the within-site test")
    a = ap.parse_args()
    os.chdir(HERE)
    v = load(a.csv)
    print(f"{v.datetime_utc.nunique()} scenes, {v.t.min().date()} .. {v.t.max().date()}")

    summ = v.groupby("zone").agg(scenes=("t", "size"), pct_scenes_with_vessels=("occupied", "mean"),
                                 mean_vessel_m2=("vessel_m2", "mean"), mean_vessel_ppm=("vessel_ppm", "mean"),
                                 median_targets=("n_targets", "median"))
    summ["pct_scenes_with_vessels"] = (100 * summ.pct_scenes_with_vessels).round(0)
    ss = pd.read_csv(os.path.join(a.results, "site_summary.csv")).set_index("site")
    summ = summ.join(ss[["median_lrr_m_yr", "pct_beyond_floor"]])
    summ.round(2).to_csv(os.path.join(a.results, "vessels_site_summary.csv"))
    print("\n== vessel pressure per zone (S1, 06:09 local) ==\n" + summ.round(2).to_string())

    tgt = summ.drop(index=["island", "rock_control"], errors="ignore").dropna(subset=["median_lrr_m_yr"])
    for col in ["median_lrr_m_yr", "pct_beyond_floor"]:
        r, p = spearmanr(tgt.mean_vessel_ppm, tgt[col])
        print(f"across {len(tgt)} target sites: Spearman(vessel density, {col}) = {r:+.2f} (p = {p:.2f}; descriptive only)")

    ann = v.groupby([v.t.dt.year.rename("year"), "zone"]).vessel_ppm.mean().unstack()
    ann.round(1).to_csv(os.path.join(a.results, "vessels_annual.csv"))

    an = shoreline_anomaly(a.site, a.results)
    j = pd.concat([ann[a.site].rename("vessel_ppm"), an], axis=1).dropna()
    for lag in (0, 1):
        x = j.vessel_ppm.shift(lag).dropna(); y = j.shore_anom_m.loc[x.index]
        if len(x) >= 5:
            r, p = spearmanr(x, y)
            print(f"{a.site}: annual vessel density vs shoreline anomaly, lag {lag} yr: rho = {r:+.2f} (p = {p:.2f}, n = {len(x)})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    for z in [c for c in ann.columns if c != "island"]:
        ax[0].plot(ann.index, ann[z], marker="o", ms=3, lw=1.2, label=NAMES.get(z, z),
                   color="#c2410c" if z == a.site else None, zorder=3 if z == a.site else 2)
    ax[0].set_ylabel("Vessel area (m² per km² of sea)"); ax[0].set_title("(a) Annual vessel presence, Sentinel-1")
    ax[0].legend(fontsize=7, ncol=2, frameon=False); ax[0].grid(alpha=.3)
    ax2 = ax[1]; ax2.bar(j.index, j.shore_anom_m, color="#9ca3af", label="Shoreline anomaly")
    ax2.set_ylabel("Shoreline anomaly (m, + seaward)"); ax2b = ax2.twinx()
    ax2b.plot(j.index, j.vessel_ppm, color="#c2410c", marker="o", label="Vessel area")
    ax2b.set_ylabel("Vessel area (m²/km²)"); ax2.set_title(f"(b) {NAMES[a.site]}")
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "fig_vessels.png"), dpi=300)
    print("wrote results/vessels_site_summary.csv, results/vessels_annual.csv, results/fig_vessels.png")


if __name__ == "__main__":
    main()
