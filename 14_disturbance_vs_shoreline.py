"""
14_disturbance_vs_shoreline.py  —  does backshore disturbance go with shoreline change?

Input : footprints KML/KMZ (same as 13) + results/<site>/ from run_pipeline.py
Output: results/disturbance_shoreline_site.csv   (site x epoch interval)
        results/disturbance_shoreline_transect.csv (transect x epoch interval)
        results/fig_disturbance_shoreline.png

    python 14_disturbance_vs_shoreline.py footprints.kmz

For every epoch (the VHR dates of the footprints) the shoreline position on each transect is the
median of the tide-corrected Landsat waterlines within +/-WIN days of the epoch (Landsat only, so a
sensor change cannot create a difference). Between consecutive epochs we compare
  * site level     : change of disturbed area within the band vs median shoreline displacement
  * transect level : change of disturbed area in the transect's own strip (transect spacing wide,
                     BAND m landward of the baseline to the transect end) vs the transect's displacement
Spearman rank correlation, descriptive only.
"""
import argparse, json, os, importlib.util, pickle, warnings
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
WIN = 365
PRIORITY = ["structure", "building", "pond", "paved", "cleared"]
NAMES = {"hat_tham_phang": "Hat Tham Phang", "tha_yai_thim": "Tha Yai Thim", "hin_muat_sila": "Hin Muat Sila",
         "tha_wang": "Tha Wang", "laem_ngu": "Laem Ngu", "town_harbour": "Town harbour",
         "chong_khao_khat": "Chong Khao Khat", "rock_control": "Reference cliff"}


def mod(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def footprint_union(fp, site, date, epsg, classes=PRIORITY):
    """union of the given classes after the overlap priority (structure > building > paved > cleared)"""
    q = fp[(fp.site == site) & (fp.date == date) & (fp.cls != "none")].to_crs(epsg=epsg)
    taken, out = None, []
    for c in PRIORITY:
        g = unary_union(list(q[q.cls == c].geometry)) if (q.cls == c).any() else None
        if g is None: continue
        if taken is not None: g = g.difference(taken)
        taken = g if taken is None else taken.union(g)
        if c in classes: out.append(g)
    return unary_union(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kml"); ap.add_argument("--results", default="results")
    ap.add_argument("--band", type=float, default=100)
    a = ap.parse_args(); os.chdir(HERE)
    m05, m13 = mod("m05", "05_transect_rates.py"), mod("m13", "13_disturbance_area.py")
    fp = m13.read_kml(a.kml)
    coef = pickle.load(open(os.path.join(HERE, "sichang_tide_coef.pkl"), "rb"))
    site_rows, tr_rows = [], []
    for site in sorted(fp.site.unique()):
        d0 = os.path.join(a.results, site)
        if not os.path.exists(os.path.join(d0, "site_config.json")): continue
        sc = json.load(open(os.path.join(d0, "site_config.json"))); epsg = sc["epsg"]
        tr = m05.build_transects(os.path.join(d0, "baseline.geojson"), sc["spacing_m"], sc["length_m"], sc["sign"], epsg)
        bl = gpd.read_file(os.path.join(d0, "baseline.geojson")).to_crs(epsg=epsg).geometry.union_all()
        band = bl.buffer(a.band)
        sl = m05.load_shorelines(os.path.join(d0, "shorelines.geojson"), epsg)
        sl = sl[sl.sensor == "Landsat"]
        dd = m05.measure(tr, sl)
        dd = m05.apply_tide(dd, coef if sc.get("slope") else None, sc.get("slope") or 0.06)
        # strip of each transect: from BAND m landward of the origin to the seaward end, spacing wide
        strips, axes = {}, {}
        for _, r in tr.iterrows():
            (x0, y0), (x1, y1) = r.geometry.coords[0], r.geometry.coords[-1]
            L = np.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
            ln = LineString([(x0 - ux * a.band, y0 - uy * a.band), (x1, y1)])
            strips[r.transect_id] = ln.buffer(sc["spacing_m"] / 2, cap_style=2)
            axes[r.transect_id] = (x0, y0, ux, uy, L)
        dates = sorted(fp[fp.site == site].date.unique())
        pos, area_b, area_t, ustr = {}, {}, {}, {}
        for D in dates:
            w = dd[(dd.t - D).abs() <= pd.Timedelta(days=WIN)]
            pos[D] = w.groupby("transect_id").dist.median()
            u = footprint_union(fp, site, D, epsg); uall = u
            area_b[D] = 0.0 if u is None else u.intersection(band).area
            us = footprint_union(fp, site, D, epsg, ["structure"]); ustr[D] = uall
            ul = footprint_union(fp, site, D, epsg, ["building", "pond", "paved", "cleared"])
            area_t[D] = {k: ((0.0 if us is None else us.intersection(g).area),
                             (0.0 if ul is None else ul.intersection(g).area)) for k, g in strips.items()}
        for D0, D1 in zip(dates[:-1], dates[1:]):
            dx = (pos[D1] - pos[D0]).dropna()
            lab = f"{pd.Timestamp(D0).year}–{pd.Timestamp(D1).year}"
            site_rows.append(dict(site=site, interval=lab, dA_band_m2=round(area_b[D1] - area_b[D0]),
                                  A0_band_m2=round(area_b[D0]), A1_band_m2=round(area_b[D1]),
                                  dx_median_m=round(dx.median(), 1) if len(dx) else np.nan,
                                  dx_q25=round(dx.quantile(.25), 1) if len(dx) else np.nan,
                                  dx_q75=round(dx.quantile(.75), 1) if len(dx) else np.nan, n_transects=len(dx)))
            for k, v in dx.items():
                dS = area_t[D1][k][0] - area_t[D0][k][0]; dL = area_t[D1][k][1] - area_t[D0][k][1]
                # footprint of any class beyond the earlier waterline (seaward encroachment): only this can move the waterline seaward
                x0_, y0_, ux_, uy_, L_ = axes[k]; p0 = pos[D0].get(k, np.nan)
                if np.isfinite(p0) and p0 < L_:
                    sea = LineString([(x0_ + ux_ * p0, y0_ + uy_ * p0), (x0_ + ux_ * L_, y0_ + uy_ * L_)]).buffer(sc["spacing_m"] / 2, cap_style=2)
                    a1 = 0.0 if ustr[D1] is None else ustr[D1].intersection(sea).area
                    a0 = 0.0 if ustr[D0] is None else ustr[D0].intersection(sea).area
                    dSea = a1 - a0
                else:
                    dSea = 0.0
                tr_rows.append(dict(site=site, interval=lab, transect_id=k, dx_m=round(v, 1),
                                    dA_structure_m2=round(dS), dA_seaward_m2=round(dSea),
                                    dA_land_m2=round(dL), dA_strip_m2=round(dS + dL)))
    S, T = pd.DataFrame(site_rows), pd.DataFrame(tr_rows)
    S.to_csv(os.path.join(a.results, "disturbance_shoreline_site.csv"), index=False)
    T.to_csv(os.path.join(a.results, "disturbance_shoreline_transect.csv"), index=False)
    print(S.to_string(index=False))
    r, p = spearmanr(S.dA_band_m2, S.dx_median_m, nan_policy="omit")
    print(f"\nsite x interval (n = {S.dx_median_m.notna().sum()}): Spearman(dA band, median dx) = {r:+.2f} (p = {p:.2f})")
    Tn = T[T.site != "town_harbour"]
    for lab, q in [("all transects", T), ("without town harbour", Tn), ("disturbed strips only (dA != 0)", T[T.dA_strip_m2 != 0])]:
        r, p = spearmanr(q.dA_strip_m2, q.dx_m)
        print(f"transect x interval, {lab} (n = {len(q)}): Spearman(dA strip, dx) = {r:+.2f} (p = {p:.3f})")
    T = T.assign(group=np.select([T.dA_structure_m2 > 50, T.dA_land_m2 > 50, T.dA_land_m2 < -50],
                                 ["new shore structure", "backshore gained", "backshore lost"], "unchanged"))
    g = T.assign(dist=T.group)
    print("\nshoreline displacement by strip disturbance change (median m, n):")
    print(g.groupby("dist").dx_m.agg(["median", "size"]).round(1).to_string())
    print(g.groupby("dist").dx_m.describe().round(1).to_string())
    for o in ["backshore gained", "backshore lost", "new shore structure"]:
        from scipy.stats import mannwhitneyu
        a_, b_ = g[g.dist == o].dx_m, g[g.dist == "unchanged"].dx_m
        if len(a_) > 2: print(f"{o} vs unchanged: Mann-Whitney p = {mannwhitneyu(a_, b_).pvalue:.3g}")
    print(g[g.dist == "new shore structure"][["site", "interval", "transect_id", "dx_m", "dA_structure_m2", "dA_seaward_m2"]].to_string(index=False))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw={"wspace": .3, "width_ratios": [1, 1.15]})
    cs = plt.get_cmap("tab10"); mk = {"2009–2014": "o", "2011–2014": "o", "2014–2019": "s", "2019–2024": "^"}
    for i, (s, q) in enumerate(S.groupby("site")):
        for _, r in q.iterrows():
            ax[0].errorbar(r.dA_band_m2 / 1000, r.dx_median_m, yerr=[[r.dx_median_m - r.dx_q25], [r.dx_q75 - r.dx_median_m]],
                           fmt=mk.get(r.interval, "o"), color=cs(i), ms=4, lw=.6, capsize=0,
                           label=NAMES.get(s, s) if r.interval == q.interval.iloc[0] else None)
    ax[0].axhline(0, c="#9ca3af", lw=.6); ax[0].axvline(0, c="#9ca3af", lw=.6)
    ax[0].set_xlabel("Change in disturbed area within 100 m (10³ m²)"); ax[0].set_ylabel("Shoreline displacement (m, + seaward)")
    ax[0].set_title("(a) Site × epoch interval", loc="left", fontsize=8, weight="bold"); ax[0].legend(fontsize=5.5, frameon=False, ncol=2)
    for lab, c in [("gained", "#c2410c"), ("unchanged", "#9ca3af"), ("lost", "#15803d")]:
        pass
    order = ["backshore lost", "unchanged", "backshore gained", "new shore structure"]
    r_, p_ = spearmanr(T.dA_seaward_m2, T.dx_m); print(f"Spearman(seaward encroachment, dx), all strips: {r_:+.2f} (p = {p_:.2g})")
    data = [g[g.dist == o].dx_m for o in order]
    ax[1].boxplot(data, showfliers=False, widths=.5, medianprops=dict(color="k"))
    for i, d in enumerate(data):
        ax[1].scatter(np.random.default_rng(1).normal(i + 1, .06, len(d)), d, s=3, color="#6b7280", alpha=.5, lw=0, zorder=3)
    ax[1].set_xticks(range(1, 5), [o.replace(" ", "\n", 1) + f"\n(n = {len(d)})" for o, d in zip(order, data)], fontsize=6.5)
    ax[1].axhline(0, c="#9ca3af", lw=.6); ax[1].set_ylabel("Transect displacement (m, + seaward)")
    ax[1].set_xlabel("Change in the transect strip between epochs", fontsize=7)
    ax[1].set_title("(b) Transect × interval", loc="left", fontsize=8, weight="bold")
    fig.savefig(os.path.join(a.results, "fig_disturbance_shoreline.png"), dpi=300, bbox_inches="tight")
    print("wrote results/disturbance_shoreline_site.csv, _transect.csv, fig_disturbance_shoreline.png")


if __name__ == "__main__":
    main()
