"""
13_disturbance_area.py  —  human-disturbance footprint per site and epoch,
from polygons digitised in Google Earth Pro.

Polygon names:  <site>_<YYYYMMDD>_<class>[_<n>]      e.g. hin_muat_sila_20190129_building_3
classes:
  building   roofs and built platforms on land
  paved      roads, car parks, paved yards
  cleared    bare or graded ground cleared by people (not natural rock or beach)
  pond       excavated detention pond or reservoir
  structure  coastal structures: piers, jetties, seawalls, revetments, reclaimed land
An epoch with nothing to digitise is recorded by one polygon named <site>_<date>_none
(any small shape) so that "zero" is not confused with "missing".

    python 13_disturbance_area.py footprints.kml          (or .kmz)
Output: results/disturbance_area.csv, results/disturbance_change.csv, results/fig_disturbance.png
"""
import sys, re, os, zipfile, json, argparse
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["building", "paved", "cleared", "structure", "pond"]
COLORS = {"building": "#b91c1c", "paved": "#6b7280", "cleared": "#d97706", "structure": "#1d4ed8", "pond": "#0d9488"}
NAME_RE = re.compile(r"^([a-z_]+?)_(\d{8})_(building|paved|cleared|structure|pond|none)(?:_\w+)?$")


def read_kml(path):
    if path.lower().endswith(".kmz"):
        with zipfile.ZipFile(path) as z:
            txt = z.read([n for n in z.namelist() if n.endswith(".kml")][0]).decode("utf-8")
    else:
        txt = open(path, encoding="utf-8").read()
    rows, bad = [], []
    for pm in re.findall(r"<Placemark.*?</Placemark>", txt, re.S):
        nm = re.search(r"<name>(.*?)</name>", pm, re.S)
        nm = nm.group(1).strip() if nm else ""
        m = NAME_RE.match(nm)
        if not m:
            bad.append(nm); continue
        site, date, cls = m.groups()
        rings = re.findall(r"<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", pm, re.S)
        for c in rings:
            pts = [tuple(map(float, s.split(",")[:2])) for s in c.split()]
            if len(pts) >= 4:
                rows.append(dict(site=site, date=pd.Timestamp(date), cls=cls, geometry=Polygon(pts).buffer(0)))
        if cls == "none" and not rings:
            rows.append(dict(site=site, date=pd.Timestamp(date), cls="none", geometry=None))
    if bad:
        print(f"[warn] {len(bad)} placemarks with names not matching <site>_<YYYYMMDD>_<class>: {bad[:8]}")
    return gpd.GeoDataFrame(rows, crs=4326)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kml"); ap.add_argument("--results", default="results")
    ap.add_argument("--band", type=float, default=100, help="backshore band half-width around the baseline (m)")
    a = ap.parse_args()
    os.chdir(HERE)
    cfg = json.load(open("sites.json"))
    g = read_kml(a.kml)
    g = g.to_crs(32647)
    rows = []
    for (site, date), grp in g.groupby(["site", "date"]):
        if site not in cfg["sites"]:
            print(f"[warn] unknown site {site}"); continue
        bx = gpd.GeoSeries([box(*cfg["sites"][site]["box"])], crs=4326).to_crs(32647).iloc[0]
        bl_path = os.path.join(a.results, site, "baseline.geojson")
        band = None
        if os.path.exists(bl_path):
            bl = gpd.read_file(bl_path).to_crs(32647).geometry.union_all()
            band = bl.buffer(a.band).intersection(bx)
        taken = None   # overlaps go to the higher-priority class: structure > building > pond > paved > cleared
        for cls in ["structure", "building", "pond", "paved", "cleared"]:
            geoms = [x for x in grp[grp.cls == cls].geometry if x is not None]
            u = unary_union(geoms).intersection(bx) if geoms else None
            if u is not None and taken is not None:
                u = u.difference(taken)
            if u is not None:
                taken = u if taken is None else taken.union(u)
            rows.append(dict(site=site, date=date.date(), cls=cls,
                             area_m2=round(u.area, 0) if u else 0.0,
                             band_area_m2=round(u.intersection(band).area, 0) if (u and band is not None) else (0.0 if band is not None else np.nan),
                             n_polygons=len(geoms)))
    t = pd.DataFrame(rows)
    os.makedirs(a.results, exist_ok=True)
    t.to_csv(os.path.join(a.results, "disturbance_area.csv"), index=False)
    wide = t.pivot_table(index=["site", "date"], columns="cls", values="area_m2", fill_value=0)
    wide["total_m2"] = wide.sum(axis=1)
    bandw = t.pivot_table(index=["site", "date"], values="band_area_m2", aggfunc="sum").rename(columns={"band_area_m2": f"total_within_{int(a.band)}m_m2"})
    wide = wide.join(bandw)
    print("\n== disturbance footprint (m2) inside each site box ==\n" + wide.round(0).astype("Int64").to_string())

    ch = []
    for site, w in wide.groupby(level=0):
        w = w.droplevel(0).sort_index()
        if len(w) < 2: continue
        first, last = w.iloc[0], w.iloc[-1]
        ch.append(dict(site=site, first=w.index[0], last=w.index[-1],
                       **{f"d_{c}_m2": last.get(c, 0) - first.get(c, 0) for c in CLASSES}, d_total_m2=last["total_m2"] - first["total_m2"]))
    ch = pd.DataFrame(ch); ch.to_csv(os.path.join(a.results, "disturbance_change.csv"), index=False)
    print("\n== change first -> last epoch (m2) ==\n" + ch.round(0).to_string(index=False))

    sites = list(dict.fromkeys(wide.index.get_level_values(0)))
    fig, axes = plt.subplots(2, (len(sites) + 1) // 2, figsize=(2.6 * ((len(sites) + 1) // 2), 5.5), sharey=True, squeeze=False)
    for ax, site in zip(axes.ravel(), sites):
        w = wide.loc[site]; bottom = np.zeros(len(w)); x = np.arange(len(w))
        for c in CLASSES:
            vals = w.get(c, pd.Series(0, index=w.index)).values / 1e3
            ax.bar(x, vals, bottom=bottom, color=COLORS[c], label=c, width=0.7); bottom += vals
        ax.set_xticks(x); ax.set_xticklabels([str(d)[:4] for d in w.index], fontsize=8)
        ax.set_title(site.replace("_", " "), fontsize=9); ax.grid(axis="y", alpha=.3)
    for ax in axes.ravel()[len(sites):]: ax.axis("off")
    axes[0, 0].set_ylabel("Footprint (×1000 m²)"); axes[1, 0].set_ylabel("Footprint (×1000 m²)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "fig_disturbance.png"), dpi=300)
    print("wrote results/disturbance_area.csv, results/disturbance_change.csv, results/fig_disturbance.png")


if __name__ == "__main__":
    main()
