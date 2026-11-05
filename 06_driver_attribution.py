"""
06_driver_attribution.py  —  associate shoreline-change rates with drivers.

Overlays the per-transect rates (from 05_transect_rates.py) with the SHARED LULC
layer as an explanatory COVARIATE ONLY, plus optional proximity to human
structures (piers/reclamation). Output is descriptive association — group means
and a rank test — NOT a re-derived LULC model. This keeps the paper distinct from
the companion soil-erosion study (salami safeguard).

    TRANSPARENCY: static land cover is ESA WorldCover 2021 (Zanaga et al., 2022),
    the same public product used descriptively in the companion erosion study
    (Tanachaichoksirikun et al., in prep). There it describes island-wide cover
    for RUSLE sub-basins; here it is only a backshore covariate for shoreline
    transects. The temporal driver series (08) is derived independently.

    python 06_driver_attribution.py transects.geojson lulc.tif drivers_out \
           [--classmap classmap.csv --buffer 60 --structures piers.geojson \
            --near 150 --epsg 32647]

Inputs
  transects : _transects.geojson from script 05 (has lrr, ci95, significant).
  lulc      : LULC class layer — GeoTIFF (categorical raster) OR a vector file
              with a `class` column. Used ONLY as covariate.
  classmap  : optional CSV `code,name,developed` (developed = 1 for urban/harbour
              /bare-from-deforestation, 0 for natural). Drives the developed-vs-
              natural contrast.
  structures: optional points/lines of piers/reclamation (optional `year`).
"""
import argparse
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from scipy import stats


def backshore_points(tr):
    """Landward end of each transect (origin coord) = the backshore to sample."""
    return gpd.GeoDataFrame(
        {"transect_id": tr.transect_id},
        geometry=[Point(g.coords[0]) for g in tr.geometry], crs=tr.crs)


def sample_raster(buffers, raster_path):
    import rasterio
    from rasterio.mask import mask as rio_mask
    out = []
    with rasterio.open(raster_path) as src:
        nod = src.nodata
        bufs = buffers.to_crs(src.crs)
        for geom in bufs.geometry:
            try:
                arr, _ = rio_mask(src, [geom.__geo_interface__], crop=True)
                v = arr[0].ravel()
                if nod is not None:
                    v = v[v != nod]
                out.append(int(stats.mode(v, keepdims=False).mode) if v.size else None)
            except Exception:
                out.append(None)
    return out


def sample_vector(buffers, lulc_path, epsg):
    lulc = gpd.read_file(lulc_path).to_crs(epsg=epsg)
    col = "class" if "class" in lulc.columns else lulc.columns[0]
    out = []
    for geom in buffers.geometry:
        hit = lulc[lulc.intersects(geom)]
        if hit.empty:
            out.append(None); continue
        areas = hit.geometry.intersection(geom).area
        out.append(hit.loc[areas.idxmax(), col])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transects"); ap.add_argument("lulc"); ap.add_argument("out_prefix")
    ap.add_argument("--classmap"); ap.add_argument("--buffer", type=float, default=60)
    ap.add_argument("--structures"); ap.add_argument("--near", type=float, default=150)
    ap.add_argument("--epsg", type=int, default=32647)
    a = ap.parse_args()

    tr = gpd.read_file(a.transects).to_crs(epsg=a.epsg)
    bs = backshore_points(tr)
    bufs = bs.copy(); bufs["geometry"] = bs.buffer(a.buffer)

    # --- adjacent LULC (covariate) ---
    if a.lulc.lower().endswith((".tif", ".tiff")):
        tr["adj_code"] = sample_raster(bufs, a.lulc)
    else:
        tr["adj_code"] = sample_vector(bufs, a.lulc, a.epsg)

    if a.classmap:
        cm = pd.read_csv(a.classmap)
        name = dict(zip(cm.code, cm.name)); dev = dict(zip(cm.code, cm.developed))
        tr["adj_name"] = tr["adj_code"].map(name)
        tr["developed"] = tr["adj_code"].map(dev).astype("Int64")
    else:
        tr["adj_name"] = tr["adj_code"].astype(str); tr["developed"] = pd.NA

    # --- optional proximity to structures ---
    if a.structures:
        st = gpd.read_file(a.structures).to_crs(epsg=a.epsg)
        tr["dist_struct_m"] = tr.geometry.apply(lambda g: st.distance(g).min()).round(1)
        tr["near_structure"] = tr["dist_struct_m"] <= a.near
        if "year" in st.columns:
            tr["near_struct_year"] = tr.geometry.apply(
                lambda g: st.loc[st.distance(g).idxmin(), "year"])

    keep = [c for c in ["transect_id", "lrr", "lrr_se", "ci95", "epr", "significant",
                        "adj_code", "adj_name", "developed", "dist_struct_m",
                        "near_structure", "near_struct_year"] if c in tr.columns]
    per = tr[keep].copy()
    per.to_csv(f"{a.out_prefix}_drivers.csv", index=False)

    # --- descriptive summaries (covariate only; no LULC tables reproduced) ---
    def grp(df, by):
        g = df.dropna(subset=[by]).groupby(by)
        s = g.agg(n=("lrr", "size"), mean_lrr=("lrr", "mean"),
                  mean_ci=("ci95", "mean"))
        if "significant" in df: s["pct_sig"] = g["significant"].mean().mul(100).round(0)
        return s.round(3)

    print("\n== rate by adjacent land use (covariate) ==")
    by_lulc = grp(per, "adj_name"); print(by_lulc.to_string())
    by_lulc.to_csv(f"{a.out_prefix}_summary_by_lulc.csv")

    if per["developed"].notna().any():
        d = per.dropna(subset=["developed"])
        a_ = d.loc[d.developed == 1, "lrr"].dropna()
        b_ = d.loc[d.developed == 0, "lrr"].dropna()
        print("\n== developed vs natural backshore ==")
        print(f"developed: n={len(a_)} mean LRR {a_.mean():.3f} m/yr")
        print(f"natural  : n={len(b_)} mean LRR {b_.mean():.3f} m/yr")
        if len(a_) >= 3 and len(b_) >= 3:
            u, p = stats.mannwhitneyu(a_, b_, alternative="two-sided")
            print(f"Mann-Whitney U p = {p:.4f} (descriptive association, not causal)")

    if "near_structure" in per.columns:
        near = per[per.near_structure]; far = per[~per.near_structure]
        print(f"\n== near structures (<= {a.near} m): n={len(near)} "
              f"mean LRR {near.lrr.mean():.3f} | far: n={len(far)} "
              f"mean LRR {far.lrr.mean():.3f} m/yr ==")

    print("\nTRANSPARENCY: static land cover = ESA WorldCover 2021 (Zanaga et al., 2022), "
          "also used descriptively in the companion erosion study (Tanachaichoksirikun "
          "et al., in prep); here only a backshore covariate.")
    print("wrote", f"{a.out_prefix}_drivers.csv", "and",
          f"{a.out_prefix}_summary_by_lulc.csv")


if __name__ == "__main__":
    main()
