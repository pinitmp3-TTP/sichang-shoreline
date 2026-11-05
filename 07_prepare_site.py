"""
07_prepare_site.py  —  merge, QC and auto-baseline one site's shorelines.

Takes the per-sensor shoreline files exported by 04_waterline_gee.py (water
polygons, each tagged with datetime_utc), and prepares everything script 05
needs, so no baseline has to be drawn by hand:

  1. merge all sensors for the site; keep only SEA water (polygons touching the
     site box edge) and drop specks below --min-area (SLC-off stripes, ponds,
     cloud-shadow false water);
  2. QC: drop scenes whose land area is an outlier (robust z > --mad), which
     catches residual cloud / failed Otsu splits;
  3. land-frequency map on a --grid metre raster -> "land core" = pixels that
     are land in >= --core of the kept scenes (robust, unlike a plain
     intersection that one bad scene can wreck);
  4. baseline = land core shrunk inland by --offset m, its ring minus the parts
     along the site box edge; oriented, and the transect --sign that points
     SEAWARD found by vote along the line;
  5. transect length = farthest shoreline excursion + margin.

Outputs in --out:  shorelines.geojson, baseline.geojson, image_times.csv,
                   scenes_per_year.csv, site_config.json (sign, length, spacing,
                   scene counts). Scenes before sites.json "start_year" are dropped.

    python 07_prepare_site.py --site hat_tham_phang \
        --inputs downloads/shoreline_landsat_hat_tham_phang.shp \
                 downloads/shoreline_s2_hat_tham_phang.shp \
        --out sites/hat_tham_phang

Box, type and spacing are read from sites.json (same file 04 uses).
"""
import argparse, json, os
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box as shp_box, LineString, Point, shape
from shapely.ops import linemerge, unary_union


def load_inputs(paths, epsg):
    parts = []
    for p in paths:
        g = gpd.read_file(p)
        if g.empty:
            continue
        if "datetime_utc" not in g.columns and "datetime_u" in g.columns:
            g = g.rename(columns={"datetime_u": "datetime_utc"})   # SHP-truncated name
        if "sensor" not in g.columns:
            g["sensor"] = os.path.basename(p).split("_")[1] if "_" in os.path.basename(p) else "unknown"
        parts.append(g.to_crs(epsg=epsg))
    if not parts:
        raise SystemExit("no shoreline features found in the inputs")
    g = pd.concat(parts, ignore_index=True)
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].explode(index_parts=False)
    return gpd.GeoDataFrame(g, crs=f"EPSG:{epsg}")


def per_scene_water(g, site_box, min_area, keep_inland):
    edge = site_box.boundary.buffer(1.0)
    g = g[g.area >= min_area]
    if not keep_inland:
        g = g[g.intersects(edge)]                         # sea = touches the box edge
    scenes = (g.dissolve(by=["datetime_utc", "sensor"], as_index=False)
                [["datetime_utc", "sensor", "geometry"]])
    scenes["geometry"] = scenes.geometry.intersection(site_box)
    scenes["land_area"] = site_box.area - scenes.area
    return scenes[~scenes.geometry.is_empty].reset_index(drop=True)


def qc(scenes, mad_k):
    a = scenes["land_area"].to_numpy()
    med = np.median(a); mad = np.median(np.abs(a - med)) * 1.4826 or 1.0
    z = np.abs(a - med) / mad
    keep = z <= mad_k
    return scenes[keep].reset_index(drop=True), scenes[~keep].reset_index(drop=True)


def land_frequency(scenes, site_box, grid):
    from rasterio import features
    from rasterio.transform import from_origin
    minx, miny, maxx, maxy = site_box.bounds
    w, h = int(np.ceil((maxx - minx) / grid)), int(np.ceil((maxy - miny) / grid))
    tr = from_origin(minx, maxy, grid, grid)
    freq = np.zeros((h, w), dtype=np.float32)
    for geom in scenes.geometry:
        water = features.rasterize([(geom, 1)], out_shape=(h, w), transform=tr,
                                   fill=0, dtype="uint8")
        freq += (water == 0)
    return freq / len(scenes), tr


def polygonize(mask, tr):
    from rasterio import features
    polys = [shape(geom) for geom, val in
             features.shapes(mask.astype("uint8"), mask=mask, transform=tr) if val == 1]
    return unary_union(polys) if polys else None


def normal_point(line, d, dist, sign):
    """Same perpendicular convention as 05_transect_rates.py."""
    p = line.interpolate(d)
    a = line.interpolate(max(d - 25.0, 0)); b = line.interpolate(min(d + 25.0, line.length))
    dx, dy = b.x - a.x, b.y - a.y
    n = np.hypot(dx, dy) or 1.0
    return Point(p.x + (-dy / n) * sign * dist, p.y + (dx / n) * sign * dist)


def chaikin(line, n=3):
    """Corner-cutting smoothing: removes the pixel staircase of a raster-derived line."""
    c = np.asarray(line.coords)
    for _ in range(n):
        q = 0.75 * c[:-1] + 0.25 * c[1:]
        r = 0.25 * c[:-1] + 0.75 * c[1:]
        c = np.vstack([c[:1], np.column_stack([q, r]).reshape(-1, 2), c[-1:]])
    return LineString(c)


def build_baseline(core, site_box, offset, smooth=30.0):
    # closing + opening by `smooth` m removes notches and spurs narrower than ~2x smooth
    core = core.buffer(smooth).buffer(-2 * smooth).buffer(smooth)
    inner = core.buffer(-offset)
    if inner.is_empty:
        raise SystemExit(f"land core vanishes when shrunk {offset} m; lower --offset")
    if inner.geom_type == "MultiPolygon":
        inner = max(inner.geoms, key=lambda g: g.area)
    ring = LineString(inner.exterior.coords)
    near_edge = site_box.exterior.buffer(offset + 5)
    coast = ring.difference(near_edge)                     # drop the box-edge parts
    if coast.is_empty:
        raise SystemExit("baseline has no coastal part; check the site box")
    if coast.geom_type == "MultiLineString":
        merged = linemerge(coast)
        coast = merged if merged.geom_type == "LineString" else max(merged.geoms, key=lambda g: g.length)
    return chaikin(coast.simplify(10.0), 3)


def seaward_sign(line, core, probe):
    votes = []
    for f in np.linspace(0.1, 0.9, 9):
        p = normal_point(line, f * line.length, probe, +1)
        votes.append(-1 if core.contains(p) else +1)       # inside land core -> +1 points landward
    return int(np.sign(sum(votes)) or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites-json", default="sites.json")
    ap.add_argument("--epsg", type=int, default=32647)
    ap.add_argument("--min-area", type=float, default=2000, help="m2; drop smaller water specks")
    ap.add_argument("--keep-inland", action="store_true")
    ap.add_argument("--mad", type=float, default=3.0)
    ap.add_argument("--grid", type=float, default=5.0)
    ap.add_argument("--core", type=float, default=0.9, help="land-frequency threshold for land core")
    ap.add_argument("--offset", type=float, default=40.0, help="baseline set-back inland (m)")
    a = ap.parse_args()

    cfg = json.load(open(a.sites_json))
    site = cfg["sites"][a.site]
    site_box = gpd.GeoSeries([shp_box(*site["box"])], crs="EPSG:4326").to_crs(epsg=a.epsg).iloc[0]
    os.makedirs(a.out, exist_ok=True)

    raw = load_inputs(a.inputs, a.epsg)
    scenes = per_scene_water(raw, site_box, a.min_area, a.keep_inland)
    sy = cfg.get("start_year")
    if sy:
        n0 = len(scenes)
        scenes = scenes[pd.to_datetime(scenes.datetime_utc).dt.year >= int(sy)].reset_index(drop=True)
        print(f"  start_year {sy}: dropped {n0 - len(scenes)} earlier scenes")
    kept, dropped = qc(scenes, a.mad)
    yrs = pd.to_datetime(kept.datetime_utc).dt.year
    per_year = (pd.crosstab(yrs, kept.sensor).reindex(
        range(yrs.min(), yrs.max() + 1), fill_value=0))
    per_year["total"] = per_year.sum(axis=1)
    per_year.index.name = "year"
    per_year.to_csv(os.path.join(a.out, "scenes_per_year.csv"))
    if len(kept) < 5:
        raise SystemExit(f"only {len(kept)} usable scenes after QC; check inputs / thresholds")

    freq, tr = land_frequency(kept, site_box, a.grid)
    core = polygonize(freq >= a.core, tr)
    ever_land = polygonize(freq >= 0.1, tr)
    baseline = build_baseline(core, site_box, a.offset)
    sign = seaward_sign(baseline, core, a.offset * 2)

    # transect length: farthest seaward excursion of any frequent shoreline + margin
    samples = [baseline.interpolate(f * baseline.length) for f in np.linspace(0, 1, 25)]
    reach = max(ever_land.boundary.distance(p) for p in samples) if ever_land else 200
    length = float(min(max(reach + 60, 120), 600))

    kept[["datetime_utc", "sensor", "geometry"]].to_file(
        os.path.join(a.out, "shorelines.geojson"), driver="GeoJSON")
    gpd.GeoDataFrame({"site": [a.site]}, geometry=[baseline], crs=f"EPSG:{a.epsg}").to_file(
        os.path.join(a.out, "baseline.geojson"), driver="GeoJSON")
    kept[["datetime_utc", "sensor"]].to_csv(os.path.join(a.out, "image_times.csv"), index=False)

    spacing = cfg.get("transect_spacing_m", {}).get(site["type"], 25)
    conf = {"site": a.site, "type": site["type"], "slope": site.get("slope"),
            "sign": sign, "length_m": round(length, 1), "spacing_m": spacing,
            "epsg": a.epsg, "n_scenes_in": int(len(scenes)), "n_kept": int(len(kept)),
            "n_dropped_qc": int(len(dropped)),
            "dropped_scenes": dropped["datetime_utc"].astype(str).tolist(),
            "by_sensor": kept["sensor"].value_counts().to_dict()}
    json.dump(conf, open(os.path.join(a.out, "site_config.json"), "w"), indent=2)

    print(f"[{a.site}] scenes in {len(scenes)} | kept {len(kept)} | QC-dropped {len(dropped)}")
    print(f"  baseline {baseline.length:.0f} m | seaward sign {sign:+d} | transect length {length:.0f} m"
          f" | spacing {spacing} m -> ~{int(baseline.length // spacing) + 1} transects")
    print(f"  sensors: {conf['by_sensor']}")
    thin = per_year.index[per_year.total < 2].tolist()
    print(f"  years with < 2 scenes after QC: {thin or 'none'} (see scenes_per_year.csv -> start_year)")
    print("  wrote shorelines.geojson, baseline.geojson, image_times.csv, site_config.json ->", a.out)


if __name__ == "__main__":
    main()
