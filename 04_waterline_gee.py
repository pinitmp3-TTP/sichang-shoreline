"""
04_waterline_gee.py  —  Si Chang shoreline extraction in Google Earth Engine.

Pulls Landsat 5/7/8/9, Sentinel-2 and Sentinel-1, extracts the instantaneous
waterline per scene (optical: MNDWI + per-image Otsu; SAR: VV sigma0 + Otsu),
and exports the vectorised shorelines to Drive with the acquisition time on every
feature so they join straight to the tide pipeline (03_predict_at_overpass.py).

RUN THIS YOURSELF: it needs an authenticated Earth Engine project.
    pip install earthengine-api
    earthengine authenticate
    python 04_waterline_gee.py            # starts the export tasks

NOTE ON METHOD: this is PIXEL-resolution shoreline (Otsu threshold -> water mask
-> vectorised boundary). Sub-pixel shorelines (~10 m RMSE) come from CoastSat,
run separately on sandy sites. Use this GEE layer for SAR, for cloud-season
gap-filling, and for the non-sandy / human-modified sites where CoastSat's sand
classifier under-performs.

Author: IEWS / KMITL   |   companion to the Si Chang shoreline method spec.
"""
import ee, json, os

# ----------------------------------------------------------------------------
# CONFIG  — everything you edit (ee_project, sites, Drive folder, S1 orbit) is in sites.json.
# ----------------------------------------------------------------------------
START_YEAR, END_YEAR = 1988, 2026
CHUNK_YEARS = 6                           # one export task per period (avoids EE timeouts)
MAX_CLOUD = 40                            # AOI-level % cloud cap (optical)
EXPORT_SCALE = 10                         # vectorising scale (m)
MIN_VALID = 0.6                           # min fraction of clear (unmasked) pixels in the site box
START, END = f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"

# Site boxes: ONE source of truth shared with 07/run_pipeline (sites.json).
_CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sites.json")))
AOI_BBOX = _CFG["aoi"]
SITES = {name: s for name, s in _CFG["sites"].items()}   # {name: {box, type, role, ...}}
EE_PROJECT = _CFG["ee_project"]                           # set in sites.json
DRIVE_FOLDER = _CFG.get("drive_folder", "GEE_Sichang")    # -> My Drive/GEE_Sichang
# Sentinel-1 geometry: ONE pass + ONE relative orbit for the whole series.
# Set by 00_data_inventory.py; "auto" = choose the pass/orbit with most scenes here.
S1_PASS = _CFG.get("s1", {}).get("pass", "auto")
S1_ORBIT = _CFG.get("s1", {}).get("orbit", "auto")

# Common band names after harmonisation.
OPT_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


# ----------------------------------------------------------------------------
# Otsu threshold (Donchyts 2016) — server-side, from a 1-band histogram dict.
# ----------------------------------------------------------------------------
def otsu(histogram):
    histogram = ee.Dictionary(histogram)
    counts = ee.Array(histogram.get("histogram"))
    means = ee.Array(histogram.get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sums = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sums.divide(total)
    indices = ee.List.sequence(1, size)

    def bss(i):
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)
        b_count = total.subtract(a_count)
        b_mean = sums.subtract(a_count.multiply(a_mean)).divide(b_count)
        return a_count.multiply(a_mean.subtract(mean).pow(2)).add(
               b_count.multiply(b_mean.subtract(mean).pow(2)))

    bsss = indices.map(lambda i: bss(ee.Number(i)))
    return means.sort(ee.Array(bsss)).get([-1])   # mean at max between-class variance


def dyn_threshold(index_img, geom, scale):
    """Per-image Otsu threshold of a single-band index over geom."""
    hist = index_img.reduceRegion(
        reducer=ee.Reducer.histogram(255, 0.001),
        geometry=geom, scale=scale, bestEffort=True, maxPixels=1e9)
    return ee.Number(otsu(hist.get(index_img.bandNames().get(0))))


# ----------------------------------------------------------------------------
# Optical: Landsat C2 L2 + Sentinel-2 SR, harmonised, cloud-masked, MNDWI.
# ----------------------------------------------------------------------------
def mask_landsat(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
             .And(qa.bitwiseAnd(1 << 2).eq(0))   # cirrus
             .And(qa.bitwiseAnd(1 << 3).eq(0))   # cloud
             .And(qa.bitwiseAnd(1 << 4).eq(0)))  # cloud shadow
    sr = img.select("SR_B.").multiply(2.75e-5).add(-0.2)
    return img.addBands(sr, None, True).updateMask(clear)


def landsat_collection(aoi):
    def harmonise(coll, bands):
        return coll.map(mask_landsat).select(bands, OPT_BANDS)
    tm = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]   # L5/L7
    oli = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]  # L8/L9
    l5 = harmonise(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"), tm)
    l7 = harmonise(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"), tm)
    l8 = harmonise(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"), oli)
    l9 = harmonise(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"), oli)
    return (l5.merge(l7).merge(l8).merge(l9)
            .filterBounds(aoi).filterDate(START, END)
            .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD)))


def mask_s2(img):
    scl = img.select("SCL")
    bad = [3, 8, 9, 10, 11]   # shadow, cloud med/high, cirrus, snow
    keep = scl.remap(bad, [0] * len(bad), 1)
    sr = img.select(["B2", "B3", "B4", "B8", "B11", "B12"], OPT_BANDS).divide(10000)
    return sr.updateMask(keep).copyProperties(img, ["system:time_start"])


def s2_collection(aoi):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi).filterDate("2017-01-01", END)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
            .map(mask_s2))


def keep_clear(coll, geom):
    """Drop scenes whose clear-pixel fraction in the site box is below MIN_VALID.

    A scene fully masked by cloud over the box gives an empty histogram, and one
    such scene makes Otsu (and so the WHOLE export task) fail. Pixels outside the
    scene footprint count as not clear (unmask(0)).
    """
    def frac(img):
        f = (img.select("green").mask().unmask(0)
             .reduceRegion(ee.Reducer.mean(), geom, 30, bestEffort=True, maxPixels=1e9)
             .get("green"))
        return img.set("valid_frac", f)
    return coll.map(frac).filter(ee.Filter.gte("valid_frac", MIN_VALID))


def optical_shorelines(coll, geom, sensor):
    """Per clear image: MNDWI -> Otsu -> water polygon boundary, tagged with time."""
    def extract(img):
        mndwi = img.normalizedDifference(["green", "swir1"]).rename("MNDWI").clip(geom)
        thr = dyn_threshold(mndwi, geom, EXPORT_SCALE)
        water = mndwi.gt(thr).selfMask().rename("water")
        vec = water.reduceToVectors(
            geometry=geom, scale=EXPORT_SCALE, geometryType="polygon",
            eightConnected=True, maxPixels=1e9, bestEffort=True)
        t = img.date().format("YYYY-MM-dd HH:mm:ss")
        return vec.map(lambda f: f.set({
            "datetime_utc": t, "sensor": sensor,
            "scene_id": img.get("system:index"), "otsu": thr,
            "valid_frac": img.get("valid_frac")}))
    return coll.map(extract).flatten()


# ----------------------------------------------------------------------------
# SAR: Sentinel-1 GRD, one pass, speckle-filtered VV, Otsu water mask.
# ----------------------------------------------------------------------------
def s1_base(aoi):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi).filterDate("2014-01-01", END)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))


def pick_s1_geometry(aoi):
    """(pass, orbit) with the most scenes over the AOI — used when config says auto."""
    coll = s1_base(aoi)
    keys = coll.aggregate_array("orbitProperties_pass").zip(
        coll.aggregate_array("relativeOrbitNumber_start")).getInfo()
    counts = {}
    for p, o in keys:
        counts[(p, int(o))] = counts.get((p, int(o)), 0) + 1
    if not counts:
        raise SystemExit("no Sentinel-1 scenes over the AOI")
    for (p, o), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  S1 {p:10s} orbit {o:3d}: {n} scenes")
    return max(counts, key=counts.get)


def s1_collection(aoi):
    global S1_PASS, S1_ORBIT
    if S1_PASS == "auto" or S1_ORBIT == "auto":
        S1_PASS, S1_ORBIT = pick_s1_geometry(aoi)
        print(f"  -> using S1 {S1_PASS} orbit {S1_ORBIT} (most scenes); "
              "run 00_data_inventory.py to record it in sites.json")
    return (s1_base(aoi)
            .filter(ee.Filter.eq("orbitProperties_pass", S1_PASS))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", int(S1_ORBIT)))
            .select("VV"))


def sar_shorelines(coll, geom):
    """Per image: speckle-filter VV -> Otsu -> water boundary, tagged with time."""
    def extract(img):
        vv = img.select("VV").focal_median(50, "circle", "meters").clip(geom)
        thr = dyn_threshold(vv, geom, EXPORT_SCALE)
        water = vv.lt(thr).selfMask().rename("water")   # water = low backscatter
        vec = water.reduceToVectors(
            geometry=geom, scale=EXPORT_SCALE, geometryType="polygon",
            eightConnected=True, maxPixels=1e9, bestEffort=True)
        t = img.date().format("YYYY-MM-dd HH:mm:ss")
        return vec.map(lambda f: f.set({
            "datetime_utc": t, "sensor": "S1",
            "scene_id": img.get("system:index"), "otsu": thr}))
    return coll.map(extract).flatten()


# ----------------------------------------------------------------------------
# Drive exports — one FeatureCollection per sensor x site.
# ----------------------------------------------------------------------------
def export_fc(fc, name):
    # GeoJSON, NOT shapefile: SHP truncates field names to 10 chars, which would
    # turn 'datetime_utc' into 'datetime_u' and break the tide join downstream.
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=name, folder=DRIVE_FOLDER,
        fileFormat="GeoJSON")
    task.start()
    print("  started export:", name)


def periods():
    y = START_YEAR
    while y <= END_YEAR:
        y1 = min(y + CHUNK_YEARS - 1, END_YEAR)
        yield y, y1
        y = y1 + 1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", nargs="+", default=["landsat", "s2", "s1"],
                    choices=["landsat", "s2", "s1"], help="re-export only these")
    ap.add_argument("--sites", nargs="+", help="subset of site names (default: all)")
    a = ap.parse_args()

    ee.Initialize(project=EE_PROJECT)
    aoi = ee.Geometry.Rectangle(AOI_BBOX)
    ls, s2 = landsat_collection(aoi), s2_collection(aoi)
    s1 = s1_collection(aoi) if "s1" in a.sensors else None
    n, skipped = 0, 0
    for site, s in SITES.items():
        if a.sites and site not in a.sites:
            continue
        geom = ee.Geometry.Rectangle(s["box"])
        print(f"\n== {site} ({s['type']}) ==")
        for y0, y1 in periods():
            d0, d1 = f"{y0}-01-01", f"{y1 + 1}-01-01"
            tag = f"{site}_{y0}_{y1}"
            jobs = []
            if "landsat" in a.sensors:
                jobs.append(("landsat", "Landsat", keep_clear(ls.filterBounds(geom).filterDate(d0, d1), geom)))
            if "s2" in a.sensors and y1 >= 2017:
                jobs.append(("s2", "S2", keep_clear(s2.filterBounds(geom).filterDate(d0, d1), geom)))
            for key, label, coll in jobs:
                k = coll.size().getInfo()
                if k == 0:
                    print(f"  skip shoreline_{key}_{tag}: no clear scene"); skipped += 1; continue
                print(f"  {k:3d} clear scenes ->", end=" ")
                export_fc(optical_shorelines(coll, geom, label), f"shoreline_{key}_{tag}"); n += 1
            # SAR only where terrain allows: never on rock/cliff sites (layover)
            if s1 is not None and y1 >= 2014 and s["type"] != "rock":
                export_fc(sar_shorelines(s1.filterBounds(geom).filterDate(d0, d1), geom),
                          f"shoreline_s1_{tag}"); n += 1
    print(f"\n{n} tasks started, {skipped} empty periods skipped — monitor in the Tasks tab. "
          f"Files land in My Drive/{DRIVE_FOLDER}.")


if __name__ == "__main__":
    main()
