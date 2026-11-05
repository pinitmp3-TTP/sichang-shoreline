"""
11_vessels_s1_gee.py  —  vessel / barge presence around each site from Sentinel-1.

Why: Google Earth shows transshipment barges moored 200–400 m off Tha Yai Thim and
Hin Muat Sila on most dates. This script measures how often and how much of the sea
around each site is occupied by vessels, 2015–2026, so vessel pressure can be
compared with shoreline change (12_vessel_analysis.py).

Method (per Sentinel-1 scene, same geometry as the shoreline record):
  * sea mask = ESA WorldCover 2021 water, minus a 30 m strip next to land
    (removes the shore, piers and vessels berthed at piers)
  * bright-target detection, CFAR-style: VV > sea median + DVV dB AND
    VH > sea median + DVH dB (both scene-adaptive), and VV > -8 dB absolute;
    targets must cover >= MIN_PIX connected 10 m pixels
  * per zone: vessel area (m2), number of targets, sea area, sea medians
Zones: each site box buffered by ZONE_BUFFER_M, plus the whole AOI ("island").

Output: one CSV in My Drive/<drive_folder>:  vessels_s1.csv
    python 11_vessels_s1_gee.py            (starts one export task)
    python 11_vessels_s1_gee.py --preview  (prints 3 scenes, no export)
"""
import ee, json, os, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "sites.json")))
EE_PROJECT, DRIVE_FOLDER = CFG["ee_project"], CFG.get("drive_folder", "GEE_Sichang")
S1_PASS, S1_ORBIT = CFG["s1"]["pass"], int(CFG["s1"]["orbit"])
START, END = "2015-01-01", "2026-12-31"
ZONE_BUFFER_M = 500      # sea within 500 m of each site box
SHORE_EXCL_M = 30        # ignore the 30 m next to land (piers, berthed vessels)
DVV, DVH = 12, 10        # dB above the scene's sea median
VV_ABS = -8              # absolute VV floor (dB)
MIN_PIX = 2              # connected 10 m pixels per target (>= 200 m2)
SCALE = 10


def zones():
    feats = [ee.Feature(ee.Geometry.Rectangle(s["box"]).buffer(ZONE_BUFFER_M), {"zone": k})
             for k, s in CFG["sites"].items()]
    feats.append(ee.Feature(ee.Geometry.Rectangle(CFG["aoi"]), {"zone": "island"}))
    return ee.FeatureCollection(feats)


def sea_mask(aoi):
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi.buffer(2000))
    land = wc.neq(80).unmask(0)   # open sea may be no-data in WorldCover: treat as water
    near_land = land.focal_max(SHORE_EXCL_M, "circle", "meters")
    return near_land.Not().rename("sea")


def s1(aoi):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(START, END)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("orbitProperties_pass", S1_PASS))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", S1_ORBIT))
            .select(["VV", "VH"]))


def per_scene(aoi, sea, zfc):
    def f(img):
        sea_img = img.updateMask(sea)
        med = sea_img.reduceRegion(ee.Reducer.median(), aoi, 20, bestEffort=True, maxPixels=1e9)
        vv_med, vh_med = ee.Number(med.get("VV")), ee.Number(med.get("VH"))
        thr_vv = vv_med.add(DVV).max(VV_ABS)
        det = (img.select("VV").gt(thr_vv)
               .And(img.select("VH").gt(vh_med.add(DVH)))
               .And(sea))
        det = det.updateMask(det.connectedPixelCount(64, True).gte(MIN_PIX)).selfMask()
        labels = det.connectedComponents(ee.Kernel.plus(1), 256).select("labels")
        stack = (ee.Image.pixelArea().updateMask(det).rename("vessel_m2")
                 .addBands(labels.rename("label"))
                 .addBands(ee.Image.pixelArea().updateMask(sea).rename("sea_m2")))
        red = (ee.Reducer.sum().combine(ee.Reducer.countDistinct(), sharedInputs=True))
        out = stack.reduceRegions(collection=zfc, reducer=red, scale=SCALE)
        t = img.date().format("YYYY-MM-dd HH:mm:ss")
        return out.map(lambda z: ee.Feature(None, {
            "datetime_utc": t, "zone": z.get("zone"),
            "vessel_m2": z.get("vessel_m2_sum"), "n_targets": z.get("label_count"),
            "sea_m2": z.get("sea_m2_sum"), "vv_sea_med": vv_med, "vh_sea_med": vh_med,
            "scene_id": img.get("system:index")}))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="print 3 scenes instead of exporting")
    a = ap.parse_args()
    ee.Initialize(project=EE_PROJECT)
    aoi = ee.Geometry.Rectangle(CFG["aoi"])
    coll, sea, zfc = s1(aoi), sea_mask(aoi), zones()
    print(f"S1 {S1_PASS} orbit {S1_ORBIT}: {coll.size().getInfo()} scenes {START}..{END}")
    fc = coll.map(per_scene(aoi, sea, zfc)).flatten()
    if a.preview:
        rows = ee.FeatureCollection(coll.limit(3).map(per_scene(aoi, sea, zfc)).flatten()).getInfo()["features"]
        for r in rows:
            p = r["properties"]
            print(f'{p["datetime_utc"]} {p["zone"]:16s} vessel {p["vessel_m2"] or 0:8.0f} m2  targets {p["n_targets"] or 0:3}  '
                  f'sea {p["sea_m2"] or 0:9.0f} m2  VV med {p["vv_sea_med"]:.1f} dB')
        return
    task = ee.batch.Export.table.toDrive(collection=fc, description="vessels_s1", folder=DRIVE_FOLDER,
                                         fileFormat="CSV",
                                         selectors=["datetime_utc", "zone", "vessel_m2", "n_targets", "sea_m2",
                                                    "vv_sea_med", "vh_sea_med", "scene_id"])
    task.start()
    print(f"started export: vessels_s1 -> My Drive/{DRIVE_FOLDER}/vessels_s1.csv")


if __name__ == "__main__":
    main()
