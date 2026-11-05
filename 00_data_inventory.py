"""
00_data_inventory.py  —  what imagery exists over Si Chang, before exporting.

Answers two set-up decisions with numbers, prints only (no Drive export):

  1. Sentinel-1 geometry: scene count per pass x relative orbit over the AOI;
     the one with most scenes (ties -> longer time span) is written into
     sites.json ("s1": {"pass", "orbit"}) so 04 uses it for the whole series.
  2. Optical record: scenes per year and sensor (scene cloud cover < MAX_CLOUD),
     to judge where the usable record really starts (start_year decision).

    python 00_data_inventory.py            # prints tables, updates sites.json
    python 00_data_inventory.py --no-write # prints only

Writes inventory_s1_orbits.csv and inventory_optical_by_year.csv.
Needs an authenticated Earth Engine project (ee_project in sites.json).
"""
import argparse, json, os
import ee
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CLOUD = 40


def s1_inventory(aoi):
    c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
         .filter(ee.Filter.eq("instrumentMode", "IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
    rows = ee.List([c.aggregate_array("orbitProperties_pass"),
                    c.aggregate_array("relativeOrbitNumber_start"),
                    c.aggregate_array("system:time_start")]).getInfo()
    df = pd.DataFrame({"pass": rows[0], "orbit": [int(o) for o in rows[1]],
                       "t": pd.to_datetime(rows[2], unit="ms")})
    inv = (df.groupby(["pass", "orbit"])
             .agg(scenes=("t", "size"), first=("t", "min"), last=("t", "max"))
             .reset_index())
    inv["span_yr"] = ((inv["last"] - inv["first"]).dt.days / 365.25).round(1)
    inv["first"] = inv["first"].dt.date; inv["last"] = inv["last"].dt.date
    return inv.sort_values(["scenes", "span_yr"], ascending=False).reset_index(drop=True)


def optical_inventory(aoi):
    specs = [("L5", "LANDSAT/LT05/C02/T1_L2", "CLOUD_COVER"),
             ("L7", "LANDSAT/LE07/C02/T1_L2", "CLOUD_COVER"),
             ("L8", "LANDSAT/LC08/C02/T1_L2", "CLOUD_COVER"),
             ("L9", "LANDSAT/LC09/C02/T1_L2", "CLOUD_COVER"),
             ("S2", "COPERNICUS/S2_SR_HARMONIZED", "CLOUDY_PIXEL_PERCENTAGE")]
    parts = []
    for name, cid, cprop in specs:
        c = ee.ImageCollection(cid).filterBounds(aoi).filterDate("1984-01-01", "2027-01-01")
        t, cc = ee.List([c.aggregate_array("system:time_start"),
                         c.aggregate_array(cprop)]).getInfo()
        if not t:
            continue
        d = pd.DataFrame({"t": pd.to_datetime(t, unit="ms"), "cloud": cc})
        d["sensor"] = name
        parts.append(d)
    d = pd.concat(parts)
    d["year"] = d.t.dt.year
    d["usable"] = d.cloud < MAX_CLOUD
    tab = d[d.usable].pivot_table(index="year", columns="sensor", values="t",
                                  aggfunc="count", fill_value=0)
    tab["landsat_total"] = tab[[c for c in tab.columns if c.startswith("L")]].sum(axis=1)
    tab["all_total"] = tab.drop(columns="landsat_total").sum(axis=1)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cfg_path = os.path.join(HERE, "sites.json")
    cfg = json.load(open(cfg_path))
    ee.Initialize(project=cfg["ee_project"])
    aoi = ee.Geometry.Rectangle(cfg["aoi"])

    print("== Sentinel-1 IW/VV over Si Chang, by pass x relative orbit ==")
    s1 = s1_inventory(aoi)
    print(s1.to_string(index=False))
    s1.to_csv(os.path.join(HERE, "inventory_s1_orbits.csv"), index=False)
    best = s1.iloc[0]
    print(f"\n-> recommended: {best['pass']} orbit {best['orbit']} "
          f"({best['scenes']} scenes, {best['first']} to {best['last']})")

    print(f"\n== Optical scenes per year with scene cloud < {MAX_CLOUD}% ==")
    opt = optical_inventory(aoi)
    print(opt.to_string())
    opt.to_csv(os.path.join(HERE, "inventory_optical_by_year.csv"))
    thin = opt.index[opt.landsat_total < 2].tolist()
    print(f"\nyears with < 2 usable Landsat scenes: {thin or 'none'}")
    print("start_year: decide after the first full run from results/<site>/scenes_per_year.csv "
          "(per-site counts after QC); this table is scene-level only.")

    if not a.no_write:
        cfg["s1"] = {"pass": str(best["pass"]), "orbit": int(best["orbit"])}
        json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
        print(f"\nsites.json updated: s1 = {cfg['s1']}")


if __name__ == "__main__":
    main()
