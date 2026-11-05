"""
08_backshore_change_gee.py  —  annual human-development signal behind each beach.

A single land-cover snapshot (WorldCover 2021) cannot say WHEN the backshore was
developed, so it cannot support a human-driven attribution on its own. This
script builds a consistent annual series from the same Landsat record as the
shorelines (1988-present):

  backshore = land (WorldCover) inside the site box within BAND_M of the sea
  per year  = greenest-pixel (max NDVI) composite of all cloud-masked Landsat
              scenes that year -> ndvi_mean (mean annual max NDVI),
              built_frac (annual max NDVI < 0.30, non-water)

and exports one CSV (site, year, n_images, ndvi_mean, built_frac) to Drive
(My Drive/GEE_Sichang). 09_timing_check.py then compares the year built-up
jumps with the shoreline break year.

This layer is derived here, independently of the companion erosion study
(which uses WorldCover 2021 as a static map only), so it adds no overlap.

    python 08_backshore_change_gee.py
"""
import importlib.util, json, os
import ee

HERE = os.path.dirname(os.path.abspath(__file__))
BAND_M = 150          # backshore depth inland from the water line (m)
SCALE = 30            # Landsat native


def load_04():
    spec = importlib.util.spec_from_file_location("w04", os.path.join(HERE, "04_waterline_gee.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def backshore_mask(geom):
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    water = wc.eq(80)
    near_sea = water.focal_max(BAND_M, "circle", "meters")
    return wc.neq(80).And(near_sea).clip(geom).selfMask()


def annual_stats(ls, site, geom, years, bands):
    """Per site-year: greenest-pixel composite over the WHOLE year.

    Si Chang's forest is deciduous: in a dry-season composite leaf-off forest has
    low NDVI and positive NDBI and is miscounted as built-up (first run: every site
    'lost' built-up after 1991). Taking each pixel's greenest observation of the
    year removes phenology: vegetation is green at some point, built surfaces never
    are. built = annual max NDVI < 0.30 on non-water backshore.
    """
    mask = backshore_mask(geom)
    add_ndvi = lambda i: i.addBands(i.normalizedDifference(["nir", "red"]).rename("ndvi"))
    # fully masked placeholder: a year with no scenes gives nulls, not a band error
    empty = ee.ImageCollection([ee.Image.constant([0] * (len(bands) + 1))
                                .rename(bands + ["ndvi"]).toFloat().updateMask(0)])

    def one(y):
        y = ee.Number(y)
        start = ee.Date.fromYMD(y, 1, 1)
        end = ee.Date.fromYMD(y.add(1), 1, 1)
        # cast to plain float32 so all scenes and the placeholder share one band type
        coll = (ls.filterBounds(geom).filterDate(start, end)
                .map(lambda i: add_ndvi(i.select(bands)).toFloat()))
        n = coll.size()
        g = coll.merge(empty).qualityMosaic("ndvi")
        mndwi = g.normalizedDifference(["green", "swir1"])
        built = g.select("ndvi").lt(0.30).And(mndwi.lt(0)).rename("built")
        r = g.select("ndvi").addBands(built).updateMask(mask).reduceRegion(
            ee.Reducer.mean(), geom, SCALE, maxPixels=1e9, bestEffort=True)
        return ee.Feature(None, {"site": site, "year": y, "n_images": n,
                                 "ndvi_mean": r.get("ndvi"), "built_frac": r.get("built")})

    return ee.FeatureCollection(ee.List(years).map(one))


def main():
    w = load_04()
    cfg = json.load(open(os.path.join(HERE, "sites.json")))
    ee.Initialize(project=cfg["ee_project"])
    aoi = ee.Geometry.Rectangle(cfg["aoi"])
    ls = w.landsat_collection(aoi)
    years = list(range(w.START_YEAR, w.END_YEAR + 1))
    fcs = [annual_stats(ls, name, ee.Geometry.Rectangle(s["box"]), years, w.OPT_BANDS)
           for name, s in cfg["sites"].items()]
    fc = ee.FeatureCollection(fcs).flatten()
    task = ee.batch.Export.table.toDrive(
        collection=fc, description="backshore_annual", folder=cfg.get("drive_folder", "GEE_Sichang"),
        fileFormat="CSV", selectors=["site", "year", "n_images", "ndvi_mean", "built_frac"])
    task.start()
    print(f"started export: backshore_annual.csv -> My Drive/{cfg.get('drive_folder', 'GEE_Sichang')} "
          f"({len(cfg['sites'])} sites x {len(years)} years). Put it in ./downloads/ with the shorelines.")


if __name__ == "__main__":
    main()
