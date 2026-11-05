"""
10_estimate_slope.py  —  beach slope from the shoreline record itself.

For each transect at tide-corrected sites, the raw (uncorrected) waterline
position from optical scenes is regressed on tide level with a linear time term
and a Sentinel-2 offset:

    raw = a + b*t + c*z_tide (+ d*S2)        tan(beta) = -1 / c

(a higher tide moves the waterline landward, so c < 0). Transects with a
significant tide response (c < 0, t-stat < -3) give tan(beta); the site value is
their median, with a bootstrap 95% CI over transects. A site where fewer than
half of the transects (or fewer than three) respond gets slope = null (no tide
correction), as at a cliff.
Known abrupt steps (break_year in sites.json) are absorbed by a step term.

    python 10_estimate_slope.py results            # print table
    python 10_estimate_slope.py results --write    # also write slopes to sites.json

Run AFTER a first run_pipeline.py, then run run_pipeline.py again.
On Si Chang (3 Nov 2026): tan(beta) 0.09-0.28 (about 1:4 to 1:11), steeper than
the 0.06 placeholder; regression CIs narrowed at every corrected site.
"""
import argparse, importlib.util, json, os, pickle
import numpy as np, pandas as pd, utide

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-t", type=float, default=3.0)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("t05", os.path.join(HERE, "05_transect_rates.py"))
    t05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t05)
    cfg_path = os.path.join(HERE, "sites.json"); cfg = json.load(open(cfg_path, encoding="utf-8"))
    coef = pickle.load(open(os.path.join(HERE, "sichang_tide_coef.pkl"), "rb"))
    rng = np.random.default_rng(0)
    out = []
    for site, s in cfg["sites"].items():
        if s["type"] == "rock":
            continue
        sc = json.load(open(os.path.join(a.results, site, "site_config.json")))
        sl = t05.load_shorelines(os.path.join(a.results, site, "shorelines.geojson"), sc["epsg"])
        tr = t05.build_transects(os.path.join(a.results, site, "baseline.geojson"),
                                 sc["spacing_m"], sc["length_m"], sc["sign"], sc["epsg"])
        df = t05.measure(tr, sl)
        df = df[df.sensor.isin(["Landsat", "S2"])].copy()
        df["z"] = utide.reconstruct(pd.to_datetime(df.t).to_numpy(), coef, verbose=False).h
        df["yr"] = pd.to_datetime(df.t).map(t05.decimal_year)
        by = s.get("break_year")
        n_tr = df.transect_id.nunique()
        resp = []
        for tid, g in df.groupby("transect_id"):
            if len(g) < 50:
                continue
            cols = [np.ones(len(g)), g.yr - 2007, g.z, (g.sensor == "S2").astype(float)]
            if isinstance(by, int):
                cols.append((g.yr >= by).astype(float))
            X = np.column_stack(cols)
            beta, *_ = np.linalg.lstsq(X, g.raw_dist.to_numpy(), rcond=None)
            r = g.raw_dist.to_numpy() - X @ beta
            se = np.sqrt(r.var(ddof=X.shape[1]) * np.linalg.inv(X.T @ X)[2, 2])
            if beta[2] < 0 and beta[2] / se < -a.min_t:
                resp.append(-1 / beta[2])
        if len(resp) >= max(3, n_tr / 2):          # most of the site must respond
            boot = [np.median(rng.choice(resp, len(resp))) for _ in range(2000)]
            val = round(float(np.median(resp)), 3)
            ci = f"{np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}"
        else:
            val, ci = None, None
        out.append({"site": site, "transects": n_tr, "responding": len(resp), "tan_beta": val, "ci95": ci,
                    "slope_1_in": round(1 / val) if val else None, "previous": s.get("slope")})
        if a.write:
            cfg["sites"][site]["slope"] = val
    t = pd.DataFrame(out); print(t.to_string(index=False))
    t.to_csv(os.path.join(a.results, "beach_slopes.csv"), index=False)
    if a.write:
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("\nsites.json updated (slope = null -> no tide correction). Now rerun run_pipeline.py.")


if __name__ == "__main__":
    main()
