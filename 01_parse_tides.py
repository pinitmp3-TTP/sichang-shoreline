"""
01_parse_tides.py  —  Ko Si Chang tide table (PDF) -> clean hourly CSV.

Parses the Hydrographic Department 2026 hourly prediction for Ko Si Chang
(heights in metres ABOVE MEAN SEA LEVEL, Thai Standard Time = UTC+7) into a
tidy table, and writes datetime in both ICT and UTC.

Usage:  python 01_parse_tides.py SC2026msl.pdf  ko_sichang_2026_hourly.csv
Tested: 8760 rows (365 x 24), 0 missing hours.
"""
import sys, re
import pandas as pd

ICT_OFFSET = pd.Timedelta(hours=7)          # Thai Standard Time -> UTC
MONTH_PAGES = {m: 10 + m for m in range(1, 13)}  # PDF page index 11..22 = Jan..Dec
ROW_RE = re.compile(r'^\s*(\d{1,2})\s+((?:-?\d+\.\d+\s*){24})$')


def parse(pdf_path: str) -> pd.DataFrame:
    import pdfplumber
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for month, pidx in MONTH_PAGES.items():
            for ln in (pdf.pages[pidx].extract_text() or "").splitlines():
                m = ROW_RE.match(ln)
                if not m:
                    continue
                day = int(m.group(1))
                vals = [float(x) for x in m.group(2).split()]
                if len(vals) != 24:
                    continue
                for hr, v in enumerate(vals):
                    rows.append((pd.Timestamp(2026, month, day, hr), v))
    df = (pd.DataFrame(rows, columns=["datetime_ict", "height_m"])
            .drop_duplicates("datetime_ict")
            .sort_values("datetime_ict").reset_index(drop=True))
    df["datetime_utc"] = df["datetime_ict"] - ICT_OFFSET
    return df[["datetime_ict", "datetime_utc", "height_m"]]


def validate(df: pd.DataFrame) -> None:
    exp = pd.date_range("2026-01-01", "2026-12-31 23:00", freq="h")
    missing = set(exp) - set(df.datetime_ict)
    assert not missing, f"{len(missing)} missing hours, e.g. {sorted(missing)[:3]}"
    assert len(df) == 8760, f"expected 8760 rows, got {len(df)}"
    print(f"OK  {len(df)} rows | height {df.height_m.min()} to {df.height_m.max()} m "
          f"(MSL) | mean {df.height_m.mean():.3f}")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "SC2026msl.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "ko_sichang_2026_hourly.csv"
    df = parse(pdf_path)
    validate(df)
    df.to_csv(out, index=False)
    print("wrote", out)
