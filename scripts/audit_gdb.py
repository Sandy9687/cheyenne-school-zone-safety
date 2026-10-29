"""
audit_gdb.py - Inventory and quality audit of the Cheyenne School Zone Safety geodatabase.

Reads the file geodatabase (read-only) and writes:
  inventory.csv  - one row per layer/table (records, geometry, CRS, extent, empty geometries)
  fields.csv     - one row per field (type, nulls, unique values, example, duplicates for ID-like fields)
  report.txt     - readable summary, including date ranges and suspicious coordinates

Needs: pip install geopandas pyogrio pandas
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

# ---- PATHS ------------------------------------------------------------------
# GDB_PATH comes from config_local.py (repo root, ignored by Git) so each computer
# can point to its own copy of the geodatabase. This script only READS the .gdb.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from config_local import GDB_PATH
except ImportError:
    raise SystemExit(
        "Missing config_local.py in the repo root. "
        "Copy config_local.example.py to config_local.py and set GDB_PATH."
    )
GDB_PATH = Path(GDB_PATH)
OUT_DIR = ROOT / "outputs" / "audit"
# -----------------------------------------------------------------------------

ID_HINTS = ("id", "case", "number", "num", "key")
DATE_HINTS = ("date", "time", "year", "month")


def is_id_like(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in ID_HINTS)


def is_date_like(col: str, series: pd.Series) -> bool:
    c = col.lower()
    return pd.api.types.is_datetime64_any_dtype(series) or any(h in c for h in DATE_HINTS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not GDB_PATH.exists():
        raise SystemExit(f"Cannot find the geodatabase at: {GDB_PATH}")

    layers = pyogrio.list_layers(GDB_PATH)  # rows of [name, geometry type or None]
    inventory, fields, notes = [], [], []

    for name, geom_type in layers:
        df = pyogrio.read_dataframe(GDB_PATH, layer=name)
        spatial = isinstance(df, gpd.GeoDataFrame)
        data_cols = [c for c in df.columns if not (spatial and c == df.geometry.name)]

        row = {
            "layer": name,
            "geometry_type": geom_type if geom_type else "table (no geometry)",
            "records": len(df),
            "n_fields": len(data_cols),
            "crs": df.crs.to_string() if spatial and df.crs else "n/a",
            "null_geometries": int(df.geometry.isna().sum()) if spatial else "n/a",
            "empty_geometries": int(df.geometry.is_empty.sum()) if spatial else "n/a",
            "extent_minx_miny_maxx_maxy": (
                [round(v, 4) for v in df.total_bounds] if spatial and len(df) else "n/a"
            ),
        }
        inventory.append(row)

        # Point layers: look for (0,0) or missing coordinates
        if spatial and geom_type and "Point" in str(geom_type) and len(df):
            pts = df.geometry.dropna()
            pts = pts[~pts.is_empty]
            zero = int(((pts.x == 0) | (pts.y == 0)).sum())
            if zero:
                notes.append(f"[{name}] {zero} points have a zero X or Y coordinate (likely bad geocodes).")

        for col in data_cols:
            s = df[col]
            nulls = int(s.isna().sum())
            entry = {
                "layer": name,
                "field": col,
                "dtype": str(s.dtype),
                "nulls": nulls,
                "pct_null": round(100 * nulls / len(df), 1) if len(df) else 0,
                "unique_values": int(s.nunique(dropna=True)),
                "example": s.dropna().iloc[0] if nulls < len(df) else None,
                "duplicates_if_id_like": int(s.dropna().duplicated().sum()) if is_id_like(col) else "",
            }
            fields.append(entry)

            if is_date_like(col, s) and nulls < len(df):
                try:
                    d = pd.to_datetime(s, errors="coerce")
                    if d.notna().any():
                        notes.append(f"[{name}.{col}] dates run {d.min()} to {d.max()} ({int(d.isna().sum())} unparseable/null).")
                except Exception:
                    pass

    inv_df = pd.DataFrame(inventory)
    fld_df = pd.DataFrame(fields)
    inv_df.to_csv(OUT_DIR / "inventory.csv", index=False)
    fld_df.to_csv(OUT_DIR / "fields.csv", index=False)

    with open(OUT_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Geodatabase: {GDB_PATH}\n\n=== INVENTORY ===\n")
        f.write(inv_df.to_string(index=False))
        f.write("\n\n=== NOTES (dates, coordinates) ===\n")
        f.write("\n".join(notes) if notes else "None flagged.")
        f.write("\n\n=== FIELDS WITH >20% NULLS ===\n")
        f.write(fld_df[fld_df["pct_null"] > 20].to_string(index=False) or "None.")
        f.write("\n")

    print(inv_df.to_string(index=False))
    print("\n".join(notes))
    print(f"\nDone. Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
