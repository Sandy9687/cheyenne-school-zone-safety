"""
prepare_cheyenne.py - Build a clean working GeoPackage for the Cheyenne analysis.

What it does (the original .gdb is only READ, never changed):
  1. Reprojects every spatial layer to NAD83 / Wyoming East (ftUS), EPSG:3736.
  2. Drops empty geometries and points with a zero X or Y coordinate.
  3. Flattens 3D (Z) geometries to 2D.
  4. Keeps only features within BUFFER_FT of the Cheyenne boundary (so roads that
     feed into the city and schools near the edge are not cut off), and adds an
     in_city column (True = inside the boundary itself).
  5. Copies the non-spatial tables (crash / involved / vehicle) as they are.
  6. Writes data/cheyenne_working.gpkg plus summary files in outputs/prep/.

Run:  python scripts\\prepare_cheyenne.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

# ---- SETTINGS ---------------------------------------------------------------
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

TARGET_CRS = "EPSG:3736"      # NAD83 / Wyoming East (ftUS)
BUFFER_FT = 5280              # keep features within 1 mile of the city boundary
OUT_GPKG = ROOT / "data" / "cheyenne_working.gpkg"
OUT_DIR = ROOT / "outputs" / "prep"

BOUNDARY_LAYER = "Cheyenne_Boundary"
SPATIAL_LAYERS = [
    "Cheyenne_Crash_Locations_2010_2019",
    "CrashPoints",
    "Cheyenne_Schools",
    "Schools_All",
    "Crosswalks",
    "Sidewalks",
    "Pedestrian_Signals",
    "AADT",
    "SpeedLimits",
    "FunctionalClassifications",
]
TABLE_LAYERS = [
    "Cheyenne_Crashes_2010_2019",
    "Cheyenne_Involved_2010_2019",
    "Cheyenne_Vehicle_2010_2019",
]
# -----------------------------------------------------------------------------


def to_2d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove Z values so clipping and distance work the same for every layer."""
    geom = gpd.GeoSeries(shapely.force_2d(gdf.geometry.values), index=gdf.index, crs=gdf.crs)
    return gdf.set_geometry(geom)


def write_layer(df, name: str) -> bool:
    try:
        pyogrio.write_dataframe(df, OUT_GPKG, layer=name, driver="GPKG")
        return True
    except Exception as exc:  # keep going, report the problem
        print(f"   !! could not write {name}: {exc}")
        return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if not GDB_PATH.exists():
        raise SystemExit(f"Cannot find the geodatabase at: {GDB_PATH}")
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    # ---- Study area ---------------------------------------------------------
    print("Reading Cheyenne boundary...")
    b = pyogrio.read_dataframe(GDB_PATH, layer=BOUNDARY_LAYER)
    b = to_2d(b).to_crs(TARGET_CRS)
    b["geometry"] = b.geometry.make_valid()
    b["area_sq_mi"] = (b.geometry.area / 27_878_400).round(3)  # sq ft -> sq mi
    b.drop(columns="geometry").to_csv(OUT_DIR / "boundary_features.csv", index=False)
    print(b.drop(columns="geometry").to_string(index=False))

    city = shapely.union_all(b.geometry.values)
    area = shapely.buffer(city, BUFFER_FT)
    write_layer(gpd.GeoDataFrame({"name": ["Cheyenne boundary (all features dissolved)"]},
                                 geometry=[city], crs=TARGET_CRS), "study_area")
    write_layer(gpd.GeoDataFrame({"name": [f"Study area + {BUFFER_FT} ft buffer"]},
                                 geometry=[area], crs=TARGET_CRS), "study_area_buffer")
    write_layer(b, "boundary_features")

    summary = []

    # ---- Spatial layers -----------------------------------------------------
    for name in SPATIAL_LAYERS:
        print(f"Processing {name}...")
        gdf = pyogrio.read_dataframe(GDB_PATH, layer=name)
        n_raw = len(gdf)
        crs_in = gdf.crs.name if gdf.crs else "none"

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf = to_2d(gdf)
        if (gdf.geom_type == "Point").all() and len(gdf):
            gdf = gdf[(gdf.geometry.x != 0) & (gdf.geometry.y != 0)].copy()
        n_clean = len(gdf)

        gdf = gdf.to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(area)].copy()
        gdf["in_city"] = gdf.geometry.intersects(city)

        write_layer(gdf, name)
        summary.append({
            "layer": name,
            "rows_in_gdb": n_raw,
            "rows_after_cleaning": n_clean,
            "dropped_empty_or_zero": n_raw - n_clean,
            "rows_kept_in_study_area": len(gdf),
            "rows_inside_city": int(gdf["in_city"].sum()),
            "original_crs": crs_in,
        })

        if name == "CrashPoints" and "date_year" in gdf.columns:
            (gdf[gdf["in_city"]]["date_year"].value_counts().sort_index()
             .rename("crashes_inside_city").to_csv(OUT_DIR / "crashpoints_by_year.csv"))
        if name == "Cheyenne_Schools":
            gdf.drop(columns="geometry").to_csv(OUT_DIR / "cheyenne_schools.csv", index=False)

    # ---- Tables (no geometry) ----------------------------------------------
    for name in TABLE_LAYERS:
        print(f"Copying table {name}...")
        df = pyogrio.read_dataframe(GDB_PATH, layer=name)
        ok = write_layer(df, name)
        summary.append({
            "layer": name, "rows_in_gdb": len(df), "rows_after_cleaning": len(df),
            "dropped_empty_or_zero": 0,
            "rows_kept_in_study_area": len(df) if ok else "write failed",
            "rows_inside_city": "table", "original_crs": "n/a",
        })

    s = pd.DataFrame(summary)
    s.to_csv(OUT_DIR / "prep_summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(s.to_string(index=False))
    print(f"\nDone. Working data: {OUT_GPKG}\nSummaries: {OUT_DIR}")


if __name__ == "__main__":
    main()
