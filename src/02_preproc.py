"""
-------------------------------------------------------------------------------
Imagery Preprocessing & Tiling Pipeline
-------------------------------------------------------------------------------

This module transforms raw sensor data (Sentinel-2, Landsat, Planet, HLS)
into analysis-ready, normalized 256x256 chips for deep learning.

It harmonizes divergent directory structures and file formats into a
unified tiling grid, ensuring spatial consistency across years and
sensors.

Supported Imagery:
    - Sentinel-2: Annual median composites (10 bands + NDVI + MNDWI).
    - PlanetScope: High-res mosaics (4 bands + NDVI + Proxy MNDWI).
    - Landsat 5/7/8: Annual composites (Scaled spectral + Indices).
    - HLS-L30: Harmonized Landsat Sentinel data.

Key Operations:
    - **Grid Alignment**: Builds a project-wide shapefile grid (0.5° cells)
    - **Normalization**: Computes global P2/P98 stats (Prerun) and *then*
      normalizes all cells to [-1, 1] consistently.

Usage:
    #1. Generate grid and stats (Prerun)
    python 02_preproc.py --sensor s2 --project avocados \
        --year 2019 --prerun

    #2. Run tiling
    python 02_preproc.py --sensor s2 --project avocados \
        --year 2019
"""

import argparse
import concurrent.futures as cf
import csv
import os
import time
import traceback
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as rio_transform
from shapely.geometry import box
from tqdm.auto import tqdm

from mlgis_helpers.cfg import load_config
from mlgis_helpers.preproc_planet_helpers import _collect_planet_files

# Configuration
# -------------

NODATA = -9999.0
P_LOW, P_HIGH = 2.0, 98.0
WGS84 = 'EPSG:4326'
CHUNK_DEG = 0.5  # Grid cell size in degrees (0.5 deg = ~55km at equator)
SAMPLE_FILES = 3 # Use up to 3 files per sensor-year to estimate pctiles
SAMPLE_PIXELS = 13000  # ~39,000 total across 3 files for 1% MOE @95%
N_WORKERS = 16  # Default number of parallel workers

RAW_SENSOR_SPECS = {
    's2': {
        'files': {
            # Source matches Step 1 output
            'source': '*ANNUAL_MEDIAN*.tif'
        },
        'bands': OrderedDict([
            ('B02', 1), ('B03', 2), ('B04', 3), ('B05', 4), ('B06', 5),
            ('B07', 6), ('B08', 7), ('B8A', 8), ('B11', 9), ('B12', 10)
        ]),
        'indices': OrderedDict([
            ('NDVI',  ('B08', 'B04')),  # (NIR - Red) / (NIR + Red)
            ('MNDWI', ('B03', 'B11'))  # (Green - SWIR) / (Green + SWIR)
        ])
    },
    'planet': {
        'files': {'source': '*.tif'},
        'bands': OrderedDict([('B1', 1), ('B2', 2), ('B3', 3), ('B4', 4)]),
        'indices': OrderedDict([
            ('NDVI',       ('B4', 'B3')),
            ('ProxyMNDWI', ('B2', 'B4')),
        ])
    },
    'l5': {
        'files': {'source': '*_ANNUAL_*.tif'},
        'bands': OrderedDict([
            ('B1', 1), ('B2', 2), ('B3', 3), ('B4', 4),
            ('B5', 5), ('B7', 6)
        ]),
        'indices': OrderedDict([
            ('NDVI',  ('B4', 'B3')),
            ('MNDWI', ('B2', 'B5'))
        ])
    },
    'l7': {
        'files': {'source': '*_ANNUAL_*.tif'},
        'bands': OrderedDict([
            ('B1', 1), ('B2', 2), ('B3', 3), ('B4', 4), ('B5', 5), ('B7', 7)
        ]),
        'indices': OrderedDict([
            ('NDVI',  ('B4', 'B3')),
            ('MNDWI', ('B2', 'B5'))
        ])
    },
    'l8': {
        'files': {'source': '*_ANNUAL_*.tif'},
        'bands': OrderedDict([
            ('B2', 2), ('B3', 3), ('B4', 4), ('B5', 5), ('B6', 6), ('B7', 7)
        ]),
        'indices': OrderedDict([
            ('NDVI',  ('B5', 'B4')),
            ('MNDWI', ('B3', 'B6'))
        ])
    },
    'hls': {
        'files': {'source': '*_ANNUAL_*.tif'},
        'bands': OrderedDict([
            ('B1', 1), ('B2', 2), ('B3', 3), ('B4', 4), ('B5', 5), ('B7', 7)
        ]),
        'indices': OrderedDict([
            ('NDVI',  ('B4', 'B3')),
            ('MNDWI', ('B2', 'B5'))
        ])
    },
}

SENSOR_CODE = {
    's2': 'S2',
    'planet': 'PL',
    'l5': 'L5',
    'l7': 'L7',
    'l8': 'L8',
    'hls': 'HLS-L',
}

# Basic Helpers
# -------------
def _secs(t0: float) -> str:
    """Helper to format elapsed time."""
    return f'{time.time() - t0:.1f}s'

def _build_sensor_specs(raw_specs) -> Dict[str, Dict]:
    """Compiles raw configuration into structured sensor specs."""
    specs: Dict[str, Dict] = {}
    for key, cfg in raw_specs.items():
        files = cfg['files']
        file_glob = files['source']

        # Handle Indices: Support both Dict (New) and List (Legacy)
        raw_indices = cfg.get('indices', {})
        final_indices = tuple(raw_indices.values())

        specs[key] = {
            'file_glob': file_glob,
            'band_map': cfg['bands'],
            'indices': final_indices
        }
    return specs

# Grid Generation Helpers
# ------------------------
def _compute_global_bounds(sources: List[Path]) -> Tuple[float, float,
                                                         float, float]:
    """
    Computes the union bounds (min_x, min_y, max_x, max_y) of all sources
    in WGS84 (EPSG:4326).
    """
    min_x, min_y, max_x, max_y = None, None, None, None
    for fp in sources:
        with rasterio.open(fp) as src:
            bounds = src.bounds
            crs = src.crs or WGS84
            xs, ys = rio_transform(
                crs,
                WGS84,
                [bounds.left, bounds.left, bounds.right, bounds.right],
                [bounds.bottom, bounds.top, bounds.bottom, bounds.top],
            )
            local_min_x, local_max_x = min(xs), max(xs)
            local_min_y, local_max_y = min(ys), max(ys)
            min_x = local_min_x if min_x is None else min(min_x, local_min_x)
            max_x = local_max_x if max_x is None else max(max_x, local_max_x)
            min_y = local_min_y if min_y is None else min(min_y, local_min_y)
            max_y = local_max_y if max_y is None else max(max_y, local_max_y)
    if min_x is None or min_y is None or max_x is None or max_y is None:
        raise ValueError('Failed to derive bounds from sources')
    return float(min_x), float(min_y), float(max_x), float(max_y)


def _build_grid_shapefile(grid_path: Path,
                           bounds: Tuple[float, float, float, float]) -> None:
    """
    Generate grid shapefile if needed. Uses file locking to prevent parallel
    jobs from building the shapefile if they start simultaneously.
    """

    grid_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = grid_path.with_suffix('.lock')
    lock_acquired = False
    lock_start = time.time()

    # Try to create/acquire lock file (wait up to 120 seconds).
    # O_CREAT creates if it doesn't exist, O_EXCL fails if exists,
    # O_WRONLY opens for write only
    # This makes jobs take turns instead of all writing at once
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
            break
        except FileExistsError:
            if time.time() - lock_start > 120:
                raise TimeoutError(f'Timed out waiting for lock: {lock_path}')
            time.sleep(0.5)

    try:
        # After getting a lock, check if another job already wrote the grid
        # Expected behavior: job 1 writes grid, jobs 2-N see it and skip write
        if grid_path.exists():
            existing = gpd.read_file(grid_path)
            print(f'SAFE MATCH: Reusing {grid_path} ({len(existing)} cells)')
            return

        # Grid Generation
        # ---------------
        min_x, min_y, max_x, max_y = bounds
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f'Invalid bounds for grid generation: {bounds}')
        x_chunks = max(1, int(np.ceil((max_x - min_x) / CHUNK_DEG)))
        y_chunks = max(1, int(np.ceil((max_y - min_y) / CHUNK_DEG)))
        geometries = []
        records = []
        for xi in range(x_chunks):
            for yi in range(y_chunks):
                chunk_min_x = min_x + xi * CHUNK_DEG
                chunk_max_x = min(max_x, chunk_min_x + CHUNK_DEG)
                chunk_min_y = min_y + yi * CHUNK_DEG
                chunk_max_y = min(max_y, chunk_min_y + CHUNK_DEG)

                if chunk_min_x >= chunk_max_x or chunk_min_y >= chunk_max_y:
                    continue

                geometries.append(
                    box(chunk_min_x, chunk_min_y, chunk_max_x, chunk_max_y)
                )
                records.append({
                    'x_idx': xi,
                    'y_idx': yi,
                    'chunk_id': f'{xi}_{yi}',
                })
        if not geometries:
            raise ValueError('No geometries generated for grid shapefile')
        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=WGS84)
        gdf.to_file(grid_path, driver='ESRI Shapefile')
        print(f'Generated grid shapefile with {len(gdf)} cells at {grid_path}')

    finally:
        # Always delete lock file, even if write failed or chunk got skipped
        if lock_acquired:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


# Fetch Data & Compute Stats
# --------------------------
def discover_sources(spec_key: str, raw_dir: Path, year: int) -> List[Path]:
    """
    Locates sensor data, respecting year-specific path structures.
    """
    sensor_spec = SENSOR_SPECS[spec_key]
    file_glob = sensor_spec['file_glob']

    if spec_key == 'planet':
        # Planet: custom /16/mun/sereno_YY structure
        sources = _collect_planet_files(raw_dir, year)
    else:
        # Standard sensors: {year}-{SENSOR} directories
        suffix = SENSOR_CODE.get(spec_key, spec_key.upper())
        search_dir = raw_dir / f'{year}-{suffix}'

        if not search_dir.is_dir():
            raise FileNotFoundError(f'No dir for {spec_key}: {search_dir}')

        matches = sorted(search_dir.glob(file_glob))
        sources = [p for p in matches if not p.name.startswith('._')]

    if not sources:
        raise FileNotFoundError(f'No {spec_key} files found for {year}')

    print(f'Found {len(sources)} {spec_key} files. Example: {sources[0].name}')
    return sources


def compute_percentile_limits(
    spec_key: str,
    files: List[Path],
    *,
    nodata_fallback: float, # Used if file metadata lacks nodata value
) -> Dict[str, Tuple[float, float]]:
    """
    Computes global P2 and P98 percentiles for normalization.
    Samples pixels deterministically from the largest input files available.
    """
    sensor_spec = SENSOR_SPECS[spec_key]
    band_map = sensor_spec['band_map']

    # Select largest files for sampling
    if len(files) <= SAMPLE_FILES:
        sampled_paths = files
    else:
        file_sizes = [(f, f.stat().st_size) for f in files]
        file_sizes.sort(key=lambda x: x[1], reverse=True) # Sort by second item
        sampled_paths = [f for f, _ in file_sizes[:SAMPLE_FILES]]

    # Buckets to fill with sampled values
    buckets = {b: [] for b in band_map}

    for fp in sampled_paths:
        with rasterio.open(fp) as src:
            h, w = src.height, src.width
            if h == 0 or w == 0:
                continue

            # Deterministic grid sampling
            total_pixels = h * w
            if total_pixels <= SAMPLE_PIXELS:
                stride = 1
            else:
                stride = int(np.sqrt(total_pixels / SAMPLE_PIXELS))
                stride = max(stride, 10)  # Sample at least every 10th pixel
            print(f'  Sampling {fp.name} (stride={stride})')

            nod = src.nodata if src.nodata is not None else nodata_fallback

            # Subset all bands at once with stride
            all_data = src.read()[:, ::stride, ::stride]

            for bname, bidx in band_map.items():
                if bidx > src.count:
                    continue
                arr = all_data[bidx - 1].ravel() # ravel() flattens to 1D
                valid = np.isfinite(arr) & (arr != nod)
                if spec_key == 'planet':
                    valid = valid & (arr >= 0.0) & (arr <= 1.0)
                vals = arr[valid]
                if vals.size:
                    buckets[bname].append(vals.astype(np.float32, copy=False))

    limits: Dict[str, Tuple[float, float]] = {}
    for bname, arrays in buckets.items():
        if arrays:
            v = np.concatenate(arrays)
            limits[bname] = (
                float(np.percentile(v, P_LOW)),
                float(np.percentile(v, P_HIGH)),
            )
        else:
            limits[bname] = (0.0, 1.0)

    return limits


def save_stats_csv(path: Path, limits: Dict[str, Tuple[float, float]]) -> None:
    """Saves normalization statistics to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['band', 'p2', 'p98'])
        for band in sorted(limits.keys()): # Sort for deterministic output
            p2, p98 = limits[band]
            writer.writerow([band, p2, p98])


def load_stats_csv(
    path: Path, spec_key: str
) -> Dict[str, Tuple[float, float]]:
    """
    Loads pre-computed P2/P98 normalization stats.
    """
    band_names = list(SENSOR_SPECS[spec_key]['band_map'].keys())

    limits = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            band = row['band']
            if band in band_names:
                limits[band] = (float(row['p2']), float(row['p98']))

    missing = [b for b in band_names if b not in limits]
    if missing:
        raise ValueError(f'Stats CSV missing bands for {spec_key}: {missing}')

    return limits


# Helpers for Chunk (Grid Cell) Processing
# ------------------------------------------------------
def _rescale(
    mosaic: np.ndarray,
    limits: Tuple[float, float],
    nodata: float
) -> np.ndarray:
    """
    Rescales a band to the [-1, 1] range based on P2/P98 limits,
    clipping values outside the P2-P98 range.
    """
    p2, p98 = limits
    span = p98 - p2
    out = np.full_like(mosaic, nodata, dtype=np.float32)

    # Avoid division by zero if a band is constant
    if span <= 1e-6:
        return out

    valid = np.isfinite(mosaic) & (mosaic != nodata)
    if np.any(valid):
        clipped = np.clip(mosaic, p2, p98)
        # Scale to [-1, 1]
        out[valid] = 2.0 * (clipped[valid] - p2) / span - 1.0

    return out


def _nd(a: Optional[np.ndarray],
        b: Optional[np.ndarray], nodata: float) -> Optional[np.ndarray]:
    """Computes Normalized Difference: (a - b) / (a + b)."""
    if a is None or b is None:
        return None
    out = np.full(a.shape, nodata, dtype=np.float32)
    den = a + b
    # Binary mask indicating where valid computation can occur
    mask = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a != nodata)
        & (b != nodata)
        & (np.abs(den) > 1e-6)
        )
    if np.any(mask):
        out[mask] = np.clip((a[mask] - b[mask]) / den[mask], -1.0, 1.0)
    return out


def _iter_grid_bounds(grid_path: Path) -> Iterable[Tuple[str, Tuple[float, float, float, float]]]:  # noqa: E501
    gdf = gpd.read_file(grid_path)
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        cid = str(row.get('chunk_id', idx))
        yield cid, geom.bounds


def _set_chunk_shape(reference_fp: Path) -> Tuple[int, int]:
    """
    Computes the target pixel shape (H, W) for a chunk based on CHUNK_DEG.
    """
    with rasterio.open(reference_fp) as src:
        h, w = src.height, src.width

        # Sample the center to minimize distortion effects at edges
        r, c = max(0, h // 2), max(0, w // 2)

        # Get coordinates for three points to estimate pixel size
        # (r, c), (r, c+1), (r+1, c)
        x0, y0 = src.xy(r, c)
        x1, y1 = src.xy(r, min(c + 1, max(0, w - 1)))
        x2, y2 = src.xy(min(r + 1, max(0, h - 1)), c)

        # Transform to WGS84 to get degree size
        lons, lats = rio_transform(src.crs, WGS84, [x0, x1, x2], [y0, y1, y2])

        # Calculate resolution (delta degrees per pixel)
        dlon = max(1e-9, abs(lons[1] - lons[0]))
        dlat = max(1e-9, abs(lats[2] - lats[0]))

        # Calculate target dimensions
        width = max(1, int(round(CHUNK_DEG / dlon)))
        height = max(1, int(round(CHUNK_DEG / dlat)))

        return height, width

def _get_cached_dataset(fp: Path) -> Optional[rasterio.DatasetReader]:
    """
    Retrieves a dataset from the worker cache or opens it.
    Updates the cache and bounds registry upon successful open.
    (Worker caches are used when we parallelize with ProcessPoolExecutor.)
    """
    fp_str = str(fp)
    # Check cache first, then open if not found
    ds = _worker_cache.get(fp_str)
    if ds is not None:
        return ds
    try:
        ds = rasterio.open(fp)
    except Exception:
        return None
    # Update cache and bounds
    _worker_cache[fp_str] = ds
    try:
        _worker_bounds[fp_str] = ds.bounds
    except Exception:
        pass
    return ds


def _read_band_vrt(
    fp: Path,
    band_idx: int,
    bounds: Tuple[float, float, float, float],
    shape_hw: Tuple[int, int],
    *,
    nodata: float,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Reads a specific band, warping it to the target bounds/shape on the fly.
    Tries to use the worker cache first (faster).
    """
    dataset = _get_cached_dataset(fp)
    if dataset is None:
        return None, nodata
    return _read_band_vrt_from_dataset(
        dataset, band_idx, bounds, shape_hw,
        nodata=nodata, resampling=resampling
    )

def _read_band_vrt_from_dataset(
    src: rasterio.DatasetReader,
    band_idx: int,
    bounds: Tuple[float, float, float, float],
    shape_hw: Tuple[int, int],
    *,
    nodata: float,
    resampling: Resampling,
) -> Tuple[Optional[np.ndarray], float]:
    """Internal helper to perform the VRT warp on an open dataset."""
    # Create transform for the chunk with rasterio.transform.from_bounds()
    transform = from_bounds(*bounds, shape_hw[1], shape_hw[0])
    # Try the file's internal nodata value, fallback to global default
    nodata_value = src.nodata if src.nodata is not None else nodata
    with WarpedVRT(
        src,
        crs=WGS84,
        transform=transform,
        width=shape_hw[1],
        height=shape_hw[0],
        resampling=resampling,
        nodata=nodata_value,
    ) as vrt:
        if band_idx > vrt.count:
            return None, nodata
        arr = vrt.read(band_idx, out_shape=(shape_hw[0], shape_hw[1]))
        return arr, nodata_value


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    *,
    nodata: float,
    compress: str = 'deflate',
) -> None:
    """
    Writes a multi-band array to a Cloud Optimized GeoTIFF (COG).
    Write to .tmp then rename to prevent partial file corruption (eg, on OOM).
    """
    # Create transform (min_x, min_y, max_x, max_y) -> (width, height)
    transform = from_bounds(*bounds, data.shape[2], data.shape[1])
    profile = {
        'driver': 'GTiff',
        'height': data.shape[1],
        'width': data.shape[2],
        'count': data.shape[0],
        'dtype': data.dtype,
        'crs': WGS84,
        'transform': transform,
        'compress': compress,
        'tiled': True,
        'nodata': nodata,
        'BIGTIFF': 'IF_SAFER',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with rasterio.open(tmp, 'w', **profile) as dst:
        dst.write(data)
    os.replace(tmp, path)

# Init dicts of rasterio.DatasetReader and bounds (tuples) for worker processes
_worker_cache = {}
_worker_bounds = {}

def _get_intersecting_sources(
    sources: List[Path],
    chunk_bounds: Tuple[float, float, float, float]
) -> List[Path]:
    """Select sources that intersect with chunk bounds using cached bounds."""
    chunk_box = box(*chunk_bounds)
    intersecting = []
    for fp in sources:
        fp_str = str(fp)
        # Fast: Check memory cache for bounds
        bounds = _worker_bounds.get(fp_str)
        # Slow: open file to get bounds (will be used below *if* _init failed)
        if bounds is None:
            ds = _get_cached_dataset(fp)
            if ds is None:
                continue
            bounds = ds.bounds
            _worker_bounds[fp_str] = bounds
        src_box = box(*bounds)
        if chunk_box.intersects(src_box):
            intersecting.append(fp)
    return intersecting


def _close_worker_cache() -> None:
    """Closes all open file handles in the worker process."""
    for ds in _worker_cache.values():
        try:
            ds.close()
        except Exception:
            pass
    _worker_cache.clear()
    _worker_bounds.clear()


def _init_worker_cache(source_paths: List[str], mask_paths: List[str]) -> None:
    """
    Multiprocessing initializer (will have a pool of N_WORKERS).
    Pre-scans all file bounds so workers can filter intersections,
    without opening the actual file handles later.
    """
    global _worker_cache, _worker_bounds
    # Ensure a clean start
    _close_worker_cache()
    unique_paths = list(dict.fromkeys(source_paths + mask_paths))
    for path in unique_paths:
        try:
            # Open to read bounds, then immediately close via context manager.
            # We DON'T keep 1000s of file handles open, only the bounds.
            with rasterio.open(path) as ds:
                _worker_bounds[path] = ds.bounds
        except Exception:
            # If a file is corrupt, ignore. Will fail gracefully later.
            pass

def _build_aoi_mask(
    aoi_rasters: List[str],
    bounds: Tuple[float, float, float, float],
    shape_hw: Tuple[int, int],
    exclude_values: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """
    Builds a boolean validity mask from Area of Interest (AOI) raster files.
    """
    if not aoi_rasters:
        return None
    # Filter to only open relevant AOI files
    aoi_paths = [Path(p) for p in aoi_rasters]
    local_aois = _get_intersecting_sources(aoi_paths, bounds)
    if not local_aois:
        return None
    # Initialize mask as "Everything is Valid" (True)
    aoi_mask = np.ones(shape_hw, dtype=bool)
    # Prepare exclusion values (Default: 0 is invalid/background)
    invalid_vals = np.array(exclude_values) if exclude_values else None
    for aoi_fp in local_aois:
        # Read using Nearest Neighbor (critical for categorical/boundary data)
        arr, nd = _read_band_vrt(
            aoi_fp, 1, bounds, shape_hw,
            nodata=NODATA, resampling=Resampling.nearest
        )
        if arr is None:
            continue
        valid_data = np.isfinite(arr)
        arr_int = arr.astype(np.int32, copy=False)
        # Determine "Bad" pixels (to set False in the mask)
        if invalid_vals is not None and invalid_vals.size:
            is_excluded = valid_data & np.isin(arr_int, invalid_vals)
        else:
            is_excluded = valid_data & (arr_int == 0)
        # Update the master AOI mask
        aoi_mask[is_excluded] = False
    return aoi_mask

# Main Processing Functions
# -------------------------
def process_chunk(job: Dict[str, Any]) -> Tuple[str, str]:
    """
    Processes raw imagery over a single 0.5deg grid cell.

    Algorithm:
    1.  Fetch data in chunk: Intersects chunk bounds with the file registry.
    2.  AOI Filtering: (Optional) Exclude non-AOI pixels.
    3.  Mosaicking: Aggregates overlapping source images.
    4.  Normalization: Rescales raw data to [-1, 1] using P2/P98 pctiles.
    5.  Build new features: Computes spectral indices (e.g., NDVI).
    6.  Saving: Saves as a tiled Cloud Optimized GeoTIFF (COG).

    Args:
        job (dict): Configuration dictionary containing:
            - chunk_id (str): Unique identifier (e.g., "3_4").
            - bounds (tuple): (min_x, min_y, max_x, max_y) in WGS84.
            - sources (List[str]): Paths to raw imagery files.
            - out_dir (str): Target directory for output COGs.
            - filename_template (str): (e.g., "chunk_{chunk_id}.tif").
            - spec_key (str): Sensor config key (e.g., 's2', 'planet').
            - shape_hw (tuple): Target pixel dimensions (height, width).
            - limits (dict): Normalization stats {band: (p2, p98)}.
            - overwrite (bool): If True, re-processes existing outputs.
            - aoi_rasters (List[str], optional): Paths to AOI masks.
            - aoi_exclude_values (List[int], optional): Pixel vals to mask out.

    Returns:
        tuple: (chunk_id, status_code)
            - chunk_id (str): The ID of the processed chunk.
            - status_code (str): 'ok', 'skip' (exists/empty), or 'error'.
    """
    start = time.time()
    chunk_id = job['chunk_id']
    overwrite = job['overwrite']
    out_dir = Path(job['out_dir'])
    filename_template = job['filename_template']
    out_path = out_dir / filename_template.format(chunk_id=chunk_id)

    # Skip if already done unless overwrite is True
    if out_path.exists() and not overwrite:
        return chunk_id, 'skip'

    try:
        # Unpack config (will raise KeyError if job dict is malformed)
        bounds = job['bounds']
        sources = [Path(p) for p in job['sources']]
        spec_key = job['spec_key']
        shape_hw = job['shape_hw']
        limits = job['limits']

        # 1. Fetch data in chunk
        # ----------------------
        local_sources = _get_intersecting_sources(sources, bounds)
        if not local_sources:
            return chunk_id, 'skip'

        # 2. AOI Filtering
        # -----------------
        aoi_mask = _build_aoi_mask(
            job.get('aoi_rasters', []),
            bounds,
            shape_hw,
            exclude_values=job.get('aoi_exclude_values')
        )

        # If AOI mask exists and is entirely False, skip chunk.
        if aoi_mask is not None and not np.any(aoi_mask):
            return chunk_id, 'skip'

        # Prep: Data Cube Allocation
        spec = SENSOR_SPECS[spec_key]
        band_map = spec['band_map']
        band_names = list(band_map.keys())
        indices = spec['indices']

        total_channels = len(band_names) + len(indices)
        output_cube = np.full((total_channels, shape_hw[0], shape_hw[1]),
                               NODATA, dtype=np.float32)

        raw_bands: Dict[str, np.ndarray] = {}
        target_transform = from_bounds(*bounds, shape_hw[1], shape_hw[0])

        # 3. Mosaicking
        # -------------
        with ExitStack() as stack:
            source_vrts: List[Tuple[WarpedVRT, Optional[float]]] = []
            for fp in local_sources:
                ds = _get_cached_dataset(fp)
                if ds is None:
                    continue
                nod = ds.nodata if ds.nodata is not None else NODATA
                vrt = stack.enter_context(
                    WarpedVRT(
                        ds, crs=WGS84, transform=target_transform,
                        width=shape_hw[1], height=shape_hw[0],
                        resampling=Resampling.bilinear, nodata=nod,
                    )
                )
                source_vrts.append((vrt, nod))

            if not source_vrts:
                return chunk_id, 'skip'

            for idx, band_name in enumerate(band_names):
                band_idx = band_map[band_name]
                mosaic = np.full(shape_hw, NODATA, dtype=np.float32)
                filled = np.zeros(shape_hw, dtype=bool)
                for vrt, nd in source_vrts:
                    if filled.all():
                        break
                    if band_idx > vrt.count:
                        continue
                    arr = vrt.read(band_idx, out_shape=(shape_hw[0], shape_hw[1]))
                    arr = arr.astype(np.float32, copy=False)

                    # Internal Validity Check (Finite + Nodata)
                    valid_pixel = np.isfinite(arr)
                    if nd is not None:
                        valid_pixel = valid_pixel & (arr != nd)

                    # Artifact Removal (only for PlanetScope data)
                    if spec_key == 'planet':
                        valid_pixel &= (arr >= 0.0) & (arr <= 1.0)

                    # Apply AOI Mask if step 2 generated one
                    if aoi_mask is not None:
                        valid_pixel = valid_pixel & aoi_mask

                    to_fill = valid_pixel & (~filled)
                    if np.any(to_fill):
                        mosaic[to_fill] = arr[to_fill]
                        filled |= to_fill

                if aoi_mask is not None:
                    mosaic[~aoi_mask] = NODATA

                raw_bands[band_name] = mosaic

                # 4. Normalization
                # ----------------
                output_cube[idx] = _rescale(mosaic, limits[band_name], NODATA)

        # 5. Build New Features (Indices)
        base_idx = len(band_names)
        for i, (num_band, den_band) in enumerate(indices):
            val = _nd(raw_bands.get(num_band), raw_bands.get(den_band), NODATA)
            if val is not None:
                output_cube[base_idx + i] = val

        # 6. Saving as .tif
        # -----------------
        _write_geotiff(out_path, output_cube, bounds, nodata=NODATA)
        print(f'Chunk {chunk_id}: wrote {out_path.name} in {_secs(start)}')
        return chunk_id, 'ok'

    except Exception as e:
        print(f'ERROR processing chunk {chunk_id}: {e}')
        traceback.print_exc()
        # Clean up partial files (_write_geotiff uses .tmp for atomic writes)
        tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
        for p in (tmp_path, out_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        return chunk_id, 'error'

def process_year(
    *,
    spec_key: str,
    raw_dir: Path,
    out_dir: Path,
    year: int,
    project: str,
    host: str = 'quest',
    grid_path: Optional[Path] = None,
    overwrite: bool = False,
    use_percentiles_from: Optional[Path] = None,
    filename_template: str = 'chunk_{chunk_id}.tif',
) -> None:
    """
    Orchestrates the parallel tiling job for a specific year.
    """
    start_time = time.time()

    # 1. Validation & Setup
    # ---------------------
    if spec_key not in SENSOR_SPECS:
        raise ValueError(f"Unknown spec_key '{spec_key}'")

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Processing {project} / {spec_key} / {year} ===")

    # 2. Discover Data
    # ----------------
    sources = discover_sources(spec_key, raw_dir, year)

    # Project-Specific Override: Avocados S2 only uses annual medians
    if project == 'avocados' and spec_key == 's2':
        sources = [s for s in sources if 'ANNUAL_MEDIAN' in s.name]

    print(f"  Sources: {len(sources)} files detected")

    # 3. Load Dependencies (Stats & Grid)
    # -----------------------------------
    spec = SENSOR_SPECS[spec_key]

    # Resolve Percentiles Source
    if use_percentiles_from:
        stats_path = Path(use_percentiles_from)
        print(f"  Stats:   Forced override from {stats_path}")
    else:
        stats_path = out_dir / 'percentiles.csv'
        print(f"  Stats:   Loading from {stats_path}")

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Percentiles missing at {stats_path}. "
            f"CRITICAL: Run with --prerun first to generate statistics."
        )
    limits = load_stats_csv(stats_path, spec_key)

    # Resolve Grid
    if not grid_path:
        grid_path = out_dir.parent / f'{spec_key}_grid_{CHUNK_DEG:.2f}.shp'

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid missing at {grid_path}. "
            f"CRITICAL: Run with --prerun first to generate grid."
        )
    # 4. Prepare Job Queue
    # --------------------
    # Calculate target shape once (expensive operation)
    shape_hw = _set_chunk_shape(sources[0])
    print(f"  Target:  {shape_hw[1]}x{shape_hw[0]} pixels @ {CHUNK_DEG}°")

    # Resolve AOI Files (from Config)
    cfg = load_config()
    project_cfg = cfg['HOSTS'][host].get(project, {})

    # AOI Logic: "mask_sources" in previous steps is now "aoi_rasters"
    # We look for 'roi_rasters' or 'aoi_rasters' in the project config
    aoi_rasters = []
    aoi_cfg_val = project_cfg.get('aoi_rasters') or project_cfg.get('roi_rasters') # noqa: E501
    if aoi_cfg_val:
        # If it's a single string, wrap it; if list, use as is
        if isinstance(aoi_cfg_val, str):
            aoi_rasters = [aoi_cfg_val]
        else:
            aoi_rasters = aoi_cfg_val
        print(f"  AOI:     {len(aoi_rasters)} boundary files configured")

    # Filter chunks based on environment variable (for Slurm arrays)
    allowed_chunks = None
    if os.getenv('PREPROC_CHUNK_IDS'):
        raw_ids = os.environ['PREPROC_CHUNK_IDS'].split(',')
        allowed_chunks = {cid.strip() for cid in raw_ids if cid.strip()}
        print(f"  SUBSET:  Restricted to {len(allowed_chunks)} chunks")

    chunk_jobs = []
    skipped_count = 0
    # Iterate grid and build arguments
    for chunk_id, bounds in _iter_grid_bounds(grid_path):
        # Filter: Specific Chunks
        if allowed_chunks is not None and chunk_id not in allowed_chunks:
            continue

        # Filter: Existence
        out_path = out_dir / filename_template.format(chunk_id=chunk_id)
        if out_path.exists() and not overwrite:
            skipped_count += 1
            continue

        # Pack the Job Dictionary (Strings for Multiprocessing safety)
        job = {
            'chunk_id': chunk_id,
            'bounds': bounds,
            'spec_key': spec_key,
            'sources': [str(p) for p in sources],
            'out_dir': str(out_dir),
            'filename_template': filename_template,
            'shape_hw': shape_hw,
            'limits': limits,
            'overwrite': overwrite,
            'aoi_rasters': aoi_rasters, # List of strings
            'aoi_exclude_values': [0]
        }
        chunk_jobs.append(job)

    if not chunk_jobs:
        print(f"  Status:  All skipped ({skipped_count} existing). Job done.")
        return

    # 5. Execute Parallel Processing
    # ------------------------------
    print(f"  Queue: {len(chunk_jobs)} chunks to proc; {skipped_count} skipped") # noqa: E501
    print(f"  Workers: {N_WORKERS}")

    results = []

    # Initialize shared memory resources
    # Pass flattened list of files (sources + masks) to initialize bounds cache
    all_files = [str(p) for p in sources] + aoi_rasters
    _init_worker_cache(all_files, [])

    try:
        if N_WORKERS == 1:
            # Serial execution for debugging
            for job in tqdm(chunk_jobs, desc="Processing (Serial)"):
                res = process_chunk(job)
                results.append(res)
        else:
            # Parallel execution
            with cf.ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_worker_cache,
                initargs=(all_files, [])
            ) as executor:
                for res in tqdm(executor.map(process_chunk, chunk_jobs),
                                total=len(chunk_jobs), desc="Processing"):
                    results.append(res)
    finally:
        _close_worker_cache()


    # 6. Final Report
    # ---------------
    success = sum(1 for _, status in results if status == 'ok')
    errors = [cid for cid, status in results if status == 'error']

    print(f"\n=== Summary ({_secs(start_time)}) ===")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors:  {len(errors)}")

    if errors:
        print("\n  FAILED CHUNKS:")
        for cid in errors:
            print(f"    - {cid}")


def _resolve_grid_path(project_cfg: Dict[str, str], allow_missing: bool = False) -> Path: # noqa: E501
    """Return the project-defined grid shapefile."""
    configured = project_cfg.get('grid_shapefile')
    if not configured:
        # Auto-construct based on project root if missing
        root = Path(project_cfg.get('root', '.'))
        return root / f'grid_{CHUNK_DEG:.2f}.shp'

    path = Path(configured)
    if path.exists() or allow_missing:
        return path
    raise FileNotFoundError(f'Configured grid path not found: {path}')


def _resolve_from_config(
    *,
    host: str,
    project: str,
    year: Optional[int],
    spec_key: str,
    allow_missing_grid: bool = False,
) -> Tuple[str, Path, Path, Path]:
    if spec_key not in SENSOR_SPECS:
        msg = (f"Unknown spec_key '{spec_key}'. "
               f"Opts: {', '.join(sorted(SENSOR_SPECS))}")
        raise ValueError(msg)
    cfg = load_config()
    host_cfg = cfg['HOSTS'].get(host)
    if host_cfg is None:
        raise ValueError(f"Unknown host '{host}' in configuration")
    project_cfg = host_cfg.get(project)
    if project_cfg is None:
        raise ValueError(f"Unknown project '{project}' for host '{host}'")

    raw_primary = project_cfg.get('raw_folder')
    if not raw_primary:
        raise FileNotFoundError("Project config missing 'raw_folder' entry")
    raw_dir = Path(raw_primary)

    if not raw_dir.exists():
        raise FileNotFoundError(f'Raw imagery dir does not exist: {raw_dir}')

    grid_path = _resolve_grid_path(project_cfg,
                                   allow_missing=allow_missing_grid)
    out_dir = Path(project_cfg['proc_folder'])
    if year is not None:
        # Standard output structure: {year}-{SENSOR_CODE}
        sensor_code = SENSOR_CODE.get(spec_key, spec_key.upper())
        out_dir = out_dir / f'{year}-{sensor_code}'

    return spec_key, raw_dir, grid_path, out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Unified imagery prep')
    parser.add_argument('--sensor', choices=sorted(SENSOR_SPECS.keys()),
                        required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--project', required=True,
                        help='Project key mapping to config.yaml paths')
    parser.add_argument('--host', default='quest', help='Host environment key')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--prerun', action='store_true',
                        help='Generate grid and stats ONLY, then exit.')
    parser.add_argument('--use-percentiles-from', type=str, default=None,
                        help='Override stats file path.')
    return parser

# Define and call main function
# -----------------------------
def main():
    parser = _build_parser()
    args = parser.parse_args()

    spec_key, raw_dir, grid_path, out_dir = _resolve_from_config(
        host=args.host,
        project=args.project,
        year=args.year,
        spec_key=args.sensor,
        allow_missing_grid=args.prerun,
    )

    # PRERUN MODE: Setup Inputs
    # -------------------------
    if args.prerun:
        print('=== PRERUN: Grid & Stats Generation ===')
        print(f'Project: {args.project}')
        print(f'Grid:    {grid_path}')
        print(f'Stats:   {out_dir / "percentiles.csv"}')

        # 1. Grid Generation
        # ------------------
        if grid_path.exists() and not args.overwrite:
            print('Grid exists. Skipping generation.')
        else:
            # Derive bounds from all sources
            sources = discover_sources(spec_key, raw_dir, args.year)
            bounds = _compute_global_bounds(sources)
            _build_grid_shapefile(grid_path, bounds)

        # 2. Stats Computation
        # --------------------
        stats_path = out_dir / 'percentiles.csv'
        if stats_path.exists() and not args.overwrite:
            print('Stats exist. Skipping computation.')
        else:
            sources = discover_sources(spec_key, raw_dir, args.year)
            if sources:
                nodata_val = NODATA
                print(f'Computing stats from {len(sources)} files...')
                limits = compute_percentile_limits(
                    spec_key,
                    sources,
                    nodata_fallback=nodata_val
                )
                save_stats_csv(stats_path, limits)
                print(f'Saved: {stats_path}')
            else:
                print('Error: No sources found for stats computation.')
        return

    # MAIN EXECUTION: Run Pipeline (Requires having grid & stats)
    # -----------------------------------------------------------
    process_year(
        spec_key=spec_key,
        raw_dir=raw_dir,
        grid_path=grid_path,
        out_dir=out_dir,
        year=args.year,
        project=args.project,
        host=args.host,
        overwrite=args.overwrite,
        use_percentiles_from=args.use_percentiles_from,
    )


SENSOR_SPECS = _build_sensor_specs(RAW_SENSOR_SPECS)
if __name__ == '__main__':
    main()
