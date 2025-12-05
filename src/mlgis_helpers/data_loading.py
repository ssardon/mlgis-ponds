"""
GIS Data Loading & Utilities.
Handles raster I/O, shapefile rasterization, and processing of shp/raster data.
"""

import os
import time
import traceback
import warnings
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Optional, Union, Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from scipy.ndimage import (
    binary_closing, 
    binary_dilation, 
    binary_fill_holes,
    distance_transform_edt,
    label as nd_label
)
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm.auto import tqdm

# Cache for heavy water buffer shapefile data
_WATER_BUFFER_CACHE = {}


# Worker function extract_patches_from_file(), and its helpers
# Used by load_patches() in this module, which is in turn called by
# preprocess_tfrecord.create_tfrecords() to build training/validation datasets.
# ---------------------------------------------------------------------------
def _read_bands_float32(image_path: str, bands: list) -> np.ndarray:
    """
    Read selected bands into an HxWxC float32 array (no re-standardizing).
    Special handling for band 13 which computes NBR = (B8-B12)/(B8+B12).
    """
    with rasterio.open(image_path) as src:
        # Check if NBR (band 13) is requested
        if 13 in bands:
            # We need to compute NBR from B8 and B12
            bands_to_read = []
            nbr_position = None

            for i, b in enumerate(bands):
                if b == 13:
                    nbr_position = i
                    # Read B8 and B12 to compute NBR
                    if 8 not in bands_to_read:
                        bands_to_read.append(8)
                    if 12 not in bands_to_read:
                        bands_to_read.append(12)
                else:
                    if b not in bands_to_read:
                        bands_to_read.append(b)

            # Read all needed bands
            data = src.read(bands_to_read).transpose((1, 2, 0)).astype(np.float32)

            # Compute NBR
            b8_idx = bands_to_read.index(8)
            b12_idx = bands_to_read.index(12)
            b8 = data[:, :, b8_idx]
            b12 = data[:, :, b12_idx]

            # NBR = (NIR - SWIR2) / (NIR + SWIR2)
            # Handle division by zero
            denominator = b8 + b12
            nbr = np.where(
                np.abs(denominator) > 1e-6,
                (b8 - b12) / denominator,
                0.0  # avoid dividing by ~0 when both bands are near zero
            )

            # Build output array with bands in requested order
            output = np.zeros((data.shape[0], data.shape[1], len(bands)), dtype=np.float32)

            for i, b in enumerate(bands):
                if b == 13:
                    output[:, :, i] = nbr
                else:
                    idx = bands_to_read.index(b)
                    output[:, :, i] = data[:, :, idx]

            return output
        else:
            # Normal band reading
            return src.read(bands).transpose((1, 2, 0)).astype(np.float32)


def _set_missing_to_nodata(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, -9999.0)

def extract_patches_from_file(args: tuple) -> Tuple[np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """
    Worker function: Extracts patches from a single TIFF file.

    Notice this is quite different from the 'process_chunk' worker function in
    '02_preproc.py', which extracts large 0.5 degree chunks from many raw
    imagery files covering the entire area of interest. Here, we take a given
    chunk (a single TIFF) and chop it into much smaller patches (currently
    using 128 pixel chunks, so 1.28 km each at Sentinel-2's 10m resolution).

    Args:
        args (tuple): Packed arguments for multiprocessing serialization:
            - idx (int): Worker index (for logging purposes)
            - image_file (str): Path to the source GeoTIFF raster
            - shape_path (str): Path to the vector label Shapefile
            - patch_size (int): Height/Width of square patches in pixels
            - stride (int): Sliding window step size in pixels
            - bands (List[int]): 1-based indices of spectral bands to read
            - cache_dir (str): Path to store intermediate .npz files
            - task (str): Task identifier (e.g., 'ponds', 'mines')

    Returns:
        tuple: (images, masks, coords)
            - images (np.ndarray): Tensor (N, ps, ps, Channels)
            - masks (np.ndarray): Tensor (N, ps, ps)
            - coords (np.ndarray): Array (N, 2) with coords of patch centroids
            (ps = patch_size; masks are binary rasters built from labels)


    Operations:
    1. Checks NPZ cache (returns early if found).
    2. Reads raster bands.
    3. Rasterizes vector labels (shapefile) to create mask.
    4. Extracts strided patches.
    5. Filters based on content (NODATA check, positive label presence).
    6. Saves result to NPZ cache.
    """
    idx, image_file, shape_path, patch_size, stride, bands, cache_dir, task = args

    fname = os.path.basename(image_file)
    print(f"\n--- Processing file {idx+1}: {fname} ---")

    # 1. Check Cache
    # --------------
    # Naming convention: {filename}_{size}px_stride{stride}.npz
    stem = os.path.splitext(fname)[0]
    cache_path = os.path.join(cache_dir,
                              f"{stem}_{patch_size}px_stride{stride}.npz")

    if os.path.exists(cache_path):
        print("  Loading from cache")
        with np.load(cache_path) as data:
            imgs = data["img"]
            msks = data["msk"]
            coords = data.get("patch_coords", np.array([]))

        # Quick sanity check / cleanup on cached data
        if np.any(imgs <= -9999):
            imgs = np.where(imgs <= -9999, -1.0, imgs)
            imgs = np.clip(imgs.astype(np.float32), -1.0, 1.0)

        return imgs, msks, coords

    try:
        # 2. Load & Preprocess Image
        # --------------------------
        # Note: read_bands_float32 must be defined below
        img_arr = _read_bands_float32(image_file, bands)
        img_arr = _set_missing_to_nodata(img_arr) # Helper to fix Infs/NaNs

        # 3. Rasterize Vector Labels
        # --------------------------
        gdf = gpd.read_file(shape_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        with rasterio.open(image_file) as src:
            # Ensure CRS match
            if gdf.crs != src.crs:
                gdf_proj = gdf.to_crs(src.crs)
            else:
                gdf_proj = gdf

            # Only rasterize polygons that actually touch the input tile
            tile_box = box(*src.bounds)
            local_gdf = gdf_proj[gdf_proj.intersects(tile_box)]

            if not local_gdf.empty:
                print(f"  Rasterizing {len(local_gdf)} polygons")
                mask_arr = rasterize(
                    [(g, 1) for g in local_gdf.geometry],
                    out_shape=src.shape,
                    transform=src.transform,
                    fill=0,
                    dtype=np.uint8
                )
            else:
                print("  No intersection; empty mask")
                mask_arr = np.zeros(src.shape, dtype=np.uint8)

            img_transform = src.transform

        # 4. Patch Extraction Loop
        # ------------------------
        h, w = img_arr.shape[:2]
        if h < patch_size or w < patch_size:
            print("  Tile smaller than patch_size; skipping.")
            return np.array([]), np.array([]), np.array([])

        # Task-Specific Logic (Mining vs Agriculture)
        keep_random_frac = 1.0
        mask_neighbor = None

        if task == "mines":
            keep_random_frac = 0.10 # Downsample negatives
            print("  Mining mode: Computing 1km proximity mask...")
            is_pos = mask_arr > 0
            # Distance transform on INVERTED mask (distance to nearest True)
            # sampling=10.0 assumes ~10m resolution (Sentinel-2)
            dist_map = distance_transform_edt(~is_pos, sampling=10.0)
            mask_neighbor = dist_map <= 1000.0

        patches_img = []
        patches_msk = []
        patches_xy = []

        # Counters
        stats = {'pos': 0, 'neighbor': 0, 'random': 0, 'nodata': 0}

        # Iterate sliding window
        for r in range(0, h - patch_size + 1, stride):
            for c in range(0, w - patch_size + 1, stride):
                # Slice
                ip = img_arr[r:r+patch_size, c:c+patch_size]
                mp = mask_arr[r:r+patch_size, c:c+patch_size]

                # Filter: NODATA Check (>10% bad pixels in ANY band)
                is_nodata = (ip <= -9999)
                # Check mean nodata per band
                # axis=(0,1) collapses H,W -> leaves (Bands,)
                band_nodata_frac = is_nodata.mean(axis=(0,1))

                if np.any(band_nodata_frac > 0.10):
                    stats['nodata'] += 1
                    continue

                # Filter: Content Check
                has_pos = mp.sum() > 0
                is_neighbor = False

                if not has_pos and mask_neighbor is not None:
                    np_patch = mask_neighbor[r:r+patch_size, c:c+patch_size]
                    is_neighbor = np_patch.any()

                # Decision: Keep or Drop?
                keep = False
                if has_pos:
                    keep = True
                    stats['pos'] += 1
                elif is_neighbor:
                    keep = True
                    stats['neighbor'] += 1
                elif np.random.random() < keep_random_frac:
                    keep = True
                    stats['random'] += 1

                if keep:
                    patches_img.append(ip)
                    patches_msk.append(mp)

                    # Calculate center coordinates (Geo)
                    cr = r + patch_size // 2
                    cc = c + patch_size // 2
                    px, py = rasterio.transform.xy(img_transform, cr, cc)
                    patches_xy.append((px, py))

        # 5. Finalize & Save to Cache
        # ---------------------------
        if task == "mines":
            print(f"  Stats: {stats['pos']} pos, {stats['neighbor']} near, {stats['random']} rand")
        if stats['nodata'] > 0:
            print(f"  Skipped {stats['nodata']} patches due to NODATA content.")

        if not patches_img:
            return np.array([]), np.array([]), np.array([])

        final_imgs = np.array(patches_img, dtype=np.float32)
        final_msks = np.array(patches_msk, dtype=np.uint8)
        final_xys = np.array(patches_xy)

        # Final safety guard
        if np.any(final_imgs <= -9999):
            final_imgs = np.where(final_imgs <= -9999, -1.0, final_imgs)

        # Atomic Write (temp file + rename, to avoid partial writes to cache)
        with NamedTemporaryFile(dir=os.path.dirname(cache_path),
                                suffix=".npz", delete=False) as tmp:
            np.savez(tmp.name, img=final_imgs,
                     msk=final_msks, patch_coords=final_xys)
        os.replace(tmp.name, cache_path)

        print(f"  Cached {len(final_imgs)} patches.")
        return final_imgs, final_msks, final_xys

    except Exception:
        print(f"  ERROR processing {fname}")
        traceback.print_exc()
        return np.array([]), np.array([]), np.array([])


#--------------------------------------------------


# Lightweight postproc utilities (CRS, intersections, grouping)
def ensure_crs(gdf, target=None, fallback='EPSG:4326'):
    gdf = gdf.set_crs(fallback) if gdf.crs is None else gdf
    return gdf if target is None or gdf.crs == target else gdf.to_crs(target)

def fix_invalid(gdf):
    inv = ~gdf.geometry.is_valid
    if getattr(inv, 'any', lambda: False)():
        gdf = gdf.copy()
        gdf.loc[inv, 'geometry'] = gdf.loc[inv, 'geometry'].buffer(0)  # zero-width buffer fixes self-intersections from digitizing glitches
    return gdf

def to_metric_pair(a, b):
    if str(b.crs).startswith('EPSG:326'):
        return a, b
    utm = b.estimate_utm_crs()
    return a.to_crs(utm), b.to_crs(utm)


def _calculate_area_m2(gdf: gpd.GeoDataFrame, preferred_projected_crs=None):
    """Return per-feature areas in square meters, reprojecting when needed."""
    if gdf.empty:
        return gdf.geometry.area

    if gdf.crs is None:
        raise ValueError("Cannot compute area_m2 without a defined CRS on the GeoDataFrame")

    if gdf.crs.is_projected:
        return gdf.geometry.area

    target_crs = preferred_projected_crs
    if target_crs is None:
        target_crs = gdf.estimate_utm_crs()

    if target_crs is None:
        raise ValueError("Unable to determine a projected CRS for area calculation")

    projected_geometry = gdf.geometry.to_crs(target_crs)
    return projected_geometry.area

def overlay_area(left, right, prefix='intersection'):
    """
    Compute intersection of two GeoDataFrames with area columns.
    Uses optimized sjoin + intersection approach for large datasets.
    """
    # For small datasets, use standard overlay
    if len(left) * len(right) < 100000:
        inter = gpd.overlay(left, right, how='intersection')
        inter[f'{prefix}_area_m2'] = inter.geometry.area
        inter[f'{prefix}_area_ha'] = inter[f'{prefix}_area_m2'] / 10000
        return inter

    # For large datasets, use optimized approach
    print(f"Using optimized overlay (sjoin + intersection) for {len(left)} x {len(right)} features...")

    # Step 1: Find candidate pairs using spatial join (uses R-tree index)
    joined = gpd.sjoin(left, right, how='inner', predicate='intersects')

    if len(joined) == 0:
        # No intersections, return empty result with correct schema
        result = gpd.GeoDataFrame()
        result[f'{prefix}_area_m2'] = []
        result[f'{prefix}_area_ha'] = []
        return result

    # Step 2: Merge right columns and compute intersections
    # Get right columns for each match
    result = joined.merge(right, left_on='index_right', right_index=True, suffixes=('', '_right'))

    # Get geometries for intersection
    left_geoms = left.loc[result.index, 'geometry'].values
    right_geoms = result['geometry_right'].values

    # Vectorized intersection
    try:
        from shapely import intersection
        inter_geoms = intersection(left_geoms, right_geoms)
    except (ImportError, AttributeError):
        # Fallback for older shapely versions
        inter_geoms = np.array([left_geoms[i].intersection(right_geoms[i]) for i in range(len(left_geoms))])

    # Replace geometry with intersection and drop geometry_right
    result = result.drop(columns=[c for c in result.columns if c.endswith('_right') and c != 'index_right'])
    result['geometry'] = inter_geoms

    # Filter out empty geometries and reset to GeoDataFrame
    inter_series = gpd.GeoSeries(inter_geoms, index=result.index, crs=left.crs)
    non_empty = ~inter_series.is_empty & inter_series.notna()
    result = result.loc[non_empty].copy()
    inter_series = inter_series.loc[non_empty]
    result = gpd.GeoDataFrame(result, geometry=inter_series, crs=left.crs)

    # Add area columns
    result[f'{prefix}_area_m2'] = result.geometry.area
    result[f'{prefix}_area_ha'] = result[f'{prefix}_area_m2'] / 10000

    return result

# Shared grid indexing for full coverage (regular stride + snap to edge)
def grid_indices(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    idx = list(range(0, length - patch + 1, stride))
    last = length - patch
    if idx[-1] != last:
        idx.append(last)
    return idx

# Shared vectorization: threshold  shapes  min-area  water-filter
def vectorize_predictions(prob_map, crs, transform, threshold=0.5, min_area_m2=100, merge_distance_m=50, task=None, skip_water_filter=False, verbose=True):
    def _vprint(msg):
        if verbose:
            print(msg, flush=True)

    _vprint(f"--- Vectorizing Results (Threshold={threshold}) ---")

    # For mining: use provided threshold (no override)
    if task and 'mine' in task.lower():
        _vprint(f" Mining mode: Using threshold of {threshold}")
        mask = (prob_map > threshold).astype(np.uint8)

        _vprint(" Mining mode: Smoothing mask with morphological operations...")

        # Apply closing to connect nearby detections (reduced from 5x5 to 3x3, and 2->1 iterations)
        mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=1)  # link nearby blobs with one cheap morphological closing
        # Fill holes within detected regions
        mask = binary_fill_holes(mask)  # fill interior holes so polygons are solid
        mask = mask.astype(np.uint8)
        _vprint("✅ Mask smoothing complete")
    else:
        mask = (prob_map > threshold).astype(np.uint8)

    # Extract shapes and calculate mean probability for each
    from rasterio.features import rasterize
    from scipy.ndimage import label as nd_label

    # OPTIMIZED: Label connected components and extract all geometries in one pass
    labeled_mask, num_features = nd_label(mask, structure=np.ones((3,3)))

    # Extract ALL geometries at once (much faster than per-polygon extraction)
    shapes_gen = rasterio.features.shapes(labeled_mask.astype(np.int32), transform=transform)

    results = []
    for shape, region_id in shapes_gen:
        region_id = int(region_id)
        if region_id > 0:  # Skip background (0)
            # Calculate mean probability for this region
            region_mask_bool = (labeled_mask == region_id)
            mean_prob = float(np.mean(prob_map[region_mask_bool]))

            results.append({
                'geometry': shape,
                'properties': {'mean_prob': mean_prob}
            })

    if not results:
        _vprint("No features found above the threshold.")
        return None

    gdf = gpd.GeoDataFrame.from_features(results, crs=crs)
    initial_count = len(gdf)
    area_crs_hint = None
    _vprint(f"Found {initial_count} raw polygons.")

    # For mining detection, aggressively merge nearby polygons FIRST before area filtering
    if task and 'mine' in task.lower():
        # Use 20m default merge distance for mines (can be overridden)
        if merge_distance_m == 50:  # Use our default for mining
            merge_distance_m = 20

        _vprint(f" Mining detection mode: Merging polygons within {merge_distance_m}m...")

        # Convert to UTM for accurate distance calculations
        utm_crs = gdf.estimate_utm_crs()
        area_crs_hint = utm_crs
        gdf_utm = gdf.to_crs(utm_crs)

        # Buffer to merge nearby polygons
        buffered = gdf_utm.copy()
        buffered['geometry'] = buffered.geometry.buffer(merge_distance_m)

        # Use spatial index for efficient overlap detection
        from shapely.strtree import STRtree
        from shapely.ops import unary_union

        # Build spatial index on buffered geometries
        geom_list = buffered.geometry.tolist()
        tree = STRtree(geom_list)

        # Track which polygons have been assigned to groups
        assigned = set()
        buffered['group_id'] = -1
        group_id = 0

        # Process each polygon
        total_polys = len(buffered)
        progress_interval = max(1, total_polys // 5)  # Report every 20%
        for idx in range(total_polys):
            if idx in assigned:
                continue

            # Progress reporting
            if verbose and idx > 0 and idx % progress_interval == 0:
                pct = 100 * idx / total_polys
                print(f"   Merging progress: {pct:.0f}% ({idx}/{total_polys} polygons processed)")

            # Start a new group with this polygon
            current_group = {idx}
            assigned.add(idx)
            to_check = [idx]

            # Find all connected polygons using BFS
            while to_check:
                check_idx = to_check.pop()
                geom = geom_list[check_idx]

                # Find overlapping polygons using spatial index
                overlapping_indices = tree.query(geom, predicate='intersects')

                for overlap_idx in overlapping_indices:
                    if overlap_idx not in assigned:
                        current_group.add(overlap_idx)
                        assigned.add(overlap_idx)
                        to_check.append(overlap_idx)

            # Assign group ID to all polygons in this group
            for group_idx in current_group:
                buffered.loc[buffered.index[group_idx], 'group_id'] = group_id

            group_id += 1

        # Dissolve by group, taking max probability per group
        dissolved = buffered.dissolve(by='group_id', aggfunc={'mean_prob': 'max'})

        # Unbuffer to approximately original size
        dissolved['geometry'] = dissolved.geometry.buffer(-merge_distance_m)

        # Create new GeoDataFrame and explode multipolygons
        gdf = dissolved[['geometry', 'mean_prob']].explode(index_parts=False).reset_index(drop=True)

        # Convert back to original CRS
        gdf = gdf.to_crs(crs)

        _vprint(f"✅ Merged {initial_count} polygons into {len(gdf)} larger features")

    # Compute polygon area in square meters using a projected CRS
    gdf['area_m2'] = _calculate_area_m2(gdf, preferred_projected_crs=area_crs_hint)

    # For mining, use much larger minimum area threshold (10,000 m)
    if task and 'mine' in task.lower():
        min_area_m2 = 10000  # 10k square meters minimum for mines
        _vprint(f" Mining mode: Using minimum area threshold of {min_area_m2} m")

    gdf = gdf[gdf['area_m2'] >= min_area_m2]  # drop fragments smaller than the configured minimum area
    _vprint(f"Kept {len(gdf)} polygons after filtering by min_area > {min_area_m2} m.")

    # Filter out polygons near existing water bodies using shared helper
    # Skip water filtering for mining tasks or if explicitly disabled
    if skip_water_filter:
        _vprint(" Skipping water body filtering (will be done in post-processing)")
    elif not (task and 'mine' in task.lower()):
        gdf = filter_natural_water(gdf, buffer_distance_m=30)
    else:
        _vprint("Skipping water body filtering for mining detection")

    return gdf


def load_patches(image_path, shape_path, patch_size,
                 out_dir, bands, cache_dir,
                 stride_ratio, quick_mode=False,
                 quick_patches=128, buffer_radius=None, task=None):
    """Return cached or freshly extracted patches.

    Set `quick_mode=True` to keep ~`quick_patches` positive patches.
    Handles single files, directories, or lists of files.

    Args:
        image_path: Can be:
            - str: Path to single TIFF file
            - str: Path to directory containing TIFF files
            - list: List of TIFF file paths
        shape_path: Path to shapefile
        patch_size: Size of patches to extract
        out_dir: Output directory for caching
        bands: Band indices to use
        quick_mode: If True, subsample patches
        quick_patches: Number of patches to keep in quick mode
        buffer_radius: Buffer radius for points
        cache_dir: Optional directory for persistent patch cache
    """
    # Normalize input to list of files
    if isinstance(image_path, str):
        if os.path.isdir(image_path):
            # Directory of TIFFs
            tiff_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)
                               if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.')])
            if not tiff_files:
                raise ValueError(f"No TIFF files found in directory: {image_path}")
        else:
            # Single file
            tiff_files = [image_path]
    elif isinstance(image_path, list):
        # List of files
        tiff_files = image_path
        print(f"Processing {len(tiff_files)} specified TIFF files")
    else:
        raise ValueError("image_path must be a string (file/directory) or list of files")

    # In quick mode with multiple files, sample a subset of chunks first
    if quick_mode and len(tiff_files) > 10:
        # Sample up to 10 chunks for quick mode
        np.random.seed(42)  # For reproducibility
        sample_size = min(10, len(tiff_files))
        tiff_files = np.random.choice(tiff_files, size=sample_size, replace=False).tolist()  # sample a reproducible subset of chunks for quick mode
        print(f"Quick mode: Sampling {sample_size} chunks from {len(image_path if isinstance(image_path, list) else os.listdir(image_path))} total")

    # Process each file and aggregate patches
    all_img_patches = []
    all_msk_patches = []
    stride = int(patch_size * stride_ratio)  # int() rounds DOWN  # int() rounds DOWN - Relative to patch_size
    total_patches_collected = 0

    # Setup persistent cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Prepare args for each chunk
    chunk_args = [
        (idx, image_file, shape_path, patch_size, stride, bands, cache_dir, task)
        for idx, image_file in enumerate(tiff_files)
    ]

    # Use parallel processing for multiple chunks (skip in quick mode)
    use_parallel = (not quick_mode and len(tiff_files) > 1)

    if use_parallel:
        from multiprocessing import Pool
        print(f"\nProcessing {len(tiff_files)} chunks in parallel...")
        with Pool() as pool:
            chunk_results = pool.map(extract_patches_from_file, chunk_args)
    else:
        # Process sequentially (for single chunk or quick mode)
        chunk_results = []
        for args in chunk_args:
            result = extract_patches_from_file(args)
            chunk_results.append(result)
            # Extract imgs to check patch count (handles both 2-tuple and 3-tuple)
            imgs = result[0] if isinstance(result, tuple) and len(result) > 0 else np.array([])
            total_patches_collected += len(imgs)
            if quick_mode and total_patches_collected >= quick_patches * 2:
                print("Quick mode: collected enough patches, stopping early")
                break

    # Aggregate results from all chunks
    all_patch_coords = []
    for result in chunk_results:
        if len(result) == 3:  # New format with coordinates
            imgs, msks, coords = result
        else:  # Backward compatibility
            imgs, msks = result
            coords = np.array([])
        if len(imgs) > 0:
            all_img_patches.append(imgs)
            all_msk_patches.append(msks)
            if len(coords) > 0:
                all_patch_coords.append(coords)

    # Concatenate all patches
    if not all_img_patches:
        print("WARNING: No patches extracted from any files!")
        return np.array([]), np.array([]), np.array([])


    image_patches = np.concatenate(all_img_patches, axis=0)
    mask_patches = np.concatenate(all_msk_patches, axis=0)
    # Concatenate coordinates if available
    if all_patch_coords:
        patch_coords = np.concatenate(all_patch_coords, axis=0)
    else:
        patch_coords = np.array([])


    print(f"\n=== TOTAL: {len(image_patches)} patches from {len(all_img_patches)} files ===")

    # Overall diagnostics
    positive_pixel_counts = [np.sum(m) for m in mask_patches]
    patches_with_any_positives = np.sum([count > 0 for count in positive_pixel_counts])
    total_positive_pixels = np.sum(positive_pixel_counts)
    print(f"DIAGNOSTIC: {patches_with_any_positives} out of {len(mask_patches)} patches contain positive pixels")
    print(f"DIAGNOSTIC: Total positive pixels across all patches: {total_positive_pixels}")
    if patches_with_any_positives > 0:
        print(f"DIAGNOSTIC: Average positive pixels per patch with positives: {total_positive_pixels/patches_with_any_positives:.1f}")

    # Quick mode subsampling (applies to aggregated patches)
    if quick_mode:
        pos_mask = (mask_patches.sum(axis=(1, 2)) > 0)
        pos_idx = np.flatnonzero(pos_mask)
        if pos_idx.size == 0:
            pos_idx = np.arange(len(image_patches))
        np.random.shuffle(pos_idx)
        keep_idx = pos_idx[:min(len(pos_idx), quick_patches)]
        print(f"\nQuick mode: keeping {len(keep_idx)} patches (prioritizing positives)")
        image_patches = image_patches[keep_idx]
        mask_patches = mask_patches[keep_idx]
        # Also filter coordinates if available
        if len(patch_coords) > 0:
            patch_coords = patch_coords[keep_idx]

    # Return coordinates as third array for use by filter functions
    # This allows filter_to_ario_proximity to access actual patch locations
    return image_patches, mask_patches, patch_coords


def simplify_masks(masks, dilation_radius=3):
    """
    Simplify masks by dilating positive pixels to create larger, more learnable targets.

    Args:
        masks: Binary masks array (a 2D raster of 0s and 1s)
        dilation_radius: Radius for dilation operation (a "smoothing" algorithm: more radius, more agressive smoothing)

    Returns:
        Dilated masks with same shape as input
    """
    if dilation_radius <= 0:
        return masks

    structure = np.ones((dilation_radius*2+1, dilation_radius*2+1), dtype=bool)  # square structuring element spanning the requested radius

    if masks.ndim == 2:
        return binary_dilation(masks, structure=structure).astype(masks.dtype)
    elif masks.ndim == 3:
        result = np.zeros_like(masks)
        for i in range(len(masks)):
            result[i] = binary_dilation(masks[i], structure=structure).astype(masks.dtype)  # apply dilation slice-by-slice for 3D masks
        return result
    elif masks.ndim == 4:
        result = np.zeros_like(masks)
        for i in range(len(masks)):
            result[i, :, :, 0] = binary_dilation(masks[i, :, :, 0], structure=structure).astype(masks.dtype)  # handle NHWC layout where mask channel is last
        return result
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")


def _check_feature_dynamic_range(image_patches):
    """
    Check dynamic range of features to detect potential data issues.
    From Test 6 of original diagnostic tests.

    Args:
        image_patches: Array of image patches (N, H, W, C)
    """
    print("\nFeature dynamic range check:")
    for band in range(image_patches.shape[-1]):
        band_data = image_patches[:, :, :, band]
        percentiles = np.percentile(band_data, [0.1, 50, 99.9])

        print(f"  Band {band}: [{percentiles[0]:.3f}, {percentiles[1]:.3f}, {percentiles[2]:.3f}] (0.1%, 50%, 99.9%)")

        # Check for degenerate cases
        if np.all(band_data == band_data.flat[0]):
            print(f"    WARNING: Band {band} is constant! This will likely cause training issues.")
        elif percentiles[2] - percentiles[0] < 0.1:
            print(f"    WARNING: Band {band} has very narrow range. Consider checking normalization.")


def neg_downsampling(image_patches, mask_patches, neg_ratio):
    """Apply negative downsampling to balance the dataset."""

    if len(image_patches) == 0:
        print("Negative down-sampling skipped: no patches to process.")
        return image_patches, mask_patches

    # Use vectorized operations for speed and type safety
    if len(mask_patches.shape) == 4:
        pos_mask = (mask_patches.sum(axis=(1, 2, 3)) > 0)
    else:
        pos_mask = (mask_patches.sum(axis=(1, 2)) > 0)

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]

    print(f"\n--- Negative downsampling ---")
    print(f"Before: {len(pos_idx)} positive patches, {len(neg_idx)} negative patches")

    if len(pos_idx) > 0 and len(neg_idx) > 0:
        keep_neg = min(len(neg_idx), int(len(pos_idx) * neg_ratio))

        if keep_neg < len(neg_idx):
            np.random.shuffle(neg_idx)
            neg_idx = neg_idx[:keep_neg]

        keep_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(keep_idx)

        image_patches = image_patches[keep_idx]
        mask_patches = mask_patches[keep_idx]

        print(f"After: {len(pos_idx)} positives, {keep_neg} negatives.")
    else:
        print(f"No downsampling needed (no positives or no negatives).")

    return image_patches, mask_patches


def filter_to_ario_proximity(patches, masks, patch_coords, image_path, patch_size, ario_shapefile, stride_ratio,
                            max_distance_km=1, chunk_bounds=None):
    """
    Filter patches to only those within a certain distance of Ario municipality.

    Args:
        patches: Image patches array
        masks: Mask patches array
        patch_coords: Array of (x, y) coordinates for each patch (or empty array)
        image_path: Path to the source image for georeferencing
        patch_size: Size of each patch
        ario_shapefile: Path to Ario municipality shapefile
        max_distance_km: Maximum distance from Ario in kilometers
        chunk_bounds: Optional tuple of (minx, miny, maxx, maxy) for chunk bounds

    Returns:
        Filtered patches and masks arrays
    """
    print(f"\n--- Filtering patches to within {max_distance_km}km of Ario ---")

    # Load Ario shapefile
    ario_gdf = gpd.read_file(ario_shapefile)

    # Ensure CRS is defined
    if ario_gdf.crs is None:
        print("WARNING: Ario shapefile has no CRS, assuming EPSG:4326")
        ario_gdf = ario_gdf.set_crs('EPSG:4326')

    # Get image info
    with rasterio.open(image_path) as src:
        img_crs = src.crs
        img_bounds = src.bounds
        img_transform = src.transform
        img_height, img_width = src.shape

    # Reproject Ario to image CRS if needed
    if ario_gdf.crs != img_crs:
        ario_gdf = ario_gdf.to_crs(img_crs)

    # Get Ario boundary
    ario_boundary = ario_gdf.unary_union

    # Check if coordinates were provided
    if patch_coords is not None and len(patch_coords) > 0:
        print("  Using provided patch coordinates from extraction")
        patch_centers = patch_coords
        if len(patch_centers) != len(patches):
            print(f"  WARNING: Coordinate count ({len(patch_centers)}) != patch count ({len(patches)})")
            print("  Falling back to grid reconstruction")
            patch_centers = None
    else:
        patch_centers = None

    # If no coordinates provided, reconstruct grid (less accurate due to filtering)
    if patch_centers is None:
        print("  WARNING: No stored coordinates, reconstructing grid (may be inaccurate due to patch filtering)")
        # Calculate grid of patch centers in PIXEL space
        stride = int(patch_size * stride_ratio)  # int() rounds DOWN  # int() rounds DOWN - Relative to patch_size
        n_patches_x = (img_width - patch_size) // stride + 1
        n_patches_y = (img_height - patch_size) // stride + 1

        # Generate patch centers in pixel coordinates
        patch_rows = np.arange(0, img_height - patch_size + 1, stride)
        patch_cols = np.arange(0, img_width - patch_size + 1, stride)

        # Convert pixel centers to geographic coordinates
        patch_centers = []
        for row in patch_rows:
            for col in patch_cols:
                # Center of patch in pixel coordinates
                center_row = row + patch_size // 2
                center_col = col + patch_size // 2
                # Convert to geographic coordinates
                x, y = rasterio.transform.xy(img_transform, center_row, center_col)  # convert patch center row/col into map coordinates
                patch_centers.append((x, y))

        patch_centers = np.array(patch_centers)

    # Convert to GeoDataFrame for distance calculation
    patch_points = [Point(x, y) for x, y in patch_centers]
    patches_gdf = gpd.GeoDataFrame(geometry=patch_points, crs=img_crs)

    # Project to UTM for accurate distance calculations (in meters)
    # Use UTM zone based on Ario centroid
    ario_centroid = ario_boundary.centroid
    # Calculate proper UTM zone (1-60)
    zone = int((ario_centroid.x + 180) / 6) + 1
    zone = max(1, min(60, zone))  # Clamp to valid range
    # Use 326xx for north, 327xx for south hemisphere
    if ario_centroid.y >= 0:
        utm_crs = f'EPSG:326{zone:02d}'
    else:
        utm_crs = f'EPSG:327{zone:02d}'

    # Reproject both to UTM
    patches_gdf_utm = patches_gdf.to_crs(utm_crs)
    ario_gdf_utm = ario_gdf.to_crs(utm_crs)
    ario_boundary_utm = ario_gdf_utm.unary_union

    # Calculate distances in meters
    distances = patches_gdf_utm.geometry.distance(ario_boundary_utm)

    # Find patches within threshold distance
    max_distance_meters = max_distance_km * 1000  # Convert km to meters
    within_distance = distances <= max_distance_meters

    # Get indices of patches to keep
    keep_indices = np.where(within_distance)[0]

    # When using stored coordinates, we already have the right mapping
    # No need to check expected vs actual counts
    total_patches = len(patches)

    # Ensure indices are valid for the actual patch count
    keep_indices = keep_indices[keep_indices < total_patches]

    # Filter patches and masks
    filtered_patches = patches[keep_indices]
    filtered_masks = masks[keep_indices]

    # Filter coordinates if they were provided
    if patch_coords is not None and len(patch_coords) > 0:
        filtered_coords = patch_coords[keep_indices]
    else:
        filtered_coords = np.array([])

    print(f"Kept {len(filtered_patches)} out of {len(patches)} patches")
    if len(keep_indices) > 0:
        print(f"Distance range: {distances[keep_indices].min():.0f}m - {distances[keep_indices].max():.0f}m")

    # Run feature dynamic range check on filtered patches
    if len(filtered_patches) > 0:
        _check_feature_dynamic_range(filtered_patches)

    return filtered_patches, filtered_masks, filtered_coords


def mask_probability_raster_with_water(prob_map, prob_map_crs, prob_map_transform, water_shapefile_path, buffer_distance_m=30):
    """
    Mask probability raster by setting pixels overlapping natural water to 0.

    Args:
        prob_map: Probability raster array (H, W)
        prob_map_crs: CRS of the probability map
        prob_map_transform: Affine transform of the probability map
        water_shapefile_path: Path to natural water shapefile
        buffer_distance_m: Buffer distance in meters (default: 30m, already applied to shapefile)

    Returns:
        Masked probability raster with water areas set to 0
    """
    if not os.path.exists(water_shapefile_path):
        print(f"WARNING: Water shapefile not found: {water_shapefile_path}")
        print("Proceeding without water masking")
        return prob_map

    print(f"--- Masking Probability Raster with Natural Water ---")
    print(f" Loading water shapefile: {water_shapefile_path}")

    try:
        # Load water shapefile
        water_gdf = gpd.read_file(water_shapefile_path)

        if water_gdf.empty:
            print(" Water shapefile is empty, no masking needed")
            return prob_map

        print(f" Loaded {len(water_gdf)} water features")

        # Ensure CRS
        if water_gdf.crs is None:
            print(" WARNING: Water shapefile has no CRS, assuming EPSG:4326")
            water_gdf = water_gdf.set_crs('EPSG:4326')

        # Reproject water to match probability map CRS if needed
        if water_gdf.crs != prob_map_crs:
            print(f" Reprojecting water from {water_gdf.crs} to {prob_map_crs}")
            water_gdf = water_gdf.to_crs(prob_map_crs)

        # Rasterize water shapefile to match probability map dimensions
        print(f" Rasterizing water mask to {prob_map.shape}")
        water_mask = rasterize(
            [(geom, 1) for geom in water_gdf.geometry],
            out_shape=prob_map.shape,
            transform=prob_map_transform,
            fill=0,
            dtype=np.uint8
        )

        # Count masked pixels
        masked_pixels = np.sum(water_mask > 0)
        total_pixels = prob_map.size
        masked_pct = 100 * masked_pixels / total_pixels

        print(f" Masking {masked_pixels:,} pixels ({masked_pct:.2f}% of raster)")

        # Apply mask: set probability to 0 where water exists
        prob_map_masked = prob_map.copy()
        prob_map_masked[water_mask > 0] = 0

        print(f" Water masking complete")
        return prob_map_masked

    except Exception as e:
        print(f"ERROR: Failed to mask probability raster: {e}")
        print("Proceeding without water masking")
        import traceback
        traceback.print_exc()
        return prob_map

def filter_natural_water(gdf, buffer_distance_m=30, chunk_name=None):
    """
    Remove predicted polygons that are within buffer_distance_m of existing water bodies.

    If chunk_name is provided, uses per-chunk water shapefile (avoids GPFS contention).
    Otherwise uses monolithic pickle cache (legacy behavior).

    Shared utility used by inference and post-processing to avoid duplication.
    """
    print(f"--- Filtering Near Existing Water Bodies (Buffer={buffer_distance_m}m) ---")

    import pickle
    import os
    import geopandas as gpd

    # Host-specific cache paths - priority: Quest -> Farm -> Mac
    if os.path.exists("/projects/p32315/avocado-ponds/CNN Inputs/aux"):  # Quest
        cache_dir = "/projects/p32315/avocado-ponds/CNN Inputs/aux"
    elif os.path.exists("/group/jsayregrp/Sebastian/avocados/labels"):  # Farm
        cache_dir = "/group/jsayregrp/Sebastian/avocados/labels"
    else:  # Mac
        cache_dir = "/Users/Sebastian/Dropbox/JMP/data/Remotely Sensed Ponds/aux"

    initial_count = len(gdf)

    try:
        # Convert to UTM for accurate buffering in meters
        utm_crs = gdf.estimate_utm_crs()
        print(f"Converting to UTM CRS: {utm_crs}")

        # NEW: Per-chunk water shapefiles (avoids 40x concurrent pickle loads)
        if chunk_name:
            water_dir = os.path.join(cache_dir, "natural_water")
            water_shp = os.path.join(water_dir, f"{chunk_name}.shp")

            if not os.path.exists(water_shp):
                print(f" WARNING: Chunk water shapefile missing: {water_shp}")
                print("Proceeding without water body filtering")
                return gdf

            print(f" Loading chunk water shapefile: {chunk_name}.shp", flush=True)
            water_gdf = gpd.read_file(water_shp)

            if water_gdf.crs != utm_crs:
                water_gdf = water_gdf.to_crs(utm_crs)

            # Buffer if needed
            if buffer_distance_m > 0:
                water_gdf['geometry'] = water_gdf.geometry.buffer(buffer_distance_m)

            water_buffers = water_gdf
            water_union = None  # Not needed for per-chunk approach

        else:
            # LEGACY: Monolithic pickle cache
            cache_key = f"water_union_utm{str(utm_crs).split(':')[-1]}_buffer{buffer_distance_m}m.pkl"
            cache_path = os.path.join(cache_dir, cache_key)
            cache_id = (cache_path, str(utm_crs))

            if not os.path.exists(cache_path):
                print(f" WARNING: Water buffer cache missing: {cache_path}")
                print("Note: Water filtering is only relevant for avocado detection, not mining")
                raise FileNotFoundError(f"Missing water buffer cache: {cache_key}")

            global _WATER_BUFFER_CACHE
            cache_entry = _WATER_BUFFER_CACHE.get(cache_id)

            if cache_entry is not None:
                water_union = cache_entry.get('union')
                water_buffers = cache_entry.get('gdf')
                print("✅ Water buffers loaded from memory cache")
            else:
                print(f" Loading prebuilt water buffers from cache: {cache_key}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                water_union = None
                water_buffers = None

                if isinstance(cached_data, dict):
                    water_union = cached_data.get('union')
                    water_buffers = cached_data.get('gdf')
                    if water_union is not None:
                        print("✅ Water union loaded from disk cache")
                    if water_buffers is not None:
                        print("✅ Water geometries loaded from disk cache")
                else:
                    # Old format: GeoDataFrame only, need to union (slow)
                    print(f" Old cache format detected, computing union...")
                    water_buffers = cached_data
                    water_union = cached_data.geometry.unary_union
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump({'gdf': water_buffers}, f)
                        print(f"✅ Cache updated with water geometries (no union needed for sjoin)")
                    except Exception as cache_err:
                        print(f" WARNING: Could not update water cache on disk: {cache_err}")

                _WATER_BUFFER_CACHE[cache_id] = {'gdf': water_buffers}

        # Fall back to union if we somehow have no component geometries
        if water_buffers is None:
            print(" WARNING: Missing individual water geometries in cache; using union fallback (slow)")
            if water_union is None:
                raise RuntimeError("Water buffer cache missing both union and geometries")
            gdf_utm = gdf.to_crs(utm_crs)
            print("Filtering predictions against water body buffers (union fallback)...")
            mask_keep = ~gdf_utm.geometry.intersects(water_union)
            gdf_filtered_utm = gdf_utm[mask_keep]
        else:
            # Convert predictions and buffers to UTM for intersection
            gdf_utm = gdf.to_crs(utm_crs)
            water_buffers_utm = water_buffers.to_crs(utm_crs)

            # Use spatial join (R-tree under the hood) to find ponds near water
            print("Filtering predictions against water body buffers (sjoin)...")
            ponds_with_idx = gdf_utm[['geometry']].copy()
            ponds_with_idx['pond_idx'] = ponds_with_idx.index

            matches = gpd.sjoin(
                ponds_with_idx,
                water_buffers_utm[['geometry']],
                how='inner',
                predicate='intersects'
            )

            if matches.empty:
                gdf_filtered_utm = gdf_utm
            else:
                ponds_to_drop = matches['pond_idx'].unique()
                mask_keep = ~gdf_utm.index.isin(ponds_to_drop)
                gdf_filtered_utm = gdf_utm[mask_keep]

        # Convert back to original CRS
        gdf_filtered = gdf_filtered_utm.to_crs(gdf.crs)

        filtered_count = len(gdf_filtered)
        removed_count = initial_count - filtered_count

        print(f"Removed {removed_count} polygons near existing water bodies")
        print(f"Kept {filtered_count} polygons away from existing water bodies")

        return gdf_filtered

    except Exception as e:
        print(f"Warning: Could not filter near existing water bodies: {e}")
        print("Proceeding without water body filtering")
        return gdf
