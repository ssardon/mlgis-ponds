"""
TFRecord Preprocessor
Extracts patches from satellite imagery and writes to TFRecord format.
Uses NPZ cache when available to skip re-extraction from raw TIFFs.
"""
import glob
import os
import shutil

import numpy as np
import tensorflow as tf

from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Helpers of 'create_tfrecords()' orchestrator function
# ---------------------------------------------------
from mlgis_helpers.data_management import (
    load_patches,
    simplify_masks,
    filter_to_ario_proximity
)

NODATA = -9999.0

def _print_patch_statistics(msk_patches: np.ndarray, split_name: str) -> None:
    """
    Analyzes class balance in mask patches (Pixel counting).
    BY "mask patches", we mean binary images where positive pixels indicate
    presence of the target class (ponds).
    """
    pixel_counts = {0: 0, 1: 0, 2: 0, 3: 0, 'more': 0}
    positive_indices = []

    for i, msk in enumerate(msk_patches):
        pos_pixels = msk.sum()
        if pos_pixels > 0:
            positive_indices.append(i)
            if pos_pixels == 1:
                pixel_counts[1] += 1
            elif pos_pixels == 2:
                pixel_counts[2] += 1
            elif pos_pixels == 3:
                pixel_counts[3] += 1
            else:
                pixel_counts['more'] += 1
        else:
            pixel_counts[0] += 1
    pos_rate = 100 * len(positive_indices) / len(msk_patches)
    print(f"\n=== PATCH STATISTICS for {split_name} ===")
    print(f"  Total patches: {len(msk_patches)}")
    print(f"  Positives:     {len(positive_indices)} ({pos_rate:.2f}%)")
    print(f"  Pixel dist:    {pixel_counts}")

def _write_patches_to_tfrecord(
    images: np.ndarray,
    masks: np.ndarray,
    output_path: str,
    split_name: str,
    dilation_radius: int = 0
) -> int:
    """
    Writes numpy arrays to a TFRecord file using Example protos.
    Enforces [-1, 1] clipping for safety.
    """
    print(f"\n--- Writing {split_name.upper()} split to TFRecord ---")

    # Safety: Clean NODATA and clip to canonical input range
    images = np.where(images <= NODATA, -1.0, images)
    images = np.clip(images, -1.0, 1.0).astype(np.float32)

    # Optional: Morphological dilation to thicken features
    if dilation_radius > 0:
        print(f"  Applying mask dilation (r={dilation_radius})")
        masks = simplify_masks(masks, dilation_radius=dilation_radius)

    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for img, msk in zip(images, masks):
            # Serialize to Byte Features
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(img.astype(np.float32)).numpy()]
                )),
                'mask': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(msk.astype(np.uint8)).numpy()]
                ))
            }
            example = tf.train.Example(features=
                                       tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            count += 1

    print(f"  {split_name.upper()}: {count:,} patches committed.")
    return count


def _process_split(
    image_path: Union[str, Sequence[str]],
    base_params: Dict[str, Any],
    quick_mode: bool,
    ario_shapefile: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper: Loads patches and applies geographic filtering.
    """
    # 1. Load Patches
    # In quick mode, we only load 100 patches to test the pipeline
    if isinstance(image_path, (list, tuple)):
        display_name = ", ".join(os.path.basename(p) for p in image_path)
    else:
        display_name = os.path.basename(image_path)
    print(f"Loading patches from: {display_name}...")
    result = load_patches(
        image_path,
        quick_mode=quick_mode,
        quick_patches=100 if quick_mode else None,
        **base_params
    )

    # Handle variable return signature from data_loading
    if len(result) == 3:
        img, msk, coords = result
        has_coords = True
    else:
        img, msk = result
        coords = np.array([])
        has_coords = False

    # 2. Apply Geographic Filtering (Ario Proximity)
    if ario_shapefile:
        if not os.path.exists(ario_shapefile):
            raise FileNotFoundError(
                f"Ario shapefile provided but not found: {ario_shapefile}"
            )

        if quick_mode:
            print("  Skipping Ario Filter (Quick Mode active)")
        else:
            if not has_coords:
                raise ValueError(
                    "Ario filter requested but load_patches did not return "
                    "coordinates; cannot apply proximity filter."
                )

            print("  Applying Ario Geographic Filter...")
            filtered_result = filter_to_ario_proximity(
                img, msk, coords,
                image_path=image_path,
                ario_shapefile=ario_shapefile,
                max_distance_km=1
            )

            if len(filtered_result) == 3:
                img, msk, _ = filtered_result
            else:
                img, msk = filtered_result

            print(f"  Filter Complete. Remaining: {len(img)} patches")

    return img, msk


# Orchestrator Function
# ---------------------

def create_tfrecords(
    train_image_path: str,
    val_image_path: str,
    cache_dir: str,
    task: str,
    base_params: Dict[str, Any],
    config: Dict[str, Any],
    quick_mode: bool = False,
    ario_shapefile: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Orchestrates the generation of TFRecords.
    Handles caching, nuclear cleanup, and splitting.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Cache Management (full deletion if 'overwrite_cache' is set to True)
    # Note: cache_dir is already per-task (e.g. .../persistent_cached_patches_{task})
    if config.get('GLOBAL', {}).get('overwrite_cache', False):
        print("\n--- FULL CACHE CLEARING ---")

        # Remove all tfrecord subdirs
        for name in os.listdir(cache_dir):
            if name.startswith('tfrecords_'):
                target = os.path.join(cache_dir, name)
                if os.path.isdir(target):
                    print(f"Removing: {target}")
                    shutil.rmtree(target)

        # Remove NPZ patch caches
        for npz in glob.glob(os.path.join(cache_dir, '*.npz')):
            print(f"Removing: {npz}")
            os.remove(npz)

        print("Cache cleared. Starting fresh.")

    # 2. Define Output Paths
    prefix = 'quick_' if quick_mode else ''
    subdir = f'tfrecords_{prefix}{task}'
    record_dir = os.path.join(cache_dir, subdir)
    os.makedirs(record_dir, exist_ok=True)

    train_out = os.path.join(record_dir, 'train.tfrecord')
    val_out = os.path.join(record_dir, 'val.tfrecord')

    # 3. Check Cache
    if os.path.exists(train_out) and os.path.exists(val_out):
        print("\n--- Using Cached TFRecords ---")
        print(f"Train: {train_out}")
        print(f"Val:   {val_out}")
        return train_out, val_out

    # 4. Process & Write
    print(f"\n--Generating TFRecords ({'QUICK' if quick_mode else 'FULL'})--")

    # Process Train
    t_img, t_msk = _process_split(train_image_path, base_params,
                                  quick_mode, ario_shapefile)
    _print_patch_statistics(t_msk, "train")

    d_train = int(config.get('GLOBAL', {}).get('mask_dilation_train', 0))
    train_count = _write_patches_to_tfrecord(
        t_img, t_msk, train_out, 'train', d_train
    )

    # Free memory
    del t_img, t_msk

    # Process Val
    v_img, v_msk = _process_split(val_image_path, base_params,
                                  quick_mode, ario_shapefile)
    _print_patch_statistics(v_msk, "val")

    d_val = int(config.get('GLOBAL', {}).get('mask_dilation_val', 0))
    val_count = _write_patches_to_tfrecord(
        v_img, v_msk, val_out, 'val', d_val
    )

    # Write metadata for fast loading (avoids re-scanning TFRecords)
    import json
    metadata_path = os.path.join(record_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({'train_count': train_count, 'val_count': val_count}, f)
    print(f"Metadata written: {metadata_path}")

    return train_out, val_out
