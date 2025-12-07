"""
Evaluation metrics and plotting utilities for avocado pond detection.
Preserves all original metrics from masterml.py
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from tqdm.auto import tqdm


def plot_metric_progress(progress, out_path, title, config):
    """
    Plot validation metrics and cumulative wall-time versus epoch.
    """
    # Align list lengths to avoid plotting errors
    required_keys = ["epoch", "auc", "precision", "recall", "f1", "time"]
    available_keys = [k for k in required_keys if k in progress and len(progress[k]) > 0]
    
    if "train_auc" in progress and len(progress["train_auc"]) > 0:
        available_keys.append("train_auc")
    
    min_len = min(len(progress[k]) for k in available_keys)
    
    for k in available_keys:
        progress[k] = progress[k][:min_len]

    epochs = progress["epoch"]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary y-axis: metrics
    ax1.plot(epochs, progress["auc"], label="Val AUC", linewidth=2)
    if "train_auc" in progress and len(progress["train_auc"]) >= min_len:
        ax1.plot(epochs, progress["train_auc"], label="Train AUC", linestyle='--', alpha=0.7)
    ax1.plot(epochs, progress["precision"], label="Precision")
    ax1.plot(epochs, progress["recall"], label="Recall")
    ax1.plot(epochs, progress["f1"], label="F1-score")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric value")

    # Secondary y-axis: cumulative time
    ax2 = ax1.twinx()
    ax2.plot(epochs, progress["time"], "k--", label="Time (s, right axis)")
    ax2.set_ylabel("Seconds")

    # Unified legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=False)
    ax1.set_title(title or "Validation metrics over epochs")
    
    # Set x-axis to show full epoch range from config
    max_epochs = config['GLOBAL']['num_epochs']  # Direct access
    ax1.set_xlim(0, max_epochs + 1)
    ax1.set_xticks(range(0, max_epochs + 1, max(1, max_epochs // 10)))

    # Add settings text
    try:
        from .cfg import get_task_config
        task_config = get_task_config(config['TASKS'], config['task'])
        patch_size_text = f"PATCH_SIZE={task_config['patch_size']}; "
    except:
        patch_size_text = ""
    settings_text = (f"Settings: {patch_size_text}"
                    f"BATCH_SIZE={config['GLOBAL']['batch_size']}; "
                    f"epochs={config['GLOBAL']['num_epochs']} "
                    f"(patience={config['GLOBAL']['patience']})")
    fig.text(0.5, 0.01, settings_text, ha='center', fontsize=8, style='italic')

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Metric progress plot saved to: {out_path}")


def plot_roc_comparison(metrics_list, output_path, config, task_config=None):
    """
    Generate and save a classic ROC curve comparison plot.
    """
    print(f"Generating ROC plot at: {output_path}")
    fig = plt.figure(figsize=(8, 6))
    colours = ['#FF5733', '#33A5FF', '#33FF57', '#FFC300']

    for i, m in enumerate(metrics_list):
        plt.plot(m["fpr"], m["tpr"],
                 label=f"{m['name']} (AUC = {m['auc']:.4f})",
                 linewidth=2.0,
                 color=colours[i % len(colours)])

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.3)

    # Add settings text
    patch_size_text = f"PATCH_SIZE={task_config['patch_size']}; " if task_config else ""
    settings_text = (f"Settings: {patch_size_text}"
                    f"BATCH_SIZE={config['GLOBAL']['batch_size']}; "
                    f"epochs={config['GLOBAL']['num_epochs']} "
                    f"(patience={config['GLOBAL']['patience']})")
    fig.text(0.5, 0.01, settings_text, ha='center', fontsize=8, style='italic')

    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(" ROC plot saved successfully!")


def threshold_at_precision(y_true, y_score, target_p=0.90):
    """Find threshold that achieves target precision (to reduce false positives)."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order].astype(np.int32)
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(tp + fp, 1)
    ok = np.where(prec >= target_p)[0]
    if len(ok) == 0:
        return 1.0
    return float(y_score[order][ok[-1]])


def optimize_threshold(y_true, y_pred, thresholds=np.linspace(0.05, 0.95, 19)):
    """
    Find optimal threshold for binary classification
    """

    best_thr, best_f1, best_prec, best_rec = 0.5, -1.0, 0.0, 0.0

    for thr in thresholds:
        y_bin = (y_pred > thr).astype(int)
        f1_tmp = f1_score(y_true, y_bin, zero_division=0)
        if f1_tmp > best_f1:
            best_f1   = f1_tmp
            best_thr  = thr
            best_prec = precision_score(y_true, y_bin, zero_division=0)
            best_rec  = recall_score(y_true, y_bin, zero_division=0)

    return best_thr, best_f1, best_prec, best_rec


def calculate_metrics(y_true, y_pred, threshold=0.5, patch_shape=None):
    """
    Calculate comprehensive metrics for binary classification.
    Now computes both pixel-level and patch-level AUC to match Aug-11 baseline.
    
    Args:
        y_true: Ground truth labels (flattened or patch format)
        y_pred: Predictions (flattened or patch format)  
        threshold: Classification threshold
        patch_shape: If provided, reshape and compute patch-level metrics (B, H, W) or (B, H, W, 1)
    """
    # Handle NaN predictions
    if np.isnan(y_pred).any():
        print(f"WARNING: {np.isnan(y_pred).sum()} NaN predictions found, replacing with 0.5")
        y_pred = np.nan_to_num(y_pred, nan=0.5)
    
    # Ensure flattened for pixel-level metrics
    y_true_flat = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    y_binary_flat = (y_pred_flat > threshold).astype(int)

    # Pixel-level metrics (original behavior)
    pixel_auc = roc_auc_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_binary_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_binary_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_binary_flat, zero_division=0)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)

    # Patch-level metrics if patch shape provided
    patch_auc = None
    patch_precision = None  
    patch_recall = None
    patch_f1 = None
    
    if patch_shape is not None:
        # Reshape to patch format
        if len(patch_shape) == 3:  # (B, H, W)
            y_true_patches = y_true_flat.reshape(patch_shape)
            y_pred_patches = y_pred_flat.reshape(patch_shape)
        elif len(patch_shape) == 4:  # (B, H, W, 1)
            y_true_patches = y_true_flat.reshape(patch_shape)
            y_pred_patches = y_pred_flat.reshape(patch_shape)
        else:
            print(f"WARNING: Unsupported patch_shape {patch_shape}, skipping patch-level metrics")
            y_true_patches = None
            y_pred_patches = None
            
        if y_true_patches is not None:
            # Patch-level presence/absence (any positive pixel in patch)
            if len(patch_shape) == 4:  # (B, H, W, 1)
                patch_true = np.any(y_true_patches > 0.5, axis=(1, 2, 3)).astype(np.float32)
                patch_pred_scores = np.max(y_pred_patches, axis=(1, 2, 3))  # Max aggregation
            else:  # (B, H, W)
                patch_true = np.any(y_true_patches > 0.5, axis=(1, 2)).astype(np.float32)
                patch_pred_scores = np.max(y_pred_patches, axis=(1, 2))  # Max aggregation
                
            # Patch-level AUC
            if len(np.unique(patch_true)) > 1:  # Need both positive and negative patches
                patch_auc = roc_auc_score(patch_true, patch_pred_scores)
                
                # Patch-level classification metrics
                patch_pred_binary = (patch_pred_scores > threshold).astype(int)
                patch_precision = precision_score(patch_true, patch_pred_binary, zero_division=0)
                patch_recall = recall_score(patch_true, patch_pred_binary, zero_division=0)
                patch_f1 = f1_score(patch_true, patch_pred_binary, zero_division=0)
            else:
                print("WARNING: All patches have same label, cannot compute patch-level AUC")
                patch_auc = float('nan')

    # Print both metrics for comparison
    print(f"Pixel-level AUC: {pixel_auc:.4f}")
    if patch_auc is not None:
        print(f"Patch-level AUC: {patch_auc:.4f}")

    return {
        'auc': pixel_auc,  # Keep pixel AUC as primary for backward compatibility
        'pixel_auc': pixel_auc,
        'patch_auc': patch_auc,
        # short keys expected by print_summary
        'prec': precision,
        'rec': recall,
        # long keys kept for backwardcompatibility
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'patch_precision': patch_precision,
        'patch_recall': patch_recall,
        'patch_f1': patch_f1,
        'fpr': fpr,
        'tpr': tpr,
        'threshold': threshold
    }


def print_summary(results):
    """
    Print performance for all models in a tabular format.
    """
    print("\n" + "=" * 75)
    print("{:<12} {:<9} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "AUC", "Precision", "Recall", "F1", "TrainSec", "PredSec"
    ))
    print("-" * 75)
    for m in results:
        print("{:<12} {:<9.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.1f} {:<10.1f}".format(
            m["name"],
            m["auc"],
            m["prec"],
            m["rec"],
            m["f1"],
            m["train_time"],
            m["pred_time"]
        ))
    print("=" * 75)


def write_metrics_txt(out_dir: str, metrics: dict, filename: str = 'test_metrics.txt') -> str:
    """Write scalar metrics to a simple text file and return its path

    Writes: AUC, Precision, Recall, F1, Threshold if present in metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w') as f:
        f.write("Test Set Metrics\n")
        f.write("================\n")
        for key in ['auc', 'precision', 'recall', 'f1', 'threshold']:
            if key in metrics:
                f.write(f"{key.capitalize()}: {metrics[key]:.4f}\n")
    print(f"\nTest metrics saved to: {path}")
    return path


def validate_inference_style(model, config, task_config, paths):
    """
    Run inference-style validation on validation chunks with sliding-window averaging.

    This mimics the full inference pipeline (04_inference.py) to compute metrics on
    post-averaging probability maps, matching what Step 5 post-processing will see.

    Returns dict with post-averaging metrics:
        - auc_postproc: AUC on averaged probability maps
        - threshold_postproc: Optimal F1 threshold for post-averaged maps
        - f1_postproc: F1 score at optimal threshold
        - precision_postproc: Precision at optimal threshold
        - recall_postproc: Recall at optimal threshold
    """
    from mlgis_helpers.data_management import read_bands_float32, grid_indices

    print("\n" + "="*60)
    print("INFERENCE-STYLE VALIDATION (post-averaging metrics)")
    print("="*60)

    # Get validation chunks from config
    val_chunks = task_config.get('val_chunks', [])
    if not val_chunks:
        print("WARNING: No val_chunks found in config, skipping inference-style validation")
        return None

    # Get paths from the paths dictionary (already resolved)
    imagery_dir = paths.get('imagery_data_dir')
    labels_path = paths.get('shape_path')

    if not imagery_dir or not os.path.exists(imagery_dir):
        print(f"WARNING: Imagery directory not found at {imagery_dir}, skipping inference-style validation")
        return None

    if not labels_path or not os.path.exists(labels_path):
        print(f"WARNING: Labels file not found at {labels_path}, skipping inference-style validation")
        return None

    # Inference parameters
    patch_size = int(task_config['patch_size'])
    stride_ratio = config['GLOBAL']['stride_ratio']
    stride = int(patch_size * stride_ratio)
    bands = task_config['bands']
    batch_size = config['GLOBAL'].get('batch_size_inf', 32)

    print(f"Validation chunks: {val_chunks}")
    print(f"Patch size: {patch_size}, stride: {stride} (ratio: {stride_ratio})")
    print(f"Bands: {bands}, batch size: {batch_size}")

    # Accumulate predictions across all validation chunks
    all_y_true = []
    all_y_pred = []

    for chunk_name in val_chunks:
        # Find chunk imagery file
        chunk_pattern = f"*{chunk_name}.tif"
        chunk_files = glob.glob(os.path.join(imagery_dir, chunk_pattern))

        if not chunk_files:
            print(f"WARNING: No imagery found for chunk {chunk_name}, skipping")
            continue

        image_path = chunk_files[0]
        print(f"\nProcessing {chunk_name}...")

        # Load imagery
        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            transform, crs = src.transform, src.crs

        # Read and standardize (match 04_inference.py preprocessing)
        image_data = read_bands_float32(image_path, bands)
        image_data = np.where(image_data <= -9999, -1.0, image_data)
        image_data = np.clip(image_data, -1.0, 1.0)

        # Sliding-window inference with averaging
        prob_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.uint16)

        rows = grid_indices(height, patch_size, stride)
        cols = grid_indices(width, patch_size, stride)
        patch_coords = [(r, c) for r in rows for c in cols]

        print(f"  Predicting {len(patch_coords)} patches...")
        for i in tqdm(range(0, len(patch_coords), batch_size), desc=f"  {chunk_name}"):
            batch_coords = patch_coords[i:i + batch_size]
            if not batch_coords:
                continue
            batch = np.array([image_data[r:r+patch_size, c:c+patch_size, :]
                            for r, c in batch_coords], dtype=np.float32)
            preds = model.predict(batch, verbose=0)
            for j, (r, c) in enumerate(batch_coords):
                prob_map[r:r+patch_size, c:c+patch_size] += preds[j, :, :, 0]
                count_map[r:r+patch_size, c:c+patch_size] += 1

        # Average where predicted
        nonzero = count_map > 0
        prob_map[nonzero] = prob_map[nonzero] / count_map[nonzero]

        # Load ground truth labels
        labels_gdf = gpd.read_file(labels_path)
        if labels_gdf.crs != crs:
            labels_gdf = labels_gdf.to_crs(crs)

        # Rasterize labels
        gt_mask = rasterize(
            [(geom, 1) for geom in labels_gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Only evaluate on grid-covered pixels
        mask_eval = nonzero & (count_map > 0)
        y_true_chunk = gt_mask[mask_eval].ravel()
        y_pred_chunk = prob_map[mask_eval].ravel()

        print(f"  Evaluation pixels: {len(y_true_chunk):,} (positives: {y_true_chunk.sum():,}, {100*y_true_chunk.mean():.2f}%)")

        all_y_true.append(y_true_chunk)
        all_y_pred.append(y_pred_chunk)

    if not all_y_true:
        print("ERROR: No validation data collected, skipping inference-style validation")
        return None

    # Concatenate all chunks
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    print(f"\nTotal validation pixels: {len(y_true_all):,}")
    print(f"Positive pixels: {y_true_all.sum():,} ({100*y_true_all.mean():.2f}%)")

    # Compute metrics on post-averaged probabilities
    if len(np.unique(y_true_all)) < 2:
        print("ERROR: Need both positive and negative samples for AUC")
        return None

    auc_postproc = roc_auc_score(y_true_all, y_pred_all)

    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true_all, y_pred_all)
    if thresholds.size > 0:
        f1 = 2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
        idx = int(np.argmax(f1))
        threshold_postproc = float(thresholds[idx])
        f1_postproc = float(f1[idx])
        precision_postproc = float(precision[1:][idx])
        recall_postproc = float(recall[1:][idx])
    else:
        threshold_postproc = 0.5
        f1_postproc = 0.0
        precision_postproc = 0.0
        recall_postproc = 0.0

    print(f"\nPost-averaging metrics:")
    print(f"  AUC: {auc_postproc:.4f}")
    print(f"  Optimal threshold: {threshold_postproc:.4f}")
    print(f"  F1: {f1_postproc:.4f}")
    print(f"  Precision: {precision_postproc:.4f}")
    print(f"  Recall: {recall_postproc:.4f}")
    print("="*60)

    return {
        'auc_postproc': auc_postproc,
        'threshold_postproc': threshold_postproc,
        'f1_postproc': f1_postproc,
        'precision_postproc': precision_postproc,
        'recall_postproc': recall_postproc
    }
