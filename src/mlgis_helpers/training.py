"""
Training utilities for object detection.
Leverages data_loading.py for all data handling.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import random
import subprocess
import sys
import tensorflow as tf
import time

# Configure GPU memory growth for Metal (needs to be before any TF/Keras imports)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth and legacy Keras")
except Exception as e:
    print(f"GPU configuration warning: {e}")
    # Continue - will fall back to CPU if needed
    
import tf_keras as keras
from tf_keras import backend as K
from tf_keras import callbacks
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

#Import our own  utilities (TECHNICAL DEBT: modules should be reorganized so there's no cross-imports)
#The impot of 'evaluation' is particularly bad because that module in turn imports 'training', leading to circular dependencies.
from .evaluation   import calculate_metrics, plot_metric_progress, optimize_threshold
from .model        import get_model

# =============================================================================
# PHASE 2: TFRECORD PIPELINE FUNCTIONS
# =============================================================================

def _parse_tfrecord_fn(proto):
    """Parse TFRecord protocol buffer with proper dtype handling."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    mask = tf.io.parse_tensor(example['mask'], out_type=tf.uint8)
    
    # Set explicit shapes to fix "unknown TensorShape" errors
    # tf.io.parse_tensor reconstructs original shapes but TF graph needs explicit shapes
    image.set_shape([None, None, None])  # (H, W, C) with unknown dimensions
    mask.set_shape([None, None])         # (H, W) with unknown dimensions
    
    # Ensure proper dtypes and binarize mask once
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    mask = tf.where(mask > 0.5, 1.0, 0.0)  # Hard binarize once
    
    # Ensure mask has channel dimension for model compatibility
    mask = tf.expand_dims(mask, axis=-1)  # Always make it (H, W, 1)
    
    return image, mask

def _create_balanced_dataset(train_tfrecord_path, batch_size, shuffle=1024):
    """Creates a balanced dataset with controlled positive/negative patch ratio."""
    AUTOTUNE = tf.data.AUTOTUNE

    # Check if balanced sampling should be enabled (only for mines task)
    
    task_name = os.environ.get('TASK', 'mines')
    use_balanced = (task_name == 'mines')

    if use_balanced:
        print(f"\n--- Building BALANCED Dataset (50% positive patches) ---")

        raw_ds = (tf.data.TFRecordDataset([train_tfrecord_path], num_parallel_reads=AUTOTUNE)
                    .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE))

        # Define positive patch detector
        def is_positive_patch(x, y):
            return tf.reduce_sum(y) > 0

        # Split into positive and negative streams
        pos_ds = raw_ds.filter(is_positive_patch).shuffle(4096).repeat()
        neg_ds = raw_ds.filter(lambda x, y: tf.logical_not(is_positive_patch(x, y))).shuffle(4096).repeat()

        # Sample with 50% positive patches
        pos_frac = 0.5
        ds = tf.data.Dataset.sample_from_datasets(
            [pos_ds, neg_ds],
            weights=[pos_frac, 1.0 - pos_frac]
        )

        # Batch and prefetch
        ds = (ds
                .batch(batch_size, drop_remainder=True)
                .prefetch(AUTOTUNE))

        print(f"  Using balanced sampling with {pos_frac*100:.0f}% positive patches")

    else:
        print(f"\n--- Building Simple Uniform Dataset (no artificial balancing) ---")

        raw_ds = (tf.data.TFRecordDataset([train_tfrecord_path], num_parallel_reads=AUTOTUNE)
                    .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE))

        # Simple uniform sampling - rely on loss weighting only
        ds = (raw_ds
                .shuffle(shuffle, reshuffle_each_iteration=True)
                .repeat()
                .batch(batch_size, drop_remainder=True)
                .prefetch(AUTOTUNE))

    return ds

def _measure_positive_fraction(ds, n_batches=50):
    """Measure actual positive patch fraction in the dataset."""
    print(f"Measuring positive fraction over {n_batches} batches...")
    
    total_patches = tf.constant(0, dtype=tf.int32)
    positive_patches = tf.constant(0, dtype=tf.int32)
    
    for i, (_, y) in enumerate(ds.take(n_batches)):
        # Check each patch: any positive pixel in mask
        batch_positives = tf.reduce_sum(tf.cast(tf.reduce_any(y > 0.5, axis=(1,2)), tf.int32))
        batch_size = tf.shape(y)[0]
        
        positive_patches = positive_patches + batch_positives
        total_patches = total_patches + batch_size
        
        if i < 5:  # Show first few batches for debugging
            print(f"  Batch {i+1}: {batch_positives.numpy()}/{batch_size.numpy()} positive patches")
    
    fraction = tf.cast(positive_patches, tf.float32) / tf.cast(total_patches, tf.float32)
    print(f"Measured positive fraction: {fraction.numpy():.3f} ({positive_patches.numpy()}/{total_patches.numpy()})")
    return fraction

def _verify_band_mapping(ds, bands_config, task_name):
    """Verify band mapping consistency and detect ordering issues."""
    print(f"\n--- Verifying Band Mapping for {task_name} ---")
    print(f"Expected bands: {bands_config} = [NIR, Green, SWIR2] for s2_nir")
    
    # Sample a batch to compute band statistics
    sample_batch = next(iter(ds.take(1)))
    images, _ = sample_batch
    
    # Compute median per band across all pixels in the batch
    band_medians = []
    for b in range(images.shape[-1]):
        band_data = images[:, :, :, b]
        # Use numpy percentile instead of tf.reduce_median (which doesn't exist)
        band_data_np = band_data.numpy().flatten()
        median_val = float(np.percentile(band_data_np, 50))
        band_medians.append(median_val)
        print(f"  Band {b} (index {bands_config[b]}): median = {median_val:.4f}")
    
    # NIR band should typically have higher values than Green over vegetation
    if len(band_medians) >= 2:
        nir_median = band_medians[0]  # First band should be NIR (index 7)
        green_median = band_medians[1]  # Second band should be Green (index 2)
        
        if nir_median > green_median:
            print(f" Band ordering check PASSED: NIR ({nir_median:.4f}) > Green ({green_median:.4f})")
        else:
            print(f" WARNING: Band ordering suspicious: NIR ({nir_median:.4f}) <= Green ({green_median:.4f})")
            print("  This may indicate band mapping issues or non-vegetated areas")
    
    print("Band mapping verification complete")
    return band_medians

def _calculate_optimal_pos_weight(ds, n_batches=100):
    """Calculate optimal pos_weight from pixel-level imbalance in training data."""
    print(f"\n--- Calculating Optimal pos_weight from Pixel Imbalance ---")
    print(f"Sampling {n_batches} batches to measure pixel-level class distribution...")
    
    total_pixels = 0
    total_positive_pixels = 0
    
    for i, (_, y) in enumerate(ds.take(n_batches)):
        # Count pixels across all patches in the batch
        batch_positive_pixels = tf.reduce_sum(tf.cast(y > 0.5, tf.int32))
        batch_total_pixels = tf.reduce_prod(tf.shape(y))
        
        total_positive_pixels += batch_positive_pixels.numpy()
        total_pixels += batch_total_pixels.numpy()
        
        if i < 5:  # Show progress for first few batches
            batch_pos_frac = batch_positive_pixels.numpy() / batch_total_pixels.numpy()
            print(f"  Batch {i+1}: {batch_positive_pixels.numpy():,}/{batch_total_pixels.numpy():,} positive pixels ({batch_pos_frac:.4f})")
    
    # Calculate pixel-level positive fraction
    pixel_pos_fraction = total_positive_pixels / total_pixels
    print(f"\nPixel-level statistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Positive pixels: {total_positive_pixels:,}")
    print(f"  Positive fraction (p): {pixel_pos_fraction:.4f}")
    
    # Calculate optimal pos_weight = (1-p)/p, with reasonable cap
    if pixel_pos_fraction > 0:
        optimal_pos_weight = (1.0 - pixel_pos_fraction) / pixel_pos_fraction

        # Cap at 100 to prevent gradient swamping
        max_weight = 100.0
        if optimal_pos_weight > max_weight:
            print(f"  Raw calculated pos_weight: {optimal_pos_weight:.2f}")
            optimal_pos_weight = max_weight
            print(f"   Capped pos_weight to {max_weight} to prevent gradient swamping")
        else:
            print(f"  Calculated pos_weight: {optimal_pos_weight:.2f}")

        print(f"  (formula: (1-p)/p = {1.0 - pixel_pos_fraction:.4f}/{pixel_pos_fraction:.4f})")

        return optimal_pos_weight, pixel_pos_fraction
    else:
        print("  WARNING: No positive pixels found, using default pos_weight=1.0")
        return 1.0, 0.0


def _make_standard_dataset(val_tfrecord_path, batch_size):
    """Create finite validation dataset from TFRecord."""
    dataset = tf.data.TFRecordDataset([val_tfrecord_path], num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)  # Finite - only one pass through data
    dataset = dataset.prefetch(2)
    return dataset


def log_quantiles(data, label, percentiles=[0.1, 50, 99.9]):
    """Log data quantiles for diagnostic verification.
    
    Args:
        data: numpy array or tensorflow tensor containing image data
        label: string label for the data source
        percentiles: list of percentiles to compute
    """
    if tf.is_tensor(data):
        data = data.numpy()
    
    print(f"\n--- {label.upper()} DATA QUANTILES ---")
    
    if len(data.shape) == 4:  # Batched data (N, H, W, C)
        for band in range(data.shape[-1]):
            band_data = data[:, :, :, band]
            band_percentiles = np.percentile(band_data, percentiles)
            print(f"  Band {band}: [{band_percentiles[0]:.6f}, {band_percentiles[1]:.6f}, {band_percentiles[2]:.6f}] ({percentiles[0]}%, {percentiles[1]}%, {percentiles[2]}%)")
    elif len(data.shape) == 3:  # Single image (H, W, C)  
        for band in range(data.shape[-1]):
            band_data = data[:, :, band]
            band_percentiles = np.percentile(band_data, percentiles)
            print(f"  Band {band}: [{band_percentiles[0]:.6f}, {band_percentiles[1]:.6f}, {band_percentiles[2]:.6f}] ({percentiles[0]}%, {percentiles[1]}%, {percentiles[2]}%)")
    else:
        # Flattened data
        data_percentiles = np.percentile(data, percentiles)
        print(f"  All data: [{data_percentiles[0]:.6f}, {data_percentiles[1]:.6f}, {data_percentiles[2]:.6f}] ({percentiles[0]}%, {percentiles[1]}%, {percentiles[2]}%)")
    
    print(f"  Data shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min/Max: [{data.min():.6f}, {data.max():.6f}]")


def apply_enhanced_data_cleaning(X, y, nodata_value=-9999):
    """Fix NODATA values BEFORE clipping to prevent data contamination."""
    # FIRST replace NODATA values, THEN clip
    X_cleaned = np.where(X <= nodata_value, 0.0, X)  # Replace NODATA with 0
    X_cleaned = np.clip(X_cleaned, -1.0, 1.0)  # Then clip to valid range
    
    # Verify cleaning worked
    if np.any(X_cleaned <= -9999):
        remaining = (X_cleaned <= -9999).sum()
        raise ValueError(f"FATAL: {remaining} NODATA values remain after cleaning!")
    
    print(f"Data cleaning: range [{X_cleaned.min():.3f}, {X_cleaned.max():.3f}]")
    return X_cleaned, y


def get_task_params(config, task):
    """Get all task-specific parameters from config.
    Parameters are now in the TASKS section.
    """
    if task not in config['TASKS']:
        raise ValueError(f"Task '{task}' not found in TASKS section")
    
    task_config = config['TASKS'][task]
    
    # Required fields
    params = {
        'neg_ratio': task_config['neg_ratio'],
        'pos_weight': task_config['pos_weight'],
        'loss_function': task_config['loss_function']
    }
    
    # Optional fields (pass through if present)
    if 'architecture' in task_config:
        params['architecture'] = task_config['architecture']
    if 'hparam_overrides' in task_config:
        params['hparam_overrides'] = task_config['hparam_overrides']
    
    return params


def log_gpu_usage():
    """Log current GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            used, total, util = result.stdout.strip().split(', ')
            used_gb = float(used) / 1024
            total_gb = float(total) / 1024
            percent = float(used) / float(total) * 100
            print(f"GPU Memory: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%), Utilization: {util}%")
            return True
    except Exception as e:
        # nvidia-smi not available or error
        pass
    return False


class GPUMonitor(callbacks.Callback):
    """Callback to monitor GPU usage during training"""
    
    def __init__(self, frequency=5):
        """
        Args:
            frequency: Log GPU stats every N epochs (default: 5)
        """
        super().__init__()
        self.frequency = frequency
        self.has_gpu = False
        
    def on_train_begin(self, logs=None):
        """Check if GPU monitoring is available"""
        self.has_gpu = log_gpu_usage()
        if not self.has_gpu:
            print("GPU monitoring not available (nvidia-smi not found or no GPU)")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log GPU usage at specified frequency"""
        if self.has_gpu and (epoch + 1) % self.frequency == 0:
            print(f"\n--- Epoch {epoch + 1} GPU Stats ---")
            log_gpu_usage()
            
    def on_train_batch_end(self, batch, logs=None):
        """Log GPU usage for first batch to check initial memory allocation"""
        # Only log first batch stats at the same frequency as epoch stats
        epoch = getattr(self, '_current_epoch', 0)
        if self.has_gpu and batch == 0 and (epoch + 1) % self.frequency == 0:
            print("\n--- First Batch GPU Stats ---")
            log_gpu_usage()
    
    def on_epoch_begin(self, epoch, logs=None):
        """Track current epoch for batch callback"""
        self._current_epoch = epoch


def safe_predict(model, X, **kwargs):
    """Wrapper to handle XLA prediction issues and ensure numeric output"""
    pred = model.predict(X, **kwargs)
    
    # Ensure we have a numpy array
    if hasattr(pred, 'numpy'):
        pred = pred.numpy()
    elif not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    
    # Force conversion to float32 to avoid any string/object dtype issues
    try:
        pred = pred.astype(np.float32)
    except (ValueError, TypeError):
        # Try to extract numeric values if possible
        if pred.size > 0:
            try:
                # Handle case where predictions might be wrapped in some way
                if pred.dtype == object:
                    # Try to extract floats from object array
                    pred_list = []
                    for item in pred.flat:
                        if hasattr(item, 'numpy'):
                            pred_list.append(float(item.numpy()))
                        else:
                            pred_list.append(float(item))
                    pred = np.array(pred_list).reshape(pred.shape).astype(np.float32)
                else:
                    # Last resort - convert to string then float
                    pred = np.array([float(str(x)) for x in pred.flat]).reshape(pred.shape)
            except Exception:
                # Return zeros as fallback
                pred = np.zeros_like(X[..., 0], dtype=np.float32)
    
    return pred




# Constants 
PROFILE_BATCH = "100, 105"  # profiler step window for TensorBoard
VAL_SPLIT = 0.2             # validation split ratio


def collect_val_metrics(model, val_ds, val_steps, batch_size):
    """Collect validation metrics efficiently in a single pass."""

    print(f"\nComputing final validation metrics...")
    start_time = time.time()

    # Initialize TF metrics for pixel-level computation
    auc_metric = tf.keras.metrics.AUC()
    prec_metric = tf.keras.metrics.Precision()
    rec_metric = tf.keras.metrics.Recall()
    patch_truth, patch_scores = [], []

    # Single pass through validation data
    for x_batch, y_batch in val_ds.take(val_steps):
        y_pred = model.predict_on_batch(x_batch)
        y_true_flat = tf.reshape(y_batch, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Update pixel-level metrics
        auc_metric.update_state(y_true_flat, y_pred_flat)
        prec_metric.update_state(y_true_flat, y_pred_flat)
        rec_metric.update_state(y_true_flat, y_pred_flat)

        # Collect patch-level data for threshold optimization
        patch_truth.append(tf.reduce_any(y_batch > 0.5, axis=(1, 2, 3)).numpy().astype(np.float32))
        patch_scores.append(tf.reduce_max(y_pred, axis=(1, 2, 3)).numpy())

    # Concatenate patch arrays
    patch_truth = np.concatenate(patch_truth)
    patch_scores = np.concatenate(patch_scores)

    # Optimize threshold on patch-level data
    opt_thr, opt_f1, opt_prec, opt_rec = optimize_threshold(patch_truth, patch_scores)
    fpr, tpr, _ = roc_curve(patch_truth, patch_scores)

    elapsed = time.time() - start_time
    auc = auc_metric.result().numpy()
    print(f"  Computed metrics in {elapsed:.1f}s - AUC={auc:.4f}, F1={opt_f1:.4f}")

    return {
        "auc": auc,
        "precision": opt_prec,  # Use optimized precision
        "recall": opt_rec,      # Use optimized recall
        "f1": opt_f1,
        "threshold": opt_thr,
        "fpr": fpr,
        "tpr": tpr,
    }

# --- I) Loss function -------------------------------------------------------------
# Legacy loss functions have been moved to training_legacy.py
# We now only use weighted BCE for simplicity and stability



def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33333):
    """
    Focal Tversky Loss for extreme class imbalance.
    =0.7, =0.3 penalizes false negatives more than false positives.
    =4/3 focuses the model on hard-to-classify examples.
    """
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        
        tversky = true_pos / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
        focal_tversky = tf.pow((1 - tversky), gamma)
        
        return focal_tversky
    return _loss


def weighted_bce_loss(pos_weight):
    """
    Weighted binary cross-entropy loss function with NODATA masking and improved numerical stability.
    Returns a function that Keras can use as loss.
    """
    pw_const = tf.constant(pos_weight)

    def _loss(y_true, y_pred, sample_weight=None):
        # Force float32 throughout for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Create valid pixel mask to exclude NODATA pixels from loss computation
        # This needs access to the input image, but in loss functions we only get y_true, y_pred
        # For now, we assume all pixels are valid, but this could be enhanced by passing
        # sample_weight with valid pixel masks from the training loop
        valid_mask = tf.ones_like(y_true, dtype=tf.float32)
        if sample_weight is not None:
            valid_mask = tf.cast(sample_weight, tf.float32)

        # Stronger clipping to prevent log(0) - increased from 1e-7 to 1e-6
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)

        # Use tf.nn.sigmoid_cross_entropy_with_logits for better numerical stability
        # First convert predictions back to logits
        logits = tf.math.log(y_pred / (1.0 - y_pred))
        
        # Compute stable BCE using TF's optimized function
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,
            logits=logits
        )
        
        # Reshape for proper broadcasting
        bce = tf.reshape(bce, tf.shape(y_true)[:-1])
        y_true_sq = tf.reshape(y_true, tf.shape(y_true)[:-1])
        valid_mask_sq = tf.reshape(valid_mask, tf.shape(y_true)[:-1])
        
        # Apply class weights
        class_weights = tf.where(
            tf.equal(y_true_sq, 1.0),
            tf.cast(pw_const, tf.float32),
            tf.ones_like(y_true_sq, dtype=tf.float32)
        )
        
        # Apply valid pixel mask to exclude NODATA from loss computation
        weighted_bce = bce * class_weights * valid_mask_sq
        
        # Only compute mean over valid pixels
        total_valid = tf.reduce_sum(valid_mask_sq)
        total_loss = tf.reduce_sum(weighted_bce)
        
        # Avoid division by zero if no valid pixels
        return tf.cond(
            total_valid > 0,
            lambda: total_loss / total_valid,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
    
    return _loss






class WeightDiff(callbacks.Callback):
    """Track weight changes to detect if optimizer is applying gradients."""
    def __init__(self):
        super().__init__()
        self.w0 = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.w0 = [w.numpy().copy() for w in self.model.trainable_weights]
        
    def on_epoch_end(self, epoch, logs=None):
        if self.w0 is not None:
            deltas = [np.abs(w1 - w0).mean() for w0, w1 in zip(self.w0, self.model.trainable_weights)]
            mean_delta = float(np.mean(deltas))
            # print(f"Mean |w|: {mean_delta:.8f}")

# --- II) Main training function -------------------------------------

def train_model(
    config: dict, # Configuration dictionary, loaded from config.yaml outside the src codebase
    out_dir: str, # Path to output folder for model checkpoints and logs
    task: str, # Task name for band selection
    *, #Force the following arguments to be keyword-only
    img_patches: str | None = None,  # TFRecord path for training data
    val_img_patches: str | None = None,  # TFRecord path for validation data
    paths: dict | None = None,  # Platform-specific paths dictionary
    patch_size: int | None = None,  # Patch size for display purposes
    pretrained_model_path: str | None = None,  # Path to pre-trained model file
):
    """
    Train a UNet segmentation model using TFRecord pipeline.

    Parameters
    ----------
    img_patches : str
        Path to training TFRecord file
    val_img_patches : str  
        Path to validation TFRecord file

    Returns
    -------
    dict
        {
            "model":    keras.Model,
            "metrics":  dict,
            "progress": dict
        }
    """
    
    # Get band configuration for this task from TASKS section
    bands = config["TASKS"][task]["bands"]
    num_channels = len(bands)
    
    # TFRecord-only pipeline (streaming is deprecated)
    train_tfrecord_path = img_patches
    val_tfrecord_path = val_img_patches
    
    print(f"Train TFRecord: {train_tfrecord_path}")
    print(f"Val TFRecord: {val_tfrecord_path}")
    
    # Create balanced TFRecord datasets with explicit 2-stream sampling
    train_ds = _create_balanced_dataset(train_tfrecord_path, config["GLOBAL"]["batch_size"])
    val_ds = _make_standard_dataset(val_tfrecord_path, config["GLOBAL"]["batch_size"])
    
    # Measure actual positive fraction (natural distribution)
    measured_frac = _measure_positive_fraction(train_ds, n_batches=20)
    print(f"Natural positive fraction: {measured_frac.numpy():.3f} (no artificial balancing)")
    
    # Verify band mapping consistency
    task_config = config['TASKS'][task]
    bands_list = task_config['bands']
    _verify_band_mapping(val_ds, bands_list, task)
    
    # Calculate optimal pos_weight from pixel-level imbalance
    optimal_pos_weight, pixel_pos_frac = _calculate_optimal_pos_weight(train_ds, n_batches=50)
    config_pos_weight = task_config.get('pos_weight', 3.0)
    
    print(f"\npos_weight configuration:")
    print(f"  Config pos_weight: {config_pos_weight}")
    print(f"  Calculated pos_weight: {optimal_pos_weight:.2f} (from pixel imbalance)")

    # Use the CALCULATED pos_weight
    print(f"  Using CALCULATED pos_weight: {optimal_pos_weight:.2f}")
    task_config['pos_weight'] = optimal_pos_weight  # Store for later use
    
    # Calculate steps_per_epoch for TFRecord mode to prevent "ran out of data" error
    # Count total records in training TFRecord to ensure sufficient batches
    raw_train_count = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord_path))
    steps_per_epoch = max(10, raw_train_count // config["GLOBAL"]["batch_size"])
    print(f"\nTFRecord training configuration:")
    print(f"  Training patches: {raw_train_count}")
    print(f"  Batch size: {config['GLOBAL']['batch_size']}")  
    print(f"  Steps per epoch: {steps_per_epoch}")
    
        
    # --- QUANTILE AUDIT: Verify TFRecord pipeline matches Aug-11 baseline ---
    print("\n=== QUANTILE AUDIT: TFRecord Pipeline Distribution ===")
    
    # Sample first batch from balanced training dataset for audit
    audit_batch = next(iter(train_ds))
    train_images_audit, train_masks_audit = audit_batch
    log_quantiles(train_images_audit, "TFRecord Balanced Train Batch", percentiles=[0.1, 50, 99.9])
    
    # Count positive vs negative patches in the audit batch
    audit_pos_patches = tf.reduce_sum(tf.cast(tf.reduce_any(train_masks_audit > 0, axis=[1,2]), tf.int32))
    audit_neg_patches = tf.shape(train_masks_audit)[0] - audit_pos_patches
    actual_pos_percent = audit_pos_patches.numpy() / tf.shape(train_masks_audit)[0].numpy() * 100
    print(f"  Patch balance in audit batch: {audit_pos_patches.numpy()} positive, {audit_neg_patches.numpy()} negative ({actual_pos_percent:.1f}% positive)")
    print(f"  Using natural patch distribution with pos_weight={optimal_pos_weight:.2f} loss weighting")
        
    # Verify underlying TFRecord has sufficient positive patches
    print("  Checking source TFRecord patch distribution...")
    raw_train_ds = tf.data.TFRecordDataset(train_tfrecord_path).map(_parse_tfrecord_fn)
    total_patches = 0
    total_positive = 0
    for _, mask in raw_train_ds.take(500):  # Sample to avoid memory issues
        is_positive = tf.reduce_any(mask > 0)
        total_patches += 1
        if is_positive:
            total_positive += 1
    source_pos_percent = (total_positive / total_patches) * 100
    print(f"  Source TFRecord distribution: {total_positive}/{total_patches} positive ({source_pos_percent:.1f}%)")
    
    # Sample validation batch for comparison
    val_audit_batch = next(iter(val_ds))
    val_images_audit, val_masks_audit = val_audit_batch
    log_quantiles(val_images_audit, "TFRecord Validation Batch", percentiles=[0.1, 50, 99.9])
    print("=== END QUANTILE AUDIT ===\n")
    
    # --- BATCH/STEP SIZING FOR INFINITE DATASETS ---
    def _count_examples(tfrecord_path: str) -> int:
        c = 0
        for _ in tf.data.TFRecordDataset(tfrecord_path):
            c += 1
        return c

    train_examples = _count_examples(train_tfrecord_path)
    val_examples   = _count_examples(val_tfrecord_path)
    bs = int(config["GLOBAL"]["batch_size"])

    # Use floor division since we're using drop_remainder=True 
    steps_per_epoch = max(1, train_examples // bs)
    val_steps       = max(1, val_examples   // bs)

    # Make quick mode actually quick
    if config.get("quick_mode", False):
        steps_per_epoch = min(steps_per_epoch, 60)
        val_steps       = min(val_steps, 30)

    print(f"Steps per epoch (train): {steps_per_epoch} | validation steps: {val_steps}")
    
    # Use TFRecord streaming for both training and validation (memory efficient)
    print("\n--- Using TFRecord streaming for validation (memory efficient) ---")
    
    # DIAGNOSTIC: Verify TFRecord parsing consistency  
    print("\n--- DIAGNOSTIC: Verifying TFRecord parsing consistency ---")
    sample_count = 0
    for image, mask in val_ds.take(1):  # Take 1 batch 
        log_quantiles(image.numpy(), "tfrecord_batch_sample")
        sample_count += image.shape[0]
        break
    print(f"Verified {sample_count} samples from TFRecord batch")
    
    # Use TFRecord streaming for both training and validation (no in-memory conversion)
    X_train = y_train = None
    X_val = y_val = None

    # TFRecord data is already cleaned and validated during preprocessing

    # Get task-specific parameters including architecture
    task_params = get_task_params(config, task)
    architecture = task_params['architecture']  # Architecture must be defined in task config

    # Strict enforcement for ResNet50 architecture
    if architecture == 'resnet50_unet' and num_channels != 3:
        raise ValueError(
            f"ResNet50 architecture requires exactly 3 bands (RGB), but got {num_channels} bands: {bands}. "
            f"Please update config.yaml to use exactly 3 bands for task '{task}'"
        )

    # Print all hyperparameters
    print("\n" + "="*60)
    print("MODEL AND TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model architecture: {architecture}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Batch size: {config['GLOBAL']['batch_size']}")
    print(f"Base Positive weight: {task_params['pos_weight']}")
    print(f"Number of epochs: {config['GLOBAL']['num_epochs']}")
    print(f"Patience: {config['GLOBAL']['patience']}")
    print(f"Negative downsampling: {config['GLOBAL']['neg_downsample']} (ratio: {task_params['neg_ratio']})")
    print(f"Input shape: {patch_size}x{patch_size}x{num_channels}")
    print(f"Selected bands: {bands}")
    print("="*60 + "\n")

    input_shape = (
        patch_size, # Squared patches of 'patch_size' pixels per side
        patch_size,
        num_channels, # Number of channels based on selected bands
    )
    
    # Load pre-trained model if provided, otherwise build new model
    if pretrained_model_path:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pre-trained model not found: {pretrained_model_path}")
        
        print(f"Loading pre-trained model from: {pretrained_model_path}")
        model = keras.models.load_model(pretrained_model_path, compile=False)
        print(f"Successfully loaded pre-trained model")
        
        # Verify model input shape matches expected configuration
        expected_shape = (patch_size, patch_size, num_channels)
        if model.input_shape[1:] != expected_shape:
            raise ValueError(f"Pre-trained model input shape {model.input_shape[1:]} does not match expected {expected_shape}")
    else:
        model = get_model(config, paths=paths, architecture=architecture)
        print(f"Building new model: {architecture}...")

    # Print model summary
    print(f"Model parameters: {model.count_params():,}")

    print(f"\n--- Applying Hyperparameters for Architecture: {architecture} ---")

    # 1. Load the default hparams for that architecture
    hparams = config['GLOBAL']['ARCHITECTURE_HPARAMS'][architecture].copy()
    print(f"Loaded default hparams: {hparams}")

    # DEBUG: Show task_params keys to understand why override detection fails
    print(f"DEBUG task_params keys: {list(task_params.keys())}")
    print(f"DEBUG 'hparam_overrides' in task_params: {'hparam_overrides' in task_params}")

    # 2. Check for and apply any task-specific overrides
    if 'hparam_overrides' in task_params:
        overrides = task_params['hparam_overrides']
        hparams.update(overrides)
        print(f"Applied task-specific overrides: {overrides}")
    else:
        print("No task-specific overrides found")

    # 3. Get the final learning rate from the processed hparams
    learning_rate = float(hparams['learning_rate'])
    print(f"Final learning rate for this run: {learning_rate}")
    
    # Use the calculated pos_weight from pixel imbalance (updated earlier in TFRecord pipeline)
    pos_weight = task_params.get('pos_weight', 3.0)  # Falls back to config value if not calculated
    
    # A learning rate schedule is required for stable Transformer fine-tuning.
    using_lr_schedule = False
    if architecture.startswith('segformer'):
        print("Applying Transformer fine-tuning protocol: AdamW with learning rate decay.")
        
        total_steps = steps_per_epoch * config['GLOBAL']['num_epochs']
        
        # Create a schedule that linearly decays the learning rate from its peak to zero.
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=float(learning_rate),
            decay_steps=int(total_steps),
            end_learning_rate=0.0,
            power=1.0  # power=1.0 is a linear decay
        )
        
        # Use the same keras that we imported (tf_keras)
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.05,
            clipnorm=1.0  # Gradient clipping is crucial
        )
        using_lr_schedule = True
        print(f"Optimizer: AdamW with linear LR decay ({learning_rate}  0.0 over {total_steps} steps, clipnorm=1.0)")
    else:
        # Keep the existing simple optimizer for UNet
        print(f"Using standard AdamW optimizer with fixed learning rate: {learning_rate}")
        # Use the same keras that we imported (tf_keras)
        optimizer = keras.optimizers.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=0.05,
            clipnorm=1.0
        )
        print(f"Optimizer: AdamW(lr={learning_rate}, weight_decay=0.05, clipnorm=1.0)")
    
    print(f"Positive Weight: {pos_weight}")
    
    # Aug-11 Baseline Verification
    original_pos_weight = 3.0
    if abs(pos_weight - original_pos_weight) < 0.001:
        print(f" VERIFIED: pos_weight={pos_weight} matches Aug-11 baseline (AUC=0.9851)")
    else:
        print(f" NOTE: pos_weight={pos_weight} differs from Aug-11 baseline (pos_weight={original_pos_weight})")
        print(f"  This is expected when using calculated pos_weight from current data distribution.")
    
    # Use the existing weighted BCE loss
    loss_fn = weighted_bce_loss(pos_weight)
    print(f"Using weighted_bce for {task} with pos_weight={pos_weight}")
    
    
    # Disable XLA to eliminate first-step compile delay
    print("Compiling model (XLA disabled to prevent first-step blocking)...")
    model.compile( # Compile method from Keras initializes the model with the optimizer, loss function, and metrics
        optimizer=optimizer,
        loss=loss_fn,  # Now uses the selected loss function
        metrics=[keras.metrics.AUC(name="auc")],
        jit_compile=False  # Disable XLA compilation
    )
    
    # Log initial GPU usage after model compilation
    print("\n--- Initial GPU Stats (After Model Compilation) ---")
    log_gpu_usage()
    
    # === DIAGNOSTIC BLOCK 1: Confirm something is trainable ===
    print("\n" + "="*60)
    print("DIAGNOSTIC: TRAINABLE PARAMETERS")
    print("="*60)
    n_vars = sum(tf.size(v).numpy() for v in model.trainable_variables)
    n_frozen = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(f"Trainable: {len(model.trainable_variables)} tensors ({n_vars:,} scalars)")
    print(f"Frozen: {len(model.non_trainable_variables)} tensors ({n_frozen:,} scalars)")
    print(f"Total parameters: {model.count_params():,}")
    
    if n_vars == 0:
        print("CRITICAL ERROR: No trainable parameters! Model is completely frozen!")
        return {"model": None, "metrics": {}, "progress": {}}
    else:
        print("Model has trainable parameters")
    print("="*60)
    
    # DEBUG: Check trainable parameters
    if config['GLOBAL']['debug_mode']:
        n_vars = sum(tf.size(v).numpy() for v in model.trainable_variables)
        n_frozen = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
        print(f"\nDEBUG MODEL INFO:")
        print(f"   Trainable: {len(model.trainable_variables)} tensors ({n_vars:,} scalars)")
        print(f"   Frozen: {len(model.non_trainable_variables)} tensors ({n_frozen:,} scalars)")
        print(f"   Total parameters: {model.count_params():,}")

    # Load checkpoint if resuming
    initial_epoch = 0
    if 'resume_from' in config['GLOBAL']:  # Only check existence, no default
        checkpoint_path = config['GLOBAL']['resume_from']
        if os.path.exists(checkpoint_path):
            print(f"\n--- RESUMING FROM CHECKPOINT ---")
            print(f"Loading weights from: {checkpoint_path}")
            model.load_weights(checkpoint_path)
            
            # Try to determine the epoch from the checkpoint path or filename
            # For now, we'll start from epoch 0 but you could enhance this
            # to store epoch info in a separate file alongside the checkpoint
            print(f"Weights loaded successfully. Training will continue from epoch {initial_epoch}")
            print(f"Note: Optimizer state is not restored, only model weights")

    # TFRecord mode (always true now):
    if X_val is not None:
        test_pred = safe_predict(model, X_val[:min(5, len(X_val))], verbose=0) # Initial predictions (sanity check)
        if isinstance(test_pred, np.ndarray):
            min_val = float(test_pred.min())
            max_val = float(test_pred.max())
            print(f"Initial predictions: [{min_val:.4f}, {max_val:.4f}]") #Helps debugging
        else:
            print(f"Initial predictions: {test_pred} (type: {type(test_pred)})") # Debug unexpected type
    elif X_train is not None:
            test_pred = safe_predict(model, X_train[:min(5, len(X_train))], verbose=0)
            if isinstance(test_pred, np.ndarray):
                min_val = float(test_pred.min())
                max_val = float(test_pred.max())
                print(f"Initial predictions (train): [{min_val:.4f}, {max_val:.4f}]")
            else:
                print(f"Initial predictions (train): {test_pred} (type: {type(test_pred)})")
    else:
        print("Skipping initial prediction sanity check (TFRecord mode)")

    # === DIAGNOSTIC BLOCK 2: Smoke-test the optimizer ===
    print("\n" + "="*60)
    print("DIAGNOSTIC: GRADIENT TAPE SMOKE TEST")
    print("="*60)
    try:
        with tf.GradientTape() as tape:
            if X_train is None:
                # Get one batch from streaming dataset or TFRecord
                x, y = next(iter(train_ds.take(1)))
            else:
                # Use first batch from memory
                batch_size = config["GLOBAL"]["batch_size"]
                x = X_train[:batch_size]
                y = y_train[:batch_size]
                if len(y.shape) == 3:  # Add channel dimension if needed
                    y = y[..., np.newaxis]
                x = tf.constant(x, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.float32)
            
            y_hat = model(x, training=True)
            loss = model.compiled_loss(y, y_hat)
        
        grads = tape.gradient(loss, model.trainable_variables)
        g_norm = tf.linalg.global_norm(grads)
        
        print(f"Loss value: {loss.numpy():.6f}")
        print(f"Gradient norm: {g_norm.numpy():.6f}")
        
        if g_norm.numpy() == 0:
            print("CRITICAL ERROR: Gradient norm is 0! Gradients not flowing!")
        elif not tf.math.is_finite(g_norm):
            print("CRITICAL ERROR: Gradient norm is NaN/Inf!")
        else:
            print("Gradients are flowing correctly")
            
        # Check individual gradient statistics
        grad_stats = []
        for i, grad in enumerate(grads):
            if grad is not None:
                grad_mean = tf.reduce_mean(tf.abs(grad))
                grad_stats.append(grad_mean.numpy())
        
        if grad_stats:
            print(f"Mean |gradient|: {np.mean(grad_stats):.8f}")
            print(f"Gradient range: [{np.min(grad_stats):.8f}, {np.max(grad_stats):.8f}]")
        
    except Exception as e:
        print(f"ERROR in gradient test: {e}")
    print("="*60)

    cb_list, tracking = _setup_callbacks(model, X_val, y_val, out_dir, config, learning_rate, val_ds=val_ds, using_lr_schedule=using_lr_schedule) # Callbacks (useful functions that run at certain points during training, e.g. early stopping, model checkpointing, etc.)

    print(f"Training for {config['GLOBAL']['num_epochs']} epochs...")
    train_start = time.time()
    
    # Calculate steps per epoch for streaming mode
    fit_kwargs = {
        "epochs": config["GLOBAL"]["num_epochs"],
        "callbacks": cb_list,
        "verbose": 2,  # 2 = one line per epoch
        "initial_epoch": initial_epoch,  # For checkpoint resuming
    }
    
    if X_train is None and 'steps_per_epoch' in locals():
        # TFRecord mode: use calculated steps
        fit_kwargs["steps_per_epoch"] = steps_per_epoch
        if X_val is None:  # Streaming validation
            fit_kwargs["validation_steps"] = val_steps
            print(f"TFRecord mode: {steps_per_epoch} steps/epoch, validation_steps={val_steps} (streaming validation)")
        else:  # In-memory validation
            print(f"TFRecord mode: {steps_per_epoch} steps/epoch, validation_steps=None (using in-memory validation)")
    
    # TFRecord steps calculation was already done earlier in the function
    
    # <<< STREAMING-COMPATIBLE DIAGNOSTIC BLOCK >>>
    if val_ds:
        print("\n--- Performing VALIDATION SET PIXEL CHECK (Streaming) ---")
        n_batches_to_check = 200  # We will inspect the first 200 batches
        total_pos_pixels = 0
        total_pixels = 0

        # Take a sample from the validation dataset to inspect its contents
        for image, mask in val_ds.take(n_batches_to_check):
            total_pixels += tf.size(mask, out_type=tf.int64).numpy()
            total_pos_pixels += tf.reduce_sum(tf.cast(mask > 0, tf.int64)).numpy()

        total_neg_pixels = total_pixels - total_pos_pixels
        
        print(f"PIXEL CHECK on first {n_batches_to_check} batches: pos:{total_pos_pixels:,}  neg:{total_neg_pixels:,}")
        
        # FATAL: Kill if validation is constant
        if total_pos_pixels == 0 or total_neg_pixels == 0:
            raise ValueError(f"FATAL: Validation set is constant! pos:{total_pos_pixels} neg:{total_neg_pixels}")
    # <<< END DIAGNOSTIC BLOCK >>>
    
    # Single training phase for all architectures
    if X_val is not None and y_val is not None:
        # Use the in-memory NumPy arrays for validation. This is consistent and required by your callbacks.
        print("Using in-memory NumPy arrays for validation data.")
        fit_kwargs.pop("validation_steps", None)  # NumPy arrays do not use 'steps'.
        validation_data_source = (X_val, y_val)
    else:
        # Fallback to the dataset object if NumPy arrays were not created (e.g., streaming mode).
        print("Using tf.data.Dataset for validation data.")
        validation_data_source = val_ds

    # Full training with proper configuration
    history = model.fit(
        train_ds,
        validation_data=validation_data_source,
        **fit_kwargs
    )

    train_time = time.time() - train_start
    
    # Log final GPU usage after training
    print("\n--- Final GPU Stats (After Training) ---")
    log_gpu_usage()
    
    return _save_final_results( # Save final metrics and plots to 'out_dir' and return a summary dictionary
        model, X_val, y_val, history, tracking, out_dir, config, train_time,
        val_ds=val_ds if X_val is None else None,  # Use streaming if no in-memory validation
        val_steps=val_steps if X_val is None else None  # Use streaming steps if no in-memory validation
    )


# --- III) Helper functions -----------------------------------------------------

def _make_tb_callback(out_dir: str, run_name: str, *, profile_batch: str = "100, 105"):
    """
    Create a TensorBoard callback and return it together with its log directory.

    Parameters
    ----------
    out_dir : str
        Root output directory for this training run.
    run_name : str
        Subfolder prefix (e.g. "unet") so different models dont overwrite logs.
    profile_batch : str, optional
        Profiling window for TensorBoard (e.g. "100, 105"  profile steps 100105).

    Returns
    -------
    tuple[callbacks.TensorBoard, str]
        (tensorboard_callback, log_dir)
    """
    log_dir = os.path.join(
        out_dir,
        "logs",
        run_name,
        time.strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    tb_cb = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        profile_batch=profile_batch,
        update_freq="epoch",
    )
    return tb_cb, log_dir

def _augment_data(x, y):
    """Aggressive data augmentation to prevent overfitting."""
    # Move augmentation to GPU for better performance
    with tf.device('/GPU:0'):
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y)
        
        # Random vertical flip  
        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_up_down(x)
            y = tf.image.flip_up_down(y)
        
        # Random rotation (90 degree increments)
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        x = tf.image.rot90(x, k)
        y = tf.image.rot90(y, k)
        
        # Random brightness and contrast (only on image, not mask)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        x = tf.clip_by_value(x, -1.0, 1.0)  # Keep in valid [-1, 1] range
    
    # Add Gaussian noise for robustness to historical imagery
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.05)
        x = tf.clip_by_value(x + noise, -1.0, 1.0)
    
    return x, y

def _generate_patience_curve(history, epoch_times, out_dir, config):
    """Generate patience curve showing AUC vs runtime for different patience levels.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        The history object from model.fit()
    epoch_times : list
        List of wall-clock time in seconds for each epoch
    out_dir : str
        Output directory for saving the plot
    config : dict
        Configuration dictionary
    """
    val_auc_history = history.history.get('val_auc', [])
    if not val_auc_history:
        print("No validation AUC history available for patience curve")
        return
    
    max_patience = config['GLOBAL']['patience']
    
    # Simulation Logic
    patience_values = []
    stop_times = []
    best_aucs = []
    stop_epochs = []
    
    print(f"\nSimulating EarlyStopping for patience levels 1 to {max_patience}")
    for patience in range(1, max_patience + 1):
        wait_counter = 0
        best_auc_so_far = 0
        best_epoch = 0
        
        # Find where training would have stopped
        stop_epoch = len(val_auc_history)
        for epoch, current_auc in enumerate(val_auc_history):
            if current_auc > best_auc_so_far:
                best_auc_so_far = current_auc
                best_epoch = epoch
                wait_counter = 0
            else:
                wait_counter += 1
            
            if wait_counter >= patience:
                stop_epoch = epoch + 1  # Stop after this epoch
                break
        
        # The final AUC is the best one found up to the point of stopping
        final_auc = max(val_auc_history[:stop_epoch])
        
        # The runtime is the cumulative time at that epoch (times are already cumulative)
        # epoch_times[i] contains the cumulative time at the end of epoch i+1
        final_runtime = epoch_times[stop_epoch - 1] if stop_epoch > 0 else 0
        
        patience_values.append(patience)
        stop_times.append(final_runtime)
        best_aucs.append(final_auc)
        stop_epochs.append(stop_epoch)
    
    print("Simulation complete.")
    
    # Plotting Logic - clean style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the trajectory line - simple blue line
    ax.plot(stop_times, best_aucs, marker='o', linestyle='-', color='#1f77b4', 
            linewidth=2, markersize=4, label='Performance Trajectory')
    
    # Highlight the actual patience used with red star
    actual_idx = max_patience - 1
    ax.plot(stop_times[actual_idx], best_aucs[actual_idx], 
           color='red', marker='*', markersize=12, 
           label=f'Actual (p={max_patience})')
    
    # Add annotations for specific patience values
    # Annotate p=1,2,3,4,5 then every 5: p=10,15,20,...
    annotate_points = []
    for p in range(1, 6):  # p=1,2,3,4,5
        if p <= max_patience:
            annotate_points.append(p)
    
    for p in range(10, max_patience + 1, 5):  # p=10,15,20,... up to max_patience
        annotate_points.append(p)
    
    # Add the actual patience if not already included
    if max_patience not in annotate_points:
        annotate_points.append(max_patience)
    
    # Apply annotations
    for p in annotate_points:
        if p <= len(patience_values):
            idx = p - 1  # Convert to 0-based index
            ax.text(stop_times[idx], best_aucs[idx] + 0.0002, f'p={p}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Clean formatting
    ax.set_title('Patience Curve')
    ax.set_xlabel('Total Training Time (seconds)')
    ax.set_ylabel('Best Validation AUC')
    ax.legend()
    
    # Save the artifact
    output_path = os.path.join(out_dir, 'patience_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nArtifact generated and saved to: {output_path}")
    
    # Print simple patience curve summary
    print(f"\nPatience curve generated with {max_patience} patience levels")
    print(f"Current patience setting: {max_patience}")
    print(f"Best AUC achieved: {best_aucs[-1]:.4f} at epoch {stop_epochs[-1]}")

def _save_final_results(model, X_val, y_val, history, tracking, out_dir, config, train_time, val_ds=None, val_steps=None):
    """Save final metrics and plots."""
    if X_val is not None and y_val is not None:
        # Get final metrics and track prediction time
        pred_start = time.time()
        val_preds_raw = safe_predict(model, X_val, verbose=0).astype(np.float32)
        pred_time = time.time() - pred_start
        
        # Pass patch shape for both pixel-level and patch-level AUC calculation
        patch_shape = val_preds_raw.shape  # Should be (B, H, W, 1) or (B, H, W)
        final_metrics = calculate_metrics(
            y_val.flatten(), 
            val_preds_raw.flatten(), 
            tracking["best_thr_series"][-1] if tracking["best_thr_series"] else 0.5,
            patch_shape=patch_shape
        )
        
        print(f"\nFinal results: AUC={final_metrics['auc']:.4f}, "
              f"F1={tracking['best_f1_series'][-1] if tracking['best_f1_series'] else 0:.4f}")
    else:
        # Streaming mode - collect metrics from val_ds
        if val_ds is not None and val_steps is not None:
            pred_start = time.time()
            metrics = collect_val_metrics(model, val_ds, val_steps, config['GLOBAL']['batch_size'])
            pred_time = time.time() - pred_start

            final_metrics = {
                'auc': metrics['auc'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'prec': metrics['precision'],  # For print_summary
                'rec': metrics['recall'],    # For print_summary
                'f1': metrics['f1'],
                'threshold': metrics['threshold'],
                'fpr': metrics['fpr'],
                'tpr': metrics['tpr'],
                'name': 'U-Net (streaming)'
            }
            print(f"\nFinal streaming val metrics: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f} (thr={metrics['threshold']:.2f})")
        else:
            # No validation available at all
            nan = float('nan')
            print("\nTraining completed without validation metrics")
            final_metrics = {
                "auc": nan,
                "precision": nan,
                "recall": nan,
                "prec": nan,
                "rec": nan,
                "f1": nan,
                "threshold": 0.5,
                "name": "U-Net (no validation)",
                "fpr": np.array([0, 1]),
                "tpr": np.array([0, 1])
            }
            pred_time = 0
    
    # Create progress dict for plotting
    if tracking["best_f1_series"]:
        progress = {
            "epoch": list(range(1, len(tracking["best_f1_series"]) + 1)),
            "auc": history.history["val_auc"],  # Direct access
            "train_auc": history.history["auc"],  # Direct access
            "precision": tracking["best_prec_series"],
            "recall": tracking["best_rec_series"],
            "f1": tracking["best_f1_series"],
            "time": tracking["times"],
        }
        
        # Save plot
        plot_metric_progress(
            progress,
            out_path=os.path.join(out_dir, "training_progress.png"),
            title="Training Progress",
            config=config,
        )
        
        # Generate patience curve diagnostic
        if tracking["times"]:
            _generate_patience_curve(history, tracking["times"], out_dir, config)
    else:
        # No validation metrics to plot
        progress = {
            "epoch": list(range(1, len(tracking["times"]) + 1)),
            "time": tracking["times"],
        }
    
    # Add timing information to metrics
    final_metrics["train_time"] = train_time
    final_metrics["pred_time"] = pred_time
    
    return {
        "model": model,
        "metrics": final_metrics,
        "progress": progress,
    }

def _setup_callbacks(model, X_val, y_val, out_dir, config, learning_rate, val_ds=None, using_lr_schedule=False):
    """Create all training callbacks."""
    # Tracking state
    tracking = {
        "best_thr_series": [],
        "best_prec_series": [],
        "best_rec_series": [],
        "best_f1_series": [],
        "current_best_thr": 0.5,
        "times": [],
    }
    
    # Time tracking with debugging - track CUMULATIVE time from training start
    _t_start = time.time()  # Training start time
    _t_last = [time.time()]  # For individual epoch timing
    
    def track_time_debug(epoch, logs):
        # Track both individual epoch time and cumulative time
        epoch_time = time.time() - _t_last[0]
        cumulative_time = time.time() - _t_start
        tracking["times"].append(cumulative_time)  # Store CUMULATIVE time
    
        #print(f"[TIME DEBUG] Epoch {epoch+1}: {epoch_time:.1f}s (epoch) | Cumulative: {cumulative_time:.1f}s | Total epochs: {len(tracking['times'])}")
    
    time_cb = callbacks.LambdaCallback(
        on_epoch_begin=lambda *_: _t_last.__setitem__(0, time.time()),
        on_epoch_end=track_time_debug,
    )
    
    # Simple learning rate scheduling - just keep the existing warmup + plateau
    target_learning_rate = learning_rate
    
    def warmup_scheduler(epoch, lr):
        """Linear warmup for first 5 epochs to ensure stable start"""
        if epoch < 5:
            target_lr = float(target_learning_rate)
            return target_lr * (epoch + 1) / 5
        # After warmup, pass through current lr (let RLROP take effect)
        return lr
    
    # DISABLED: Learning rate scheduling causes training to continue forever
    # Just use simple early stopping with fixed learning rate
    # warmup_cb = callbacks.LearningRateScheduler(warmup_scheduler)
    # reduce_lr_cb = callbacks.ReduceLROnPlateau(
    #     monitor='val_auc',
    #     factor=0.3,
    #     patience=8,
    #     min_lr=1e-6,
    #     mode='max',
    #     verbose=1
    # )
    # lr_scheduler_cb = [warmup_cb, reduce_lr_cb]
    lr_scheduler_cb = []  # No LR scheduling
    
    # Weight change tracking for debugging
    weight_diff_cb = WeightDiff()
    
    # GPU monitoring callback
    gpu_monitor_cb = GPUMonitor(frequency=10)  # Log GPU stats every 10 epochs
    
    # Build callback list with scheduler
    cb_list = [time_cb, weight_diff_cb, gpu_monitor_cb]
    
    # Only add reactive schedulers if we are NOT using a pre-defined schedule in the optimizer.
    if not using_lr_schedule:
        if isinstance(lr_scheduler_cb, list):
            cb_list.extend(lr_scheduler_cb)
        else:
            cb_list.append(lr_scheduler_cb)
    
    
    # Only add validation-dependent callbacks if we have validation data loaded in memory
    if X_val is not None and y_val is not None:
        # Metrics optimization
        def _opt_metrics(epoch, _logs):
            val_preds = safe_predict(model, X_val, verbose=0).astype(np.float32).flatten()
            if np.isnan(val_preds).any():
                print("NaN detected -- stopping.")
                model.stop_training = True
                return
            
            y_true = y_val.flatten()
            
            # Re-optimize threshold every 10 epochs
            if epoch % 10 == 0:  # Optimize every 10 epochs
                tracking["current_best_thr"], best_f1, best_prec, best_rec = optimize_threshold(y_true, val_preds)
            else:
                thr = tracking["current_best_thr"]
                y_bin = (val_preds > thr).astype(int)
                best_prec = precision_score(y_true, y_bin, zero_division=0)
                best_rec = recall_score(y_true, y_bin, zero_division=0)
                best_f1 = f1_score(y_true, y_bin, zero_division=0)
            
            # Save metrics
            tracking["best_thr_series"].append(tracking["current_best_thr"])
            tracking["best_prec_series"].append(best_prec)
            tracking["best_rec_series"].append(best_rec)
            tracking["best_f1_series"].append(best_f1)
            
            print(f"[Epoch {epoch+1}] thr={tracking['current_best_thr']:.2f} "
                  f"F1={best_f1:.4f} P={best_prec:.4f} R={best_rec:.4f}")
        
        # DISABLED: Custom validation callback causing shape errors
        # opt_cb = callbacks.LambdaCallback(on_epoch_end=_opt_metrics)
        # cb_list.append(opt_cb)
        
        # Add validation-dependent callbacks
        cb_list.extend([
            callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=config["GLOBAL"]["patience"], 
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(out_dir, "best.keras"),
                save_best_only=True,
                monitor="val_auc",
                mode="max",
            ),
        ])
    else:
        # Streaming mode or no validation data
        if val_ds is not None:
            # Streaming mode with validation - use standard callbacks without custom metrics
            cb_list.extend([
                callbacks.EarlyStopping(
                    monitor='val_auc',
                    mode='max',
                    patience=config["GLOBAL"]["patience"], 
                    restore_best_weights=True
                ),
                callbacks.ModelCheckpoint(
                    filepath=os.path.join(out_dir, "best.keras"),
                    save_best_only=True,
                    monitor="val_auc",
                    mode="max",
                ),
            ])
        else:
            # No validation - just save the final model at the end
            cb_list.append(
                callbacks.ModelCheckpoint(
                    filepath=os.path.join(out_dir, "final.keras"),
                    save_best_only=False,  # Save every epoch (or just the last)
                    save_freq='epoch'
                )
            )
    
    # Optional TensorBoard
    if config['GLOBAL']['use_tensorboard']:
        tb_cb, log_dir = _make_tb_callback(out_dir, "unet", profile_batch=PROFILE_BATCH)
        print(f"TensorBoard logs: {log_dir}")
        cb_list.append(tb_cb)
    
    return cb_list, tracking


def setup_tensorflow(host: str) -> None:
    """Configure TensorFlow logging and GPU memory growth."""
    # Enable deterministic operations for reproducibility
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # Older TF versions

    # Quiet TF logs
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass

    # GPU setup
    try:
        gpus = tf.config.list_physical_devices('GPU')
    except Exception:
        gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            try:
                # Avoid pre-allocating entire GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"Found {len(gpus)} GPU(s).")
    else:
        if host in ('farm', 'quest'):
            print("FATAL: No GPUs detected on a designated GPU host.")
            sys.exit(1)
        else:
            print("No GPUs found; running on CPU.")


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
