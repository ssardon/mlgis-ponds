"""
Training utilities for object detection.
Orchestrates TFRecord ingestion, model compilation, training loop, and
artifact saving.
"""

import os
import random
import subprocess
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import backend as K
from tf_keras import callbacks
from sklearn.metrics import roc_curve

# Local Application Imports
from .evaluation import (plot_patience_curve,
                         plot_metric_progress,
                         optimize_threshold
)
from .model import get_model

# Constants
AUTOTUNE = tf.data.AUTOTUNE
PROFILE_BATCH = "100, 105"  # profiler step window for TensorBoard


# I. SETUP & CONFIGURATION
# ------------------------

def setup_tensorflow(host: str) -> None:
    """
    Configure TensorFlow logging, determinism, and GPU memory growth.

    Raises RuntimeError if running on HPC ('farm', 'quest') without GPU.
    """
    # 1. Logging and Determinism
    # Suppress TensorFlow INFO and WARNING logs
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    # Enforce deterministic operations for reproducibility
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.experimental.enable_op_determinism()

    # 2. GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            try:
                # Prevent TF from grabbing all GPU VRAM upfront.
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"GPU Config Warning: {e}")
        print(f"TensorFlow Configured: Found {len(gpus)} GPU(s).")
    else:
        # Fail fast if GPUs are not availabile on HPCs (defaults to CPU)
        if host in ('farm', 'quest'):
            raise RuntimeError(
                f"CRITICAL: No GPUs detected on HPC node ({host}). "
                "Check CUDA module loading or SLURM allocation."
            )
        print(f"TensorFlow Configured: CPU Mode (Host: {host}).")


def seed_everything(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# II. Loss functions
# -----------------
def weighted_bce_loss(pos_weight):
    """
    Weighted BCE loss with optional NODATA masking via sample_weight.
    Uses logit-space computation for numerical stability.
    """
    pw_const = tf.constant(pos_weight)

    def _loss(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Valid pixel mask (pass NODATA mask via sample_weight)
        valid_mask = tf.ones_like(y_true, dtype=tf.float32)
        if sample_weight is not None:
            valid_mask = tf.cast(sample_weight, tf.float32)

        # Clip and convert to logits for stable BCE
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        logits = tf.math.log(y_pred / (1.0 - y_pred))
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                      logits=logits)

        # Reshape for broadcasting
        bce = tf.reshape(bce, tf.shape(y_true)[:-1])
        y_true_sq = tf.reshape(y_true, tf.shape(y_true)[:-1])
        valid_mask_sq = tf.reshape(valid_mask, tf.shape(y_true)[:-1])

        # Apply class weights
        class_weights = tf.where(
            tf.equal(y_true_sq, 1.0),
            tf.cast(pw_const, tf.float32),
            tf.ones_like(y_true_sq, dtype=tf.float32)
        )
        weighted_bce = bce * class_weights * valid_mask_sq

        # Mean over valid pixels
        total_valid = tf.reduce_sum(valid_mask_sq)
        total_loss = tf.reduce_sum(weighted_bce)
        return tf.cond(
            total_valid > 0,
            lambda: total_loss / total_valid,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    return _loss

def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=1.33333):
    """
    Focal Tversky Loss for extreme class imbalance.
    alpha > beta penalizes false negatives more than false positives.

    EXPERIMENTAL: As of 2025-12, gives worse results than weighted BCE.
    """
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), K.epsilon(),
                                  1.0 - K.epsilon())

        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))

        tversky = tp / (tp + alpha * fn + beta * fp + K.epsilon())
        return tf.pow(1 - tversky, gamma)

    return _loss

# III. Callbacks
# --------------
# Monitor resource usage to adjust HPC requests (on SLURM) if needed
class GPUMonitor(callbacks.Callback):
    """Callback to monitor GPU usage during training."""

    def __init__(self, frequency: int = 5):
        super().__init__()
        self.frequency = frequency
        self.has_gpu = False

    def on_train_begin(self, logs=None):
        """Check if GPU monitoring is available via nvidia-smi."""
        self.has_gpu = log_gpu_usage()
        if not self.has_gpu:
            print(
                "GPU monitoring not available "
                "(nvidia-smi not found or no GPU)"
            )

    def on_epoch_end(self, epoch, logs=None):
        """Log GPU usage at specified frequency."""
        if self.has_gpu and (epoch + 1) % self.frequency == 0:
            print(f"\n--- Epoch {epoch + 1} GPU Stats ---")
            log_gpu_usage()

    def on_train_batch_end(self, batch, logs=None):
        """Log GPU usage for first batch to check initial memory allocation."""
        # Only log first batch stats at the same frequency as epoch stats
        epoch = getattr(self, '_current_epoch', 0)
        if self.has_gpu and batch == 0 and (epoch + 1) % self.frequency == 0:
            print("\n--- First Batch GPU Stats ---")
            log_gpu_usage()

    def on_epoch_begin(self, epoch, logs=None):
        """Track current epoch for batch callback."""
        self._current_epoch = epoch


# Debugging callback to track weight changes
class WeightDiff(callbacks.Callback):
    """
    Track weight changes to detect if optimizer is applying gradients.
    At various points, this has helped identify issues with optimizers
    not updating gradients, but as of 2025-12-09 it is not actively used.
    """

    def __init__(self):
        super().__init__()
        self.w0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self.w0 = [w.numpy().copy() for w in self.model.trainable_weights]

    def on_epoch_end(self, epoch, logs=None):
        if self.w0 is not None:
            deltas = [ # noqa: F841
                np.abs(w1 - w0).mean()
                for w0, w1 in zip(self.w0, self.model.trainable_weights)
            ]
            # print(f"Mean |w|: {float(np.mean(deltas)):.8f}")


def _make_tb_callback(
    out_dir: str,
    run_name: str,
    *,
    profile_batch: str = "100, 105"
) -> Tuple[callbacks.TensorBoard, str]:
    """Create a TensorBoard callback and return it with its log directory."""
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


def _setup_callbacks(model, out_dir, config, val_ds=None):
    """Create training callbacks. Returns (callback_list, tracking_dict)."""
    tracking = {"times": []}
    _t_start = time.time()

    def track_time(epoch, logs):
        tracking["times"].append(time.time() - _t_start)

    cb_list = [
        callbacks.LambdaCallback(on_epoch_end=track_time),
        WeightDiff(),
        GPUMonitor(frequency=10),
    ]

    # Validation-dependent callbacks
    if val_ds is not None:
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
                mode="max"
            ),
        ])
    else:
        cb_list.append(
            callbacks.ModelCheckpoint(
                filepath=os.path.join(out_dir, "final.keras"),
                save_best_only=False,
                save_freq='epoch'
            )
        )

    if config['GLOBAL'].get('use_tensorboard'):
        tb_cb, log_dir = _make_tb_callback(
            out_dir, "unet", profile_batch=PROFILE_BATCH
        )
        print(f"TensorBoard logs: {log_dir}")
        cb_list.append(tb_cb)

    return cb_list, tracking

# --------------------------------------------------------------------
# Sections IV-VII contain helpers for the orchestrator in section VIII
# --------------------------------------------------------------------

# IV. Data Loading Helpers
# ------------------------
def _parse_tfrecord_fn(proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parses a single TFRecord example into (image, mask) tensors.

    Pipeline:
    1. Parse Example proto using FixedLenFeature.
    2. Decode serialized ByteString to Tensor.
    3. Set static shapes (rank only) for TF Graph compilation.
    4. Normalize types and ensure mask has channel dimension.
    """
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(proto, feature_description)

    # Decode raw bytes
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    mask = tf.io.parse_tensor(example['mask'], out_type=tf.uint8)

    # Critical: Set explicit shapes so TF Graph knows the rank (H, W, C)
    image.set_shape([None, None, None])
    mask.set_shape([None, None])

    # Standardization
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    # Hard binarization (anything > 0 becomes 1.0)
    mask = tf.where(mask > 0.5, 1.0, 0.0)

    # Add Channel Dimension: (H, W) -> (H, W, 1)
    mask = tf.expand_dims(mask, axis=-1)

    return image, mask


def _create_balanced_dataset(
    tfrecord_path: str,
    batch_size: int,
    task: str = 'ponds'
) -> tf.data.Dataset:
    """
    Creates a training dataset pipeline.

    Strategies:
    - Default: Uniform sampling with shuffling.
    - 'mines': 50/50 balanced sampling (Positive vs Negative patches).
    """
    # Basic pipeline: Read -> Parse
    raw_ds = (
        tf.data.TFRecordDataset([tfrecord_path], num_parallel_reads=AUTOTUNE)
        .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    )

    # Default: Uniform sampling
    print("\n--- Pipeline: Uniform Sampling ---")
    ds = raw_ds.shuffle(1024, reshuffle_each_iteration=True).repeat()


    return ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)


def _make_standard_dataset(val_tfrecord_path, batch_size):
    """Create finite validation dataset from TFRecord."""
    dataset = tf.data.TFRecordDataset(
        [val_tfrecord_path], num_parallel_reads=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        _parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)  # Finite - only one pass through data
    dataset = dataset.prefetch(2)
    return dataset


# V. Configuration & Hyperparameter Selection Helpers
# --------------------------------------------------------------
def get_task_params(config, task):
    """
    Get all task-specific parameters from config.
    Parameters are now in the TASKS section.
    """
    if task not in config['TASKS']:
        raise ValueError(f"Task '{task}' not found in TASKS section")

    task_config = config['TASKS'][task]

    # Required fields
    params = {
        'pos_weight': task_config['pos_weight'],
        'architecture': task_config.get('architecture', 'unet_tiny'),
        'neg_ratio': task_config.get('neg_ratio', 1.0)
    }

    # Optional fields (pass through if present)
    if 'hparam_overrides' in task_config:
        params['hparam_overrides'] = task_config['hparam_overrides']

    return params


def _calculate_optimal_pos_weight(ds, n_batches=50):
    """
    Calculate optimal pos_weight from pixel-level imbalance.
    Also logs patch-level fraction for diagnostics (single pass).
    """
    print(f"\n--- Calculating Optimal pos_weight ({n_batches} batches) ---")

    total_pixels = 0
    positive_pixels = 0
    total_patches = 0
    positive_patches = 0

    for i, (_, y) in enumerate(ds.take(n_batches)):
        batch_pos_pix = tf.reduce_sum(tf.cast(y > 0.5, tf.int64)).numpy()
        batch_total_pix = tf.reduce_prod(tf.shape(y)).numpy()
        positive_pixels += batch_pos_pix
        total_pixels += batch_total_pix

        has_pos = tf.reduce_any(y > 0.5, axis=(1, 2, 3))
        positive_patches += tf.reduce_sum(tf.cast(has_pos, tf.int32)).numpy()
        total_patches += tf.shape(y)[0].numpy()

        if i < 3:
            print(
                f"  Batch {i+1}: {batch_pos_pix:,}/{batch_total_pix:,} px, "
                f"{positive_patches}/{total_patches} patches"
            )

    pixel_frac = positive_pixels / total_pixels if total_pixels > 0 else 0.0
    patch_frac = positive_patches / total_patches if total_patches > 0 else 0.0

    if pixel_frac > 0:
        pos_weight = min(100.0, (1.0 - pixel_frac) / pixel_frac)
    else:
        print("  WARNING: No positive pixels, using default pos_weight=1.0")
        pos_weight = 1.0

    print(f"\nPixel: p={pixel_frac:.4f}, pos_weight={pos_weight:.2f}")
    print(f"Patch: frac={patch_frac:.3f}")
    return pos_weight, pixel_frac


# VI: Diagnostic Helpers (Debugging and Verification)
# -------------------------------------------------------------
def log_gpu_usage():
    """Log current GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total, util = result.stdout.strip().split(', ')
            used_gb = float(used) / 1024
            total_gb = float(total) / 1024
            percent = float(used) / float(total) * 100
            print(
                f"GPU Memory: {used_gb:.1f}/{total_gb:.1f} GB "
                f"({percent:.1f}%), Utilization: {util}%"
            )
            return True
    except Exception:
        pass
    return False


def log_quantiles(data, label, percentiles=[0.1, 50, 99.9]):
    """Log data quantiles for diagnostic verification."""
    if tf.is_tensor(data):
        data = data.numpy()

    print(f"\n--- {label.upper()} DATA QUANTILES ---")

    if len(data.shape) == 4:  # Batched data (N, H, W, C)
        for band in range(data.shape[-1]):
            band_data = data[:, :, :, band]
            p_vals = np.percentile(band_data, percentiles)
            print(
                f"  Band {band}: [{p_vals[0]:.6f}, {p_vals[1]:.6f}, "
                f"{p_vals[2]:.6f}]"
            )
    elif len(data.shape) == 3:  # Single image (H, W, C)
        for band in range(data.shape[-1]):
            band_data = data[:, :, band]
            p_vals = np.percentile(band_data, percentiles)
            print(
                f"  Band {band}: [{p_vals[0]:.6f}, {p_vals[1]:.6f}, "
                f"{p_vals[2]:.6f}]"
            )
    else:
        # Flattened data
        p_vals = np.percentile(data, percentiles)
        print(
            f"  All data: [{p_vals[0]:.6f}, {p_vals[1]:.6f}, "
            f"{p_vals[2]:.6f}]"
        )

    print(f"  Data shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min/Max: [{data.min():.6f}, {data.max():.6f}]")


def _verify_band_mapping(ds, bands_config, task_name):
    """Verify band mapping consistency and detect ordering issues."""
    print(f"\n--- Verifying Band Mapping for {task_name} ---")
    print(f"Expected bands: {bands_config}")

    sample_batch = next(iter(ds.take(1)))
    images, _ = sample_batch

    band_medians = []
    for b in range(images.shape[-1]):
        band_data = images[:, :, :, b]
        band_data_np = band_data.numpy().flatten()
        median_val = float(np.percentile(band_data_np, 50))
        band_medians.append(median_val)
        print(
            f"  Band {b} (index {bands_config[b]}): median = {median_val:.4f}"
        )

    # Heuristic: NIR (often index 7) > Green (often index 2) over vegetation
    if len(band_medians) >= 2:
        nir_median = band_medians[0]
        green_median = band_medians[1]

        if nir_median > green_median:
            print(" Band ordering check PASSED: NIR > Green")
        else:
            print(" WARNING: Band ordering suspicious: NIR <= Green")

    print("Band mapping verification complete")
    return band_medians


# VII. Evaluation Helpers (using evaluation.py)
# ---------------------------------------------
def safe_predict(model, X, **kwargs):
    """Wrapper to handle XLA prediction issues and ensure numeric output."""
    pred = model.predict(X, **kwargs)

    if hasattr(pred, 'numpy'):
        pred = pred.numpy()
    elif not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    try:
        pred = pred.astype(np.float32)
    except (ValueError, TypeError):
        if pred.size > 0:
            try:
                # Handle wrapped objects
                if pred.dtype == object:
                    pred_list = []
                    for item in pred.flat:
                        if hasattr(item, 'numpy'):
                            pred_list.append(float(item.numpy()))
                        else:
                            pred_list.append(float(item))
                    pred = np.array(pred_list).reshape(pred.shape).astype(
                        np.float32
                    )
                else:
                    pred = np.array(
                        [float(str(x)) for x in pred.flat]
                    ).reshape(pred.shape)
            except Exception:
                pred = np.zeros_like(X[..., 0], dtype=np.float32)

    return pred


def collect_val_metrics(model, val_ds, val_steps):
    """Collect validation metrics efficiently in a single pass."""
    print("\nComputing final validation metrics...")
    start_time = time.time()

    auc_metric = tf.keras.metrics.AUC()
    patch_truth, patch_scores = [], []

    for x_batch, y_batch in val_ds.take(val_steps):
        y_pred = model.predict_on_batch(x_batch)
        y_true_flat = tf.reshape(y_batch, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        auc_metric.update_state(y_true_flat, y_pred_flat)
        patch_truth.append(
            tf.reduce_any(y_batch > 0.5, axis=(1, 2, 3)).numpy().astype(
                np.float32
            )
        )
        patch_scores.append(tf.reduce_max(y_pred, axis=(1, 2, 3)).numpy())

    patch_truth = np.concatenate(patch_truth)
    patch_scores = np.concatenate(patch_scores)

    opt_thr, opt_f1, opt_prec, opt_rec = optimize_threshold(
        patch_truth, patch_scores
    )
    fpr, tpr, _ = roc_curve(patch_truth, patch_scores)

    elapsed = time.time() - start_time
    auc = auc_metric.result().numpy()
    print(
        f"  Computed metrics in {elapsed:.1f}s - AUC={auc:.4f}, "
        f"F1={opt_f1:.4f}"
    )

    return {
        "auc": auc,
        "precision": opt_prec,
        "recall": opt_rec,
        "f1": opt_f1,
        "threshold": opt_thr,
        "fpr": fpr,
        "tpr": tpr,
    }


def _save_final_results(
    model, history, tracking, out_dir, config, train_time,
    val_ds=None, val_steps=None
):
    """Save final metrics and plots."""
    if val_ds is not None and val_steps is not None:
        pred_start = time.time()
        metrics = collect_val_metrics(model, val_ds, val_steps)
        pred_time = time.time() - pred_start

        final_metrics = {
            'auc': metrics['auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'prec': metrics['precision'],
            'rec': metrics['recall'],
            'f1': metrics['f1'],
            'threshold': metrics['threshold'],
            'fpr': metrics['fpr'],
            'tpr': metrics['tpr'],
            'name': 'U-Net (streaming)'
        }
        print(
            f"\nFinal streaming val metrics: AUC={metrics['auc']:.4f}, "
            f"F1={metrics['f1']:.4f} (thr={metrics['threshold']:.2f})"
        )
    else:
        nan = float('nan')
        print("\nTraining completed without validation metrics")
        final_metrics = {
            "auc": nan, "precision": nan, "recall": nan,
            "prec": nan, "rec": nan, "f1": nan,
            "threshold": 0.5,
            "name": "U-Net (no validation)",
            "fpr": np.array([0, 1]), "tpr": np.array([0, 1])
        }
        pred_time = 0.0

    progress = {}
    if tracking["times"]:
        progress = {
            "epoch": list(range(1, len(tracking["times"]) + 1)),
            "time": tracking["times"],
        }

        if "val_auc" in history.history:
            progress["auc"] = history.history["val_auc"]
            progress["train_auc"] = history.history["auc"]

        plot_metric_progress(
            progress,
            out_path=os.path.join(out_dir, "training_progress.png"),
            title="Training Progress",
            config=config,
        )

        plot_patience_curve(history, tracking["times"], out_dir, config)

    final_metrics["train_time"] = train_time
    final_metrics["pred_time"] = pred_time

    return {
        "model": model,
        "metrics": final_metrics,
        "progress": progress,
    }


# VIII. Main Orchestrator
# ---------------------------------

def train_model(
    config: dict,
    out_dir: str,
    task: str,
    *,
    img_patches: str | None = None,
    val_img_patches: str | None = None,
    paths: dict | None = None,
    patch_size: int | None = None,
    pretrained_model_path: str | None = None,
):
    """
    Train a segmentation model using TFRecord pipeline.
    Refactored for maintainability and streaming efficiency.
    """
    # 1. Configuration & Setup
    task_config = config['TASKS'][task]
    bands = task_config["bands"]
    num_channels = len(bands)

    print(f"Train TFRecord: {img_patches}")
    print(f"Val TFRecord: {val_img_patches}")

    # 2. Dataset Creation
    # Build raw dataset first for statistics (before any balancing)
    raw_ds = (
        tf.data.TFRecordDataset([img_patches], num_parallel_reads=AUTOTUNE)
        .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .batch(config["GLOBAL"]["batch_size"])
    )

    # 3. Diagnostics (on raw, unbalanced data - single pass)
    optimal_pos_weight, _ = _calculate_optimal_pos_weight(raw_ds, n_batches=50)
    task_config['pos_weight'] = optimal_pos_weight

    # Now create training dataset (potentially balanced for mines)
    train_ds = _create_balanced_dataset(
        img_patches, config["GLOBAL"]["batch_size"], task=task
    )
    val_ds = _make_standard_dataset(
        val_img_patches, config["GLOBAL"]["batch_size"]
    )

    _verify_band_mapping(val_ds, bands, task)

    # 4. Step Calculation (read from metadata if available)
    import json
    metadata_path = os.path.join(os.path.dirname(img_patches), 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        train_examples = meta['train_count']
        val_examples = meta['val_count']
        print(f"Loaded counts from metadata: train={train_examples}, val={val_examples}")
    else:
        print("No metadata.json found, scanning TFRecords...")
        train_examples = sum(1 for _ in tf.data.TFRecordDataset(img_patches))
        val_examples = sum(1 for _ in tf.data.TFRecordDataset(val_img_patches))

    bs = config["GLOBAL"]["batch_size"]

    steps_per_epoch = max(1, train_examples // bs)
    val_steps = max(1, val_examples // bs)

    if config.get("quick_mode", False):
        steps_per_epoch = min(steps_per_epoch, 60)
        val_steps = min(val_steps, 30)

    print(f"Steps: train={steps_per_epoch}, val={val_steps}")

    # 5. Model Building
    task_params = get_task_params(config, task)
    architecture = task_params['architecture']

    if architecture == 'resnet50_unet' and num_channels != 3:
        raise ValueError("ResNet50 requires exactly 3 bands.")

    input_shape = (patch_size, patch_size, num_channels)
    print(f"\nBuilding model: {architecture}, input_shape={input_shape}")

    if pretrained_model_path:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Missing: {pretrained_model_path}")
        model = keras.models.load_model(pretrained_model_path, compile=False)
        if model.input_shape[1:] != input_shape:
            raise ValueError(
                f"Pretrained model shape {model.input_shape[1:]} != "
                f"expected {input_shape}"
            )
    else:
        model = get_model(
            config, paths=paths, architecture=architecture,
            input_shape=input_shape
        )

    # 6. Compilation
    hparams = config['GLOBAL']['ARCHITECTURE_HPARAMS'][architecture].copy()
    if 'hparam_overrides' in task_params:
        hparams.update(task_params['hparam_overrides'])

    learning_rate = float(hparams['learning_rate'])

    if architecture.startswith('segformer'):
        total_steps = steps_per_epoch * config['GLOBAL']['num_epochs']
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=int(total_steps),
            end_learning_rate=0.0, power=1.0
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=0.05, clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=0.05, clipnorm=1.0
        )

    loss_fn = weighted_bce_loss(optimal_pos_weight)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[keras.metrics.AUC(name="auc")],
        jit_compile=False
    )

    # 7. Pre-flight Checks
    log_gpu_usage()

    initial_epoch = 0
    if 'resume_from' in config['GLOBAL']:
        checkpoint_path = config['GLOBAL']['resume_from']
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model.load_weights(checkpoint_path)
            print("Model weights loaded. Starting from epoch 0.")

    print("\nDIAGNOSTIC: Validation Pixel Check")
    n_batches_check = 200
    pos_pix = 0
    for _, mask in val_ds.take(n_batches_check):
        pos_pix += tf.reduce_sum(tf.cast(mask > 0, tf.int64)).numpy()

    print(f"Checked {n_batches_check} batches: {pos_pix} positive pixels")
    if pos_pix == 0:
        raise ValueError("FATAL: Validation set has NO positive pixels!")

    print("\nDIAGNOSTIC: Gradient Tape Smoke Test")
    try:
        with tf.GradientTape() as tape:
            x, y = next(iter(train_ds.take(1)))
            y_hat = model(x, training=True)
            loss = model.compiled_loss(y, y_hat)

        grads = tape.gradient(loss, model.trainable_variables)
        g_norm = tf.linalg.global_norm(grads)

        print(f"Loss: {loss.numpy():.6f}, Grad Norm: {g_norm.numpy():.6f}")

        if g_norm.numpy() == 0:
            print("CRITICAL: Gradient norm is 0!")
        elif not tf.math.is_finite(g_norm):
            print("CRITICAL: Gradient norm is NaN/Inf!")
    except Exception as e:
        print(f"ERROR in gradient test: {e}")

    # 8. Callbacks
    cb_list, tracking = _setup_callbacks(
        model, out_dir, config, val_ds=val_ds
    )

    # 9. Training Loop
    print(f"Training for {config['GLOBAL']['num_epochs']} epochs...")
    train_start = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["GLOBAL"]["num_epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cb_list,
        verbose=2,
        initial_epoch=initial_epoch
    )

    train_time = time.time() - train_start
    log_gpu_usage()

    # 10. Save & Return
    return _save_final_results(
        model, history, tracking, out_dir, config, train_time,
        val_ds=val_ds, val_steps=val_steps
    )
