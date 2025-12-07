"""
-------------------------------------------------------------------------------
Training Program for Irrigation Pond Detection (Segmentation Model, Sentinel-2)
-------------------------------------------------------------------------------

This module orchestrates the training of a Deep Learning models (U-Net) to
detecting agricultural infrastructure (irrigation ponds) in satellite imagery.

It uses a host-aware configuration system that manages paths across
local development (Mac) and twodiffer HPC clusters (Northwestern's Quest and
UC Davis's FARM). It can easily accomodate additional hosts.

Key Capabilities:
    - Data Pipeline: Consumes optimized TFRecords for high-throughput,
      out-of-core training on large geospatial datasets.
    - Experiment Tracking: Automatically logs hyperparameters, ROC curves,
      and pixel-level metrics (Precision/Recall/F1).
    - Inference Validation: Runs a secondary validation loop using
      inference-style sliding windows to determine optimal thresholds.
    - Reproducibility: Enforces global seeding and environment isolation

Usage:
    #1. Standard Training (Local)
    python 03_train_model.py --task pondsNIR-S2024 --arch unet_tiny

    #2. Quick Debug Run (Fast, few epochs on a small subset)
    python 03_train_model.py --task pondsNIR-S2024 --quick

    #3. HPC Execution (Quest Node)
    python 03_train_model.py --host quest --task pondsNIR-S2024 --project avocados
"""

import json
import os
import time

os.environ['TF_USE_LEGACY_KERAS'] = 'True'
from tf_keras import backend as K
from mlgis_helpers import cfg, evaluation, training
from mlgis_helpers.preprocess_tfrecord import create_tfrecords


# Setup
# -----
DEFAULTS = {
    'host': 'mac',
    'project': 'avocados',
    'task': 'pondsNIR-S2024',
    'arch': 'unet_tiny', #Other (worse) CNNs: 'unet', 'unet_heavy', 'resnet50'
    'quick': False,
    'pretrained_model': None
}

def prepare_environment_and_config(args):
    """Configure environment and host-aware paths."""
    config, paths = cfg.resolve_config_and_paths(args)
    training.setup_tensorflow(args.host)
    print(f"Host:    {args.host}")
    print(f"Project: {args.project}")
    print(f"Task:    {args.task}")
    print(f"Output:  {paths['out_dir']}")
    return config, paths

# Data Loading
# ------------
def load_and_prep_data(config, paths, task, args):
    """
    Load train/val data via TFRecord pipeline.
    Returns paths to the TFRecord files (which contain both images and masks).
    """
    task_config = config['TASKS'][task]

    base_params = {
        'bands': task_config['bands'],
        'patch_size': task_config['patch_size'],
        # Task can override global stride settings
        'stride_ratio': task_config.get('stride_ratio',
                                        config['GLOBAL']['stride_ratio']),
        'cache_dir': paths['cache_dir'],
        'shape_path': paths['shape_path'],
        'out_dir': paths['out_dir'],
        'task': task
    }

    # Generate or Load TFRecords
    train_tfrecord, val_tfrecord = create_tfrecords(
        train_image_path=paths['train_imagery_path'],
        val_image_path=paths['val_imagery_path'],
        cache_dir=paths['cache_dir'],
        task=task,
        base_params=base_params,
        quick_mode=args.quick,
        ario_shapefile=paths.get('Ario_shapefile'),
        config=config
    )
    return train_tfrecord, val_tfrecord


# Training the CNN and evaluate its performance on validation data
# ----------------------------------------------------------------
def run_pipeline(config, paths, task, arch,
                 train_tfrecord, val_tfrecord, args):
    """Train a model, test it on val data, and save results."""
    print("\n--- Starting Model Training ---")
    task_config = config['TASKS'][task]
    patch_size = task_config['patch_size']

    # Train
    # -----
    result = training.train_model(
        config,
        paths['out_dir'],
        task,
        img_patches=train_tfrecord,
        val_img_patches=val_tfrecord,
        paths=paths,
        patch_size=patch_size,
        pretrained_model_path=args.pretrained_model,
    )

    if result["model"] is None:
        print("Training failed - no results to display")
        return

    # Test on Validation Data
    # -----------------------
    train_metrics = result["metrics"]
    model = result["model"]
    train_metrics["name"] = f"{task_config['architecture']}"
    evaluation.print_summary([train_metrics])
    roc_path = os.path.join(paths['out_dir'], "roc_curve.png")
    evaluation.plot_roc_comparison([train_metrics], roc_path,
                                   config, task_config)
    # Simulate sliding windows (as in inference) to get F1-maximizing threshold
    config['current_host'] = config.get('host', 'mac')
    post_metrics = evaluation.validate_inference_style(model,
                                                       config,
                                                       task_config,
                                                       paths)
    _save_validation_artifacts(paths['out_dir'],
                               train_metrics, post_metrics)
    print(f"\nValidation artifacts saved to: {paths['out_dir']}")

def _save_validation_artifacts(
    out_dir: str,
    train_metrics: dict,
    post_metrics: dict
) -> None:
    "Save validation metrics to JSON and TXT."
    metrics_path_json = os.path.join(out_dir, 'val_metrics.json')
    metrics_path_txt = os.path.join(out_dir, 'val_metrics.txt')

    core_keys = ('auc', 'precision', 'recall', 'f1', 'threshold')
    metrics_dict = {k: float(train_metrics[k])
                    for k in core_keys if k in train_metrics}

    if post_metrics:
        metrics_dict.update(post_metrics)

    # Write JSON (Easy to parse)
    with open(metrics_path_json, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Write TXT (Human Readable)
    with open(metrics_path_txt, 'w') as f:
        f.write("--- Pixel-Level Metrics ---\n")
        for k in core_keys:
            if k in train_metrics:
                f.write(f"{k.capitalize()}: {train_metrics[k]:.4f}\n")
        if post_metrics:
            f.write("\n--- Inference-Style Metrics (Post-Averaging) ---\n")
            # Filter for specific post-proc keys to keep order
            pp_keys = ('auc_postproc', 'precision_postproc',
                       'recall_postproc', 'f1_postproc', 'threshold_postproc')
            for k in pp_keys:
                if k in post_metrics:
                    f.write(f"{k}: {post_metrics[k]:.4f}\n")

# Define and call main function
# -----------------------------
def main():
    """
    Main Entry Point.
    Orchestrates: Setup -> Data Loading -> Training -> Cleanup.
    """
    script_start_time = time.time()

    # 1. Setup
    # --------
    training.seed_everything(42)
    K.set_image_data_format('channels_last')

    args = cfg.parse_args(DEFAULTS)
    config, paths = prepare_environment_and_config(args)

    # 2. Data Loading
    # ---------------
    train_tfrecord, val_tfrecord = load_and_prep_data(
        config, paths, args.task, args
    )

    # 3. Training and Evaluation
    # --------------------------
    run_pipeline(
        config,
        paths,
        args.task,
        args.arch,
        train_tfrecord,
        val_tfrecord,
        args
    )

    elapsed = time.time() - script_start_time
    print(f"\nDONE! TIME ELAPSED: {elapsed/60:.2f} minutes ({elapsed:.1f}s)")

if __name__ == "__main__":
    main()
