"""
Configuration management for ML pipeline.
Handles argument parsing, YAML loading, and path resolution.
"""
import argparse
import os
import time
from argparse import Namespace as _NS
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

def parse_args(defaults: Dict[str, Any]) -> argparse.Namespace:
    """
    Parses CLI arguments, merging with provided defaults.
    Note: The 'choices' list are constraints, enforced strictly.
    To run on a new environment (e.g., 'windows', 'mac_2'),
    you must explicitly add it to the choices list below.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, choices=['mac', 'quest', 'farm'])
    parser.add_argument("--project", type=str, choices=['mines', 'avocados'])
    parser.add_argument("--task", type=str)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse_cache", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def load_config() -> Dict[str, Any]:
    """
    Load the YAML configuration.
    No arguments needed: finds config/config.yaml relative to this module.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    target_path = project_root / "config" / "config.yaml"
    if not target_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Config file missing at {target_path}.\n"
            "This pipeline enforces the structure:\n"
            "  root/\n"
            "    config/config.yaml\n"
            "    src/mlgis_helpers/cfg.py"
        )

    with open(target_path, 'r') as f:
        return yaml.safe_load(f)


def build_paths(config: Dict[str, Any],
                host: str, project: str, task: str) -> Dict[str, Any]:
    """
    Builds absolute paths based on the host+project configuration.
    """
    if host not in config['HOSTS']:
        raise ValueError(f"Host '{host}' not found in config.yaml.")
    if project not in config['HOSTS'][host]:
        raise ValueError(f"Project '{project}' not configured for host '{host}'.")

    host_config = config['HOSTS'][host][project]
    root_path = Path(host_config['root'])
    proc_folder = Path(host_config['proc_folder'])
    labels_folder = Path(host_config['labels_folder'])
    cache_base = Path(host_config['cache_dir'])
    cache_dir = cache_base / f"persistent_cached_patches_{task}"

    # Initialize paths dict
    paths = {
        'host': host,
        'task': task,
        'root_dir': str(root_path),
        'imagery_data_dir': str(proc_folder),
        'labels_data_dir': str(labels_folder),
        'cache_dir': str(cache_dir)
    }

    # Handle Task-Specific Paths
    if task:
        if task not in config['TASKS']:
            raise ValueError(f"Task '{task}' not found in TASKS config.")
        task_config = config['TASKS'][task]
        # Imagery
        if 'imagery_folder' in task_config:
            task_imagery = proc_folder / task_config['imagery_folder']
        else:
            task_imagery = proc_folder
        paths['imagery_data_dir'] = str(task_imagery)
        # Labels
        paths['shape_path'] = str(labels_folder / task_config['labels_file'])
        # (Optional) 1996 Labels
        if task_config.get('labels_file_1996'):
            paths['shape_path_1996'] = str(
                labels_folder / task_config['labels_file_1996']
            )

        # Train/Val Imagery Helpers
        def _resolve_img(entry):
            if isinstance(entry, list):
                return [str(task_imagery / e) for e in entry]
            return str(task_imagery / entry)
        if 'train_img' in task_config:
            paths['train_imagery_path'] = _resolve_img(
                task_config['train_img']
            )
        if 'val_img' in task_config:
            paths['val_imagery_path'] = _resolve_img(task_config['val_img'])

        # Output Directory (different if training multiple cnns for HP tuning)
        if os.environ.get('HP_OUTPUT_DIR'):
            output_dir = Path(os.environ['HP_OUTPUT_DIR'])
        else:
            runtime_tag = time.strftime("%Y%m%d-%H%M%S")
            output_dir = root_path / "out" / "mlgis_output" / task / f"run_{runtime_tag}" # noqa: E501
        paths['out_dir'] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Diagnostics
        if 'diagnostic_csv' in task_config:
            diag_root = host_config.get('diagnostics_dir')
            if diag_root:
                paths['diagnostic_csv'] = str(
                    Path(diag_root) / task_config['diagnostic_csv']
                    )

    # Boundary File (Ario municipality: the labeled subregion for train/val)
    if config['GLOBAL']['ONLY_Ario']:
        paths['Ario_shapefile'] = str(host_config['ario_boundary'])

    return paths


def ensure_proj_data(config: Dict[str, Any], host: str) -> None:
    """
    Many bugs can be solved by explicitly setting a proj path on config.yaml.
    Sets PROJ_DATA/PROJ_LIB environment variables if such path is found.
    """
    host_cfg = config['HOSTS'].get(host, {})
    proj_path_str = host_cfg.get('proj_data')
    if proj_path_str:
        proj_path = Path(proj_path_str)
        if (proj_path / "proj.db").exists():
            os.environ['PROJ_DATA'] = str(proj_path)
            os.environ['PROJ_LIB'] = str(proj_path)
        else:
            print(f"WARNING: Configured PROJ_DATA {proj_path} not found.")


def parse_dict(defaults: Optional[Dict[str, Any]] = None, **overrides) -> _NS:
    """
    Create a dot-access namespace from defaults + overrides.
    Useful for testing without CLI args (e.g., in notebooks).
    """
    if defaults is None:
        defaults = {}
    merged = dict(defaults)
    merged.update(overrides)
    return _NS(**merged)


# Orchestrator function
# ---------------------
def resolve_config_and_paths(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    """
    Orchestrator: Loads config, merges CLI args, sets up env, and builds paths.
    """
    # 1. Load Base Config
    config = load_config()

    # 2. Merge CLI Overrides into Config
    if 'GLOBAL' not in config:
        config['GLOBAL'] = {}

    config['host'] = args.host
    config['task'] = args.task

    if args.arch:
        config['GLOBAL']['architecture'] = args.arch
    if args.quick:
        config['quick_mode'] = True
        config['GLOBAL']['num_epochs'] = config['GLOBAL']['quick_epochs']
    if args.patience is not None:
        config['GLOBAL']['patience'] = args.patience
        print(f"CLI override: patience set to {args.patience}")
    if getattr(args, 'overwrite', False):
        config['GLOBAL']['overwrite_cache'] = True
        print("CLI override: forcing cache rebuild (--overwrite)")
    if getattr(args, 'reuse_cache', False):
        config['GLOBAL']['reuse_cache'] = True

    # 3. Setup Environment Variables
    ensure_proj_data(config, args.host)

    # 4. Build Absolute Paths
    paths = build_paths(config, args.host, args.project, args.task)

    return config, paths
