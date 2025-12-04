# mlgis-ponds

Deep learning pipeline for detecting irrigation ponds in satellite imagery. Built for my PhD dissertation studying how trade affects agricultural development in Mexico.

## Overview

This toolkit uses U-Net segmentation models to identify irrigation infrastructure (ponds) in Sentinel-2 and Landsat imagery. The detected ponds are aggregated to census enumeration areas to build a panel dataset of agricultural investment.

## Pipeline Stages

1. **Data Download** (`01_downloads_gee_sentinel2.py`) - Export annual composites from Google Earth Engine
2. **Preprocessing** (`02_preproc.py`) - Cloud masking, tiling, and TFRecord creation
3. **Training** (`03_main.py`) - CNN (U-Net) model training and evaluation

## Project Structure

```
mlgis-ponds/
├── config/
│   └── config.yaml          # Paths and hyperparameters per host
├── src/
│   ├── 01_downloads_gee_sentinel2.py
│   ├── 02_preproc.py
│   ├── 03_main.py
│   └── mlgis_helpers/
│       ├── cfg.py           # Configuration parser
│       ├── data_loading.py  # TFRecord data pipeline
│       ├── evaluation.py    # Metrics and ROC curves
│       ├── model.py         # U-Net architecture
│       ├── preprocess_tfrecord.py
│       └── training.py      # Training loop
└── requirements.txt
```

## Usage

```bash
# Preprocess imagery
python src/02_preproc.py

# Train model
python src/03_main.py --task pondsNIR-S2024
```

## License & Citation

MIT License.

If you use this code, please cite:

Sardon, Sebastian (2025). "Trade, Land Consolidation, and Agricultural Productivity." Working Paper, Northwestern University.
