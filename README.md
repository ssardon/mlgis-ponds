# mlgis-ponds

Deep learning pipeline for detecting irrigation ponds in satellite imagery. This toolkit was developed for my PhD dissertation studying how international trade affects agricultural investment and development in Mexico.

## Motivation

Irrigation ponds are a key form of agricultural capital investment, yet no systematic data exists on their spatial distribution or temporal evolution. This project uses convolutional neural networks to detect ponds in satellite imagery, enabling the construction of panel datasets that track agricultural investment at fine geographic scales.

## Results

The model achieves strong out-of-sample performance (AUC = 0.99) on held-out validation tiles:

<p align="center">
  <img src="assets/roc_curve_S2.png" width="500">
</p>

Example detection in Michoacan, Mexico. Left: Sentinel-2 imagery showing an irrigation pond surrounded by avocado orchards. Right: Model predictions with confidence levels.

<p align="center">
  <img src="assets/detection_example.png" width="700">
</p>

## Setup

Create the `mlgis` conda environment:

```bash
conda env create -f config/mlgis.yaml
conda activate mlgis
```

## Pipeline

1. **Data Download** (`01_downloads_gee_sentinel2.py`) - Export annual composites from Google Earth Engine
2. **Preprocessing** (`02_preproc.py`) - Cloud masking, tiling, and TFRecord creation
3. **Training** (`03_main.py`) - U-Net model training and evaluation

## Project Structure

```
mlgis-ponds/
├── config/
│   ├── config.yaml       # Paths and hyperparameters
│   └── mlgis.yaml        # Conda environment
├── src/
│   ├── 01_downloads_gee_sentinel2.py
│   ├── 02_preproc.py
│   ├── 03_main.py
│   └── mlgis_helpers/
│       ├── cfg.py
│       ├── data_loading.py
│       ├── evaluation.py
│       ├── model.py
│       ├── preprocess_tfrecord.py
│       └── training.py
└── assets/
    ├── detection_example.png
    └── roc_curve_S2.png
```

## Usage

```bash
conda activate mlgis
python src/03_main.py --task pondsNIR-S2024
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

> Sardon, Sebastian (2025). "Trade, Land Consolidation, and Agricultural Productivity." Working Paper, Northwestern University.
