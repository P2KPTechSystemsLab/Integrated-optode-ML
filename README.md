
# 📈 pH Prediction and Visualization via Neural Networks

## 🧠 Overview

This repository provides automated tools for **training**, **inference**, and **visualization** of pH values from RGB images using neural networks.

Key Capabilities:

* Time-resolved and treatment-specific pH monitoring
* RGB-to-pH mapping with quantifiable performance metrics
* Two operational modes for different experimental needs:

  * **Fast Run Model** – Optimized for high-throughput, multi-treatment datasets
  * **Track History** – Ideal for longitudinal studies and single-sample visual outputs



## 🚀 Features

* Neural network for **RGB-to-pH regression**
* Dataset creation from **standard pH calibration images**
* **Model persistence** for reusability
* **Batch prediction** on experimental datasets
* **Export of pH heatmaps** in `.svg` and `.tiff` formats
* Training loss visualization for model evaluation

---

## 📂 Directory Structure

```
.
├── main_model_fast_run.py       # Entry point for Fast Run execution
├── track_history.py             # Entry point for history tracking
├── model.pk00.keras             # Trained model (generated if not found)
├── pHCalib/                     # Standard pH reference images (e.g., 6_5.png)
├── experimental_data/           # Experimental input images (*.png)
├── ProcessedFigures/            # Fast Run output visualizations
├── outputs/                     # Track History prediction results
└── README.md                    # This documentation
```

---

## 📦 Dependencies

Install required packages via pip:

```bash
pip install numpy tensorflow scikit-learn matplotlib opencv-python scikit-image pillow tqdm
```



## 📁 Input Data Format

### Calibration Images (`/pHCalib/`)

* Format: `.png` or `.tiff`
* Filenames should include pH value (e.g., `6_5.png` → pH 6.5)

### Experimental Images (`/experimental_data/`)

* Organized by **treatment**, **timepoint**, and **replicate**
* Example path: `/experimental_data/Ctrl/SD_1/R1.png`

---

## 🔬 Methodology

### 1. Dataset Construction

Function: `create_datasets_from_sample_ph()`

* Loads calibration images
* Crops and normalizes RGB values
* Attaches pH labels
* Output: `(N_pixels × N_images, 4)` → `[R, G, B, pH]`

---

### 2. Model Architecture

| Layer  | Type  | Units | Activation | Dropout |
| ------ | ----- | ----- | ---------- | ------- |
| Input  | Dense | 64    | ReLU       | 0.2     |
| Hidden | Dense | 32    | ReLU       | 0.2     |
| Output | Dense | 1     | Linear     | 0       |

* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Epochs:** 20
* **Batch Size:** 50
* **Split:** 90% Train / 10% Test

---

### 3. Prediction & Visualization

* Images are cropped, normalized, flattened
* Inferred through trained model
* Reshaped into 2D heatmaps
* Heatmaps saved as:

  * `.tiff` (raw matrix)
  * `.svg` (colorized with `RdYlGn`; range: pH 5.4–8.0)

---

### 4. Loss Curve Plotting

* Training vs. validation loss plotted via `matplotlib`
* Assists in convergence and model performance evaluation

---

## ⚙️ Execution

### A. Fast Run — Batch Processing

```bash
python main_model_fast_run.py
```

Performs:

* Dataset creation from `/pHCalib`
* Model training or loading
* Prediction on experimental images
* Output to `/ProcessedFigures/`

**Visualization layout:**

* **Rows:** Raw → Predicted pH → Enhanced Raw → Enhanced pH
* **Columns:** Replicates (R1–R4) across Days (1, 16, 64)

---

### B. Track History — Single Sample with Visual Exports

```bash
python track_history.py
```

Performs:

* Dataset creation
* Model training or loading
* Prediction on `.png` images in `/experimental_data`
* Heatmap export to `/outputs/` as `.svg` and `.tiff`
* Final loss curve displayed

---

## 🖼️ Example Outputs

* `sample_pre.svg` – Colorized pH map
* `sample_pre.tiff` – Raw pH matrix
* `training_curve.png` – Training loss over epochs

---

## 📊 Performance

* **Typical validation loss:** `~0.0076` pH² units (MSE)
* Resolves subtle spatial pH gradients in:

  * Environmental imaging
  * Biological assays
  * Chemical reaction monitoring

---

## ⚠️ Limitations

* **Calibration-dependent**: Model  predicts within the pH range it was trained on (e.g., 5.5–8.0)
* Adjust calibration images accordingly if working outside that range

---

## 🧪 Applications

* Environmental monitoring (acidification, eutrophication)
* Toxicological bioassays
* Educational demos for colorimetric sensors

---

## 📚 Citation & Contact

For academic use 
