pH Prediction and Visualization via Neural Networks
Overview


Automated training, inference, and visualization

Time-resolved and treatment-specific pH monitoring

RGB-to-pH mapping with quantifiable performance metrics

There are two operational modes in this repository:

Fast Run Model — optimized for high-throughput, multi-treatment experimental datasets

Track History — ideal for continuous monitoring, longitudinal studies, and visual output generation

Features
Neural network for RGB-to-pH regression

Dataset creation from standard pH calibration images

Model persistence and reusability

Batch prediction on experimental images

Export of pH heatmaps as .svg and .tiff

Training loss visualization for model evaluation

Directory Structure
bash
Copy
Edit
.
├── main_model_fast_run.py       # Entry point for Fast Run execution
├── track_history.py             # Entry point for history tracking
├── model.pk00.keras             # Trained model (generated if not found)
├── pHCalib/                     # Standard pH reference images (e.g., 6_5.png)
├── experimental_data/           # Folder for experimental images (*.png)
├── ProcessedFigures/            # Fast Run output visualizations
├── outputs/                     # Track History prediction results
└── README.md                    # This documentation
Dependencies
Install required packages via pip:

bash
Copy
Edit
pip install numpy tensorflow scikit-learn matplotlib opencv-python scikit-image pillow tqdm
Input Data Format
Calibration Images (/pHCalib):
RGB .png or .tiff images labeled by pH value in the filename, e.g., 6_5.png → pH 6.5

Experimental Images (/experimental_data or nested under treatment/day folders):
.png images organized by treatment (e.g., Ctrl, RM), timepoint (e.g., SD_1), and replicate (R1–R4)

Methodology
1. Dataset Construction
Function: create_datasets_from_sample_ph()

Loads calibration images → crops → normalizes RGB → attaches pH labels

Output: np.ndarray of shape (N_pixels × N_images, 4) → [R, G, B, pH]

2. Model Architecture
Layer	Type	Units	Activation	Dropout
Input	Dense	64	ReLU	0.2
Hidden	Dense	32	ReLU	0.2
Output	Dense	1	Linear	0

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Epochs: 20

Batch Size: 50

Training/Test Split: 90% / 10%

3. Prediction & Visualization
Experimental images are:

Cropped and preprocessed

Flattened and normalized

Inferred through the trained model

Reshaped into 2D heatmaps

Heatmaps saved in:

.tiff (raw matrix)

.svg (colored visualization using RdYlGn, range: pH 5.4–8.0); Requires adjustment accordingly

4. Loss Curve Plotting
Training and validation loss are plotted using matplotlib

Helps evaluate convergence and model generalization

Execution
A. Fast Run (Batch Processing for Multi-Treatment)
bash
Copy
Edit
python main_model_fast_run.py
Performs:

Dataset creation from /pHCalib

Model training or loading

pH prediction on all experimental images

Output to /ProcessedFigures/ as multi-subplot grids (per treatment & timepoint)

Visualization Layout:

Rows: Raw → Predicted pH → Enhanced raw → Enhanced pH

Columns: Replicates (R1–R4) across days (1, 16, 64)

B. Track History (Single Run + Visual Exports)
bash
Copy
Edit
python track_history.py
Performs:

Dataset creation

Model training or loading

Prediction on experimental .png images in /experimental_data

Heatmap export to /outputs/ as .svg and .tiff

Final loss curve visualization displayed via matplotlib

Example Outputs
sample_pre.svg: Colorized pH map

sample_pre.tiff: Raw pH array

Training Curve: Saved or displayed showing loss over 20 epochs

Performance
Typical validation loss: <1 to 0.0076 pH square units (MSE)

Sufficient for resolving spatial pH trends in biological and chemical imaging contexts

Limitations
Calibration Range: Only predicts within pH range of training data (e.g., 5.5–8.0), adjust accordingly


Applications
Environmental monitoring (e.g., acidification, eutrophication)

Toxicological assessment in bioassays

Educational tools for colorimetric sensor interpretation

Citation & Contact
For academic use
