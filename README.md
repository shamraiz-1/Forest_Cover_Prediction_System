# Forest_Cover_Prediction_System
an ML project using Random Forest &amp; XGBoost to predict forest types. Features an interactive GUI with CSV/GZ upload, real-time predictions, accuracy reports, and in-GUI visualizations.
Forest Cover Predictor

Forest Cover Predictor is a Python machine learning project that classifies forest cover types using Random Forest and XGBoost models. It includes a Tkinter GUI for uploading datasets, viewing predictions, accuracy metrics, and interactive plots.

Features

Predict forest cover types from cartographic and environmental features.

Compare Random Forest and XGBoost models with accuracy scores and classification reports.

Visualize confusion matrices and predicted vs actual counts directly inside the GUI.

Supports both CSV and GZ datasets.

Fully interactive and beginner-friendly GUI.

Project Structure
Forest-Cover-Predictor/
│
├─ trainandeval.py          # Script to train Random Forest and XGBoost models
├─ forestcover_gui.py       # Tkinter GUI to upload data and view predictions
├─ README.md                # Project documentation
├─ requirements.txt         # Python dependencies
└─ (models are NOT included due to size)

Installation

Clone the repository:

git clone https://github.com/YourUsername/Forest-Cover-Predictor.git
cd Forest-Cover-Predictor


Create a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux


Install dependencies:

pip install -r requirements.txt

Dataset

This project uses the UCI Covertype dataset.

You can download it here: Covertype dataset

Place the dataset (.csv or .gz) in the project folder before running the scripts.

How to Train Models

Run the training script to generate Random Forest and XGBoost models:

python trainandeval.py


This will train the models and save:

random_forest_model.joblib

xgb_model.joblib

Note: These files are not included in the repo due to size.

How to Use the GUI

Run the GUI script:

python forestcover_gui.py


Click “Upload CSV/GZ for Prediction”.

Select a dataset with all required features (including Cover_Type if you want evaluation).

View results:

Accuracy score and classification report in the text box.

Confusion matrix and predicted vs actual plots displayed inside the GUI.

Required Features in Dataset
Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,
Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4,
Soil_Type1 ... Soil_Type40, Cover_Type (optional)

Notes

If you don’t have Cover_Type in your CSV, the GUI will still predict but won’t display accuracy or confusion matrix.

XGBoost model labels are adjusted to match the dataset (1–7).

Dependencies

Python 3.10+

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

joblib

tkinter (built-in)

You can also install all dependencies via:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

License

This project is open-source under the MIT License.
