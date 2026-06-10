"""
config.py
---------
Single source of truth for the entire CKD prediction project.

Every path, constant, column name, and hyperparameter is defined here.
No other file should hardcode these values — always import from config.

Author  : EMAKPOR PAUL || ML Engineer
Project : Chronic Kidney Disease Prediction
"""

from pathlib import Path


# DIRECTORY PATHS
# Using pathlib.Path makes paths work on Windows, Mac, and Linux automatically.
# No more hardcoded C:\Users\HP\Desktop\... paths.

# Root of the project (the ml-kidney/ folder)
# __file__ is this config.py → .parent is src/ → .parent again is ml-kidney/
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR          = ROOT_DIR / "data"           # raw datasets
MODEL_DIR         = ROOT_DIR / "model"          # saved model artefacts
VIZ_DIR           = ROOT_DIR / "visualizations" # saved plots
NOTEBOOK_DIR      = ROOT_DIR / "notebooks"      # jupyter notebooks
SRC_DIR           = ROOT_DIR / "src"            # source code


# FILE PATHS

RAW_DATA_PATH = DATA_DIR / "kidney_disease.csv"  # raw input data

MODEL_PATH        = MODEL_DIR / "kidney_model.pkl"     # trained best model
SCALER_PATH       = MODEL_DIR / "scaler.pkl"           # fitted StandardScaler
ENCODER_PATH      = MODEL_DIR / "label_encoders.pkl"   # fitted LabelEncoders
FEATURE_PATH      = MODEL_DIR / "feature_names.json"   # ordered feature list
MODEL_RESULTS_PATH= MODEL_DIR / "model_results.csv"    # metrics for all models


# COLUMN DEFINITIONS
# Defined once here so renaming a column only requires one change.

# Original short column names as they appear in the raw CSV
ORIGINAL_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "classification",
]

# Human-readable column names we rename to
RENAMED_COLUMNS = [
    "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
    "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria",
    "blood_glucose_random", "blood_urea", "serum_creatinine",
    "sodium", "potassium", "haemoglobin", "packed_cell_volume",
    "white_blood_cell_count", "red_blood_cell_count",
    "hypertension", "diabetes_mellitus", "coronary_artery_disease",
    "appetite", "pedal_edema", "anemia", "class",
]

# Target column name (what we are predicting)
TARGET_COLUMN = "class"

# Numerical feature columns (continuous/discrete numeric measurements)
NUMERICAL_COLS = [
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "sugar",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "potassium",
    "haemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
]

# Categorical feature columns (text labels that need encoding)
CATEGORICAL_COLS = [
    "red_blood_cells",
    "pus_cell",
    "pus_cell_clumps",
    "bacteria",
    "hypertension",
    "diabetes_mellitus",
    "coronary_artery_disease",
    "appetite",
    "pedal_edema",
    "anemia",
]


# DATA CLEANING MAPS
# Lookup tables for fixing dirty/inconsistent values in the raw dataset.

# Tab characters and extra spaces found in categorical columns
DIRTY_VALUE_MAP = {
    "diabetes_mellitus":       {"\tno": "no", "\tyes": "yes", " yes": "yes"},
    "coronary_artery_disease": {"\tno": "no"},
}

# Target column has 3 dirty variants → normalise to 2 clean values
TARGET_CLEAN_MAP = {
    "ckd\t":  "ckd",
    "notckd": "not ckd",
    "ckd":    "ckd",
}

# Encode target as integer:
# 1 = CKD (disease PRESENT  → positive class)
# 0 = Not CKD (disease absent → negative class)
# CKD must be 1 so that precision/recall/F1 report on the disease correctly.
TARGET_LABEL_MAP = {"ckd": 1, "not ckd": 0}


# TRAIN / TEST SPLIT SETTINGS

TEST_SIZE    = 0.20   # 20% of data held out for final evaluation
RANDOM_STATE = 42     # fixed seed → reproducible splits every run

# MODEL HYPERPARAMETERS
# Centralised here so tuning changes propagate to both notebooks and training scripts.

KNN_PARAMS = {
    "n_neighbors": 5,
}

SVM_PARAMS = {
    "C":           10,
    "gamma":       0.001,
    "probability": True,   # needed for predict_proba → ROC curves
}

DECISION_TREE_PARAMS = {
    "criterion":          "gini",
    "max_depth":          7,
    "max_features":       "sqrt",
    "min_samples_leaf":   2,
    "min_samples_split":  7,
    "random_state":       RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators":      100,
    "criterion":         "gini",
    "max_depth":         10,
    "max_features":      "sqrt",
    "min_samples_split": 7,
    "random_state":      RANDOM_STATE,
}

XGB_PARAMS = {
    "objective":     "binary:logistic",
    "learning_rate": 0.01,
    "max_depth":     10,
    "n_estimators":  100,
    "eval_metric":   "logloss",   # suppresses XGBoost v2 warnings
    "random_state":  RANDOM_STATE,
}

GRADIENT_BOOSTING_PARAMS = {
    "loss":          "exponential",
    "learning_rate": 1.0,
    "n_estimators":  100,
    "random_state":  RANDOM_STATE,
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter":    1000,     # prevents convergence warnings
    "random_state": RANDOM_STATE,
}

# VISUALISATION SETTINGS

FIG_DPI      = 150            # dots per inch when saving plots
PLOT_STYLE   = "ggplot"       # matplotlib style
PALETTE      = "colorblind"   # seaborn colour palette (accessible)

# Colours used consistently across all plots
COLOR_CKD    = "#e74c3c"   # red  — CKD (disease present)
COLOR_NOTCKD = "#2980b9"   # blue — Not CKD (disease absent)