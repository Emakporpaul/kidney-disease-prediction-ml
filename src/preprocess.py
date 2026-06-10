"""
Reusable data preprocessing pipeline for the CKD prediction project.

This module handles every transformation step between raw CSV and
model-ready arrays. Functions are designed to be:

  - Called by notebooks for exploration and training
  - Called by app.py for real-time inference (single patient predictions)
  - Tested independently via tests/test_preprocess.py

Preprocessing order (critical — do not rearrange):
  1. Load raw data
  2. Coerce data types
  3. Clean dirty categorical values
  4. Train / test split        ← MUST happen before imputation
  5. Impute missing values     ← fit on train, apply to test
  6. Encode categorical cols   ← fit on train, apply to test
  7. Scale all features        ← fit on train, apply to test

Why split before imputing?
  Imputing on the full dataset before splitting causes data leakage —
  test set values influence the training imputation distribution.
  Always treat the test set as completely unseen data.

"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Import all configuration — no magic numbers or hardcoded values here
from config import (
    RAW_DATA_PATH,
    ORIGINAL_COLUMNS,
    RENAMED_COLUMNS,
    TARGET_COLUMN,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    DIRTY_VALUE_MAP,
    TARGET_CLEAN_MAP,
    TARGET_LABEL_MAP,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_DIR,
    ENCODER_PATH,
    SCALER_PATH,
    FEATURE_PATH,
)


# STEP 1 — LOAD

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw kidney disease CSV and apply clean column names.

    Parameters
    ----------
    path : Path to the CSV file (defaults to config.RAW_DATA_PATH)

    Returns
    -------
    pd.DataFrame with 25 columns (id dropped, columns renamed)
    """
    df = pd.read_csv(path)

    # Drop the id column — it's a row index, not a feature
    df.drop(columns=["id"], errors="ignore", inplace=True)

    # Rename short abbreviations to human-readable names
    # e.g. 'bp' → 'blood_pressure', 'hemo' → 'haemoglobin'
    df.columns = RENAMED_COLUMNS

    return df


# STEP 2 — TYPE COERCION

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric columns that were read as object (string) to float.

    Three columns — packed_cell_volume, white_blood_cell_count,
    red_blood_cell_count — contain occasional non-numeric characters
    in the raw CSV, causing pandas to read them as strings.
    We coerce them to float so they can be used in arithmetic operations.

    Parameters
    ----------
    df : Raw DataFrame from load_raw_data()

    Returns
    -------
    DataFrame with corrected dtypes
    """
    # Columns known to be stored as object despite being numeric
    force_numeric_cols = [
        "packed_cell_volume",
        "white_blood_cell_count",
        "red_blood_cell_count",
    ]

    for col in force_numeric_cols:
        # errors='coerce' turns unparseable values into NaN
        # rather than raising an exception
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# STEP 3 — CLEAN DIRTY VALUES

def clean_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix tab characters, extra spaces, and inconsistent labels in the dataset.

    Problems found in the raw data:
      - diabetes_mellitus     : '\\tno', '\\tyes', ' yes' (tab/space contamination)
      - coronary_artery_disease: '\\tno'
      - class (target)        : 'ckd\\t', 'notckd' (should be 'ckd', 'not ckd')

    Also encodes the target column to integers:
      1 = CKD     (disease present  → positive class)
      0 = Not CKD (disease absent   → negative class)

    Parameters
    ----------
    df : DataFrame from coerce_types()

    Returns
    -------
    Cleaned DataFrame with integer target column
    """
    df = df.copy()

    # Defensively strip leading/trailing whitespace and tab characters from
    # ALL categorical columns. The real UCI dataset only has tabs in
    # diabetes_mellitus and coronary_artery_disease, but stripping all of them
    # costs nothing and makes the pipeline robust against any future dirty data.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Apply column-specific value remappings (e.g. " yes" -> "yes")
    # after the strip so the mapping keys still match.
    for col, mapping in DIRTY_VALUE_MAP.items():
        df[col] = df[col].replace(mapping)

    # Clean and encode the target column
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()         # remove leading/trailing whitespace
        .str.lower()         # normalise to lowercase
        .replace(TARGET_CLEAN_MAP)   # fix 'ckd\t' → 'ckd', 'notckd' → 'not ckd'
        .map(TARGET_LABEL_MAP)       # 'ckd' → 1,  'not ckd' → 0
    )

    return df


# STEP 4 — TRAIN / TEST SPLIT

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and test sets.

    Uses stratified splitting to ensure both sets have the same
    class ratio as the full dataset (~62% CKD, ~38% Not CKD).

    Parameters
    ----------
    df : Cleaned DataFrame from clean_dirty_values()

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Separate features (X) from target (y)
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,       # 20% test, 80% train
        random_state=RANDOM_STATE, # fixed seed for reproducibility
        stratify=y,                # preserve class balance in both sets
    )

    return X_train, X_test, y_train, y_test


# STEP 5 — IMPUTE MISSING VALUES

def _random_sample_impute(
    train_col: pd.Series,
    target_col: pd.Series,
) -> pd.Series:
    """
    Fill NaN values in target_col by randomly sampling from train_col's
    observed (non-null) values.

    Why random sampling instead of mean/median?
    - Mean imputation creates an artificial spike at the mean
    - Random sampling preserves the original distribution shape
    - Better for columns with high missingness (e.g. rc: 32%, wc: 26%)

    Parameters
    ----------
    train_col  : The training column (used to learn the distribution)
    target_col : The column to impute (could be train or test)

    Returns
    -------
    Imputed Series with no NaN values
    """
    n_missing = target_col.isna().sum()
    if n_missing == 0:
        return target_col  # nothing to impute

    # Sample from training distribution only — no test leakage
    observed = train_col.dropna()
    fill_values = observed.sample(n_missing, replace=True, random_state=RANDOM_STATE)
    fill_values.index = target_col[target_col.isna()].index

    target_col = target_col.copy()
    target_col.loc[target_col.isna()] = fill_values

    return target_col


def impute_missing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in both train and test sets.

    Strategy:
      - Numerical columns   → random-sample from training distribution
      - Categorical columns → mode of training set

    Both strategies are fitted on training data only, then applied to test.
    This prevents any test set information from influencing training.

    Parameters
    ----------
    X_train : Training feature DataFrame
    X_test  : Test feature DataFrame

    Returns
    -------
    (X_train_imputed, X_test_imputed) — both with zero missing values
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    # Get only the feature columns that exist in our DataFrames
    num_cols = [c for c in NUMERICAL_COLS   if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    # Impute numerical columns with random sampling
    for col in num_cols:
        X_train[col] = _random_sample_impute(X_train[col], X_train[col])
        X_test[col]  = _random_sample_impute(X_train[col], X_test[col])

    # Impute categorical columns with mode (most frequent value)
    for col in cat_cols:
        mode_val = X_train[col].mode()[0]   # computed from training only
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col]  = X_test[col].fillna(mode_val)

    return X_train, X_test


# STEP 6 — ENCODE CATEGORICAL FEATURES

def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit: bool = True,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Label-encode each categorical column.

    Each column gets its OWN LabelEncoder instance.
    Using one shared encoder across all columns is a common bug —
    it only remembers the last column's mapping.

    Parameters
    ----------
    X_train  : Training features (post-imputation)
    X_test   : Test features (post-imputation)
    fit      : True  → fit new encoders on X_train (training mode)
               False → use supplied encoders (inference mode for API)
    encoders : Pre-fitted encoders dict — required when fit=False

    Returns
    -------
    (X_train_encoded, X_test_encoded, encoders_dict)
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    if fit:
        # Training mode — fit one encoder per column on training data
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])  # fit + transform train
            X_test[col]  = le.transform(X_test[col])        # transform test only
            encoders[col] = le                              # store for later use
    else:
        # Inference mode — use pre-fitted encoders (called by the API)
        if encoders is None:
            raise ValueError(
                "encoders dict must be supplied when fit=False. "
                "Load from model/label_encoders.pkl first."
            )
        for col in cat_cols:
            if col in X_train.columns:
                X_train[col] = encoders[col].transform(X_train[col])
            if col in X_test.columns:
                X_test[col] = encoders[col].transform(X_test[col])

    return X_train, X_test, encoders


# STEP 7 — SCALE FEATURES

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardise features to mean=0, std=1 using StandardScaler.

    Why scaling matters:
      - KNN uses Euclidean distance — white_blood_cell_count (range 0–7800)
        would completely dominate specific_gravity (range 1.005–1.025)
      - SVM's RBF kernel is equally scale-sensitive
      - Logistic Regression converges faster with scaled inputs
      - Tree-based models (RF, XGBoost) do NOT need scaling, but it
        doesn't hurt them either

    Parameters
    ----------
    X_train : Encoded training features
    X_test  : Encoded test features
    fit     : True  → fit scaler on X_train (training mode)
              False → use supplied scaler (inference mode for API)
    scaler  : Pre-fitted scaler — required when fit=False

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
    """
    if fit:
        # Fit ONLY on training data — using test data here would be leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        # Inference mode — apply same scaling learned from training
        if scaler is None:
            raise ValueError(
                "scaler must be supplied when fit=False. "
                "Load from model/scaler.pkl first."
            )
        X_train_scaled = scaler.transform(X_train)

    # Always transform test with the same scaler fitted on train
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# FULL PIPELINE — runs all steps in the correct order

def run_full_pipeline(
    path: Path = RAW_DATA_PATH,
    save_artefacts: bool = True,
) -> dict:
    """
    Run the complete preprocessing pipeline end-to-end.

    This is a convenience function that calls every step in the correct
    order and returns everything needed for model training.

    Parameters
    ----------
    path            : Path to raw CSV file
    save_artefacts  : If True, saves scaler, encoders, and feature names
                      to the model/ folder for use by the API

    Returns
    -------
    Dictionary with keys:
        df            — fully cleaned DataFrame
        X_train       — raw (unscaled) training features
        X_test        — raw (unscaled) test features
        y_train       — training labels
        y_test        — test labels
        X_train_sc    — scaled training features (numpy array)
        X_test_sc     — scaled test features (numpy array)
        scaler        — fitted StandardScaler
        encoders      — dict of fitted LabelEncoders
        feature_names — ordered list of feature column names
    """
    # Step 1 — Load
    df = load_raw_data(path)

    # Step 2 — Fix types
    df = coerce_types(df)

    # Step 3 — Clean dirty values & encode target
    df = clean_dirty_values(df)

    # Step 4 — Split (BEFORE imputation to prevent leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5 — Impute (fitted on train, applied to both)
    X_train, X_test = impute_missing(X_train, X_test)

    # Step 6 — Encode categoricals (fitted on train, applied to both)
    X_train, X_test, encoders = encode_categoricals(X_train, X_test, fit=True)

    # Step 7 — Scale (fitted on train, applied to both)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, fit=True)

    # Save artefacts to model/ folder so the API can load them
    if save_artefacts:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoders, f)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        with open(FEATURE_PATH, "w") as f:
            json.dump(list(X_train.columns), f)

    return {
        "df":            df,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "X_train_sc":    X_train_sc,
        "X_test_sc":     X_test_sc,
        "scaler":        scaler,
        "encoders":      encoders,
        "feature_names": list(X_train.columns),
    }


# INFERENCE PIPELINE — used by the API for single patient predictions

def preprocess_single_patient(
    patient_data: dict,
    encoders: dict,
    scaler: StandardScaler,
    feature_names: list,
) -> np.ndarray:
    """
    Preprocess a single patient's raw input data for model inference.

    This replicates every transformation applied during training so the
    model receives data in exactly the same format it was trained on.

    Parameters
    ----------
    patient_data  : Dict of raw feature values (from API request)
    encoders      : Fitted LabelEncoders loaded from model/label_encoders.pkl
    scaler        : Fitted StandardScaler loaded from model/scaler.pkl
    feature_names : Ordered feature list loaded from model/feature_names.json

    Returns
    -------
    numpy array of shape (1, n_features) ready for model.predict()
    """
    # Convert dict to single-row DataFrame in the correct column order
    df = pd.DataFrame([patient_data])[feature_names]

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Apply the same label encoding used during training
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # Apply the same scaling used during training
    return scaler.transform(df)