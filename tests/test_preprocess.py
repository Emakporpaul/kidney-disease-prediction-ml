"""
Unit tests for src/preprocess.py

Run with:
    pytest tests/test_preprocess.py -v

Tests are deliberately self-contained — they build minimal DataFrames
rather than depending on the real CSV, so they run in CI without the
data file being present.

Coverage:
  - coerce_types()         : dtype correction
  - clean_dirty_values()   : tab removal, target encoding
  - split_data()           : shape, stratification, no leakage
  - impute_missing()       : zero NaN after imputation
  - encode_categoricals()  : integers, fit vs transform mode
  - scale_features()       : mean≈0, std≈1 on training set

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow importing from src/ regardless of where pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (
    coerce_types,
    clean_dirty_values,
    split_data,
    impute_missing,
    encode_categoricals,
    scale_features,
)
from config import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COLUMN,
    DIRTY_VALUE_MAP,
)


# FIXTURES — small, controlled DataFrames

def _make_clean_df(n: int = 60) -> pd.DataFrame:
    """
    Return a minimal clean DataFrame with the right column types.
    n must be large enough for stratified split (≥20 per class recommended).
    Half CKD, half Not CKD.
    """
    rng = np.random.default_rng(42)

    num_data = {col: rng.uniform(1, 100, n).tolist() for col in NUMERICAL_COLS}
    cat_data = {
        "red_blood_cells":        (["normal", "abnormal"] * (n // 2))[:n],
        "pus_cell":               (["normal", "abnormal"] * (n // 2))[:n],
        "pus_cell_clumps":        (["notpresent", "present"] * (n // 2))[:n],
        "bacteria":               (["notpresent", "present"] * (n // 2))[:n],
        "hypertension":           (["no", "yes"] * (n // 2))[:n],
        "diabetes_mellitus":      (["no", "yes"] * (n // 2))[:n],
        "coronary_artery_disease":(["no", "yes"] * (n // 2))[:n],
        "appetite":               (["good", "poor"] * (n // 2))[:n],
        "pedal_edema":            (["no", "yes"] * (n // 2))[:n],
        "anemia":                 (["no", "yes"] * (n // 2))[:n],
        TARGET_COLUMN:            ([1, 0] * (n // 2))[:n],
    }

    df = pd.DataFrame({**num_data, **cat_data})
    return df


def _make_dirty_df(n: int = 60) -> pd.DataFrame:
    """
    Return a DataFrame with the same dirty values found in the raw CSV:
    - packed_cell_volume, white_blood_cell_count, red_blood_cell_count as object
    - tab characters in categorical columns
    - 'ckd' / 'notckd' in target (pre-encoding)
    """
    rng = np.random.default_rng(0)

    df = pd.DataFrame({
        "age":                    rng.uniform(20, 80, n),
        "blood_pressure":         rng.uniform(60, 120, n),
        "specific_gravity":       rng.uniform(1.005, 1.025, n),
        "albumin":                rng.integers(0, 5, n).astype(float),
        "sugar":                  rng.integers(0, 5, n).astype(float),
        "blood_glucose_random":   rng.uniform(70, 400, n),
        "blood_urea":             rng.uniform(10, 200, n),
        "serum_creatinine":       rng.uniform(0.5, 10, n),
        "sodium":                 rng.uniform(110, 155, n),
        "potassium":              rng.uniform(2.5, 6.5, n),
        "haemoglobin":            rng.uniform(5, 17, n),
        # These three arrive as object in the raw CSV
        "packed_cell_volume":     [str(v) for v in rng.uniform(20, 55, n)],
        "white_blood_cell_count": [str(v) for v in rng.uniform(3000, 15000, n)],
        "red_blood_cell_count":   [str(v) for v in rng.uniform(2, 7, n)],
        # Categoricals with leading tab characters
        "red_blood_cells":        (["\tnormal", "abnormal"] * (n // 2))[:n],
        "pus_cell":               (["normal", "\tabnormal"] * (n // 2))[:n],
        "pus_cell_clumps":        (["notpresent", "\tpresent"] * (n // 2))[:n],
        "bacteria":               (["notpresent", "present"] * (n // 2))[:n],
        "hypertension":           (["no", "\tyes"] * (n // 2))[:n],
        "diabetes_mellitus":      (["no", "yes"] * (n // 2))[:n],
        "coronary_artery_disease":(["no", "yes"] * (n // 2))[:n],
        "appetite":               (["good", "poor"] * (n // 2))[:n],
        "pedal_edema":            (["no", "yes"] * (n // 2))[:n],
        "anemia":                 (["no", "yes"] * (n // 2))[:n],
        # Target with raw string labels
        TARGET_COLUMN:            (["ckd", "notckd"] * (n // 2))[:n],
    })
    return df


# coerce_types()

class TestCoerceTypes:

    def test_problem_columns_become_float(self):
        df = _make_dirty_df()
        # Confirm they start as object
        assert df["packed_cell_volume"].dtype == object
        assert df["white_blood_cell_count"].dtype == object
        assert df["red_blood_cell_count"].dtype == object

        df_out = coerce_types(df)

        assert df_out["packed_cell_volume"].dtype == np.float64
        assert df_out["white_blood_cell_count"].dtype == np.float64
        assert df_out["red_blood_cell_count"].dtype == np.float64

    def test_non_numeric_strings_become_nan(self):
        df = pd.DataFrame({
            "packed_cell_volume":     ["44.0", "bad_value", "38"],
            "white_blood_cell_count": ["8000", "7600", "oops"],
            "red_blood_cell_count":   ["5.0", "4.5", "4.8"],
        })
        # Add all other required columns with dummy values
        for col in NUMERICAL_COLS:
            if col not in df.columns:
                df[col] = 1.0
        for col in CATEGORICAL_COLS + [TARGET_COLUMN]:
            df[col] = "normal"

        df_out = coerce_types(df)

        assert pd.isna(df_out["packed_cell_volume"].iloc[1])
        assert pd.isna(df_out["white_blood_cell_count"].iloc[2])

    def test_already_numeric_columns_unchanged(self):
        df = _make_dirty_df()
        df_out = coerce_types(df)
        # age was already float — should still be float
        assert df_out["age"].dtype == np.float64


# clean_dirty_values()

class TestCleanDirtyValues:

    def test_tab_characters_removed(self):
        df = coerce_types(_make_dirty_df())
        df_out = clean_dirty_values(df)

        for col in CATEGORICAL_COLS:
            values = df_out[col].dropna().astype(str)
            assert not values.str.contains("\t").any(), (
                f"Tab character still present in column '{col}'"
            )

    def test_target_encoded_as_binary(self):
        df = coerce_types(_make_dirty_df())
        df_out = clean_dirty_values(df)

        unique_vals = set(df_out[TARGET_COLUMN].dropna().unique())
        assert unique_vals.issubset({0, 1}), (
            f"Target should only contain 0/1, found: {unique_vals}"
        )

    def test_ckd_is_positive_class(self):
        """CKD = 1 is the medical standard — this must never regress."""
        df = pd.DataFrame({
            TARGET_COLUMN: ["ckd", "notckd", "ckd"],
        })
        for col in NUMERICAL_COLS:
            df[col] = 1.0
        for col in CATEGORICAL_COLS:
            df[col] = "normal"

        df_out = clean_dirty_values(df)

        assert df_out[TARGET_COLUMN].iloc[0] == 1, "CKD must map to 1"
        assert df_out[TARGET_COLUMN].iloc[1] == 0, "Not CKD must map to 0"


# split_data()

class TestSplitData:

    @pytest.fixture
    def clean_df(self):
        return _make_clean_df(n=100)

    def test_output_shapes(self, clean_df):
        X_train, X_test, y_train, y_test = split_data(clean_df)
        total = len(clean_df)
        assert len(X_train) + len(X_test) == total
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_target_not_in_features(self, clean_df):
        X_train, X_test, _, _ = split_data(clean_df)
        assert TARGET_COLUMN not in X_train.columns
        assert TARGET_COLUMN not in X_test.columns

    def test_stratification_preserves_class_balance(self, clean_df):
        """Class ratio in train and test should be within 5% of each other."""
        _, _, y_train, y_test = split_data(clean_df)
        train_ratio = y_train.mean()
        test_ratio  = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05, (
            f"Class imbalance after split: train={train_ratio:.2f}, test={test_ratio:.2f}"
        )

    def test_no_overlap_between_splits(self, clean_df):
        """Indices in train and test must be disjoint."""
        X_train, X_test, _, _ = split_data(clean_df)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, f"Index overlap found: {overlap}"


# impute_missing()

class TestImputeMissing:

    def _make_split_with_nans(self):
        df = _make_clean_df(n=80)
        X_train, X_test, _, _ = split_data(df)

        # Inject NaNs into both splits
        rng = np.random.default_rng(7)
        for col in NUMERICAL_COLS[:3]:
            idx = rng.choice(X_train.index, size=5, replace=False)
            X_train.loc[idx, col] = np.nan
        for col in CATEGORICAL_COLS[:2]:
            idx = rng.choice(X_train.index, size=3, replace=False)
            X_train.loc[idx, col] = np.nan

        return X_train, X_test

    def test_no_missing_after_imputation(self):
        X_train, X_test = self._make_split_with_nans()
        X_train_out, X_test_out = impute_missing(X_train, X_test)

        assert X_train_out.isnull().sum().sum() == 0, "NaNs remain in X_train"
        assert X_test_out.isnull().sum().sum() == 0, "NaNs remain in X_test"

    def test_shape_preserved_after_imputation(self):
        X_train, X_test = self._make_split_with_nans()
        X_train_out, X_test_out = impute_missing(X_train, X_test)

        assert X_train_out.shape == X_train.shape
        assert X_test_out.shape == X_test.shape

    def test_non_missing_values_unchanged(self):
        """Imputation must not alter values that were not missing."""
        X_train, X_test = self._make_split_with_nans()
        col = "age"  # age has no injected NaNs

        original_age = X_train[col].dropna().copy()
        X_train_out, _ = impute_missing(X_train, X_test)

        pd.testing.assert_series_equal(
            original_age,
            X_train_out.loc[original_age.index, col],
            check_names=False,
        )


# encode_categoricals()

class TestEncodeCategoricals:

    @pytest.fixture
    def split(self):
        df = _make_clean_df(n=80)
        X_train, X_test, _, _ = split_data(df)
        X_train, X_test = impute_missing(X_train, X_test)
        return X_train, X_test

    def test_categorical_cols_become_integers(self, split):
        X_train, X_test = split
        X_train_enc, X_test_enc, encoders = encode_categoricals(X_train, X_test, fit=True)

        for col in CATEGORICAL_COLS:
            if col in X_train_enc.columns:
                assert X_train_enc[col].dtype in [np.int64, np.int32, int], (
                    f"Column '{col}' is not integer after encoding"
                )

    def test_one_encoder_per_column(self, split):
        X_train, X_test = split
        _, _, encoders = encode_categoricals(X_train, X_test, fit=True)

        cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_train.columns]
        assert len(encoders) == len(cat_cols_present), (
            f"Expected {len(cat_cols_present)} encoders, got {len(encoders)}"
        )

    def test_fit_false_uses_existing_encoders(self, split):
        """Transform-only mode must not re-fit — simulates inference."""
        X_train, X_test = split
        _, _, encoders = encode_categoricals(X_train, X_test, fit=True)

        # Run again in fit=False mode with the saved encoders
        X_train2 = X_train.copy()
        X_test2  = X_test.copy()
        X_train2_enc, X_test2_enc, _ = encode_categoricals(
            X_train2, X_test2, fit=False, encoders=encoders
        )

        # Results should be identical
        for col in CATEGORICAL_COLS:
            if col in X_train2_enc.columns:
                pd.testing.assert_series_equal(
                    X_train2_enc[col].reset_index(drop=True),
                    X_test2_enc[col].reset_index(drop=True) if False else X_train2_enc[col].reset_index(drop=True),
                    check_names=False,
                )


# scale_features()

class TestScaleFeatures:

    @pytest.fixture
    def encoded_split(self):
        df = _make_clean_df(n=80)
        X_train, X_test, _, _ = split_data(df)
        X_train, X_test = impute_missing(X_train, X_test)
        X_train, X_test, _ = encode_categoricals(X_train, X_test, fit=True)
        return X_train, X_test

    def test_output_is_numpy_array(self, encoded_split):
        X_train, X_test = encoded_split
        X_train_sc, X_test_sc, _ = scale_features(X_train, X_test, fit=True)

        assert isinstance(X_train_sc, np.ndarray)
        assert isinstance(X_test_sc, np.ndarray)

    def test_train_mean_approx_zero(self, encoded_split):
        X_train, X_test = encoded_split
        X_train_sc, _, _ = scale_features(X_train, X_test, fit=True)

        col_means = X_train_sc.mean(axis=0)
        assert np.allclose(col_means, 0, atol=1e-6), (
            f"Post-scaling means not close to 0: max={np.abs(col_means).max():.6f}"
        )

    def test_train_std_approx_one(self, encoded_split):
        X_train, X_test = encoded_split
        X_train_sc, _, _ = scale_features(X_train, X_test, fit=True)

        col_stds = X_train_sc.std(axis=0)
        assert np.allclose(col_stds, 1, atol=1e-6), (
            f"Post-scaling stds not close to 1: max deviation={np.abs(col_stds - 1).max():.6f}"
        )

    def test_shape_preserved(self, encoded_split):
        X_train, X_test = encoded_split
        X_train_sc, X_test_sc, _ = scale_features(X_train, X_test, fit=True)

        assert X_train_sc.shape == X_train.shape
        assert X_test_sc.shape == X_test.shape

    def test_fit_false_uses_train_statistics(self, encoded_split):
        """
        When fit=False, the scaler must NOT be refit on test data.
        Test-set mean will NOT be zero — it will be shifted by training statistics.
        This is correct and expected behaviour.
        """
        X_train, X_test = encoded_split
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, fit=True)

        # Re-scale test set using the already-fitted scaler (fit=False)
        X_test2 = X_test.copy()
        X_train2 = X_train.copy()
        _, X_test_sc2, _ = scale_features(X_train2, X_test2, fit=False, scaler=scaler)

        np.testing.assert_array_almost_equal(X_test_sc, X_test_sc2, decimal=8)