"""
src/
----
Core source package for the CKD Prediction project.

Modules
-------
config      : Single source of truth — all paths, constants, hyperparameters.
preprocess  : Full preprocessing pipeline — load, clean, split, impute, encode, scale.
evaluate    : Evaluation utilities — metrics, confusion matrix, ROC curves, comparison plots.

Usage in notebooks
------------------
    import sys
    sys.path.append("../src")   # add src/ to Python path

    from config     import RAW_DATA_PATH, NUMERICAL_COLS
    from preprocess import run_full_pipeline
    from evaluate   import build_results_table, plot_roc_curves

Usage in scripts
----------------
    import sys
    sys.path.append("src")

    from config import MODEL_PATH
"""

# Expose top-level imports so callers can do:
#   from src import config
#   from src import preprocess
#   from src import evaluate

from . import config
from . import preprocess
from . import evaluate