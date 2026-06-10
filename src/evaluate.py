"""
evaluate.py
-----------
Evaluation utilities for the CKD prediction project.

This module provides everything needed to assess model performance:
    - Single model metrics  (accuracy, precision, recall, F1, ROC-AUC)
    - Classification report (per-class breakdown)
    - Confusion matrix plot
    - ROC curve plot        (all models on one axes)
    - Feature importance    (tree-based models)
    - Model comparison      (grouped bar chart)
    - Results table         (sorted DataFrame of all model metrics)

Why a separate module?
    Evaluation code is identical whether called from a notebook, a training
    script, or a CI test. Centralising it here avoids copy-paste and ensures
    every model is assessed with the exact same metrics.

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import metrics

from config import (
    VIZ_DIR,
    FIG_DPI,
    PLOT_STYLE,
    COLOR_CKD,
    COLOR_NOTCKD,
)

# Ensure the visualizations folder exists before any plot is saved
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# SINGLE MODEL METRICS

def get_metrics(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute a full set of evaluation metrics for one trained model.

    Metrics chosen and why:
        accuracy  — overall correct predictions (misleading on imbalanced data alone)
        precision — of all predicted CKD, how many actually have CKD
                    (low precision = too many false alarms)
        recall    — of all actual CKD patients, how many did we catch
                    (low recall = missing real patients — dangerous in clinical use)
        f1        — harmonic mean of precision and recall (balanced metric)
        roc_auc   — model's ability to rank CKD patients above non-CKD
                    across ALL decision thresholds (best single metric for
                    clinical binary classification)

    Parameters
    ----------
    name   : str         — display name for the model
    model  : fitted model — must implement predict() and predict_proba()
    X_test : np.ndarray  — scaled test features
    y_test : np.ndarray  — true labels (1=CKD, 0=Not CKD)

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)

    # predict_proba gives probability scores needed for ROC-AUC
    # decision_function is the fallback for models like LinearSVC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]  # probability of CKD (class 1)
    else:
        y_prob = model.decision_function(X_test)

    return {
        "model":     name,
        "accuracy":  round(metrics.accuracy_score(y_test, y_pred), 4),
        "precision": round(metrics.precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(metrics.recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(metrics.f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(metrics.roc_auc_score(y_test, y_prob), 4),
    }


def print_classification_report(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Print a formatted sklearn classification report for one model.

    Shows per-class precision, recall, F1, and support.
    Class labels are shown as 'Not CKD' and 'CKD' for readability.

    Parameters
    ----------
    name   : str
    model  : fitted model
    X_test : np.ndarray
    y_test : np.ndarray
    """
    y_pred = model.predict(X_test)

    print(f"\n{'═' * 58}")
    print(f"  {name}")
    print(f"{'═' * 58}")
    print(
        metrics.classification_report(
            y_test,
            y_pred,
            target_names=["Not CKD (0)", "CKD (1)"],  # explicit labels prevent confusion
        )
    )


# CONFUSION MATRIX

def plot_confusion_matrix(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save: bool = True,
) -> None:
    """
    Plot a styled confusion matrix for one model.

    Reading a confusion matrix:
        True Negative  (TN) top-left  — predicted Not CKD, actually Not CKD ✓
        False Positive (FP) top-right — predicted CKD, actually Not CKD (false alarm)
        False Negative (FN) bot-left  — predicted Not CKD, actually CKD ⚠ (dangerous)
        True Positive  (TP) bot-right — predicted CKD, actually CKD ✓

        In clinical settings, False Negatives (missed CKD patients) are the
        most costly error — we want high recall to minimise FN.

    Parameters
    ----------
    name  : str
    model : fitted model
    X_test, y_test : np.ndarray
    save  : bool — if True, saves plot to visualizations/
    """
    plt.style.use(PLOT_STYLE)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not CKD", "CKD"],
        yticklabels=["Not CKD", "CKD"],
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save:
        filename = f"cm_{name.lower().replace(' ', '_')}.png"
        fig.savefig(VIZ_DIR / filename, dpi=FIG_DPI, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_all_confusion_matrices(
    model_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save: bool = True,
) -> None:
    """
    Plot confusion matrices for all models in a single grid figure.

    Parameters
    ----------
    model_dict : dict — {model_name: fitted_model}
    X_test, y_test : np.ndarray
    save : bool
    """
    plt.style.use(PLOT_STYLE)

    n_models = len(model_dict)
    n_cols   = 4
    n_rows   = -(-n_models // n_cols)  # ceiling division

    fig = plt.figure(figsize=(16, n_rows * 4.5))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.4)

    for i, (name, model) in enumerate(model_dict.items()):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        y_pred = model.predict(X_test)
        cm     = metrics.confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not CKD", "CKD"],
            yticklabels=["Not CKD", "CKD"],
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    # Hide any unused subplot slots
    total_slots = n_rows * n_cols
    for j in range(n_models, total_slots):
        row, col = divmod(j, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    plt.suptitle(
        "Confusion Matrices — All Models",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    if save:
        fig.savefig(
            VIZ_DIR / "confusion_matrices_all.png",
            dpi=FIG_DPI,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()


# ROC CURVES

def plot_roc_curves(
    model_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save: bool = True,
) -> None:
    """
    Plot ROC curves for all models on a single axes.

    The ROC curve plots:
        x-axis — False Positive Rate (FPR) = FP / (FP + TN)
        y-axis — True Positive Rate  (TPR) = TP / (TP + FN) = Recall

    At each decision threshold the model produces one (FPR, TPR) point.
    The curve traces all thresholds from 0 to 1.

    AUC interpretation:
        1.0 — perfect classifier
        0.5 — random guessing (the diagonal dashed line)
        < 0.5 — worse than random (something is wrong)

    Parameters
    ----------
    model_dict : dict — {model_name: fitted_model}
    X_test, y_test : np.ndarray
    save : bool
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use a colourblind-friendly palette with enough colours for 7 models
    colors = plt.cm.tab10(np.linspace(0, 0.85, len(model_dict)))

    for (name, model), color in zip(model_dict.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
        auc          = metrics.roc_auc_score(y_test, y_prob)

        ax.plot(
            fpr, tpr,
            label=f"{name}  (AUC = {auc:.3f})",
            color=color,
            linewidth=2.2,
        )

    # Reference line — random classifier baseline
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="grey",
        linewidth=1.5,
        label="Random Classifier (AUC = 0.500)",
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — CKD Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save:
        fig.savefig(VIZ_DIR / "roc_curves.png", dpi=FIG_DPI, bbox_inches="tight")

    plt.show()
    plt.close()


# FEATURE IMPORTANCE

def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str = "Random Forest",
    top_n: int = None,
    save: bool = True,
) -> None:
    """
    Plot feature importances for tree-based models (RF, XGBoost, GBC, DT).

    Feature importance = average reduction in Gini impurity contributed
    by each feature across all trees. Higher = more predictive.

    This plot answers the clinical question:
        "Which lab measurements are most useful for diagnosing CKD?"

    Parameters
    ----------
    model        : fitted tree-based model with feature_importances_ attribute
    feature_names: list — ordered list matching model's input features
    model_name   : str  — used in the plot title
    top_n        : int  — show only top N features (None = show all)
    save         : bool
    """
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} does not expose feature importances — skipping.")
        return

    plt.style.use(PLOT_STYLE)

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    if top_n:
        importances = importances.tail(top_n)

    # Highlight top-quartile features in a different colour
    threshold = importances.quantile(0.75)
    colors    = [COLOR_CKD if v >= threshold else "#6c5ce7" for v in importances.values]

    fig, ax = plt.subplots(figsize=(9, max(6, len(importances) * 0.4)))

    ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
    ax.axvline(
        importances.mean(),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean importance ({importances.mean():.3f})",
    )

    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(
        f"Feature Importances — {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(VIZ_DIR / filename, dpi=FIG_DPI, bbox_inches="tight")

    plt.show()
    plt.close()


# MODEL COMPARISON

def build_results_table(
    model_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Build a sorted comparison table of metrics for all models.

    Parameters
    ----------
    model_dict : dict — {model_name: fitted_model}
    X_test, y_test : np.ndarray

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = [model, accuracy, precision, recall, f1, roc_auc]
        Sorted by roc_auc descending (best model first).
    """
    rows = [
        get_metrics(name, model, X_test, y_test)
        for name, model in model_dict.items()
    ]

    results_df = (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    return results_df


def plot_model_comparison(
    results_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """
    Grouped bar chart comparing Accuracy and ROC-AUC across all models.

    Sorting by ROC-AUC gives a clear visual ranking of model performance.
    Value labels on each bar make the chart readable without a legend table.

    Parameters
    ----------
    results_df : pd.DataFrame — output of build_results_table()
    save       : bool
    """
    plt.style.use(PLOT_STYLE)

    df      = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    x       = np.arange(len(df))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_acc = ax.bar(
        x - width / 2,
        df["accuracy"] * 100,
        width,
        label="Accuracy (%)",
        color="#6c5ce7",
        alpha=0.88,
        edgecolor="white",
    )
    bars_auc = ax.bar(
        x + width / 2,
        df["roc_auc"] * 100,
        width,
        label="ROC-AUC (%)",
        color="#00b894",
        alpha=0.88,
        edgecolor="white",
    )

    # Add value labels on top of each bar
    for bar in bars_acc:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_auc:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(50, 110)
    ax.set_title(
        "Model Comparison — Accuracy & ROC-AUC",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()

    if save:
        fig.savefig(
            VIZ_DIR / "model_comparison.png",
            dpi=FIG_DPI,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()