"""
Visualization Module
====================
Generates all plots and visualizations required for academic paper.
Supports Section 5.i, 6.e, 6.f of the paper requirements.

Includes:
  - Training/validation loss and accuracy curves
  - ROC-AUC curves
  - Precision-Recall curves
  - Confusion matrices (normalized and raw)
  - Classification reports (heatmap format)
  - Word clouds for text data
  - Distribution plots
  - Correlation matrices
  - Feature importance plots
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Default figure size and DPI for high-quality outputs
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 300


# ──────────────────────────────────────────────────────────────────────────────
# Training History Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ["loss", "accuracy"],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Plot training and validation metrics over epochs.
    
    Creates subplots for each metric showing train vs. validation curves.
    Essential for Section 6.e (Training, Testing or validation loss and accuracy graphs).
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_accuracy', etc.
        metrics: List of metric names to plot (e.g., ['loss', 'accuracy'])
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info(
        "plotting_training_history",
        metrics=metrics,
        save_path=save_path,
    )
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics, figsize[1]))
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history.get(f'train_{metrics[0]}', [])) + 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            ax.plot(epochs, history[train_key], 'b-', label=f'Training {metric}', linewidth=2)
        
        if val_key in history:
            ax.plot(epochs, history[val_key], 'r--', label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("training_history_saved", path=save_path)
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ROC and Precision-Recall Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: Union[List, np.ndarray],
    y_scores: Union[List, np.ndarray],
    pos_label: int = 1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Mathematical Basis:
    -------------------
    ROC curve plots True Positive Rate vs False Positive Rate at various thresholds.
    TPR = TP / (TP + FN) (also called Recall or Sensitivity)
    FPR = FP / (FP + TN)
    
    AUC (Area Under Curve) = ∫ TPR d(FPR)
    
    Required for Section 6.f (ROC-AUC curves).
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probabilities or decision scores
        pos_label: Label of positive class
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info("plotting_roc_curve", save_path=save_path)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("roc_curve_saved", path=save_path, auc=round(roc_auc, 3))
    
    return fig


def plot_precision_recall_curve(
    y_true: Union[List, np.ndarray],
    y_scores: Union[List, np.ndarray],
    pos_label: int = 1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Plot Precision-Recall curve.
    
    Mathematical Basis:
    -------------------
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    
    The curve shows the trade-off between precision and recall at various thresholds.
    
    Required for Section 6.f (Precision-Recall curves).
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probabilities or decision scores
        pos_label: Label of positive class
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info("plotting_precision_recall_curve", save_path=save_path)
    
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label
    )
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("pr_curve_saved", path=save_path, auc=round(pr_auc, 3))
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Mathematical Definition:
    ------------------------
    CM[i,j] = number of samples with true label i predicted as label j
    
    Normalization options:
    - 'true': normalize over true labels (recall-based)
    - 'pred': normalize over predictions (precision-based)
    - 'all': normalize over all samples
    - None: raw counts
    
    Required for Section 6.f (confusion matrices).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names for display
        normalize: Normalization mode
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    logger.info(
        "plotting_confusion_matrix",
        normalize=normalize,
        save_path=save_path,
    )
    
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels if labels else 'auto',
        yticklabels=labels if labels else 'auto',
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax,
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    title = 'Confusion Matrix'
    if normalize:
        title += f' (Normalized by {normalize})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("confusion_matrix_saved", path=save_path)
    
    return fig


def plot_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot classification report as a heatmap.
    
    Shows precision, recall, and F1-score for each class in a visual format.
    Required for Section 6.f (classification report visualizations).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names for display
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info("plotting_classification_report", save_path=save_path)
    
    # Get classification report as dict
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    
    # Extract per-class metrics
    class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_names = ['precision', 'recall', 'f1-score']
    
    # Build data matrix
    data = np.array([
        [report[cls][metric] for metric in metrics_names]
        for cls in class_names
    ])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=[m.capitalize() for m in metrics_names],
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Score'},
        ax=ax,
    )
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title('Classification Report Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("classification_report_saved", path=save_path)
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Word Cloud and Text Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def plot_wordcloud(
    text: Union[str, List[str]],
    title: str = "Word Cloud",
    max_words: int = 100,
    background_color: str = "white",
    colormap: str = "viridis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Generate and plot a word cloud from text data.
    
    Required for Section 5.i (Output of EDA in graphical representations with Word cloud).
    
    Args:
        text: Input text string or list of strings
        title: Plot title
        max_words: Maximum number of words to display
        background_color: Background color
        colormap: Matplotlib colormap name
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    try:
        from wordcloud import WordCloud
        
        logger.info(
            "generating_wordcloud",
            max_words=max_words,
            save_path=save_path,
        )
        
        # Combine text if list
        if isinstance(text, list):
            text = " ".join(text)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1200,
            height=800,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            contour_width=2,
            contour_color='steelblue',
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
            logger.info("wordcloud_saved", path=save_path)
        
        return fig
        
    except ImportError:
        logger.error(
            "wordcloud_not_installed",
            message="Install with: pip install wordcloud"
        )
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Distribution and Statistical Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_distribution(
    data: Union[List, np.ndarray],
    title: str = "Distribution Plot",
    xlabel: str = "Value",
    bins: int = 30,
    kde: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
) -> Figure:
    """
    Plot histogram with optional KDE overlay.
    
    Useful for EDA and understanding data distributions.
    Supports Section 5.i (EDA graphical representations).
    
    Args:
        data: Numerical data to plot
        title: Plot title
        xlabel: X-axis label
        bins: Number of histogram bins
        kde: Whether to overlay kernel density estimate
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info("plotting_distribution", title=title, save_path=save_path)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data, bins=bins, kde=kde, ax=ax, color='skyblue', edgecolor='black')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean and median lines
    mean_val = np.mean(data)
    median_val = np.median(data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("distribution_plot_saved", path=save_path)
    
    return fig


def plot_correlation_matrix(
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot correlation matrix as a heatmap.
    
    Useful for feature engineering and understanding feature relationships.
    Supports Section 5.g (Feature Engineering / EDA Process).
    
    Args:
        data: 2D array of numerical features (samples × features)
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info("plotting_correlation_matrix", save_path=save_path)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        xticklabels=feature_names if feature_names else range(corr_matrix.shape[0]),
        yticklabels=feature_names if feature_names else range(corr_matrix.shape[0]),
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax,
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("correlation_matrix_saved", path=save_path)
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Comparison and Benchmark Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot bar chart comparing multiple models across different metrics.
    
    Required for Section 6.i (Comparison with Other Baselines).
    
    Args:
        model_names: List of model names
        metrics: Dictionary where keys are metric names and values are lists of scores
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info(
        "plotting_model_comparison",
        num_models=len(model_names),
        metrics=list(metrics.keys()),
        save_path=save_path,
    )
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, scores) in enumerate(metrics.items()):
        ax = axes[idx]
        
        x = np.arange(len(model_names))
        bars = ax.bar(x, scores, color=sns.color_palette("husl", len(model_names)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
            )
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f'{metric_name.capitalize()} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(scores) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("model_comparison_saved", path=save_path)
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Feature Importance Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    feature_names: List[str],
    importances: Union[List, np.ndarray],
    title: str = "Feature Importance",
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot feature importance as a horizontal bar chart.
    
    Useful for understanding model behavior and feature engineering.
    Supports Section 5.g and Section 6.g (model interpretability).
    
    Args:
        feature_names: List of feature names
        importances: Importance scores (e.g., from tree-based models or SHAP)
        title: Plot title
        top_n: Show only top N features
        save_path: Path to save the figure
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    logger.info(
        "plotting_feature_importance",
        num_features=len(feature_names),
        top_n=top_n,
        save_path=save_path,
    )
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    if top_n:
        indices = indices[:top_n]
    
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importances, color='steelblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("feature_importance_saved", path=save_path)
    
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def save_all_figures(
    figures: Dict[str, Figure],
    output_dir: str,
    close_after_save: bool = True,
) -> None:
    """
    Save multiple figures to a directory.
    
    Args:
        figures: Dictionary mapping filenames to Figure objects
        output_dir: Output directory path
        close_after_save: Whether to close figures after saving
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("saving_all_figures", num_figures=len(figures), output_dir=output_dir)
    
    for filename, fig in figures.items():
        filepath = output_path / filename
        fig.savefig(filepath, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info("figure_saved", filename=filename)
        
        if close_after_save:
            plt.close(fig)
    
    logger.info("all_figures_saved", output_dir=output_dir)
