"""
Explainable AI (XAI) Module
============================
Provides model interpretability and explainability tools.
Required for Section 5.o and 6.g of academic paper.

Includes:
  - Feature importance analysis
  - SHAP (SHapley Additive exPlanations) integration
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Attention visualization for transformer models
  - Text highlighting for NLP explanations
  - Global and local model interpretability
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Integration
# ──────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP-based model explainer.
    
    SHAP (SHapley Additive exPlanations) assigns each feature an importance value
    for a particular prediction based on game theory principles.
    
    Mathematical Basis:
    -------------------
    For a feature i, SHAP value φᵢ is:
    φᵢ = Σ [|S|! * (M - |S| - 1)! / M!] * [f(S ∪ {i}) - f(S)]
    
    where S ranges over all possible subsets of features, M is total features.
    
    Required for Section 5.o and 6.g.
    """
    
    def __init__(
        self,
        model: Any,
        explainer_type: str = "tree",
        background_data: Optional[np.ndarray] = None,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            explainer_type: Type of explainer ('tree', 'kernel', 'deep', 'linear')
            background_data: Background dataset for kernel explainer
        """
        try:
            import shap
            self.shap = shap
        except ImportError:
            logger.error("shap_not_installed", message="pip install shap")
            raise
        
        self.model = model
        self.explainer_type = explainer_type
        
        # Initialize appropriate explainer
        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif explainer_type == "kernel":
            if background_data is None:
                raise ValueError("background_data required for KernelExplainer")
            self.explainer = shap.KernelExplainer(model.predict, background_data)
        elif explainer_type == "deep":
            self.explainer = shap.DeepExplainer(model, background_data)
        elif explainer_type == "linear":
            self.explainer = shap.LinearExplainer(model, background_data)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        logger.info("shap_explainer_initialized", type=explainer_type)
    
    def explain_instance(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction or batch of predictions.
        
        Args:
            X: Input features (single sample or batch)
            feature_names: Names of features
        
        Returns:
            Dictionary with SHAP values and explanations
        """
        logger.info("explaining_with_shap", num_samples=len(X) if len(X.shape) > 1 else 1)
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case
            shap_values = np.array(shap_values)
        
        explanation = {
            "shap_values": shap_values,
            "base_value": self.explainer.expected_value,
            "feature_names": feature_names,
        }
        
        logger.info("shap_explanation_generated")
        return explanation
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        aggregation: str = "mean_abs",
    ) -> Dict[str, float]:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            X: Dataset to analyze
            feature_names: Names of features
            aggregation: How to aggregate ('mean_abs', 'mean', 'sum_abs')
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.info("calculating_shap_feature_importance", aggregation=aggregation)
        
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)
        
        # Aggregate importance
        if aggregation == "mean_abs":
            importance = np.abs(shap_values).mean(axis=0)
        elif aggregation == "mean":
            importance = shap_values.mean(axis=0)
        elif aggregation == "sum_abs":
            importance = np.abs(shap_values).sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_dict = dict(zip(feature_names, importance.tolist()))
        
        logger.info("shap_feature_importance_calculated")
        return importance_dict
    
    def save_summary_plot(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: str = "shap_summary.png",
        plot_type: str = "dot",
    ) -> None:
        """
        Generate and save SHAP summary plot.
        
        Args:
            X: Dataset to visualize
            feature_names: Names of features
            save_path: Path to save plot
            plot_type: Type of plot ('dot', 'bar', 'violin')
        """
        import matplotlib.pyplot as plt
        
        logger.info("generating_shap_summary_plot", plot_type=plot_type)
        
        shap_values = self.explainer.shap_values(X)
        
        plt.figure()
        self.shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type=plot_type,
            show=False,
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("shap_summary_plot_saved", path=save_path)


# ──────────────────────────────────────────────────────────────────────────────
# LIME Integration
# ──────────────────────────────────────────────────────────────────────────────

class LIMEExplainer:
    """
    LIME-based model explainer.
    
    LIME (Local Interpretable Model-agnostic Explanations) explains individual
    predictions by approximating the model locally with an interpretable model.
    
    Mathematical Principle:
    -----------------------
    Find simple model g that minimizes:
    L(f, g, πₓ) + Ω(g)
    
    where:
    - f is the original model
    - g is the interpretable approximation
    - πₓ defines locality around instance x
    - Ω(g) is complexity penalty
    
    Required for Section 6.g.
    """
    
    def __init__(
        self,
        predict_fn: callable,
        mode: str = "classification",
    ):
        """
        Initialize LIME explainer.
        
        Args:
            predict_fn: Model prediction function
            mode: 'classification' or 'regression'
        """
        try:
            from lime import lime_tabular, lime_text
            self.lime_tabular = lime_tabular
            self.lime_text = lime_text
        except ImportError:
            logger.error("lime_not_installed", message="pip install lime")
            raise
        
        self.predict_fn = predict_fn
        self.mode = mode
        self.explainer = None
        
        logger.info("lime_explainer_initialized", mode=mode)
    
    def fit_tabular(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize LIME for tabular data.
        
        Args:
            training_data: Training dataset
            feature_names: Names of features
            class_names: Names of classes
            categorical_features: Indices of categorical features
        """
        logger.info("initializing_lime_tabular", num_features=len(feature_names))
        
        self.explainer = self.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features,
            mode=self.mode,
        )
        
        logger.info("lime_tabular_initialized")
    
    def explain_instance_tabular(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        top_labels: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single tabular prediction.
        
        Args:
            instance: Single data instance
            num_features: Number of features to include in explanation
            top_labels: Number of top predicted labels to explain
        
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            raise RuntimeError("Call fit_tabular() first")
        
        logger.info("explaining_tabular_instance", num_features=num_features)
        
        explanation = self.explainer.explain_instance(
            instance,
            self.predict_fn,
            num_features=num_features,
            top_labels=top_labels,
        )
        
        # Extract feature contributions
        feature_importance = dict(explanation.as_list())
        
        result = {
            "feature_importance": feature_importance,
            "prediction": explanation.predict_proba if self.mode == "classification" else None,
            "local_explanation": explanation,
        }
        
        logger.info("tabular_explanation_generated")
        return result
    
    def explain_text(
        self,
        text: str,
        predict_proba_fn: callable,
        class_names: List[str],
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Explain a text classification prediction.
        
        Args:
            text: Input text
            predict_proba_fn: Function that returns class probabilities
            class_names: Names of classes
            num_features: Number of features (words) to include
        
        Returns:
            Dictionary with explanation
        """
        logger.info("explaining_text_instance")
        
        text_explainer = self.lime_text.LimeTextExplainer(class_names=class_names)
        
        explanation = text_explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
        )
        
        # Extract word contributions
        word_importance = dict(explanation.as_list())
        
        result = {
            "word_importance": word_importance,
            "highlighted_text": explanation.as_html(),
        }
        
        logger.info("text_explanation_generated")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Attention Visualization (for Transformers)
# ──────────────────────────────────────────────────────────────────────────────

def visualize_attention_weights(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer: int = -1,
    head: int = 0,
    save_path: Optional[str] = None,
) -> Any:
    """
    Visualize attention weights from transformer models.
    
    Attention Mechanism Formula:
    -----------------------------
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
    
    where Q, K, V are query, key, value matrices.
    
    The attention weights (softmax scores) show which tokens the model
    focuses on when processing each token.
    
    Required for Section 6.g (attention visualization).
    
    Args:
        attention_weights: Attention weight matrix [num_layers, num_heads, seq_len, seq_len]
        tokens: List of tokens
        layer: Which layer to visualize (-1 for last)
        head: Which attention head to visualize
        save_path: Path to save visualization
    
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info(
        "visualizing_attention",
        layer=layer,
        head=head,
        num_tokens=len(tokens),
    )
    
    # Extract specific layer and head
    if len(attention_weights.shape) == 4:
        # [layers, heads, seq_len, seq_len]
        attn = attention_weights[layer, head]
    elif len(attention_weights.shape) == 3:
        # [heads, seq_len, seq_len]
        attn = attention_weights[head]
    else:
        # [seq_len, seq_len]
        attn = attention_weights
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
    )
    
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    ax.set_title(f'Attention Weights (Layer {layer}, Head {head})', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("attention_visualization_saved", path=save_path)
    
    return fig


def extract_attention_from_model(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int = -1,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract attention weights from a Hugging Face transformer model.
    
    Args:
        model: Hugging Face model with attention output
        tokenizer: Corresponding tokenizer
        text: Input text
        layer: Which layer to extract (-1 for last)
    
    Returns:
        Tuple of (attention_weights, tokens)
    """
    import torch
    
    logger.info("extracting_attention_from_model", layer=layer)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention weights
    attention = outputs.attentions  # Tuple of tensors, one per layer
    attention_weights = attention[layer][0].cpu().numpy()  # [num_heads, seq_len, seq_len]
    
    logger.info("attention_extracted", shape=attention_weights.shape)
    
    return attention_weights, tokens


# ──────────────────────────────────────────────────────────────────────────────
# Text Highlighting for Explanations
# ──────────────────────────────────────────────────────────────────────────────

def highlight_important_tokens(
    text: str,
    importance_scores: Dict[str, float],
    threshold: float = 0.1,
    format: str = "html",
) -> str:
    """
    Highlight important tokens in text based on importance scores.
    
    Args:
        text: Original text
        importance_scores: Dictionary mapping tokens/words to importance scores
        threshold: Minimum score to highlight
        format: Output format ('html', 'markdown')
    
    Returns:
        Formatted text with highlights
    """
    logger.info("highlighting_important_tokens", num_tokens=len(importance_scores))
    
    words = text.split()
    highlighted = []
    
    for word in words:
        score = importance_scores.get(word.lower().strip(), 0.0)
        
        if abs(score) > threshold:
            if format == "html":
                color = "lightgreen" if score > 0 else "lightcoral"
                opacity = min(abs(score), 1.0)
                highlighted.append(
                    f'<span style="background-color: {color}; opacity: {opacity}">{word}</span>'
                )
            elif format == "markdown":
                marker = "**" if score > 0 else "*"
                highlighted.append(f"{marker}{word}{marker}")
            else:
                highlighted.append(word)
        else:
            highlighted.append(word)
    
    return " ".join(highlighted)


# ──────────────────────────────────────────────────────────────────────────────
# Feature Importance Analysis
# ──────────────────────────────────────────────────────────────────────────────

def calculate_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: callable,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Calculate permutation-based feature importance.
    
    Principle:
    ----------
    For each feature, shuffle its values and measure the decrease in model
    performance. Features that cause large decreases are more important.
    
    Mathematical Formula:
    ---------------------
    Importance(f_i) = E[metric(X, y)] - E[metric(X_perm_i, y)]
    
    where X_perm_i is X with feature i shuffled.
    
    Args:
        model: Trained model
        X: Test features
        y: Test labels
        metric_fn: Metric function (e.g., accuracy_score)
        n_repeats: Number of shuffling repeats
        random_state: Random seed
    
    Returns:
        Dictionary with importance scores and statistics
    """
    from sklearn.inspection import permutation_importance
    
    logger.info(
        "calculating_permutation_importance",
        num_features=X.shape[1],
        n_repeats=n_repeats,
    )
    
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=metric_fn,
    )
    
    importance_stats = {
        "importances_mean": result.importances_mean.tolist(),
        "importances_std": result.importances_std.tolist(),
    }
    
    logger.info("permutation_importance_calculated")
    
    return importance_stats


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def create_explainability_report(
    shap_importance: Dict[str, float],
    lime_importance: Dict[str, float],
    top_n: int = 10,
) -> str:
    """
    Create a combined explainability report from multiple methods.
    
    Useful for Section 6.g of the paper.
    
    Args:
        shap_importance: SHAP feature importance scores
        lime_importance: LIME feature importance scores
        top_n: Number of top features to report
    
    Returns:
        Formatted report string
    """
    logger.info("creating_explainability_report", top_n=top_n)
    
    lines = [
        "=" * 70,
        "EXPLAINABILITY REPORT".center(70),
        "=" * 70,
        "",
        "Top Features by SHAP:",
        "-" * 70,
    ]
    
    # Sort SHAP
    sorted_shap = sorted(shap_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    for i, (feature, score) in enumerate(sorted_shap, 1):
        lines.append(f"  {i:2d}. {feature:30s} {score:10.4f}")
    
    lines.extend([
        "",
        "Top Features by LIME:",
        "-" * 70,
    ])
    
    # Sort LIME
    sorted_lime = sorted(lime_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    for i, (feature, score) in enumerate(sorted_lime, 1):
        lines.append(f"  {i:2d}. {feature:30s} {score:10.4f}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
