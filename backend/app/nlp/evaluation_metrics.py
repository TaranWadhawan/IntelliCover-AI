"""
Evaluation Metrics Module
==========================
Comprehensive metrics for measuring NLP model and system performance.
Required for academic paper evaluation section (Section 6.c, 6.d).

Includes:
  - Classification metrics: Precision, Recall, F1-Score, Accuracy
  - Confusion Matrix computation
  - Text generation metrics: BERTScore, BLEU, ROUGE, Perplexity
  - Regression metrics: RMSE, MAE
  - Ranking metrics: NDCG
  
All metrics include mathematical formulas in docstrings for paper documentation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Classification Metrics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = "weighted",
    labels: Optional[List] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Mathematical Formulas:
    ----------------------
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Where:
      TP = True Positives
      TN = True Negatives
      FP = False Positives
      FN = False Negatives
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'binary')
        labels: List of label names (optional)
    
    Returns:
        Dictionary containing precision, recall, f1, and accuracy scores
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    logger.info(
        "calculating_classification_metrics",
        num_samples=len(y_true),
        average=average,
    )
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support.tolist(),
    }
    
    logger.info(
        "classification_metrics_computed",
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
    
    return metrics


def calculate_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Mathematical Formula:
    ---------------------
    CM[i,j] = number of observations known to be in group i but predicted to be in group j
    
    Normalization options:
    - 'true': normalize over true labels (row-wise)
    - 'pred': normalize over predicted labels (column-wise)
    - 'all': normalize over all samples
    - None: raw counts
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalization mode
    
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    logger.info(
        "confusion_matrix_computed",
        shape=cm.shape,
        normalize=normalize,
    )
    
    return cm


# ──────────────────────────────────────────────────────────────────────────────
# Text Generation Metrics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_bertscore(
    candidates: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate BERTScore for text generation evaluation.
    
    Mathematical Basis:
    -------------------
    BERTScore computes contextualized embeddings using BERT and measures:
    - Precision: Average maximum cosine similarity of reference tokens to candidate tokens
    - Recall: Average maximum cosine similarity of candidate tokens to reference tokens
    - F1: Harmonic mean of precision and recall
    
    Args:
        candidates: List of generated texts
        references: List of reference texts
        lang: Language code (default: 'en')
        model_type: Specific BERT model to use (e.g., 'bert-base-uncased')
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    try:
        from bert_score import score as bert_score
        
        logger.info(
            "calculating_bertscore",
            num_samples=len(candidates),
            lang=lang,
            model_type=model_type or "default",
        )
        
        P, R, F1 = bert_score(
            candidates,
            references,
            lang=lang,
            model_type=model_type,
            verbose=False,
        )
        
        metrics = {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
        
        logger.info(
            "bertscore_computed",
            precision=round(metrics["bertscore_precision"], 4),
            recall=round(metrics["bertscore_recall"], 4),
            f1=round(metrics["bertscore_f1"], 4),
        )
        
        return metrics
        
    except ImportError:
        logger.error(
            "bertscore_not_installed",
            message="Install with: pip install bert-score"
        )
        raise


def calculate_bleu(
    candidates: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Calculate BLEU (Bilingual Evaluation Understudy) score.
    
    Mathematical Formula:
    ---------------------
    BLEU = BP * exp(Σ(w_n * log(p_n)))
    
    Where:
    - BP = Brevity Penalty = exp(1 - r/c) if c < r, else 1
    - p_n = n-gram precision
    - w_n = uniform weight (1/N)
    - r = reference length
    - c = candidate length
    
    Args:
        candidates: List of generated texts (one per sample)
        references: List of reference text lists (can have multiple refs per sample)
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
    
    Returns:
        Dictionary with BLEU scores for n=1 to max_n
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
        from nltk.tokenize import word_tokenize
        
        logger.info(
            "calculating_bleu",
            num_samples=len(candidates),
            max_n=max_n,
        )
        
        # Tokenize
        tokenized_candidates = [word_tokenize(c.lower()) for c in candidates]
        tokenized_references = [
            [word_tokenize(r.lower()) for r in ref_list]
            for ref_list in references
        ]
        
        # Calculate corpus BLEU
        bleu_scores = {}
        for n in range(1, max_n + 1):
            weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
            score = corpus_bleu(
                tokenized_references,
                tokenized_candidates,
                weights=weights,
            )
            bleu_scores[f"bleu_{n}"] = float(score)
        
        logger.info(
            "bleu_computed",
            bleu_4=round(bleu_scores.get("bleu_4", 0.0), 4),
        )
        
        return bleu_scores
        
    except ImportError:
        logger.error(
            "nltk_not_installed",
            message="Install with: pip install nltk"
        )
        raise


def calculate_rouge(
    candidates: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Calculate ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.
    
    Mathematical Formulas:
    ----------------------
    ROUGE-N: Recall-based n-gram overlap
    ROUGE-L: Longest Common Subsequence based F-measure
    
    ROUGE-N = Σ(CountMatch(n-gram)) / Σ(Count(n-gram_reference))
    
    Args:
        candidates: List of generated texts
        references: List of reference texts
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
        
        logger.info(
            "calculating_rouge",
            num_samples=len(candidates),
        )
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True,
        )
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for candidate, reference in zip(candidates, references):
            scores = scorer.score(reference, candidate)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        metrics = {
            "rouge1": float(np.mean(rouge1_scores)),
            "rouge2": float(np.mean(rouge2_scores)),
            "rougeL": float(np.mean(rougeL_scores)),
        }
        
        logger.info(
            "rouge_computed",
            rouge1=round(metrics["rouge1"], 4),
            rouge2=round(metrics["rouge2"], 4),
            rougeL=round(metrics["rougeL"], 4),
        )
        
        return metrics
        
    except ImportError:
        logger.error(
            "rouge_score_not_installed",
            message="Install with: pip install rouge-score"
        )
        raise


def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Calculate perplexity for language model evaluation.
    
    Mathematical Formula:
    ---------------------
    Perplexity = exp(-1/N * Σ log P(w_i | w_1...w_{i-1}))
    
    Where:
    - N = total number of tokens
    - P(w_i | w_1...w_{i-1}) = probability of token w_i given previous tokens
    
    Lower perplexity indicates better language modeling performance.
    
    Args:
        model: Pre-trained language model (HuggingFace)
        tokenizer: Corresponding tokenizer
        texts: List of texts to evaluate
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Dictionary with average perplexity and per-sample perplexities
    """
    try:
        import torch
        from torch.nn import CrossEntropyLoss
        
        logger.info(
            "calculating_perplexity",
            num_texts=len(texts),
            device=device,
        )
        
        model.eval()
        model.to(device)
        
        perplexities = []
        loss_fct = CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                ).to(device)
                
                outputs = model(**encodings, labels=encodings['input_ids'])
                
                # Calculate perplexity
                neg_log_likelihood = outputs.loss
                perplexity = torch.exp(neg_log_likelihood)
                perplexities.append(float(perplexity))
        
        avg_perplexity = float(np.mean(perplexities))
        
        logger.info(
            "perplexity_computed",
            avg_perplexity=round(avg_perplexity, 4),
        )
        
        return {
            "perplexity": avg_perplexity,
            "perplexities_per_sample": perplexities,
        }
        
    except Exception as e:
        logger.error("perplexity_calculation_failed", error=str(e))
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Regression Metrics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_regression_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Mathematical Formulas:
    ----------------------
    MSE = 1/n * Σ(y_true - y_pred)²
    RMSE = √(MSE)
    MAE = 1/n * Σ|y_true - y_pred|
    R² = 1 - (Σ(y_true - y_pred)²) / (Σ(y_true - ȳ)²)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with MSE, RMSE, MAE, and R² scores
    """
    from sklearn.metrics import r2_score
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
    
    logger.info(
        "regression_metrics_computed",
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2=round(r2, 4),
    )
    
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# ATS Scoring Evaluation (Project-Specific)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_ats_scoring(
    true_scores: List[float],
    predicted_scores: List[float],
    true_matches: List[List[str]],
    predicted_matches: List[List[str]],
) -> Dict[str, float]:
    """
    Evaluate ATS scoring accuracy for IntelliCover AI.
    
    Combines:
    1. Regression metrics for score prediction accuracy
    2. Precision/Recall for skill matching accuracy
    
    Args:
        true_scores: Ground truth ATS scores (0-100)
        predicted_scores: Predicted ATS scores (0-100)
        true_matches: Ground truth matched skills per sample
        predicted_matches: Predicted matched skills per sample
    
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    logger.info(
        "evaluating_ats_scoring",
        num_samples=len(true_scores),
    )
    
    # Score prediction accuracy
    score_metrics = calculate_regression_metrics(true_scores, predicted_scores)
    
    # Skill matching accuracy (treat as multi-label classification)
    # Calculate Jaccard similarity for each sample
    jaccard_scores = []
    for true_set, pred_set in zip(true_matches, predicted_matches):
        true_set = set(true_set)
        pred_set = set(pred_set)
        
        if len(true_set) == 0 and len(pred_set) == 0:
            jaccard = 1.0
        elif len(true_set.union(pred_set)) == 0:
            jaccard = 0.0
        else:
            jaccard = len(true_set.intersection(pred_set)) / len(true_set.union(pred_set))
        
        jaccard_scores.append(jaccard)
    
    avg_jaccard = float(np.mean(jaccard_scores))
    
    metrics = {
        **score_metrics,
        "skill_match_jaccard": avg_jaccard,
    }
    
    logger.info(
        "ats_evaluation_complete",
        score_rmse=round(score_metrics["rmse"], 2),
        skill_jaccard=round(avg_jaccard, 4),
    )
    
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def format_metrics_report(metrics: Dict, title: str = "Metrics Report") -> str:
    """
    Format metrics dictionary as a readable report for logging/paper.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Report title
    
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        title.center(60),
        "=" * 60,
    ]
    
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{name:.<40} {value:.4f}")
        elif isinstance(value, list):
            lines.append(f"{name}:")
            for i, v in enumerate(value):
                lines.append(f"  Class {i}: {v:.4f}")
        else:
            lines.append(f"{name:.<40} {value}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
