"""
Exploratory Data Analysis (EDA) and Statistics Module
======================================================
Comprehensive data analysis and descriptive statistics for academic paper.
Supports Section 5.e, 5.f, 5.i (descriptive statistics, dataset descriptions, EDA).

Includes:
  - Descriptive statistics computation and tables
  - Data profiling and quality checks
  - Dataset characterization
  - Text corpus statistics (for NLP projects)
  - Data balance analysis
  - Feature distribution analysis
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Descriptive Statistics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_descriptive_statistics(
    data: Union[List, np.ndarray, pd.Series],
    percentiles: List[float] = [25, 50, 75],
) -> Dict[str, float]:
    """
    Calculate comprehensive descriptive statistics for numerical data.
    
    Required for Section 5.e (Descriptive statistics in table).
    
    Statistics Computed:
    --------------------
    - Count, Mean, Standard Deviation
    - Min, Max, Range
    - Median (50th percentile)
    - Quartiles (25th, 75th percentiles)
    - Skewness (asymmetry of distribution)
    - Kurtosis (tailedness of distribution)
    - Coefficient of Variation (CV = std/mean)
    
    Args:
        data: Numerical data array or list
        percentiles: List of percentiles to calculate (default: [25, 50, 75])
    
    Returns:
        Dictionary with all statistical measures
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        logger.warning("empty_data_for_statistics")
        return {}
    
    stats_dict = {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.max(data) - np.min(data)),
        "variance": float(np.var(data, ddof=1)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data)),
    }
    
    # Calculate percentiles
    for p in percentiles:
        stats_dict[f"p{int(p)}"] = float(np.percentile(data, p))
    
    # Coefficient of variation
    if stats_dict["mean"] != 0:
        stats_dict["cv"] = stats_dict["std"] / stats_dict["mean"]
    else:
        stats_dict["cv"] = np.inf
    
    logger.info(
        "descriptive_statistics_calculated",
        count=stats_dict["count"],
        mean=round(stats_dict["mean"], 2),
        std=round(stats_dict["std"], 2),
    )
    
    return stats_dict


def create_statistics_table(
    data_dict: Dict[str, Union[List, np.ndarray]],
    output_format: str = "markdown",
) -> str:
    """
    Create a formatted statistics table for multiple variables.
    
    Required for Section 5.e (Descriptive statistics in table).
    
    Args:
        data_dict: Dictionary mapping variable names to data arrays
        output_format: Output format ('markdown', 'latex', 'html')
    
    Returns:
        Formatted table string
    """
    logger.info(
        "creating_statistics_table",
        num_variables=len(data_dict),
        format=output_format,
    )
    
    # Calculate stats for each variable
    stats_list = []
    for var_name, data in data_dict.items():
        stat = calculate_descriptive_statistics(data)
        stat["variable"] = var_name
        stats_list.append(stat)
    
    # Convert to DataFrame
    df = pd.DataFrame(stats_list)
    
    # Reorder columns
    cols = ["variable", "count", "mean", "std", "min", "p25", "p50", "p75", "max"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Format output
    if output_format == "markdown":
        table_str = df.to_markdown(index=False, floatfmt=".3f")
    elif output_format == "latex":
        table_str = df.to_latex(index=False, float_format="%.3f")
    elif output_format == "html":
        table_str = df.to_html(index=False, float_format="%.3f")
    else:
        table_str = df.to_string(index=False)
    
    return table_str


# ──────────────────────────────────────────────────────────────────────────────
# Text Corpus Statistics (NLP-Specific)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_text_corpus(
    texts: List[str],
    include_vocabulary: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a text corpus and compute comprehensive statistics.
    
    Essential for NLP projects. Supports Section 5.f (comprehensive description of datasets).
    
    Statistics Computed:
    --------------------
    - Total number of documents
    - Total number of tokens (words)
    - Average document length
    - Vocabulary size (unique words)
    - Token frequency distribution
    - Average word length
    - Character count statistics
    
    Args:
        texts: List of text strings
        include_vocabulary: Whether to include full vocabulary list
    
    Returns:
        Dictionary with corpus statistics
    """
    logger.info("analyzing_text_corpus", num_documents=len(texts))
    
    if not texts:
        return {}
    
    # Tokenize (simple whitespace tokenization)
    all_tokens = []
    doc_lengths = []
    char_counts = []
    
    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)
        doc_lengths.append(len(tokens))
        char_counts.append(len(text))
    
    # Count unique tokens
    token_counts = Counter(all_tokens)
    vocabulary = set(all_tokens)
    
    # Word length statistics
    word_lengths = [len(word) for word in all_tokens]
    
    stats = {
        "num_documents": len(texts),
        "total_tokens": len(all_tokens),
        "vocabulary_size": len(vocabulary),
        "avg_doc_length_tokens": float(np.mean(doc_lengths)),
        "std_doc_length_tokens": float(np.std(doc_lengths)),
        "min_doc_length": int(np.min(doc_lengths)),
        "max_doc_length": int(np.max(doc_lengths)),
        "avg_word_length": float(np.mean(word_lengths)),
        "avg_char_count": float(np.mean(char_counts)),
        "total_characters": sum(char_counts),
        "most_common_words": token_counts.most_common(20),
        "type_token_ratio": len(vocabulary) / len(all_tokens) if all_tokens else 0,
    }
    
    if include_vocabulary:
        stats["vocabulary"] = sorted(list(vocabulary))
    
    logger.info(
        "text_corpus_analyzed",
        num_documents=stats["num_documents"],
        vocabulary_size=stats["vocabulary_size"],
        avg_length=round(stats["avg_doc_length_tokens"], 2),
    )
    
    return stats


def analyze_skill_distribution(
    skill_lists: List[List[str]],
) -> Dict[str, Any]:
    """
    Analyze the distribution of skills across resumes or job descriptions.
    
    Project-specific for IntelliCover AI.
    Supports Section 5.f (dataset description).
    
    Args:
        skill_lists: List of skill lists (one per resume/JD)
    
    Returns:
        Dictionary with skill statistics
    """
    logger.info("analyzing_skill_distribution", num_samples=len(skill_lists))
    
    # Flatten all skills
    all_skills = []
    skill_counts_per_doc = []
    
    for skills in skill_lists:
        normalized_skills = [s.lower().strip() for s in skills]
        all_skills.extend(normalized_skills)
        skill_counts_per_doc.append(len(normalized_skills))
    
    # Count frequencies
    skill_freq = Counter(all_skills)
    
    stats = {
        "num_samples": len(skill_lists),
        "total_skill_mentions": len(all_skills),
        "unique_skills": len(skill_freq),
        "avg_skills_per_sample": float(np.mean(skill_counts_per_doc)),
        "std_skills_per_sample": float(np.std(skill_counts_per_doc)),
        "min_skills": int(np.min(skill_counts_per_doc)) if skill_counts_per_doc else 0,
        "max_skills": int(np.max(skill_counts_per_doc)) if skill_counts_per_doc else 0,
        "top_20_skills": skill_freq.most_common(20),
    }
    
    logger.info(
        "skill_distribution_analyzed",
        unique_skills=stats["unique_skills"],
        avg_per_sample=round(stats["avg_skills_per_sample"], 2),
    )
    
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Balance Analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_class_balance(
    labels: Union[List, np.ndarray],
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Analyze class distribution and imbalance in labeled data.
    
    Supports Section 5.j (strategies to mitigate dataset imbalance).
    
    Metrics Computed:
    -----------------
    - Class counts and proportions
    - Imbalance Ratio = max_count / min_count
    - Gini Coefficient (measure of inequality)
    
    Args:
        labels: Array of class labels
        label_names: Optional mapping from label indices to names
    
    Returns:
        Dictionary with balance statistics
    """
    logger.info("analyzing_class_balance", num_samples=len(labels))
    
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    total = len(labels)
    proportions = counts / total
    
    # Imbalance ratio
    imbalance_ratio = np.max(counts) / np.min(counts) if len(counts) > 0 else 1.0
    
    # Gini coefficient (0 = perfect balance, 1 = perfect imbalance)
    sorted_props = np.sort(proportions)
    n = len(sorted_props)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_props)) / (n * np.sum(sorted_props)) - (n + 1) / n
    
    # Build class distribution
    class_dist = []
    for label, count, prop in zip(unique_labels, counts, proportions):
        label_name = label_names.get(label, str(label)) if label_names else str(label)
        class_dist.append({
            "label": label_name,
            "count": int(count),
            "proportion": float(prop),
        })
    
    stats = {
        "num_classes": len(unique_labels),
        "total_samples": int(total),
        "class_distribution": class_dist,
        "imbalance_ratio": float(imbalance_ratio),
        "gini_coefficient": float(gini),
        "is_balanced": imbalance_ratio < 1.5,  # Heuristic threshold
    }
    
    logger.info(
        "class_balance_analyzed",
        num_classes=stats["num_classes"],
        imbalance_ratio=round(imbalance_ratio, 2),
        gini=round(gini, 3),
    )
    
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Summary Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_dataset_summary(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive dataset summary report.
    
    Combines multiple analyses for Section 5.f (comprehensive description of datasets).
    
    Args:
        data: Pandas DataFrame with dataset
        target_column: Name of target/label column (if classification)
        text_columns: Names of text columns for NLP analysis
    
    Returns:
        Dictionary with complete dataset summary
    """
    logger.info(
        "generating_dataset_summary",
        shape=data.shape,
        target_column=target_column,
    )
    
    summary = {
        "basic_info": {
            "num_samples": len(data),
            "num_features": len(data.columns),
            "feature_names": list(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
        },
        "data_quality": {
            "missing_values": data.isnull().sum().to_dict(),
            "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
            "duplicate_rows": int(data.duplicated().sum()),
        },
    }
    
    # Numerical features statistics
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numerical_features"] = {
            col: calculate_descriptive_statistics(data[col].dropna())
            for col in numeric_cols
        }
    
    # Categorical features statistics
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary["categorical_features"] = {
            col: {
                "unique_values": int(data[col].nunique()),
                "most_common": data[col].value_counts().head(10).to_dict(),
            }
            for col in categorical_cols
        }
    
    # Target variable analysis
    if target_column and target_column in data.columns:
        summary["target_analysis"] = analyze_class_balance(
            data[target_column].values
        )
    
    # Text column analysis
    if text_columns:
        summary["text_analysis"] = {}
        for col in text_columns:
            if col in data.columns:
                texts = data[col].dropna().astype(str).tolist()
                summary["text_analysis"][col] = analyze_text_corpus(texts)
    
    logger.info("dataset_summary_generated")
    
    return summary


def format_dataset_summary_report(summary: Dict, output_format: str = "markdown") -> str:
    """
    Format dataset summary as a structured report for papers.
    
    Args:
        summary: Summary dictionary from generate_dataset_summary
        output_format: Output format ('markdown', 'latex', 'text')
    
    Returns:
        Formatted report string
    """
    lines = []
    
    if output_format == "markdown":
        lines.append("# Dataset Summary Report")
        lines.append("")
        lines.append("## Basic Information")
        lines.append(f"- **Number of Samples**: {summary['basic_info']['num_samples']}")
        lines.append(f"- **Number of Features**: {summary['basic_info']['num_features']}")
        lines.append(f"- **Memory Usage**: {summary['basic_info']['memory_usage_mb']:.2f} MB")
        lines.append("")
        
        lines.append("## Data Quality")
        lines.append(f"- **Duplicate Rows**: {summary['data_quality']['duplicate_rows']}")
        missing = summary['data_quality']['missing_percentage']
        if any(v > 0 for v in missing.values()):
            lines.append("- **Missing Values**:")
            for col, pct in missing.items():
                if pct > 0:
                    lines.append(f"  - {col}: {pct:.2f}%")
        else:
            lines.append("- **Missing Values**: None")
        lines.append("")
        
        if "target_analysis" in summary:
            lines.append("## Target Variable Distribution")
            target = summary["target_analysis"]
            lines.append(f"- **Number of Classes**: {target['num_classes']}")
            lines.append(f"- **Imbalance Ratio**: {target['imbalance_ratio']:.2f}")
            lines.append(f"- **Is Balanced**: {'Yes' if target['is_balanced'] else 'No'}")
            lines.append("")
    
    else:
        # Simple text format
        lines.append("=" * 60)
        lines.append("DATASET SUMMARY REPORT".center(60))
        lines.append("=" * 60)
        lines.append(f"Samples: {summary['basic_info']['num_samples']}")
        lines.append(f"Features: {summary['basic_info']['num_features']}")
        lines.append("=" * 60)
    
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Data Splitting Documentation
# ──────────────────────────────────────────────────────────────────────────────

def document_data_split(
    train_size: int,
    val_size: int,
    test_size: int,
) -> Dict[str, Any]:
    """
    Document data splitting ratios for paper.
    
    Required for Section 5.k (Dataset splitting ratio).
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
    
    Returns:
        Dictionary with split information
    """
    total = train_size + val_size + test_size
    
    split_info = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_size": total,
        "train_ratio": train_size / total,
        "val_ratio": val_size / total,
        "test_ratio": test_size / total,
        "split_description": f"{train_size}/{val_size}/{test_size} (Train/Val/Test)",
        "split_percentages": f"{train_size/total*100:.1f}%/{val_size/total*100:.1f}%/{test_size/total*100:.1f}%",
    }
    
    logger.info(
        "data_split_documented",
        split=split_info["split_percentages"],
        total=total,
    )
    
    return split_info
