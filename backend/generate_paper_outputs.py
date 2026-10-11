"""
Paper Evaluation Suite
======================
Comprehensive script to generate all evaluation metrics, visualizations,
and analyses required for the academic paper.

This script demonstrates how to use all the evaluation modules to:
1. Generate descriptive statistics and EDA
2. Calculate comprehensive evaluation metrics
3. Create all required visualizations
4. Compare with baseline models
5. Generate explainability reports

Run this script to generate all outputs for your paper's Results section.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from app.core.config import settings

# Import all evaluation modules
from app.nlp.baseline_models import (
    BaselineComparator,
    EmbeddingBaselineScorer,
    RuleBasedSkillMatcher,
    TFIDFBaselineScorer,
    TraditionalMLBaseline,
)
from app.nlp.eda_statistics import (
    analyze_class_balance,
    analyze_skill_distribution,
    analyze_text_corpus,
    calculate_descriptive_statistics,
    create_statistics_table,
    document_data_split,
    format_dataset_summary_report,
    generate_dataset_summary,
)
from app.nlp.evaluation_metrics import (
    calculate_bertscore,
    calculate_bleu,
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_regression_metrics,
    calculate_rouge,
    evaluate_ats_scoring,
    format_metrics_report,
)
from app.nlp.explainability import (
    LIMEExplainer,
    SHAPExplainer,
    create_explainability_report,
    highlight_important_tokens,
    visualize_attention_weights,
)
from app.nlp.visualization import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_distribution,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
    plot_wordcloud,
    save_all_figures,
)


def setup_output_directories() -> Dict[str, Path]:
    """
    Create output directories for all paper materials.
    
    Returns:
        Dictionary mapping output types to directory paths
    """
    base_dir = Path("paper_outputs")
    
    dirs = {
        "base": base_dir,
        "statistics": base_dir / "statistics",
        "visualizations": base_dir / "visualizations",
        "metrics": base_dir / "metrics",
        "comparisons": base_dir / "comparisons",
        "explainability": base_dir / "explainability",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def generate_dataset_statistics(
    resumes_data: List[Dict],
    jds_data: List[Dict],
    output_dirs: Dict[str, Path],
) -> None:
    """
    Generate comprehensive dataset statistics for Section 5.
    
    Args:
        resumes_data: List of parsed resume dictionaries
        jds_data: List of parsed job description dictionaries
        output_dirs: Output directories
    """
    print("\n" + "=" * 70)
    print("GENERATING DATASET STATISTICS (Section 5.e, 5.f)")
    print("=" * 70)
    
    # Analyze resume texts
    resume_texts = [" ".join([exp.get("description", "") for exp in r.get("experience", [])]) 
                    for r in resumes_data]
    resume_stats = analyze_text_corpus(resume_texts, include_vocabulary=False)
    
    # Analyze JD texts
    jd_texts = [" ".join(jd.get("responsibilities", [])) for jd in jds_data]
    jd_stats = analyze_text_corpus(jd_texts, include_vocabulary=False)
    
    # Analyze skill distributions
    resume_skills = [r.get("skills", []) for r in resumes_data]
    jd_skills = [jd.get("required_skills", []) for jd in jds_data]
    
    resume_skill_stats = analyze_skill_distribution(resume_skills)
    jd_skill_stats = analyze_skill_distribution(jd_skills)
    
    # Save statistics
    stats = {
        "resume_corpus": resume_stats,
        "jd_corpus": jd_stats,
        "resume_skills": resume_skill_stats,
        "jd_skills": jd_skill_stats,
    }
    
    with open(output_dirs["statistics"] / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Dataset statistics saved to {output_dirs['statistics']}")
    
    # Generate descriptive statistics table
    ats_scores = [85.0, 72.3, 91.5, 68.2, 79.8]  # Example data - replace with actual
    stats_table = create_statistics_table(
        {"ATS_Scores": ats_scores},
        output_format="markdown"
    )
    
    with open(output_dirs["statistics"] / "descriptive_stats_table.md", "w") as f:
        f.write("# Descriptive Statistics Table\n\n")
        f.write(stats_table)
    
    print(f"✓ Descriptive statistics table saved")


def generate_visualizations(
    output_dirs: Dict[str, Path],
) -> None:
    """
    Generate all required visualizations for Section 5.i and Section 6.
    
    Args:
        output_dirs: Output directories
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS (Section 5.i, 6.e, 6.f)")
    print("=" * 70)
    
    figures = {}
    
    # 1. Word Cloud (Section 5.i)
    sample_text = """
    Python machine learning deep learning NLP natural language processing
    TensorFlow PyTorch scikit-learn data science AI artificial intelligence
    AWS cloud computing Docker Kubernetes FastAPI Django Flask
    """
    figures["wordcloud.png"] = plot_wordcloud(
        sample_text,
        title="Resume Skills Word Cloud",
        max_words=50,
    )
    print("✓ Word cloud generated")
    
    # 2. Training History (Section 6.e)
    # Example training history - replace with actual data
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
        "val_loss": [0.85, 0.65, 0.5, 0.45, 0.4],
        "train_accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_accuracy": [0.58, 0.68, 0.75, 0.78, 0.82],
    }
    figures["training_history.png"] = plot_training_history(
        history,
        metrics=["loss", "accuracy"],
    )
    print("✓ Training history plots generated")
    
    # 3. Confusion Matrix (Section 6.f)
    # Example data - replace with actual predictions
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10)
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0] * 10)
    
    figures["confusion_matrix.png"] = plot_confusion_matrix(
        y_true, y_pred,
        labels=["Low Match", "Medium Match", "High Match"],
        normalize="true",
    )
    print("✓ Confusion matrix generated")
    
    # 4. ROC Curve (Section 6.f)
    # Example data - replace with actual scores
    y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 10)
    y_scores = np.random.rand(100)
    
    figures["roc_curve.png"] = plot_roc_curve(y_true_binary, y_scores)
    print("✓ ROC curve generated")
    
    # 5. Precision-Recall Curve (Section 6.f)
    figures["pr_curve.png"] = plot_precision_recall_curve(y_true_binary, y_scores)
    print("✓ Precision-Recall curve generated")
    
    # 6. Feature Importance (Section 5.g, 6.g)
    feature_names = ["Hard Skills", "Semantic Match", "Education", "Years Exp", "Keywords"]
    importances = [0.50, 0.40, 0.10, 0.15, 0.25]
    
    figures["feature_importance.png"] = plot_feature_importance(
        feature_names,
        importances,
        title="ATS Scoring Component Importance",
    )
    print("✓ Feature importance plot generated")
    
    # 7. Distribution Plot (Section 5.i)
    ats_scores = np.random.normal(75, 15, 1000)  # Example - replace with actual
    figures["ats_distribution.png"] = plot_distribution(
        ats_scores,
        title="ATS Score Distribution",
        xlabel="ATS Score (0-100)",
    )
    print("✓ Distribution plot generated")
    
    # Save all figures
    save_all_figures(figures, output_dirs["visualizations"])
    print(f"\n✓ All visualizations saved to {output_dirs['visualizations']}")


def evaluate_model_performance(
    output_dirs: Dict[str, Path],
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for Section 6.c, 6.d.
    
    Args:
        output_dirs: Output directories
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 70)
    print("CALCULATING EVALUATION METRICS (Section 6.c, 6.d)")
    print("=" * 70)
    
    # Example data - replace with actual model predictions
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    
    # Classification metrics
    metrics = calculate_classification_metrics(y_true, y_pred, average="weighted")
    print("✓ Classification metrics calculated")
    
    # For text generation (cover letters)
    # Example - replace with actual generated and reference texts
    candidates = ["This is a generated cover letter." for _ in range(10)]
    references = ["This is a reference cover letter." for _ in range(10)]
    
    # BLEU Score
    bleu_scores = calculate_bleu(
        candidates,
        [[ref] for ref in references],
    )
    metrics.update(bleu_scores)
    print("✓ BLEU scores calculated")
    
    # ROUGE Score
    rouge_scores = calculate_rouge(candidates, references)
    metrics.update(rouge_scores)
    print("✓ ROUGE scores calculated")
    
    # BERTScore (optional - computationally expensive)
    try:
        bert_scores = calculate_bertscore(candidates, references)
        metrics.update(bert_scores)
        print("✓ BERTScore calculated")
    except Exception as e:
        print(f"⚠ BERTScore calculation skipped: {e}")
    
    # ATS Scoring evaluation (project-specific)
    true_scores = [85.0, 72.3, 91.5, 68.2, 79.8] * 20
    pred_scores = [83.0, 74.1, 89.3, 70.5, 78.2] * 20
    
    true_matches = [["python", "ml", "aws"]] * 100
    pred_matches = [["python", "ml", "docker"]] * 100
    
    ats_metrics = evaluate_ats_scoring(
        true_scores, pred_scores,
        true_matches, pred_matches,
    )
    metrics.update(ats_metrics)
    print("✓ ATS scoring evaluation completed")
    
    # Format and save metrics report
    report = format_metrics_report(metrics, title="IntelliCover AI Performance Metrics")
    
    with open(output_dirs["metrics"] / "evaluation_metrics.txt", "w") as f:
        f.write(report)
    
    with open(output_dirs["metrics"] / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to {output_dirs['metrics']}")
    
    return metrics


def compare_with_baselines(
    output_dirs: Dict[str, Path],
    proposed_metrics: Dict[str, float],
) -> None:
    """
    Compare proposed approach with baselines for Section 6.i, 6.j.
    
    Args:
        output_dirs: Output directories
        proposed_metrics: Metrics from proposed model
    """
    print("\n" + "=" * 70)
    print("COMPARING WITH BASELINES (Section 6.i, 6.j)")
    print("=" * 70)
    
    comparator = BaselineComparator()
    
    # Add proposed model results
    comparator.add_result("IntelliCover AI (Proposed)", proposed_metrics)
    
    # Example baseline results - replace with actual evaluations
    comparator.add_result("TF-IDF + Cosine", {
        "accuracy": 0.65,
        "precision": 0.63,
        "recall": 0.68,
        "f1_score": 0.65,
        "rmse": 12.5,
    })
    
    comparator.add_result("Word2Vec + Similarity", {
        "accuracy": 0.70,
        "precision": 0.69,
        "recall": 0.72,
        "f1_score": 0.70,
        "rmse": 10.8,
    })
    
    comparator.add_result("Rule-Based Matching", {
        "accuracy": 0.58,
        "precision": 0.60,
        "recall": 0.55,
        "f1_score": 0.57,
        "rmse": 15.3,
    })
    
    comparator.add_result("BERT-base (fine-tuned)", {
        "accuracy": 0.82,
        "precision": 0.81,
        "recall": 0.83,
        "f1_score": 0.82,
        "rmse": 7.2,
    })
    
    # Generate comparison table
    comparison_table = comparator.generate_comparison_table(format="markdown")
    
    with open(output_dirs["comparisons"] / "baseline_comparison.md", "w") as f:
        f.write("# Baseline Model Comparison (Section 6.i)\n\n")
        f.write(comparison_table)
    
    print("✓ Comparison table generated")
    
    # Generate comparison visualization
    comparison = comparator.compare()
    
    metrics_to_plot = {
        "Accuracy": comparison.get("accuracy", []),
        "F1-Score": comparison.get("f1_score", []),
        "RMSE": comparison.get("rmse", []),
    }
    
    fig = plot_model_comparison(
        comparison["models"],
        metrics_to_plot,
        title="Model Performance Comparison",
    )
    
    fig.savefig(
        output_dirs["comparisons"] / "model_comparison.png",
        dpi=300,
        bbox_inches='tight',
    )
    print("✓ Comparison visualization saved")
    
    # Generate pros/cons for each model
    with open(output_dirs["comparisons"] / "pros_cons.md", "w") as f:
        f.write("# Pros and Cons Analysis (Section 6.j)\n\n")
        
        for model_name in comparison["models"]:
            f.write(f"## {model_name}\n\n")
            
            # Get pros/cons
            analysis = comparator.get_pros_cons(model_name)
            
            f.write("**Pros:**\n")
            for pro in analysis["pros"]:
                f.write(f"- {pro}\n")
            
            f.write("\n**Cons:**\n")
            for con in analysis["cons"]:
                f.write(f"- {con}\n")
            
            f.write("\n")
    
    print("✓ Pros/cons analysis generated")
    print(f"\n✓ All comparisons saved to {output_dirs['comparisons']}")


def generate_explainability_analysis(
    output_dirs: Dict[str, Path],
) -> None:
    """
    Generate explainability visualizations and reports for Section 5.o, 6.g.
    
    Args:
        output_dirs: Output directories
    """
    print("\n" + "=" * 70)
    print("GENERATING EXPLAINABILITY ANALYSIS (Section 5.o, 6.g)")
    print("=" * 70)
    
    # Example: Feature importance from SHAP
    # Note: This requires actual model and data - this is a demonstration
    
    shap_importance = {
        "Python": 0.35,
        "Machine Learning": 0.28,
        "AWS": 0.22,
        "Docker": 0.18,
        "Years Experience": 0.15,
    }
    
    lime_importance = {
        "Python": 0.32,
        "Machine Learning": 0.30,
        "AWS": 0.20,
        "Docker": 0.16,
        "Years Experience": 0.14,
    }
    
    # Create explainability report
    report = create_explainability_report(
        shap_importance,
        lime_importance,
        top_n=10,
    )
    
    with open(output_dirs["explainability"] / "xai_report.txt", "w") as f:
        f.write(report)
    
    print("✓ Explainability report generated")
    
    # Save importance data
    with open(output_dirs["explainability"] / "feature_importance.json", "w") as f:
        json.dump({
            "shap": shap_importance,
            "lime": lime_importance,
        }, f, indent=2)
    
    print(f"✓ Explainability analysis saved to {output_dirs['explainability']}")


def document_implementation_details(
    output_dirs: Dict[str, Path],
) -> None:
    """
    Document implementation details for Section 5 and 6.
    
    Args:
        output_dirs: Output directories
    """
    print("\n" + "=" * 70)
    print("DOCUMENTING IMPLEMENTATION DETAILS")
    print("=" * 70)
    
    # Section 5.k: Dataset splitting
    split_info = document_data_split(
        train_size=8000,
        val_size=1000,
        test_size=1000,
    )
    
    with open(output_dirs["statistics"] / "data_split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    # Section 5.n: Hyperparameters
    hyperparameters = {
        "sentence_transformer": {
            "model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
        },
        "llm_suggestion": {
            "model": settings.SUGGESTION_LLM_MODEL,
            "max_new_tokens": settings.LLM_MAX_NEW_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": settings.LLM_TOP_P,
        },
        "llm_cover_letter": {
            "model": settings.COVER_LETTER_BASE_MODEL,
            "lora_adapter": settings.COVER_LETTER_LORA_ADAPTER,
            "max_new_tokens": 800,
            "temperature": 0.8,
        },
        "scoring_weights": {
            "hard_skills": 0.50,
            "semantic_experience": 0.40,
            "education": 0.10,
        },
    }
    
    with open(output_dirs["base"] / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)
    
    # Section 6.b: Software/Hardware setup
    environment_info = {
        "software": {
            "python_version": "3.10+",
            "framework": "FastAPI",
            "nlp_library": "spaCy, sentence-transformers",
            "llm_framework": "Hugging Face Transformers",
            "database": "PostgreSQL",
            "task_queue": "Celery + Redis",
        },
        "hardware": {
            "cpu": "Intel Xeon or equivalent",
            "gpu": "NVIDIA T4 or better (16GB VRAM recommended)",
            "ram": "32GB minimum",
            "storage": "50GB for models and data",
        },
        "cloud_setup": {
            "platform": "AWS / Azure / GCP",
            "compute": "GPU-enabled instances for LLM inference",
            "storage": "S3 / Blob Storage for files",
            "database": "Managed PostgreSQL",
        },
    }
    
    with open(output_dirs["base"] / "environment_setup.json", "w") as f:
        json.dump(environment_info, f, indent=2)
    
    print("✓ Implementation details documented")
    print(f"✓ Documentation saved to {output_dirs['base']}")


def main():
    """
    Main function to run the complete paper evaluation suite.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "INTELLICOVER AI - PAPER EVALUATION SUITE")
    print("=" * 70)
    
    # Setup directories
    output_dirs = setup_output_directories()
    print(f"\n✓ Output directories created at: {output_dirs['base']}")
    
    # Note: Replace these with actual data loading
    # Example data structures
    resumes_data = [
        {"skills": ["Python", "ML", "AWS"], "experience": [{"description": "Developed ML models"}]},
        {"skills": ["Java", "Spring", "Docker"], "experience": [{"description": "Built microservices"}]},
    ] * 50  # Simulate 100 resumes
    
    jds_data = [
        {"required_skills": ["Python", "ML", "AWS"], "responsibilities": ["Build AI systems"]},
        {"required_skills": ["Java", "Spring"], "responsibilities": ["Develop backend services"]},
    ] * 50  # Simulate 100 JDs
    
    # Run all evaluation steps
    generate_dataset_statistics(resumes_data, jds_data, output_dirs)
    generate_visualizations(output_dirs)
    proposed_metrics = evaluate_model_performance(output_dirs)
    compare_with_baselines(output_dirs, proposed_metrics)
    generate_explainability_analysis(output_dirs)
    document_implementation_details(output_dirs)
    
    print("\n" + "=" * 70)
    print("✓ PAPER EVALUATION SUITE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dirs['base']}")
    print("\nYou can now use these materials for your paper:")
    print("  - Statistics: Dataset descriptions (Section 5.e, 5.f)")
    print("  - Visualizations: EDA and training plots (Section 5.i, 6.e, 6.f)")
    print("  - Metrics: Performance evaluation (Section 6.c, 6.d)")
    print("  - Comparisons: Baseline comparisons (Section 6.i, 6.j)")
    print("  - Explainability: Model interpretability (Section 5.o, 6.g)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
