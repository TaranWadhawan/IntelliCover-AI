"""
Integration Test for Paper Evaluation Modules
==============================================
Quick test to verify all evaluation modules can be imported and basic functionality works.
Run this to ensure the implementation is correct before generating paper outputs.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_evaluation_metrics():
    """Test evaluation_metrics module."""
    print("\n" + "=" * 70)
    print("Testing Evaluation Metrics Module")
    print("=" * 70)
    
    try:
        from app.nlp.evaluation_metrics import (
            calculate_classification_metrics,
            calculate_regression_metrics,
            calculate_bleu,
            calculate_rouge,
            format_metrics_report,
        )
        
        # Test classification metrics
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0])
        
        metrics = calculate_classification_metrics(y_true, y_pred, average="weighted")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        print(f"✓ Classification metrics: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
        
        # Test regression metrics
        y_true_reg = np.array([85.0, 72.3, 91.5, 68.2, 79.8])
        y_pred_reg = np.array([83.0, 74.1, 89.3, 70.5, 78.2])
        
        reg_metrics = calculate_regression_metrics(y_true_reg, y_pred_reg)
        assert "rmse" in reg_metrics
        assert "mae" in reg_metrics
        print(f"✓ Regression metrics: RMSE={reg_metrics['rmse']:.3f}, MAE={reg_metrics['mae']:.3f}")
        
        # Test BLEU
        candidates = ["This is a test sentence."]
        references = [["This is a reference sentence."]]
        
        bleu = calculate_bleu(candidates, references)
        assert "bleu-1" in bleu
        print(f"✓ BLEU score: BLEU-1={bleu['bleu-1']:.3f}")
        
        # Test metrics report formatting
        all_metrics = {**metrics, **reg_metrics, **bleu}
        report = format_metrics_report(all_metrics, title="Test Report")
        assert len(report) > 0
        print("✓ Metrics report formatting works")
        
        print("\n✅ Evaluation Metrics Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation Metrics Module: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization module."""
    print("\n" + "=" * 70)
    print("Testing Visualization Module")
    print("=" * 70)
    
    try:
        from app.nlp.visualization import (
            plot_confusion_matrix,
            plot_training_history,
            plot_roc_curve,
            plot_wordcloud,
        )
        
        # Test confusion matrix
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 0])
        
        fig = plot_confusion_matrix(y_true, y_pred, labels=["Low", "Med", "High"])
        assert fig is not None
        print("✓ Confusion matrix plot created")
        
        # Test training history
        history = {
            "train_loss": [0.8, 0.6, 0.4],
            "val_loss": [0.85, 0.65, 0.5],
            "train_accuracy": [0.6, 0.7, 0.8],
            "val_accuracy": [0.58, 0.68, 0.75],
        }
        
        fig = plot_training_history(history)
        assert fig is not None
        print("✓ Training history plot created")
        
        # Test ROC curve
        y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        y_scores = np.random.rand(10)
        
        fig = plot_roc_curve(y_true_binary, y_scores)
        assert fig is not None
        print("✓ ROC curve plot created")
        
        # Test word cloud
        text = "Python ML AWS Docker Kubernetes machine learning data science"
        fig = plot_wordcloud(text, max_words=20)
        assert fig is not None
        print("✓ Word cloud plot created")
        
        print("\n✅ Visualization Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Visualization Module: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eda_statistics():
    """Test EDA statistics module."""
    print("\n" + "=" * 70)
    print("Testing EDA Statistics Module")
    print("=" * 70)
    
    try:
        from app.nlp.eda_statistics import (
            calculate_descriptive_statistics,
            analyze_text_corpus,
            analyze_skill_distribution,
            analyze_class_balance,
            document_data_split,
        )
        
        # Test descriptive statistics
        data = [85.0, 72.3, 91.5, 68.2, 79.8, 88.1, 75.6, 82.4]
        stats = calculate_descriptive_statistics(data, "ATS_Score")
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        print(f"✓ Descriptive stats: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
        
        # Test text corpus analysis
        texts = ["This is text one.", "This is text two with more words."]
        corpus_stats = analyze_text_corpus(texts, include_vocabulary=False)
        
        assert "total_documents" in corpus_stats
        assert "avg_tokens_per_doc" in corpus_stats
        print(f"✓ Text corpus: {corpus_stats['total_documents']} docs, {corpus_stats['avg_tokens_per_doc']:.1f} avg tokens")
        
        # Test skill distribution
        skill_lists = [["Python", "ML"], ["Python", "AWS"], ["Java", "Spring"]]
        skill_stats = analyze_skill_distribution(skill_lists)
        
        assert "unique_skills" in skill_stats
        assert "total_skill_mentions" in skill_stats
        print(f"✓ Skills: {skill_stats['unique_skills']} unique, {skill_stats['total_skill_mentions']} total")
        
        # Test class balance
        labels = [0, 1, 2, 0, 1, 2, 0, 0, 0, 0]
        balance = analyze_class_balance(labels, return_gini=True)
        
        assert "class_distribution" in balance
        assert "gini_coefficient" in balance
        print(f"✓ Class balance: Gini={balance['gini_coefficient']:.3f}")
        
        # Test data split documentation
        split_info = document_data_split(8000, 1000, 1000)
        assert "train" in split_info
        assert "train_ratio" in split_info
        print(f"✓ Data split: {split_info['train_ratio']:.1%} train, {split_info['val_ratio']:.1%} val, {split_info['test_ratio']:.1%} test")
        
        print("\n✅ EDA Statistics Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ EDA Statistics Module: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explainability():
    """Test explainability module."""
    print("\n" + "=" * 70)
    print("Testing Explainability Module")
    print("=" * 70)
    
    try:
        from app.nlp.explainability import (
            highlight_important_tokens,
            create_explainability_report,
        )
        
        # Test token highlighting
        text = "Python machine learning AWS cloud computing"
        importance_scores = [0.9, 0.8, 0.3, 0.7, 0.2]
        
        highlighted = highlight_important_tokens(text.split(), importance_scores, threshold=0.5)
        assert len(highlighted) > 0
        print(f"✓ Token highlighting: {len(highlighted)} important tokens identified")
        
        # Test explainability report
        shap_importance = {"Python": 0.35, "ML": 0.28, "AWS": 0.22}
        lime_importance = {"Python": 0.32, "ML": 0.30, "AWS": 0.20}
        
        report = create_explainability_report(shap_importance, lime_importance, top_n=5)
        assert len(report) > 0
        print("✓ Explainability report generated")
        
        # Note: Full SHAP/LIME testing requires actual models
        print("✓ Basic explainability functions work (SHAP/LIME require actual models)")
        
        print("\n✅ Explainability Module: PASSED (Basic)")
        return True
        
    except Exception as e:
        print(f"\n❌ Explainability Module: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_models():
    """Test baseline models module."""
    print("\n" + "=" * 70)
    print("Testing Baseline Models Module")
    print("=" * 70)
    
    try:
        from app.nlp.baseline_models import (
            TFIDFBaselineScorer,
            RuleBasedSkillMatcher,
            BaselineComparator,
        )
        
        # Test TF-IDF baseline
        tfidf = TFIDFBaselineScorer()
        
        resume_text = "Python machine learning AWS cloud"
        jd_text = "Python ML AWS Docker"
        
        tfidf.fit([resume_text, jd_text])
        score = tfidf.score(resume_text, jd_text)
        
        assert 0 <= score <= 1
        print(f"✓ TF-IDF baseline: Score={score:.3f}")
        
        # Test rule-based matcher
        matcher = RuleBasedSkillMatcher(
            skill_keywords=["python", "ml", "aws", "docker"]
        )
        
        resume_skills = ["Python", "ML", "AWS"]
        jd_skills = ["Python", "ML", "Docker"]
        
        match_score = matcher.match_skills(resume_skills, jd_skills)
        assert 0 <= match_score <= 1
        print(f"✓ Rule-based matcher: Score={match_score:.3f}")
        
        # Test comparator
        comparator = BaselineComparator()
        
        comparator.add_result("Model A", {"accuracy": 0.85, "f1_score": 0.83})
        comparator.add_result("Model B", {"accuracy": 0.80, "f1_score": 0.78})
        
        comparison = comparator.compare()
        assert len(comparison["models"]) == 2
        print(f"✓ Comparator: {len(comparison['models'])} models compared")
        
        # Test table generation
        table = comparator.generate_comparison_table(format="markdown")
        assert len(table) > 0
        print("✓ Comparison table generated")
        
        # Test pros/cons
        analysis = comparator.get_pros_cons("Model A")
        assert "pros" in analysis
        assert "cons" in analysis
        print(f"✓ Pros/cons: {len(analysis['pros'])} pros, {len(analysis['cons'])} cons")
        
        print("\n✅ Baseline Models Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Baseline Models Module: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_existing():
    """Test integration with existing modules."""
    print("\n" + "=" * 70)
    print("Testing Integration with Existing Code")
    print("=" * 70)
    
    try:
        # Try importing existing modules
        from app.nlp.scorer import SemanticMatcher
        from app.nlp.resume_parser import extract_text_from_pdf
        from app.core.config import Settings
        
        print("✓ Existing modules can still be imported")
        
        # Test that new modules don't conflict
        from app.nlp.evaluation_metrics import calculate_classification_metrics
        from app.nlp.visualization import plot_confusion_matrix
        from app.nlp.eda_statistics import calculate_descriptive_statistics
        
        print("✓ New modules don't conflict with existing code")
        
        print("\n✅ Integration Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "PAPER EVALUATION MODULES - INTEGRATION TEST")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Evaluation Metrics", test_evaluation_metrics()))
    results.append(("Visualization", test_visualization()))
    results.append(("EDA Statistics", test_eda_statistics()))
    results.append(("Explainability", test_explainability()))
    results.append(("Baseline Models", test_baseline_models()))
    results.append(("Integration", test_integration_with_existing()))
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 28 + "TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<50} {status}")
    
    print("\n" + "=" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to generate paper outputs.")
        print("\nNext steps:")
        print("1. Replace example data in generate_paper_outputs.py with actual data")
        print("2. Run: python generate_paper_outputs.py")
        print("3. Use outputs in paper_outputs/ directory for your paper")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check for any import errors or missing packages")
        print("3. Review the traceback for specific issues")
    
    print("=" * 70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
