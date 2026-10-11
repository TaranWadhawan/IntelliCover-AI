"""
Baseline Models and Comparison Module
======================================
Implements baseline models for comparison with the proposed approach.
Required for Section 6.i and 6.j (comparison with baselines, pros/cons).

Includes:
  - Simple keyword-based matching (TF-IDF + Cosine Similarity)
  - Traditional ML baselines (Logistic Regression, SVM, Random Forest)
  - Pre-trained embedding baselines (Word2Vec, GloVe)
  - Simpler LLM baselines
  - Comparison framework and result aggregation
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1: TF-IDF + Cosine Similarity
# ──────────────────────────────────────────────────────────────────────────────

class TFIDFBaselineScorer:
    """
    Simple baseline using TF-IDF and cosine similarity.
    
    Mathematical Foundation:
    ------------------------
    TF-IDF(term, doc) = TF(term, doc) × IDF(term)
    
    where:
    - TF(term, doc) = frequency of term in document
    - IDF(term) = log(N / df(term))
    - N = total documents
    - df(term) = document frequency of term
    
    Similarity = cosine(vec_resume, vec_jd) = (A·B) / (||A|| ||B||)
    
    This is a simple but effective baseline for text matching.
    """
    
    def __init__(self, max_features: int = 500):
        """
        Initialize TF-IDF baseline.
        
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
        )
        self.is_fitted = False
        
        logger.info("tfidf_baseline_initialized", max_features=max_features)
    
    def fit(self, corpus: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on a corpus.
        
        Args:
            corpus: List of text documents
        """
        logger.info("fitting_tfidf_vectorizer", corpus_size=len(corpus))
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        logger.info("tfidf_vectorizer_fitted")
    
    def calculate_match_score(
        self,
        resume_text: str,
        jd_text: str,
    ) -> float:
        """
        Calculate match score between resume and job description.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
        
        Returns:
            Similarity score (0-100)
        """
        if not self.is_fitted:
            # Fit on the fly using both texts
            self.fit([resume_text, jd_text])
        
        # Vectorize
        resume_vec = self.vectorizer.transform([resume_text])
        jd_vec = self.vectorizer.transform([jd_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_vec, jd_vec)[0][0]
        
        score = similarity * 100.0  # Convert to 0-100 scale
        
        logger.info(
            "tfidf_match_score_calculated",
            score=round(score, 2),
        )
        
        return float(score)
    
    def extract_important_terms(
        self,
        text: str,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Extract most important terms from text based on TF-IDF scores.
        
        Args:
            text: Input text
            top_n: Number of top terms to return
        
        Returns:
            List of (term, score) tuples
        """
        if not self.is_fitted:
            self.fit([text])
        
        vec = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get scores
        scores = vec.toarray()[0]
        
        # Sort and get top N
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        logger.info("important_terms_extracted", num_terms=len(top_terms))
        
        return top_terms


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 2: Traditional ML Classifiers
# ──────────────────────────────────────────────────────────────────────────────

class TraditionalMLBaseline:
    """
    Traditional machine learning baseline models.
    
    Implements:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Random Forest
    
    These serve as strong baselines for classification tasks.
    """
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        **model_kwargs,
    ):
        """
        Initialize ML baseline model.
        
        Args:
            model_type: Type of model ('logistic_regression', 'svm', 'random_forest')
            **model_kwargs: Additional arguments for the model
        """
        if model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **model_kwargs,
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                **model_kwargs,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        
        logger.info("ml_baseline_initialized", model_type=model_type)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the baseline model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(
            "training_ml_baseline",
            model=self.model_type,
            num_samples=len(X_train),
        )
        
        self.model.fit(X_train, y_train)
        
        logger.info("ml_baseline_trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return one-hot encoded predictions
            predictions = self.model.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if available).
        
        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 3: Word Embedding Baseline (Word2Vec/GloVe)
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingBaselineScorer:
    """
    Baseline using pre-trained word embeddings.
    
    Uses average word embeddings to represent documents,
    then calculates cosine similarity.
    
    Mathematical Approach:
    ----------------------
    doc_embedding = (1/n) Σ word_embedding(w_i) for all words w_i in doc
    
    This is simpler than contextualized embeddings (BERT, etc.)
    but serves as a good baseline.
    """
    
    def __init__(
        self,
        embedding_model: str = "word2vec-google-news-300",
    ):
        """
        Initialize embedding baseline.
        
        Args:
            embedding_model: Pre-trained embedding model name
        """
        try:
            import gensim.downloader as api
            logger.info("loading_word_embeddings", model=embedding_model)
            self.embeddings = api.load(embedding_model)
            logger.info("word_embeddings_loaded")
        except Exception as e:
            logger.error("failed_to_load_embeddings", error=str(e))
            raise
        
        self.embedding_dim = self.embeddings.vector_size
    
    def get_document_embedding(self, text: str) -> np.ndarray:
        """
        Get document embedding by averaging word vectors.
        
        Args:
            text: Input text
        
        Returns:
            Document embedding vector
        """
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.embeddings:
                vectors.append(self.embeddings[word])
        
        if len(vectors) == 0:
            # No words found, return zero vector
            return np.zeros(self.embedding_dim)
        
        # Average word vectors
        doc_embedding = np.mean(vectors, axis=0)
        return doc_embedding
    
    def calculate_match_score(
        self,
        resume_text: str,
        jd_text: str,
    ) -> float:
        """
        Calculate semantic match score using embeddings.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
        
        Returns:
            Similarity score (0-100)
        """
        resume_emb = self.get_document_embedding(resume_text)
        jd_emb = self.get_document_embedding(jd_text)
        
        # Cosine similarity
        similarity = np.dot(resume_emb, jd_emb) / (
            np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb) + 1e-9
        )
        
        score = similarity * 100.0
        
        logger.info("embedding_match_score_calculated", score=round(score, 2))
        
        return float(score)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 4: Rule-Based Skill Matching
# ──────────────────────────────────────────────────────────────────────────────

class RuleBasedSkillMatcher:
    """
    Simple rule-based skill matching baseline.
    
    This represents the simplest possible approach:
    - Extract skills using keyword matching
    - Calculate overlap percentage
    
    No ML involved - purely deterministic.
    """
    
    def __init__(self, skill_keywords: Optional[List[str]] = None):
        """
        Initialize rule-based matcher.
        
        Args:
            skill_keywords: List of known skills (if None, uses default set)
        """
        if skill_keywords is None:
            # Use a default set of common tech skills
            skill_keywords = [
                "python", "java", "javascript", "c++", "sql",
                "machine learning", "deep learning", "nlp",
                "aws", "azure", "docker", "kubernetes",
                "react", "angular", "node.js", "django", "flask",
                "tensorflow", "pytorch", "scikit-learn",
                "git", "agile", "scrum",
            ]
        
        self.skill_keywords = [s.lower() for s in skill_keywords]
        
        logger.info("rule_based_matcher_initialized", num_skills=len(skill_keywords))
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using keyword matching.
        
        Args:
            text: Input text
        
        Returns:
            List of matched skills
        """
        text_lower = text.lower()
        matched_skills = []
        
        for skill in self.skill_keywords:
            if skill in text_lower:
                matched_skills.append(skill)
        
        return matched_skills
    
    def calculate_match_score(
        self,
        resume_text: str,
        jd_text: str,
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate skill match score.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
        
        Returns:
            Tuple of (score, matched_skills, missing_skills)
        """
        resume_skills = set(self.extract_skills(resume_text))
        jd_skills = set(self.extract_skills(jd_text))
        
        if len(jd_skills) == 0:
            score = 100.0
            matched = []
            missing = []
        else:
            matched = list(resume_skills.intersection(jd_skills))
            missing = list(jd_skills.difference(resume_skills))
            score = (len(matched) / len(jd_skills)) * 100.0
        
        logger.info(
            "rule_based_score_calculated",
            score=round(score, 2),
            matched=len(matched),
            missing=len(missing),
        )
        
        return float(score), matched, missing


# ──────────────────────────────────────────────────────────────────────────────
# Comparison Framework
# ──────────────────────────────────────────────────────────────────────────────

class BaselineComparator:
    """
    Framework for comparing multiple baseline models with the proposed approach.
    
    Essential for Section 6.i (Comparison with Other Baselines).
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.results = {}
        logger.info("baseline_comparator_initialized")
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Add evaluation results for a model.
        
        Args:
            model_name: Name of the model/baseline
            metrics: Dictionary of metric names to scores
        """
        self.results[model_name] = metrics
        logger.info("result_added", model_name=model_name, num_metrics=len(metrics))
    
    def compare(
        self,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare all models across metrics.
        
        Args:
            metric_names: List of metrics to compare (None = all)
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("comparing_baselines", num_models=len(self.results))
        
        if not self.results:
            logger.warning("no_results_to_compare")
            return {}
        
        # Get all metric names if not specified
        if metric_names is None:
            metric_names = set()
            for metrics in self.results.values():
                metric_names.update(metrics.keys())
            metric_names = sorted(list(metric_names))
        
        # Build comparison table
        comparison = {
            "models": list(self.results.keys()),
        }
        
        for metric in metric_names:
            comparison[metric] = [
                self.results[model].get(metric, np.nan)
                for model in comparison["models"]
            ]
        
        # Find best model for each metric
        comparison["best_model_per_metric"] = {}
        for metric in metric_names:
            scores = comparison[metric]
            if all(np.isnan(scores)):
                comparison["best_model_per_metric"][metric] = None
            else:
                best_idx = np.nanargmax(scores)
                comparison["best_model_per_metric"][metric] = comparison["models"][best_idx]
        
        logger.info("baseline_comparison_complete")
        
        return comparison
    
    def generate_comparison_table(
        self,
        format: str = "markdown",
    ) -> str:
        """
        Generate formatted comparison table.
        
        Args:
            format: Output format ('markdown', 'latex', 'csv')
        
        Returns:
            Formatted table string
        """
        import pandas as pd
        
        logger.info("generating_comparison_table", format=format)
        
        comparison = self.compare()
        
        if not comparison:
            return "No results to compare."
        
        # Create DataFrame
        df = pd.DataFrame({
            metric: comparison[metric]
            for metric in comparison.keys()
            if metric not in ["models", "best_model_per_metric"]
        }, index=comparison["models"])
        
        # Add best model indicators
        for metric in df.columns:
            best_model = comparison["best_model_per_metric"].get(metric)
            if best_model:
                df.loc[best_model, metric] = f"**{df.loc[best_model, metric]:.4f}**"
        
        # Format output
        if format == "markdown":
            table_str = df.to_markdown(floatfmt=".4f")
        elif format == "latex":
            table_str = df.to_latex(float_format="%.4f")
        elif format == "csv":
            table_str = df.to_csv()
        else:
            table_str = df.to_string()
        
        return table_str
    
    def get_pros_cons(
        self,
        model_name: str,
    ) -> Dict[str, List[str]]:
        """
        Generate pros and cons for a specific model.
        
        Required for Section 6.j (Comparison based on Pros and Cons/Limitations).
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with 'pros' and 'cons' lists
        """
        logger.info("generating_pros_cons", model=model_name)
        
        # This would ideally be customized based on actual results
        # Here we provide a template that should be filled in based on evaluation
        
        if model_name not in self.results:
            logger.warning("model_not_found_in_results", model=model_name)
            return {"pros": [], "cons": []}
        
        metrics = self.results[model_name]
        
        # Generic analysis based on metrics
        pros = []
        cons = []
        
        # Example logic (customize based on your specific metrics)
        if metrics.get("accuracy", 0) > 0.85:
            pros.append("High accuracy in predictions")
        else:
            cons.append("Lower accuracy compared to other models")
        
        if metrics.get("inference_time", float('inf')) < 1.0:
            pros.append("Fast inference time suitable for real-time applications")
        else:
            cons.append("Slower inference may limit real-time use")
        
        return {
            "pros": pros,
            "cons": cons,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def create_baseline_ensemble(
    baseline_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Create an ensemble score from multiple baselines.
    
    Args:
        baseline_scores: Dictionary mapping baseline names to scores
        weights: Optional weights for each baseline (uniform if None)
    
    Returns:
        Weighted ensemble score
    """
    if weights is None:
        weights = {name: 1.0 for name in baseline_scores.keys()}
    
    weighted_sum = sum(score * weights.get(name, 1.0) 
                      for name, score in baseline_scores.items())
    total_weight = sum(weights.get(name, 1.0) for name in baseline_scores.keys())
    
    ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    logger.info(
        "ensemble_score_calculated",
        ensemble_score=round(ensemble_score, 2),
        num_baselines=len(baseline_scores),
    )
    
    return ensemble_score
