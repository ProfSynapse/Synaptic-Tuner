# Research: Expanding Toolset-Training to Traditional ML

## Executive Summary

The Toolset-Training pipeline currently handles LLM fine-tuning (SFT/KTO) and synthetic data generation. This research explores two complementary expansion vectors:

1. **Pipeline-integrated ML**: Traditional ML algorithms (LightGBM, XGBoost, CatBoost) that enhance the existing LLM pipeline -- dataset quality scoring, model routing, evaluation classification, and feature extraction.

2. **Standalone ML training**: First-class support for arbitrary prediction tasks (classification, regression, clustering) on tabular, text, or mixed data -- making Toolset-Training a general-purpose model training platform.

For pipeline integration, the recommended first step is a **dataset quality classifier** using LightGBM with TF-IDF features, integrated into the existing `SynthChat/` improvement pipeline.

For standalone ML, the recommended approach is a **config-driven training module** (`Trainers/ml/`) with YAML configs mirroring the SFT/KTO pattern, powered by scikit-learn pipelines with optional AutoML via FLAML. This provides multi-algorithm training, automated feature engineering, cross-validation, and structured output directories -- all exposed through the existing CLI.

MLflow is the recommended experiment tracking platform for both tracks, as it natively supports scikit-learn and LLM workflows under a single UI.

## Background

This research was prompted by the desire to expand beyond pure LLM fine-tuning and leverage the strengths of traditional ML where appropriate. The current pipeline structure (documented in `CLAUDE.md`) centers on:

- **tuner.py / run.sh**: CLI entry point with interactive menus
- **Trainers/**: SFT (`rtx3090_sft/`) and KTO (`rtx3090_kto/`) training
- **SynthChat/**: Synthetic data generation and quality improvement (rubric runner)
- **Evaluator/**: Model evaluation harness
- **shared/**: LLM client, validation, utilities, judge module
- **Datasets/**: JSONL conversation datasets

The key question: where does traditional ML add value alongside this LLM-centric workflow?

## Methodology

Research was conducted through:
1. Web search of current documentation, papers, and industry practices (2025-2026)
2. Analysis of the existing Toolset-Training codebase structure
3. Review of integration patterns from MLflow, RAPIDS, and ML pipeline frameworks
4. Examination of recent papers on data quality scoring and LLM routing

---

## 1. Traditional ML Algorithm Landscape

### Gradient Boosting Frameworks

The three dominant gradient boosting frameworks each have distinct strengths:

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Tree Growth** | Level-wise (balanced) | Leaf-wise (faster convergence) | Symmetric (balanced) |
| **Speed** | Fast | Fastest (GOSS + EFB) | Moderate |
| **Categorical Features** | Requires encoding | Requires encoding | Native support |
| **Missing Values** | Built-in handling | Built-in handling | Built-in handling |
| **GPU Support** | Yes | Yes | Yes |
| **Regularization** | L1 + L2 | L1 + L2 | Ordered boosting |
| **Overfitting Resistance** | Good | Good | Best (ordered boosting) |
| **Best For** | General purpose, competitions | Large datasets, fast iteration | Categorical-heavy data, minimal tuning |
| **Maturity** | Most mature, largest community | Very mature | Mature, growing |
| **PyPI Install** | `pip install xgboost` | `pip install lightgbm` | `pip install catboost` |

**Recommendation for this project**: **LightGBM** as the primary choice due to speed on large datasets (synthetic data can be voluminous), strong GPU support, and excellent scikit-learn API compatibility. CatBoost as secondary for tasks with heavy categorical features.

### Scikit-learn Ecosystem

Scikit-learn (v1.8.0, 2026) remains the foundational ML library providing:
- **Preprocessing**: `TfidfVectorizer`, `StandardScaler`, `LabelEncoder`
- **Pipeline API**: `sklearn.pipeline.Pipeline` for reproducible workflows
- **Model Selection**: `GridSearchCV`, `RandomizedSearchCV`, `cross_validate`
- **Metrics**: `confusion_matrix`, `classification_report`, `roc_auc_score`
- **Feature Selection**: `SelectKBest`, `mutual_info_classif`

All gradient boosting libraries implement scikit-learn's estimator API (`fit`/`predict`/`transform`), enabling seamless integration with pipelines and cross-validation.

### GPU Acceleration: RAPIDS cuML

NVIDIA RAPIDS cuML (v26.02) provides **zero code-change GPU acceleration** for scikit-learn:
- Drop-in replacement: `import cuml` accelerates existing scikit-learn code
- 10-50x faster performance on average for realistic workloads
- 50+ algorithms: clustering, regression, classification, dimensionality reduction
- Multi-GPU support via Dask
- Now pip-installable from PyPI (since 25.10 release)
- Compatible with existing conda/PyTorch environments

**Relevance**: Since the project already uses NVIDIA GPUs for LLM training, cuML can leverage the same hardware for traditional ML with near-zero integration cost.

---

## 2. Use Cases Within This Pipeline

### 2.1 Dataset Quality Scoring (HIGH VALUE)

**Problem**: The current `SynthChat/services/rubric_runner.py` uses LLM-as-judge for quality assessment. This is effective but slow and expensive -- each example requires an LLM inference call.

**Traditional ML Solution**: Train a lightweight classifier (LightGBM/XGBoost) on features extracted from conversation data to predict quality scores. Use LLM-as-judge labels as ground truth for training.

**Feature Engineering from JSONL Conversations**:
```python
# Example feature extraction from a conversation
features = {
    # Structural features
    "num_turns": len(conversations),
    "avg_turn_length": mean([len(c["content"]) for c in conversations]),
    "max_turn_length": max([len(c["content"]) for c in conversations]),
    "user_assistant_ratio": count_role("user") / count_role("assistant"),

    # Content features (TF-IDF on concatenated text)
    "tfidf_features": tfidf_vectorizer.transform([full_text]),

    # Tool-calling specific features
    "has_tool_call": "tool_call:" in assistant_content,
    "num_tool_calls": count_tool_calls(assistant_content),
    "has_context_param": "context" in first_tool_arg,
    "has_result": "Result:" in assistant_content,

    # Validation features (from shared/validation/)
    "xml_valid": run_xml_validation(content),
    "json_valid": run_json_validation(content),
    "schema_errors_count": count_schema_errors(content),

    # Complexity features
    "vocabulary_richness": unique_words / total_words,
    "avg_sentence_length": mean_sentence_length(content),
}
```

**Impact**: Pre-filter synthetic data at 1000x the speed of LLM-as-judge, reserving expensive LLM evaluation for borderline cases only. Research shows TF-IDF + XGBoost achieves F1 scores of 0.92+ on text classification tasks.

### 2.2 LLM Response Router/Classifier (HIGH VALUE)

**Problem**: When evaluating or generating with multiple models/backends (LM Studio, Ollama, OpenRouter), there's no intelligent routing.

**Traditional ML Solution**: Train a classifier to route prompts to the most appropriate model based on prompt features (complexity, domain, tool requirements).

**How it works**:
1. Collect prompt-model-quality triples from evaluation runs
2. Extract features from prompts (length, tool keywords, complexity metrics)
3. Train a Random Forest or LightGBM classifier to predict best model per prompt
4. Integrate into `shared/llm/` client layer for automatic routing

**Industry precedent**: RouteLLM (by LMSYS) demonstrates 30-70% cost reduction while maintaining 95% of top-model quality. NVIDIA released an LLM Router Blueprint. Random Forests have proven effective for model selection ("Routing on Random Forests").

### 2.3 Training Data Deduplication and Clustering (MEDIUM VALUE)

**Problem**: As synthetic datasets grow, near-duplicates and imbalanced topic coverage reduce training effectiveness.

**Traditional ML Solution**:
- **HDBSCAN/DBSCAN** clustering on sentence embeddings to identify topic clusters
- **MinHash/LSH** for near-duplicate detection
- **K-Means** with TF-IDF for topic distribution analysis
- **UMAP** dimensionality reduction for visualization

**Integration point**: Add as a preprocessing step in `SynthChat/` or `Tools/` before training.

### 2.4 Evaluation Classification (MEDIUM VALUE)

**Problem**: Current `Evaluator/` uses LLM-based evaluation which is slow and non-deterministic.

**Traditional ML Solution**: For structured evaluation criteria (tool-call format correctness, response structure, keyword presence), train binary classifiers that provide instant, deterministic pass/fail decisions.

**Approach**:
- Use existing evaluation results as labeled training data
- Train per-criterion classifiers (e.g., "correct tool call format", "appropriate response length")
- Reserve LLM-as-judge for subjective quality dimensions only
- Store predictions alongside LLM evaluations for comparison

### 2.5 KTO Label Prediction (MEDIUM VALUE)

**Problem**: KTO training requires interleaved `true`/`false` labeled examples. Generating `false` examples requires careful crafting.

**Traditional ML Solution**: Train a classifier on existing labeled data to predict whether a new synthetic example would be labeled `true` or `false`, helping balance the dataset and identify edge cases.

### 2.6 Anomaly Detection in Training Runs (LOW-MEDIUM VALUE)

**Problem**: Training runs can silently degrade (loss plateaus, gradient issues) without clear signals.

**Traditional ML Solution**: Train an anomaly detector (Isolation Forest, One-Class SVM) on historical training metrics to flag unusual patterns early.

### 2.7 Feature Store for Prompt Analysis (LOW VALUE, FUTURE)

**Problem**: Repeated feature extraction from the same prompts/conversations across different tasks.

**Traditional ML Solution**: Build a lightweight feature store that caches extracted features (TF-IDF vectors, embeddings, structural metrics) for reuse across quality scoring, routing, and evaluation.

---

## 3. Integration Patterns

### 3.1 MLflow for Unified Experiment Tracking

**MLflow** (v3.9.0, 2026) is the recommended platform for tracking both traditional ML and LLM experiments:

| Capability | Traditional ML | LLM Training |
|------------|---------------|--------------|
| **Autologging** | `mlflow.sklearn.autolog()` | Custom logging via callbacks |
| **Parameters** | Hyperparameters (n_estimators, lr) | LoRA rank, learning rate, epochs |
| **Metrics** | Accuracy, F1, AUC | Loss, perplexity |
| **Artifacts** | Serialized models, plots | LoRA adapters, merged models |
| **Model Registry** | Version + stage management | Same |
| **Comparison UI** | Side-by-side runs | Same |

**Integration with existing code**:
```python
# In a new Trainers/ml_classifiers/train_quality_scorer.py
import mlflow
import mlflow.sklearn

mlflow.set_experiment("quality-scoring")

with mlflow.start_run():
    mlflow.sklearn.autolog()  # Captures everything automatically

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LGBMClassifier(n_estimators=500))
    ])
    pipeline.fit(X_train, y_train)

    # Cross-validation metrics logged automatically
    # Model artifact saved automatically
```

**Key advantage**: MLflow's scikit-learn autolog captures parameters, metrics, cross-validation results, and models without manual instrumentation. The same MLflow server can track LLM fine-tuning runs, enabling cross-paradigm comparison.

### 3.2 Scikit-learn Pipeline Pattern

Use `sklearn.pipeline.Pipeline` for reproducible, serializable workflows:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

quality_pipeline = Pipeline([
    ("text_features", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("classifier", LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        device="gpu"  # Use same GPU as LLM training
    ))
])

# Serialize entire pipeline (preprocessing + model)
import joblib
joblib.dump(quality_pipeline, "models/quality_scorer.joblib")
```

### 3.3 Feature Engineering Pipeline

For conversation data, a dedicated feature extraction step is needed:

```python
class ConversationFeatureExtractor:
    """Extract tabular features from JSONL conversation data."""

    def transform(self, conversations_jsonl):
        features = []
        for line in conversations_jsonl:
            conv = json.loads(line)
            feat = {
                "num_turns": len(conv["conversations"]),
                "total_chars": sum(len(c["content"]) for c in conv["conversations"]),
                "has_tool_call": any("tool_call:" in c["content"] for c in conv["conversations"]),
                "num_tool_calls": self._count_tool_calls(conv),
                "schema_error_count": self._count_schema_errors(conv),
                "full_text": " ".join(c["content"] for c in conv["conversations"]),
                # ... more features
            }
            features.append(feat)
        return pd.DataFrame(features)
```

---

## 4. Architecture Impact

### 4.1 Proposed Directory Structure Changes

```
Toolset-Training/
├── Trainers/
│   ├── rtx3090_sft/           # Existing LLM SFT
│   ├── rtx3090_kto/           # Existing LLM KTO
│   ├── rtx3090_grpo/          # Existing LLM GRPO
│   │
│   ├── ml_classifiers/        # NEW: Traditional ML training
│   │   ├── train_quality_scorer.py
│   │   ├── train_router.py
│   │   ├── train_eval_classifier.py
│   │   ├── configs/
│   │   │   └── quality_scorer_config.py
│   │   ├── features/          # Feature extraction modules
│   │   │   ├── conversation_features.py
│   │   │   ├── text_features.py
│   │   │   └── structural_features.py
│   │   └── models/            # Saved model artifacts
│   │       └── quality_scorer.joblib
│   │
│   └── shared/                # Existing shared training code
│
├── shared/
│   ├── llm/                   # Existing
│   ├── validation/            # Existing
│   ├── judge/                 # Existing
│   │
│   ├── ml/                    # NEW: Shared ML utilities
│   │   ├── features.py        # Common feature extractors
│   │   ├── metrics.py         # Evaluation metrics + reporting
│   │   └── data_loaders.py    # JSONL -> tabular conversion
│   │
│   └── experiment_tracking/   # NEW: MLflow integration
│       ├── mlflow_config.py
│       └── tracking.py
│
├── Evaluator/
│   └── ml_evaluator.py        # NEW: Traditional ML model evaluation
│
├── Datasets/
│   ├── tools_datasets/        # Existing JSONL
│   ├── behavior_datasets/     # Existing JSONL
│   │
│   └── ml_features/           # NEW: Extracted feature caches
│       └── quality_features.parquet
│
└── tuner.py                   # Extended CLI
```

### 4.2 CLI Integration

The `tuner.py` / `run.sh` CLI should be extended with a new menu category:

```
Main Menu:
  [1] Train LLM (SFT/KTO/GRPO)
  [2] Upload to HuggingFace
  [3] Evaluate Model
  [4] Generate Synthetic Data
  [5] Improvement Engine
  [6] ML Classifiers            # NEW
      ├── Train Quality Scorer
      ├── Train Router
      ├── Train Eval Classifier
      ├── Evaluate ML Model
      └── Extract Features
  [7] MLflow Dashboard          # NEW (launches mlflow ui)
```

### 4.3 Integration Points with Existing Code

| Existing Module | Integration | How |
|-----------------|-------------|-----|
| `SynthChat/services/rubric_runner.py` | Pre-filter with quality scorer | Call ML model before LLM judge |
| `shared/llm/` client layer | Model routing | Classify prompt -> select backend |
| `Evaluator/` | Hybrid evaluation | ML for structural checks, LLM for subjective |
| `shared/validation/` | Feature source | Validation results become ML features |
| `shared/judge/` | Label source | Judge scores become ML training labels |
| `Tools/validate_syngen.py` | Enhanced validation | ML-powered quality predictions |

---

## 5. Libraries and Tooling

### Core Dependencies

| Library | Version | Purpose | Conda Compatible | GPU Support |
|---------|---------|---------|-------------------|-------------|
| **scikit-learn** | 1.8.0 | ML framework, pipelines, metrics | Yes | Via cuML |
| **lightgbm** | 4.x | Primary gradient boosting | Yes | Yes (native) |
| **xgboost** | 2.x | Alternative gradient boosting | Yes | Yes (native) |
| **catboost** | 1.x | Categorical-heavy tasks | Yes | Yes (native) |
| **mlflow** | 3.9.0 | Experiment tracking | Yes | N/A |
| **pandas** | 2.x | Data manipulation | Yes (existing) | Via cuDF |
| **joblib** | 1.x | Model serialization | Yes (existing) | N/A |
| **optuna** | 3.x | Hyperparameter optimization | Yes | N/A |
| **shap** | 0.x | Model interpretability | Yes | Yes |

### Optional / Future Dependencies

| Library | Purpose | When to Add |
|---------|---------|-------------|
| **cuml** (RAPIDS) | GPU-accelerated scikit-learn | When dataset scale demands it |
| **hdbscan** | Clustering for deduplication | When implementing data clustering |
| **umap-learn** | Dimensionality reduction / viz | When implementing data visualization |
| **sentence-transformers** | Embedding extraction | When implementing semantic features |

### Compatibility with Existing Environment

The existing conda `toolset` environment uses PyTorch + CUDA. All recommended libraries are compatible:
- LightGBM, XGBoost, CatBoost all support CUDA and coexist with PyTorch
- scikit-learn is a standard conda dependency
- MLflow has no GPU requirements but tracks GPU-based experiments
- No dependency conflicts with existing `requirements.txt`

**Installation (incremental)**:
```bash
# Minimal first step
pip install lightgbm scikit-learn mlflow optuna

# Full suite
pip install lightgbm xgboost catboost scikit-learn mlflow optuna shap
```

---

## 6. Dataset Implications

### Current Format: JSONL Conversations

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "label": true}
```

### Required: Tabular Feature Extraction

Traditional ML requires tabular data. The conversion pipeline:

```
JSONL Conversations  →  Feature Extraction  →  Parquet/CSV  →  ML Training
                         (text + structural)
```

### Feature Categories

| Category | Features | Extraction Method |
|----------|----------|-------------------|
| **Text** | TF-IDF vectors, n-grams, vocabulary stats | `TfidfVectorizer` on concatenated content |
| **Structural** | Turn count, lengths, role ratios | Direct computation from JSON |
| **Tool-specific** | Tool call count, parameter presence, result format | Regex/parsing on assistant content |
| **Validation** | Schema errors, XML validity, JSON correctness | Existing `shared/validation/` |
| **Quality signals** | Judge scores, rubric pass/fail | Existing `shared/judge/` results |
| **Embedding** | Sentence embeddings (384-768 dim) | `sentence-transformers` (optional) |

### Storage Format

| Format | Use Case | Why |
|--------|----------|-----|
| **Parquet** | Feature caches, large datasets | Columnar, compressed, fast reads |
| **CSV** | Small datasets, debugging | Human-readable |
| **Joblib** | Fitted transformers/pipelines | Python-native serialization |
| **JSONL** | Keep for LLM training (no change) | Existing format, no migration |

### Key Insight: No Migration Required

The existing JSONL datasets remain as-is for LLM training. Traditional ML operates on **derived features** extracted from those same datasets. This means:
- No breaking changes to existing data pipelines
- Feature extraction is additive (new code, no changes to existing)
- Features can be cached in `Datasets/ml_features/` as Parquet files
- Same source data serves both LLM and ML training

---

## 7. Evaluation Differences

### Traditional ML vs LLM Evaluation

| Aspect | Traditional ML | LLM Evaluation |
|--------|---------------|----------------|
| **Primary Metrics** | Accuracy, F1, AUC, MSE | Perplexity, BLEU, ROUGE, human preference |
| **Validation** | k-fold cross-validation | Held-out test sets, human evaluation |
| **Determinism** | Fully deterministic (given seed) | Non-deterministic (temperature > 0) |
| **Speed** | Milliseconds per prediction | Seconds per generation |
| **Visualization** | Confusion matrix, ROC curve, feature importance | Example outputs, rubric scores |
| **Statistical Tests** | t-test, Wilcoxon, bootstrap CI | Win rate, Elo rating, preference |
| **Interpretability** | SHAP values, feature importance | Attention visualization, probing |
| **Cost** | Near-zero (local compute) | Token-based API costs |

### Extending the Evaluator

The existing `Evaluator/` module can be extended to support ML model evaluation:

```python
# Evaluator/ml_evaluator.py
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import mlflow

class MLModelEvaluator:
    """Evaluate traditional ML models with standard metrics."""

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cross_val_scores": cross_val_score(model, X_test, y_test, cv=5).tolist(),
        }

        # Log to MLflow alongside LLM evaluation results
        mlflow.log_metrics({
            "accuracy": report["classification_report"]["accuracy"],
            "f1_weighted": report["classification_report"]["weighted avg"]["f1-score"],
            "roc_auc": report["roc_auc"],
        })

        return report
```

### Unified Reporting

Both ML and LLM evaluations should feed into a common reporting format:

```
Evaluator/results/
├── llm/                    # Existing LLM evaluation results
│   └── eval_20260312.json
├── ml/                     # NEW: ML evaluation results
│   └── quality_scorer_20260312.json
└── combined/               # NEW: Cross-paradigm comparison
    └── pipeline_report_20260312.json
```

---

## 8. Concrete Integration Ideas (Ranked by Value)

### Idea 1: Dataset Quality Scorer (Recommended First Step)

**Value**: HIGH | **Effort**: LOW | **Risk**: LOW

Train a LightGBM classifier to predict dataset quality using features from existing validation and judge results.

**Implementation**:
1. Extract features from existing JSONL datasets using `shared/validation/`
2. Use existing `shared/judge/` scores as labels
3. Train LightGBM classifier with scikit-learn pipeline
4. Integrate as pre-filter in `SynthChat/services/rubric_runner.py`
5. Track experiments with MLflow

**Expected outcome**: 100-1000x faster quality screening, reserving LLM judge for borderline cases.

**Smallest useful first step**: A single Python script that:
- Loads a JSONL dataset
- Extracts ~20 features (structural + text)
- Trains a LightGBM binary classifier (quality: good/bad)
- Reports cross-validated F1 score
- Saves the model as a joblib file

### Idea 2: Prompt Complexity Router

**Value**: HIGH | **Effort**: MEDIUM | **Risk**: LOW

Train a classifier to route prompts to appropriate models based on complexity and requirements.

**Implementation**:
1. Collect prompt-model-quality data from `Evaluator/` runs
2. Train Random Forest / LightGBM to predict best model per prompt
3. Integrate into `shared/llm/` client as routing layer

**Expected outcome**: 30-70% cost reduction on API calls while maintaining quality.

### Idea 3: Training Data Influence Scoring (NEW)

**Value**: HIGH | **Effort**: MEDIUM | **Risk**: MEDIUM

Identify which training examples contribute most to model quality using gradient-based influence estimation (TracIn) and per-example loss tracking, then train a LightGBM meta-model to predict impact scores for new data.

**Implementation**:
1. Add per-example loss logging callback to SFT/KTO training
2. Compute TracIn influence scores using saved checkpoints
3. Train LightGBM meta-model: example features + loss trajectory -> impact score
4. Use impact predictor to filter/rank future synthetic datasets

**Expected outcome**: Data-driven dataset curation -- automatically identify and remove low-impact or harmful examples, surface the highest-value training data. See **Section 23** for full research.

### Idea 4: Synthetic Data Deduplication

**Value**: MEDIUM | **Effort**: LOW | **Risk**: LOW

Use MinHash LSH or embedding clustering to identify and remove near-duplicate synthetic examples.

**Implementation**:
1. Extract TF-IDF or sentence embeddings from conversations
2. Apply HDBSCAN clustering or MinHash for duplicate detection
3. Add as preprocessing step in `Tools/`

**Expected outcome**: Cleaner training datasets, better training efficiency.

### Idea 5: Evaluation Fast-Path Classifier

**Value**: MEDIUM | **Effort**: MEDIUM | **Risk**: LOW

Train binary classifiers for deterministic evaluation criteria, using LLM-as-judge only for subjective dimensions.

**Implementation**:
1. Identify deterministic evaluation criteria (format, structure, keyword)
2. Train per-criterion classifiers on existing evaluation data
3. Integrate into `Evaluator/` alongside LLM evaluation

**Expected outcome**: 10-50x faster evaluation for structural criteria.

### Idea 6: KTO Label Balancer

**Value**: MEDIUM | **Effort**: MEDIUM | **Risk**: MEDIUM

Train a classifier to predict true/false labels for KTO training data, helping balance datasets.

**Implementation**:
1. Train on existing labeled KTO datasets
2. Use to predict labels on new synthetic data
3. Ensure proper interleaving of true/false examples

**Expected outcome**: Better balanced KTO datasets, potentially improved fine-tuning results.

### Idea 7: Training Run Anomaly Detector

**Value**: LOW-MEDIUM | **Effort**: LOW | **Risk**: LOW

Monitor training metrics with Isolation Forest to detect anomalous training behavior early.

**Implementation**:
1. Collect historical training logs from `Trainers/*/logs/`
2. Train anomaly detector on normal training patterns
3. Integrate as callback in training loops

### Idea 8: Topic Coverage Analyzer

**Value**: LOW-MEDIUM | **Effort**: LOW | **Risk**: LOW

Use clustering to visualize and analyze topic coverage in datasets, identifying gaps.

**Implementation**:
1. Embed conversations with sentence-transformers
2. Apply UMAP + HDBSCAN for topic clustering
3. Generate coverage reports showing topic distribution

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Feature engineering doesn't capture quality signals | Medium | Medium | Start with known-good features from validation; iterate based on SHAP analysis |
| ML models don't generalize to new data patterns | Medium | Medium | Use proper cross-validation; monitor drift in production |
| Dependency conflicts with existing environment | Low | High | Test in isolated conda env first; all libraries are well-tested with PyTorch |
| Over-engineering: ML where simple rules suffice | Medium | Low | Start with rule-based baseline; only add ML if it measurably improves |
| Maintenance burden of two paradigms | Medium | Medium | Use MLflow to unify tracking; keep ML models simple |

---

## 10. Compatibility Matrix

| Component | Python 3.10+ | CUDA 12.x | PyTorch 2.x | Conda | Notes |
|-----------|:------------:|:---------:|:-----------:|:-----:|-------|
| scikit-learn 1.8 | Yes | N/A | Compatible | Yes | No conflicts |
| LightGBM 4.x | Yes | Yes | Compatible | Yes | GPU optional |
| XGBoost 2.x | Yes | Yes | Compatible | Yes | GPU optional |
| CatBoost 1.x | Yes | Yes | Compatible | Yes | GPU optional |
| MLflow 3.9 | Yes | N/A | Compatible | Yes | Tracking server |
| RAPIDS cuML 26.02 | Yes | Yes | Compatible | Yes | Optional, pip install |
| Optuna 3.x | Yes | N/A | Compatible | Yes | No conflicts |
| SHAP 0.x | Yes | Yes | Compatible | Yes | GPU optional |

All libraries are compatible with the existing `toolset` conda environment.

---

## 11. Recommendations

### Incremental Adoption Path

| Phase | What | Effort | Dependencies |
|-------|------|--------|--------------|
| **Phase 0** | Install MLflow, set up experiment tracking for existing LLM training | 1-2 days | None |
| **Phase 1** | Build dataset quality scorer (LightGBM + TF-IDF) | 3-5 days | Phase 0 |
| **Phase 2** | Integrate quality scorer into SynthChat rubric runner | 2-3 days | Phase 1 |
| **Phase 3** | Build prompt router for model selection | 3-5 days | Phase 0, evaluation data |
| **Phase 4** | Build evaluation fast-path classifiers | 3-5 days | Phase 0, evaluation data |
| **Phase 5** | Add deduplication and clustering tools | 2-3 days | Phase 0 |

### Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary ML framework | LightGBM | Fastest, GPU support, scikit-learn compatible |
| Experiment tracking | MLflow | Supports both ML and LLM, open source, mature |
| Feature format | Parquet | Columnar, compressed, fast for tabular data |
| Pipeline framework | scikit-learn Pipeline | Reproducible, serializable, standard |
| Hyperparameter tuning | Optuna | Modern, efficient, MLflow integration |
| Interpretability | SHAP | Works with all tree models, GPU accelerated |

### What NOT to Do

- **Don't replace LLM evaluation entirely** -- ML classifiers complement, not replace, LLM judgment for subjective quality
- **Don't build a feature store yet** -- start with simple Parquet caches; feature stores add complexity
- **Don't migrate existing datasets** -- extract features alongside, don't change JSONL format
- **Don't add cuML/RAPIDS yet** -- standard scikit-learn + LightGBM GPU is sufficient initially
- **Don't over-engineer the CLI** -- start with direct Python scripts before adding menu integration

---

## 12. Security Considerations

| Concern | Risk | Mitigation |
|---------|------|------------|
| Model serialization (joblib/pickle) | Arbitrary code execution on load | Only load models from trusted sources; consider ONNX for sharing |
| Training data leakage | ML features may encode PII from conversations | Audit feature extractors; avoid raw text features in cached data |
| MLflow server exposure | Tracking data accessible on network | Run MLflow locally (default); add auth if shared |
| GPU memory conflicts | ML training + LLM training concurrent | Schedule separately; ML training is typically fast (minutes) |

---

---

# Part II: Standalone Traditional ML Training

The following sections cover making traditional ML a first-class training capability -- not just a helper for the LLM pipeline, but a standalone system for arbitrary prediction tasks.

---

## 13. Generic ML Training Module Architecture

### Design Philosophy

Mirror the existing LLM training pattern: **YAML config in, trained model out, timestamped run directory with logs and artifacts**.

The existing SFT config (`Trainers/rtx3090_sft/configs/config.yaml`) uses a clean hierarchy: `model:`, `lora:`, `training:`. The ML training config should follow the same pattern with ML-specific sections.

### Config-Driven Training Design

```yaml
# Trainers/ml/configs/example_classification.yaml
# ============================================================================
# ML Training Configuration
# Traditional ML classification task
# ============================================================================

# Task Configuration
task:
  type: "classification"          # classification | regression | clustering
  name: "customer_churn"          # Human-readable task name
  target_column: "churned"        # Target variable
  eval_metric: "f1_weighted"      # Primary optimization metric
  random_state: 42

# Data Configuration
data:
  train_path: "Datasets/ml/customer_churn_train.csv"
  test_path: "Datasets/ml/customer_churn_test.csv"       # Optional
  test_size: 0.2                  # Used if test_path not provided
  stratify: true                  # Stratified split for classification

# Feature Engineering
features:
  numeric:
    columns: ["age", "tenure", "monthly_charges", "total_charges"]
    imputer: "median"             # mean | median | knn | none
    scaler: "standard"            # standard | minmax | robust | none

  categorical:
    columns: ["contract_type", "payment_method", "internet_service"]
    encoder: "onehot"             # onehot | ordinal | target | none
    handle_unknown: "ignore"

  text:
    columns: ["support_notes"]
    vectorizer: "tfidf"           # tfidf | count | none
    max_features: 5000
    ngram_range: [1, 2]

  drop_columns: ["customer_id", "name"]   # Columns to exclude

# Algorithm Selection
algorithm:
  name: "lightgbm"               # lightgbm | xgboost | catboost | random_forest |
                                  # gradient_boosting | svm | logistic_regression |
                                  # neural_net | auto
  # Algorithm-specific hyperparameters
  params:
    n_estimators: 500
    learning_rate: 0.05
    num_leaves: 31
    max_depth: -1
    device: "gpu"                 # cpu | gpu

# Hyperparameter Tuning (optional)
tuning:
  enabled: true
  method: "optuna"                # optuna | grid | random
  n_trials: 100
  timeout: 3600                   # Max seconds
  search_space:                   # Override default search space
    n_estimators: [100, 1000]
    learning_rate: [0.01, 0.3]
    num_leaves: [15, 63]

# Cross-Validation
cross_validation:
  method: "stratified_kfold"      # kfold | stratified_kfold | repeated_kfold | none
  n_splits: 5
  n_repeats: 1                    # For repeated_kfold

# Evaluation
evaluation:
  metrics: ["accuracy", "f1_weighted", "roc_auc", "precision", "recall"]
  generate_plots: true            # Confusion matrix, ROC, feature importance
  shap_analysis: true             # SHAP interpretability
  shap_max_samples: 1000          # Limit for SHAP (can be slow on large datasets)

# Output Configuration
output:
  dir: "./ml_output"              # Base output directory
  save_model: "joblib"            # joblib | onnx | both
  save_pipeline: true             # Save entire pipeline (preprocessing + model)

# Experiment Tracking
tracking:
  enabled: true
  backend: "mlflow"               # mlflow | wandb | none
  experiment_name: "customer-churn"
  tags:
    project: "toolset-training"
    task_type: "classification"
```

### Algorithm Registry

The training module should support a registry of algorithms, each with default hyperparameters and search spaces:

| Algorithm | Key | Classification | Regression | GPU | Default Search Space |
|-----------|-----|:-----------:|:---------:|:---:|---------------------|
| LightGBM | `lightgbm` | Yes | Yes | Yes | n_estimators, lr, num_leaves, max_depth |
| XGBoost | `xgboost` | Yes | Yes | Yes | n_estimators, lr, max_depth, subsample |
| CatBoost | `catboost` | Yes | Yes | Yes | iterations, lr, depth, l2_leaf_reg |
| Random Forest | `random_forest` | Yes | Yes | No | n_estimators, max_depth, min_samples_split |
| Gradient Boosting | `gradient_boosting` | Yes | Yes | No | n_estimators, lr, max_depth |
| SVM | `svm` | Yes | Yes | Via cuML | C, kernel, gamma |
| Logistic Regression | `logistic_regression` | Yes | No | Via cuML | C, penalty, solver |
| Ridge/Lasso | `ridge` / `lasso` | No | Yes | Via cuML | alpha |
| Neural Net (sklearn) | `neural_net` | Yes | Yes | No | hidden_layers, lr, activation |
| K-Nearest Neighbors | `knn` | Yes | Yes | Via cuML | n_neighbors, weights, metric |
| **Auto** | `auto` | Yes | Yes | Varies | FLAML selects best |

When `algorithm.name: "auto"`, the system delegates to FLAML for automatic algorithm selection.

### How AutoGluon, FLAML, and PyCaret Handle This

These three frameworks represent different philosophies for multi-algorithm training:

**AutoGluon** (v1.5.0, AWS):
- Highest accuracy via multi-layer model ensembling (stacking)
- Presets: `best_quality`, `high_quality`, `good_quality`, `medium_quality`
- API: `TabularPredictor(label="target").fit(train_data, presets="best_quality")`
- Automatic feature type detection and preprocessing
- GPU support for neural net components
- Heaviest dependency footprint (~2GB install)

**FLAML** (v2.x, Microsoft Research):
- Budget-aware optimization -- finds strong models with minimal compute
- API: `AutoML().fit(X_train, y_train, task="classification", time_budget=60)`
- Custom learner registration: `automl.add_learner("my_model", MyEstimator)`
- Lightest dependency footprint, fastest search
- Best for resource-constrained environments (our RTX 3090 scenario)

**PyCaret** (v3.3.2):
- Low-code: `setup(data, target="label")` then `compare_models()`
- Trains and cross-validates 15+ models in one call
- Beautiful built-in reporting and visualization
- Heaviest abstraction -- hardest to customize
- Best for rapid prototyping and model comparison

**Recommendation for Toolset-Training**: Use **FLAML** as the AutoML engine.
- Lightest weight, integrates cleanly as a library (not a framework)
- Budget-aware fits our single-GPU constraint
- Custom learner API means we can register our own estimators
- scikit-learn compatible -- works within our Pipeline architecture
- Can restrict to specific algorithms: `estimator_list=["lgbm", "xgboost"]`

AutoGluon is heavier but could be offered as an optional "best quality" preset. PyCaret is too opinionated for integration into an existing CLI.

---

## 14. Dataset Ingestion for Standalone ML

### Supported Input Formats

| Format | Reader | Use Case |
|--------|--------|----------|
| **CSV** | `pd.read_csv()` | Most common tabular format |
| **Parquet** | `pd.read_parquet()` | Large datasets, columnar efficiency |
| **Excel** | `pd.read_excel()` | Business data sources |
| **JSONL** | Custom loader | Conversation data (existing format) |
| **SQLite/SQL** | `pd.read_sql()` | Database sources |

### Schema Detection and Type Inference

The data loader should automatically detect column types and suggest appropriate feature engineering:

```python
class DatasetLoader:
    """Load and analyze datasets for ML training."""

    def load(self, path: str, config: dict) -> pd.DataFrame:
        """Load dataset from any supported format."""
        ext = Path(path).suffix.lower()
        loaders = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".xlsx": pd.read_excel,
            ".jsonl": self._load_jsonl,
        }
        return loaders[ext](path)

    def infer_schema(self, df: pd.DataFrame, target: str) -> dict:
        """Auto-detect feature types and suggest preprocessing."""
        schema = {"numeric": [], "categorical": [], "text": [], "datetime": [], "drop": []}

        for col in df.columns:
            if col == target:
                continue
            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                schema["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema["datetime"].append(col)
            elif df[col].nunique() < 50 and df[col].nunique() / len(df) < 0.05:
                schema["categorical"].append(col)
            elif df[col].str.len().mean() > 100:  # Long strings = text
                schema["text"].append(col)
            else:
                schema["categorical"].append(col)

        return schema

    def suggest_task_type(self, df: pd.DataFrame, target: str) -> str:
        """Suggest classification vs regression based on target column."""
        if pd.api.types.is_numeric_dtype(df[target]):
            if df[target].nunique() <= 20:
                return "classification"
            return "regression"
        return "classification"
```

### Train/Test Splitting Strategy

```python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def split_data(df, config):
    """Split data according to config."""
    if config.get("test_path"):
        # Explicit test set provided
        train_df = df
        test_df = pd.read_csv(config["test_path"])
    else:
        # Automatic split
        stratify = df[config["target_column"]] if config.get("stratify") else None
        train_df, test_df = train_test_split(
            df,
            test_size=config.get("test_size", 0.2),
            random_state=config.get("random_state", 42),
            stratify=stratify
        )
    return train_df, test_df
```

### Coexistence with JSONL Conversation Data

The dataset ingestion layer handles both paradigms:

```
Datasets/
├── tools_datasets/           # JSONL (LLM training) -- existing
├── behavior_datasets/        # JSONL (LLM training) -- existing
├── ml_features/              # Parquet (derived features) -- existing plan
│
├── ml/                       # NEW: Standalone ML datasets
│   ├── customer_churn/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── metadata.yaml     # Dataset description, source, date
│   ├── text_classification/
│   │   ├── data.parquet
│   │   └── metadata.yaml
│   └── README.md
```

The `metadata.yaml` per dataset stores provenance:
```yaml
name: "Customer Churn"
description: "Telecom customer churn prediction dataset"
source: "Kaggle"
date_added: "2026-03-12"
rows: 7043
columns: 21
target: "churned"
task_type: "classification"
```

---

## 15. Feature Engineering Pipelines

### Config-Driven ColumnTransformer

The YAML config `features:` section drives the construction of a scikit-learn `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def build_preprocessor(feature_config: dict) -> ColumnTransformer:
    """Build a ColumnTransformer from YAML config."""
    transformers = []

    # Numeric pipeline
    if "numeric" in feature_config:
        nc = feature_config["numeric"]
        steps = []
        # Imputation
        imputer_map = {"mean": "mean", "median": "median", "knn": KNNImputer()}
        if nc.get("imputer", "none") != "none":
            imp = nc["imputer"]
            if imp == "knn":
                steps.append(("imputer", KNNImputer()))
            else:
                steps.append(("imputer", SimpleImputer(strategy=imp)))
        # Scaling
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if nc.get("scaler", "none") != "none":
            steps.append(("scaler", scaler_map[nc["scaler"]]))

        if steps:
            transformers.append(("numeric", Pipeline(steps), nc["columns"]))

    # Categorical pipeline
    if "categorical" in feature_config:
        cc = feature_config["categorical"]
        encoder_map = {
            "onehot": OneHotEncoder(handle_unknown=cc.get("handle_unknown", "ignore"),
                                     sparse_output=False),
            "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1),
        }
        encoder = encoder_map.get(cc.get("encoder", "onehot"))
        if encoder:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", encoder),
            ]), cc["columns"]))

    # Text pipeline
    if "text" in feature_config:
        tc = feature_config["text"]
        vectorizer_map = {
            "tfidf": TfidfVectorizer(
                max_features=tc.get("max_features", 5000),
                ngram_range=tuple(tc.get("ngram_range", [1, 1])),
            ),
            "count": CountVectorizer(
                max_features=tc.get("max_features", 5000),
                ngram_range=tuple(tc.get("ngram_range", [1, 1])),
            ),
        }
        for col in tc["columns"]:
            vec = vectorizer_map.get(tc.get("vectorizer", "tfidf"))
            transformers.append((f"text_{col}", vec, col))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",          # Drop columns not specified
        n_jobs=-1,
    )
```

### Reusable Feature Transformer Patterns

Common patterns that users can reference in config:

| Pattern | Config Key | What It Does |
|---------|-----------|--------------|
| **Numeric Standard** | `imputer: median, scaler: standard` | Median imputation + z-score normalization |
| **Numeric Robust** | `imputer: median, scaler: robust` | Handles outliers with IQR-based scaling |
| **Categorical OneHot** | `encoder: onehot` | Sparse binary features, handles unknowns |
| **Categorical Ordinal** | `encoder: ordinal` | Integer encoding for tree models (preferred) |
| **Text TF-IDF** | `vectorizer: tfidf, max_features: 5000` | N-gram frequency features |
| **Text Count** | `vectorizer: count` | Raw term frequencies |

### Advanced: Custom Feature Transformers

For project-specific features, users can register custom transformers:

```python
# Trainers/ml/features/custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin

class ConversationFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract features from JSONL conversation columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is a pandas Series of conversation JSON strings
        features = []
        for conv_json in X:
            conv = json.loads(conv_json)
            features.append({
                "num_turns": len(conv),
                "avg_length": np.mean([len(c["content"]) for c in conv]),
                "has_tool_call": any("tool_call:" in c["content"] for c in conv),
            })
        return pd.DataFrame(features)
```

---

## 16. AutoML Capabilities

### Comparison: Wrap vs Build

| Approach | Pros | Cons |
|----------|------|------|
| **Wrap FLAML** | Lightweight, budget-aware, scikit-learn compatible, extensible | Less powerful ensembling than AutoGluon |
| **Wrap AutoGluon** | Best accuracy, multi-layer stacking | Heavy (~2GB), opinionated, harder to integrate |
| **Wrap PyCaret** | Beautiful reporting, many models | Too opinionated, fights our CLI design |
| **Build custom** | Full control, minimal dependencies | Reinventing the wheel, maintenance burden |

### Recommended: FLAML as Primary, AutoGluon as Optional

**FLAML integration** (default `algorithm.name: "auto"`):

```python
from flaml import AutoML

def train_auto(X_train, y_train, config):
    """AutoML training via FLAML."""
    automl = AutoML()

    flaml_settings = {
        "task": config["task"]["type"],
        "metric": config["task"]["eval_metric"],
        "time_budget": config.get("tuning", {}).get("timeout", 600),
        "estimator_list": config.get("tuning", {}).get("estimator_list",
            ["lgbm", "xgboost", "catboost", "rf", "extra_tree"]),
        "eval_method": "cv",
        "n_splits": config.get("cross_validation", {}).get("n_splits", 5),
        "seed": config["task"].get("random_state", 42),
        "verbose": 1,
    }

    automl.fit(X_train, y_train, **flaml_settings)

    return automl  # .model is the best model, .best_config is its config
```

**AutoGluon integration** (optional preset `algorithm.name: "autogluon"`):

```python
from autogluon.tabular import TabularPredictor

def train_autogluon(train_df, config):
    """AutoML training via AutoGluon (optional, heavier)."""
    preset_map = {
        "fast": "medium_quality",
        "balanced": "good_quality",
        "best": "best_quality",
    }
    preset = preset_map.get(config.get("autogluon_preset", "balanced"))

    predictor = TabularPredictor(
        label=config["task"]["target_column"],
        path=config["output"]["dir"],
        eval_metric=config["task"]["eval_metric"],
    )
    predictor.fit(
        train_data=train_df,
        presets=preset,
        time_limit=config.get("tuning", {}).get("timeout", 3600),
        num_gpus=1 if config.get("algorithm", {}).get("params", {}).get("device") == "gpu" else 0,
    )
    return predictor
```

### Config for AutoML Mode

```yaml
# Auto mode with FLAML (lightweight, fast)
algorithm:
  name: "auto"
  params:
    time_budget: 600              # Max 10 minutes
    estimator_list: ["lgbm", "xgboost", "rf"]

# Or AutoGluon mode (heavier, highest accuracy)
algorithm:
  name: "autogluon"
  params:
    preset: "balanced"            # fast | balanced | best
    time_limit: 3600
```

### Additional AutoML Dependencies

| Library | Install | Size | When |
|---------|---------|------|------|
| FLAML | `pip install flaml[automl]` | ~50MB | Always (lightweight default) |
| AutoGluon | `pip install autogluon.tabular` | ~2GB | Optional (best accuracy) |

---

## 17. CLI Integration Design

### Updated Menu Structure

```
Synaptic Tuner - Main Menu
==========================
  [1] Train LLM (SFT/KTO/GRPO)
  [2] Upload to HuggingFace
  [3] Evaluate Model
  [4] Generate Synthetic Data
  [5] Improvement Engine
  [6] Train ML Model              # NEW
  [7] Evaluate ML Model           # NEW
  [8] MLflow Dashboard            # NEW
  [9] System Status

Train ML Model Submenu:
  [1] Train from config           # Uses YAML config file
  [2] Quick train (interactive)   # Guided setup, generates config
  [3] AutoML (FLAML)              # Auto algorithm selection
  [4] AutoML (AutoGluon)          # Best accuracy (if installed)
  [5] List available configs
  [6] List available datasets

Evaluate ML Model Submenu:
  [1] Evaluate model on test set
  [2] Cross-validate model
  [3] Compare models
  [4] Generate SHAP report
  [5] List trained models
```

### CLI Command Interface

```bash
# Config-driven training
./run.sh ml train --config Trainers/ml/configs/customer_churn.yaml

# Quick train with defaults
./run.sh ml train --data Datasets/ml/data.csv --target churned

# AutoML
./run.sh ml train --data Datasets/ml/data.csv --target churned --auto

# Evaluate
./run.sh ml eval --model ml_output/20260312_143000/model.joblib --data test.csv

# Compare models
./run.sh ml compare --experiment customer-churn

# List resources
./run.sh ml list models
./run.sh ml list configs
./run.sh ml list datasets

# Launch MLflow UI
./run.sh mlflow
```

### Direct Python Interface

```bash
# Via tuner.py
python tuner.py ml train --config configs/customer_churn.yaml

# Via module
python -m Trainers.ml.train --config configs/customer_churn.yaml

# Interactive
python tuner.py ml
```

### Config File Conventions

```
Trainers/ml/
├── configs/
│   ├── templates/             # Starter templates
│   │   ├── classification.yaml
│   │   ├── regression.yaml
│   │   ├── text_classification.yaml
│   │   └── automl.yaml
│   └── <user_configs>.yaml    # User-created configs
```

---

## 18. Model Output and Deployment

### Serialization Formats

| Format | Library | Size | Portability | Speed | Use Case |
|--------|---------|------|-------------|-------|----------|
| **Joblib** | joblib | Small | Python only | Fast | Default, development |
| **Pickle** | pickle | Small | Python only | Fast | Legacy compatibility |
| **ONNX** | sklearn-onnx, onnxmltools | Medium | Cross-platform | Fastest inference | Production deployment |
| **MLflow Model** | mlflow | Medium | MLflow ecosystem | Fast | Model registry |

### ONNX Export Pipeline

sklearn-onnx (v1.20.0) supports converting entire scikit-learn pipelines including LightGBM/XGBoost:

```python
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Convert pipeline to ONNX
onnx_model = to_onnx(pipeline, X_train[:1].values,
                      target_opset={"": 17, "ai.onnx.ml": 3})

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Inference with ONNX Runtime (10-100x faster than sklearn predict)
session = rt.InferenceSession("model.onnx")
pred = session.run(None, {"X": X_test.values.astype(np.float32)})
```

### MLflow Model Registry

```python
import mlflow.sklearn

# Log model to registry
with mlflow.start_run():
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="customer-churn-classifier",
        input_example=X_train[:5],
        signature=mlflow.models.infer_signature(X_train, y_pred),
    )

# Load from registry
model = mlflow.sklearn.load_model("models:/customer-churn-classifier/Production")
```

### HuggingFace Model Cards for sklearn Models

While HuggingFace Hub is primarily for deep learning models, sklearn models can be shared via the `skops` library:

```python
from skops import hub_utils, card

# Create model card
model_card = card.Card(pipeline)
model_card.metadata.license = "mit"
model_card.metadata.library_name = "scikit-learn"

# Push to HuggingFace
hub_utils.push(
    repo_id="username/customer-churn-lgbm",
    source="ml_output/20260312_143000/",
    card=model_card,
)
```

### Output Directory Structure

```
ml_output/
├── 20260312_143000/              # Timestamped run directory
│   ├── config.yaml               # Frozen copy of training config
│   ├── model.joblib              # Serialized model/pipeline
│   ├── model.onnx                # ONNX export (if configured)
│   ├── preprocessor.joblib       # Fitted preprocessor (ColumnTransformer)
│   ├── metrics.json              # All evaluation metrics
│   ├── schema.json               # Input schema (column names, types)
│   │
│   ├── plots/                    # Generated visualizations
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── feature_importance.png
│   │   ├── shap_summary.png
│   │   └── learning_curve.png
│   │
│   ├── logs/                     # Training logs
│   │   ├── training.log          # Verbose training log
│   │   └── tuning_history.json   # Hyperparameter search history
│   │
│   └── cross_validation/         # CV results
│       ├── fold_metrics.json     # Per-fold metrics
│       └── cv_summary.json       # Aggregated CV results
```

This mirrors the LLM training output pattern:

| LLM Training | ML Training | Purpose |
|--------------|-------------|---------|
| `sft_output_rtx3090/YYYYMMDD_HHMMSS/` | `ml_output/YYYYMMDD_HHMMSS/` | Timestamped run dir |
| `final_model/` | `model.joblib` + `model.onnx` | Trained model |
| `checkpoints/` | `tuning_history.json` | Intermediate state |
| `logs/training_latest.jsonl` | `logs/training.log` | Training logs |
| Config dataclass | `config.yaml` (frozen copy) | Reproducibility |

---

## 19. Evaluation for Standalone ML

### Comprehensive Metrics by Task Type

**Classification Metrics**:

| Metric | Function | When to Use |
|--------|----------|-------------|
| Accuracy | `accuracy_score` | Balanced classes |
| F1 (weighted) | `f1_score(average="weighted")` | Imbalanced classes (default) |
| F1 (macro) | `f1_score(average="macro")` | Equal weight per class |
| ROC AUC | `roc_auc_score` | Binary or multiclass (OvR) |
| Precision | `precision_score` | When false positives are costly |
| Recall | `recall_score` | When false negatives are costly |
| Log Loss | `log_loss` | Probabilistic predictions |
| Matthews Correlation | `matthews_corrcoef` | Severe class imbalance |

**Regression Metrics**:

| Metric | Function | When to Use |
|--------|----------|-------------|
| RMSE | `mean_squared_error(squared=False)` | Default, same units as target |
| MAE | `mean_absolute_error` | Robust to outliers |
| R-squared | `r2_score` | Proportion of variance explained |
| MAPE | `mean_absolute_percentage_error` | Percentage-based interpretation |

**Clustering Metrics**:

| Metric | Function | When to Use |
|--------|----------|-------------|
| Silhouette | `silhouette_score` | No ground truth available |
| Adjusted Rand | `adjusted_rand_score` | When ground truth exists |
| Calinski-Harabasz | `calinski_harabasz_score` | Cluster separation quality |

### Cross-Validation Pipeline

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

def run_cross_validation(pipeline, X, y, config):
    """Run cross-validation with multiple metrics."""
    cv = StratifiedKFold(
        n_splits=config.get("n_splits", 5),
        shuffle=True,
        random_state=config.get("random_state", 42),
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "roc_auc": "roc_auc_ovr_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
    }

    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # Summarize
    summary = {}
    for metric in scoring:
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        summary[metric] = {
            "mean": results[test_key].mean(),
            "std": results[test_key].std(),
            "per_fold": results[test_key].tolist(),
            "train_mean": results[train_key].mean(),
        }
    return summary
```

### SHAP Interpretability

```python
import shap

def generate_shap_report(model, X_test, output_dir):
    """Generate SHAP analysis plots and data."""
    # Tree-based models get fast TreeExplainer
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_test[:100])

    shap_values = explainer(X_test[:1000])  # Limit for speed

    # Summary plot (global feature importance)
    shap.summary_plot(shap_values, X_test[:1000], show=False)
    plt.savefig(f"{output_dir}/plots/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean absolute SHAP values)
    shap.plots.bar(shap_values, show=False)
    plt.savefig(f"{output_dir}/plots/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save SHAP values for downstream analysis
    return {
        "top_features": list(zip(
            X_test.columns,
            np.abs(shap_values.values).mean(axis=0)
        ))[:20]
    }
```

### Automated Report Generation

Each training run produces a JSON report and optional HTML summary:

```python
def generate_evaluation_report(metrics, cv_results, shap_data, config, output_dir):
    """Generate comprehensive evaluation report."""
    report = {
        "task": config["task"],
        "algorithm": config["algorithm"]["name"],
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(X_train.columns),
        },
        "test_metrics": metrics,
        "cross_validation": cv_results,
        "feature_importance": shap_data.get("top_features", []),
        "plots": [f for f in os.listdir(f"{output_dir}/plots/")],
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
```

---

## 20. Concrete End-to-End Examples

### Example 1: Tabular Classification -- Customer Churn

**Scenario**: Predict customer churn from structured data (demographics, usage, billing).

**Config** (`configs/customer_churn.yaml`):
```yaml
task:
  type: classification
  name: customer_churn
  target_column: churned
  eval_metric: f1_weighted

data:
  train_path: Datasets/ml/customer_churn.csv
  test_size: 0.2
  stratify: true

features:
  numeric:
    columns: [tenure, monthly_charges, total_charges]
    imputer: median
    scaler: standard
  categorical:
    columns: [contract_type, payment_method, internet_service]
    encoder: onehot

algorithm:
  name: lightgbm
  params:
    n_estimators: 500
    learning_rate: 0.05
    device: gpu

tuning:
  enabled: true
  method: optuna
  n_trials: 50

evaluation:
  metrics: [accuracy, f1_weighted, roc_auc]
  generate_plots: true
  shap_analysis: true

output:
  dir: ./ml_output
  save_model: both  # joblib + onnx
```

**Run**:
```bash
./run.sh ml train --config Trainers/ml/configs/customer_churn.yaml
```

**Expected output**:
```
ML Training: customer_churn
=============================
Algorithm: LightGBM (GPU)
Dataset: 7043 rows, 18 features
Hyperparameter tuning: 50 Optuna trials

[Tuning] Best trial: F1=0.847 (n_estimators=342, lr=0.078, leaves=24)
[CV] 5-fold: F1=0.843 +/- 0.012
[Test] Accuracy=0.891, F1=0.847, ROC-AUC=0.923

Artifacts saved to: ml_output/20260312_143000/
  model.joblib (1.2 MB)
  model.onnx (0.8 MB)
  plots/confusion_matrix.png
  plots/roc_curve.png
  plots/shap_summary.png
  metrics.json

MLflow run: http://localhost:5000/#/experiments/3/runs/abc123
```

### Example 2: Text Classification -- Dataset Quality Scoring

**Scenario**: Classify synthetic training examples as high/low quality using text features from conversations. This bridges standalone ML with the pipeline-integrated use case.

**Config** (`configs/quality_scorer.yaml`):
```yaml
task:
  type: classification
  name: dataset_quality_scorer
  target_column: quality_label
  eval_metric: f1_weighted

data:
  train_path: Datasets/ml_features/quality_labeled.parquet
  test_size: 0.2
  stratify: true

features:
  numeric:
    columns: [num_turns, avg_turn_length, vocabulary_richness,
              schema_error_count, tool_call_count]
    imputer: median
    scaler: standard
  text:
    columns: [full_text]
    vectorizer: tfidf
    max_features: 5000
    ngram_range: [1, 2]

algorithm:
  name: auto             # Let FLAML choose
  params:
    time_budget: 300     # 5 minutes

evaluation:
  metrics: [accuracy, f1_weighted, roc_auc, precision, recall]
  generate_plots: true
  shap_analysis: true

output:
  dir: ./ml_output
  save_model: joblib
```

**Run**:
```bash
./run.sh ml train --config Trainers/ml/configs/quality_scorer.yaml
```

### Example 3: Regression -- Training Loss Prediction

**Scenario**: Predict expected final training loss based on hyperparameter configuration and early training metrics (for run optimization).

**Config** (`configs/loss_predictor.yaml`):
```yaml
task:
  type: regression
  name: training_loss_prediction
  target_column: final_loss
  eval_metric: rmse

data:
  train_path: Datasets/ml/training_runs_history.csv
  test_size: 0.2

features:
  numeric:
    columns: [learning_rate, batch_size, num_epochs, lora_rank,
              loss_epoch_1, loss_epoch_2, gradient_norm_avg]
    imputer: mean
    scaler: robust
  categorical:
    columns: [model_family, optimizer, scheduler]
    encoder: ordinal

algorithm:
  name: xgboost
  params:
    n_estimators: 300
    max_depth: 6
    device: gpu

evaluation:
  metrics: [rmse, mae, r2]
  generate_plots: true
  shap_analysis: true

output:
  dir: ./ml_output
  save_model: joblib
```

---

## 21. Updated Architecture (Dual-Purpose Vision)

### Revised Directory Structure

```
Toolset-Training/
├── Trainers/
│   ├── rtx3090_sft/              # LLM: Supervised Fine-Tuning
│   ├── rtx3090_kto/              # LLM: KTO Training
│   ├── rtx3090_grpo/             # LLM: GRPO Training
│   │
│   ├── ml/                       # NEW: Traditional ML Training
│   │   ├── train.py              # Main training entry point
│   │   ├── train_auto.py         # AutoML entry point (FLAML/AutoGluon)
│   │   ├── evaluate.py           # Model evaluation
│   │   ├── predict.py            # Inference on new data
│   │   │
│   │   ├── configs/              # YAML training configs
│   │   │   ├── templates/        # Starter templates
│   │   │   │   ├── classification.yaml
│   │   │   │   ├── regression.yaml
│   │   │   │   ├── text_classification.yaml
│   │   │   │   └── automl.yaml
│   │   │   └── <user_configs>.yaml
│   │   │
│   │   ├── algorithms/           # Algorithm registry and wrappers
│   │   │   ├── registry.py       # Algorithm name -> class mapping
│   │   │   ├── lightgbm_wrapper.py
│   │   │   ├── xgboost_wrapper.py
│   │   │   └── auto_wrapper.py   # FLAML/AutoGluon integration
│   │   │
│   │   ├── features/             # Feature engineering
│   │   │   ├── builder.py        # Config -> ColumnTransformer
│   │   │   ├── custom_transformers.py
│   │   │   └── text_features.py
│   │   │
│   │   ├── evaluation/           # Evaluation and reporting
│   │   │   ├── metrics.py        # Metric computation
│   │   │   ├── plots.py          # Visualization generation
│   │   │   ├── shap_report.py    # SHAP analysis
│   │   │   └── report.py         # Report generation
│   │   │
│   │   └── data/                 # Data loading and splitting
│   │       ├── loader.py         # Multi-format data loading
│   │       ├── schema.py         # Type inference, schema detection
│   │       └── splitter.py       # Train/test splitting
│   │
│   └── shared/                   # Existing shared training code
│
├── shared/
│   ├── llm/                      # Existing
│   ├── validation/               # Existing
│   ├── judge/                    # Existing
│   │
│   ├── ml/                       # NEW: Shared ML utilities
│   │   ├── features.py           # Common feature extractors
│   │   ├── metrics.py            # Unified metric helpers
│   │   └── data_loaders.py       # JSONL -> tabular conversion
│   │
│   └── experiment_tracking/      # NEW: MLflow integration
│       ├── mlflow_config.py
│       └── tracking.py
│
├── Evaluator/
│   ├── (existing LLM evaluation) # Existing
│   └── ml_evaluator.py           # NEW: ML model evaluation
│
├── Datasets/
│   ├── tools_datasets/           # JSONL (LLM training)
│   ├── behavior_datasets/        # JSONL (LLM training)
│   ├── ml_features/              # Derived features (Parquet)
│   └── ml/                       # NEW: Standalone ML datasets
│       ├── README.md
│       └── <dataset_name>/
│           ├── data.csv
│           └── metadata.yaml
│
└── tuner.py                      # Extended CLI with ML subcommand
```

---

## 22. Updated Recommendations (Dual-Purpose Vision)

### Revised Incremental Adoption Path

| Phase | What | Effort | Track |
|-------|------|--------|-------|
| **Phase 0** | Install MLflow + core ML deps, set up experiment tracking | 1-2 days | Foundation |
| **Phase 1a** | Build `Trainers/ml/` skeleton: config loader, data loader, single algorithm (LightGBM) | 3-5 days | Standalone ML |
| **Phase 1b** | Build dataset quality scorer (pipeline-integrated use case) | 3-5 days | Pipeline ML |
| **Phase 2a** | Add algorithm registry (XGBoost, CatBoost, RF, SVM), feature engineering | 3-5 days | Standalone ML |
| **Phase 2b** | Integrate quality scorer into SynthChat rubric runner | 2-3 days | Pipeline ML |
| **Phase 3** | Add AutoML (FLAML), evaluation reporting, SHAP | 3-5 days | Standalone ML |
| **Phase 4** | CLI integration (`./run.sh ml train`, etc.) | 2-3 days | Both |
| **Phase 5** | Add ONNX export, model registry, prompt router | 3-5 days | Both |
| **Phase 6** | Add templates, docs, example datasets | 2-3 days | Both |

### Revised Technology Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary ML framework | scikit-learn + LightGBM | Pipeline architecture + fast training |
| AutoML engine | FLAML (primary), AutoGluon (optional) | Lightweight, budget-aware, extensible |
| Experiment tracking | MLflow | Unified ML + LLM tracking |
| Config format | YAML | Matches existing SFT/KTO config pattern |
| Feature format | Parquet | Columnar, fast, compressed |
| Model serialization | Joblib (default) + ONNX (production) | Development + deployment |
| Hyperparameter tuning | Optuna (manual) + FLAML (auto) | Flexibility |
| Interpretability | SHAP | Universal, GPU-accelerated |

### Revised What NOT to Do

- **Don't build a custom ML framework** -- wrap existing libraries (scikit-learn, FLAML)
- **Don't support every algorithm from day one** -- start with LightGBM, add incrementally
- **Don't build a web UI** -- CLI-first, MLflow UI covers visualization needs
- **Don't build a feature store** -- Parquet caches are sufficient initially
- **Don't try to unify LLM and ML configs** -- they're fundamentally different; separate YAML schemas with shared output conventions
- **Don't add AutoGluon to default dependencies** -- keep it optional due to 2GB footprint

### Updated Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep: trying to build a full ML platform | High | High | Strict phased approach; each phase delivers standalone value |
| Config complexity: YAML becomes unwieldy | Medium | Medium | Good defaults; templates; `--auto` for zero-config |
| AutoML overhead: FLAML/AutoGluon too heavy | Low | Medium | FLAML is lightweight; AutoGluon is optional |
| User confusion: two training paradigms | Medium | Medium | Clear CLI separation; good documentation |
| Maintenance burden | Medium | Medium | Leverage scikit-learn ecosystem; minimal custom code |

---

## 23. Training Data Influence and Attribution

### 23.1 Overview

Training Data Attribution (TDA) answers a fundamental question: **which training examples had the most (or least) impact on model quality?** For LLM fine-tuning, this translates to: which QA pairs in our SFT/KTO datasets are actually helping, which are neutral, and which are harming the model?

This is a high-value capability because:
- It enables **data-driven dataset curation** rather than heuristic-based filtering
- It creates a **feedback loop** between training outcomes and dataset quality
- It bridges **standalone ML and pipeline enhancement** -- the meta-model approach uses traditional ML (LightGBM) to predict training data value

### 23.2 Established Methods

| Method | Mechanism | Computational Cost | Accuracy | Practical for Single GPU? |
|--------|-----------|-------------------|----------|--------------------------|
| **Leave-One-Out (LOO)** | Retrain N times, each omitting one example | O(N * training_cost) | Gold standard | No (prohibitive for LLM) |
| **Influence Functions** | Second-order gradient approximation (Hessian) | O(N * d) per test point | Good for convex; noisy for deep nets | Marginal (Hessian expensive for LLMs) |
| **TracIn** | First-order gradient dot-products at checkpoints | O(N * C * d) where C=checkpoints | Good; scales well | **Yes** (recommended) |
| **Data Shapley** | Shapley values over data subsets | O(2^N) exact; Monte Carlo approximation | Theoretically optimal | No (too many retrainings) |
| **In-Run Data Shapley** (ICLR 2025) | Shapley per gradient update, accumulated | O(1) additional per step | Promising; new | Potentially (needs validation) |
| **Datamodels** | Train many models on random subsets; regress | O(K * training_cost), K~hundreds | Very good with enough models | No (hundreds of retrainings) |
| **Loss Trajectory Analysis** | Track per-example loss across training | O(N) per checkpoint evaluation | Proxy; fast | **Yes** (recommended) |
| **DDA (2024)** | Enhanced influence functions with fitting error correction | Similar to influence functions | Better than standard IF for LLMs | Marginal |

#### Method Details

**Leave-One-Out (LOO)**:
The gold standard -- retrain the model N times, each time omitting one example, and measure the change in evaluation metric. For a dataset of 2,676 examples (our SFT dataset), this would require 2,676 full training runs. Completely impractical for LLM fine-tuning.

**Influence Functions** (Koh & Liang, 2017):
Approximate the effect of removing a training point using the inverse Hessian-vector product. For LLMs, the Hessian is intractable without approximations (e.g., Kronecker factoring, LiSSA). Recent work (2025) questions whether influence functions work reliably on LLMs due to non-convexity and fitting errors. For LoRA fine-tuning, the parameter space is smaller (rank * layers), which helps but doesn't fully solve the issue.

**TracIn** (Pruthi et al., NeurIPS 2020):
Traces the influence of each training example by computing dot products of loss gradients between training and test examples at saved checkpoints. The key insight: influence is approximated as:

```
influence(z_train, z_test) = Σ_t η_t * ∇L(z_train, θ_t) · ∇L(z_test, θ_t)
```

Where η_t is the learning rate and θ_t are model parameters at checkpoint t. This is computationally feasible because:
- Uses first-order gradients only (no Hessian)
- Leverages existing training checkpoints
- Can be restricted to specific layers (e.g., LoRA layers only)

**Data Shapley** (Ghorbani & Zou, 2019):
Assigns each training point a value based on its marginal contribution to all possible subsets. Theoretically elegant but requires exponential retraining. Monte Carlo approximation helps but still needs hundreds of model retrainings.

**In-Run Data Shapley** (ICLR 2025):
Major breakthrough -- computes Shapley values during a single training run by calculating per-gradient-update Shapley values and accumulating them. Adds negligible runtime overhead. However, this is very new and needs validation for LoRA fine-tuning scenarios.

**Loss Trajectory Analysis**:
The simplest and most practical approach. Track the loss on each training example across training steps. Examples that show rapid loss decrease are being "learned"; examples whose loss stays high may be too hard, mislabeled, or conflicting with other data. No additional computation beyond evaluation at checkpoints.

### 23.3 What's Practical for LoRA Fine-Tuning on RTX 3090

Given our constraints (single RTX 3090, 24GB VRAM, LoRA fine-tuning with Unsloth/TRL), here's what's feasible:

| Approach | Feasibility | Additional Cost | Recommended? |
|----------|-------------|-----------------|-------------|
| **Per-example loss logging** | Easy | ~5% training time (extra forward pass at checkpoints) | **Yes -- do this first** |
| **TracIn (LoRA layers only)** | Moderate | ~20-30% additional compute post-training | **Yes -- primary attribution method** |
| **In-Run Data Shapley** | Experimental | Negligible if implemented correctly | Worth investigating |
| **Influence Functions** | Difficult | Hessian approximation complex for LLMs | Not recommended initially |
| **Data Shapley (MC)** | Impractical | 100+ retrainings | No |
| **LOO** | Impractical | N retrainings | No |
| **Datamodels** | Impractical | Hundreds of retrainings | No |

**Recommended two-step approach**:
1. **Step 1 (Easy)**: Per-example loss tracking during training -- add a callback to log loss for each example at each checkpoint
2. **Step 2 (Moderate)**: TracIn on LoRA parameters only -- compute gradient dot-products using saved checkpoints, restricted to LoRA adapter weights (much smaller parameter space than full model)

### 23.4 Implementation Approaches

#### Per-Example Loss Logging

Add a custom callback to the HuggingFace/Unsloth SFT Trainer:

```python
from transformers import TrainerCallback
import torch
import json

class PerExampleLossCallback(TrainerCallback):
    """Log per-example loss at each checkpoint."""

    def __init__(self, train_dataset, output_dir, eval_interval=100):
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.eval_interval = eval_interval
        self.loss_history = {}  # {example_idx: [loss_at_step_0, loss_at_step_100, ...]}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_interval != 0:
            return

        model.eval()
        step_losses = {}

        with torch.no_grad():
            for idx in range(len(self.train_dataset)):
                example = self.train_dataset[idx]
                inputs = {k: v.unsqueeze(0).to(model.device)
                          for k, v in example.items()
                          if isinstance(v, torch.Tensor)}
                outputs = model(**inputs)
                loss = outputs.loss.item()
                step_losses[idx] = loss

                if idx not in self.loss_history:
                    self.loss_history[idx] = []
                self.loss_history[idx].append({
                    "step": state.global_step,
                    "loss": loss,
                })

        model.train()

        # Save incrementally
        with open(f"{self.output_dir}/per_example_loss_step{state.global_step}.json", "w") as f:
            json.dump(step_losses, f)

    def on_train_end(self, args, state, control, **kwargs):
        """Save complete loss trajectories."""
        with open(f"{self.output_dir}/loss_trajectories.json", "w") as f:
            json.dump(self.loss_history, f, indent=2)
```

**Integration with existing SFT training**:
```python
# In Trainers/rtx3090_sft/train_sft.py
loss_callback = PerExampleLossCallback(
    train_dataset=tokenized_dataset,
    output_dir=run_dir / "influence",
    eval_interval=100,  # Log every 100 steps
)
trainer = SFTTrainer(
    model=model,
    # ... existing config ...
    callbacks=[loss_callback],
)
```

**Note on efficiency**: For our dataset size (~2,676 SFT examples), a full forward pass over all examples takes ~30-60 seconds on RTX 3090. At 100-step intervals over ~500 total steps, this adds ~2.5-5 minutes to training time (~5-10% overhead).

#### TracIn on LoRA Parameters

```python
import torch
from pathlib import Path

class LoRATracIn:
    """Compute TracIn influence scores using LoRA adapter gradients only."""

    def __init__(self, model, checkpoint_dirs, device="cuda"):
        self.model = model
        self.checkpoint_dirs = checkpoint_dirs
        self.device = device

    def _get_lora_gradients(self, model, example):
        """Compute gradients w.r.t. LoRA parameters only."""
        model.zero_grad()
        inputs = {k: v.unsqueeze(0).to(self.device)
                  for k, v in example.items()
                  if isinstance(v, torch.Tensor)}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        # Collect gradients from LoRA layers only
        grads = []
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.grad is not None:
                grads.append(param.grad.detach().flatten())

        return torch.cat(grads), loss.item()

    def compute_influence(self, train_dataset, test_examples, learning_rates):
        """
        Compute TracIn influence scores.

        influence(z_train, z_test) = Σ_t η_t * ∇L(z_train, θ_t) · ∇L(z_test, θ_t)
        """
        influence_scores = torch.zeros(len(train_dataset), len(test_examples))

        for ckpt_idx, ckpt_dir in enumerate(self.checkpoint_dirs):
            lr = learning_rates[ckpt_idx]

            # Load checkpoint
            self.model.load_adapter(ckpt_dir)
            self.model.to(self.device)

            # Compute test gradients (once per checkpoint)
            test_grads = []
            for test_ex in test_examples:
                grad, _ = self._get_lora_gradients(self.model, test_ex)
                test_grads.append(grad)

            # Compute train gradients and dot products
            for train_idx in range(len(train_dataset)):
                train_grad, _ = self._get_lora_gradients(
                    self.model, train_dataset[train_idx]
                )
                for test_idx, test_grad in enumerate(test_grads):
                    influence_scores[train_idx, test_idx] += (
                        lr * torch.dot(train_grad, test_grad).item()
                    )

        return influence_scores
```

**LoRA advantage**: LoRA adapters typically have 1-10M parameters vs 1-8B for the full model. This makes gradient computation ~100-1000x cheaper. For rank=16 targeting 8 projection matrices, gradient dimensionality is ~16 * 2 * 8 * hidden_dim, which is very manageable.

#### Using Captum's TracInCP

Meta's Captum library (v0.7.0) provides production-grade TracIn implementations:

```python
from captum.influence import TracInCP, TracInCPFast

# TracInCP: Full gradient computation (most accurate)
tracin = TracInCP(
    model=model,
    train_dataset=train_dataset,
    checkpoints=checkpoint_paths,      # List of saved checkpoint paths
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
    batch_size=32,
)

# Compute influence of all training examples on specific test examples
influence_scores = tracin.influence(
    inputs=test_inputs,
    targets=test_targets,
    top_k=50,  # Return top 50 most influential
)

# TracInCPFast: Uses only last layer gradients (faster, slightly less accurate)
tracin_fast = TracInCPFast(
    model=model,
    final_fc_layer=model.lm_head,      # Last projection layer
    train_dataset=train_dataset,
    checkpoints=checkpoint_paths,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
    batch_size=64,
)
```

### 23.5 The Meta-Model Approach

The most powerful integration: train a traditional ML model (LightGBM) to predict training data impact from easily-computed features, enabling instant impact scoring of new examples without retraining.

#### Feature Engineering for Impact Prediction

Combine structural features (already defined in Section 2.1) with loss trajectory features:

```python
def extract_influence_features(example, loss_trajectory):
    """Extract features for the impact meta-model."""
    features = {}

    # === Loss Trajectory Features ===
    losses = [entry["loss"] for entry in loss_trajectory]

    features["initial_loss"] = losses[0]                    # Loss before training
    features["final_loss"] = losses[-1]                     # Loss after training
    features["loss_delta"] = losses[0] - losses[-1]         # Total loss reduction
    features["loss_reduction_pct"] = features["loss_delta"] / (losses[0] + 1e-8)
    features["loss_at_25pct"] = losses[len(losses) // 4]    # Loss at 25% training
    features["loss_at_50pct"] = losses[len(losses) // 2]    # Loss at 50% training
    features["loss_at_75pct"] = losses[3 * len(losses) // 4]
    features["early_loss_slope"] = (losses[0] - losses[len(losses) // 4]) / (len(losses) // 4)
    features["late_loss_slope"] = (losses[len(losses) // 2] - losses[-1]) / (len(losses) // 2)
    features["loss_variance"] = np.var(losses)              # Stability of learning
    features["loss_monotonic"] = int(all(a >= b for a, b in zip(losses, losses[1:])))

    # === Structural Features (from Section 2.1) ===
    conv = example["conversations"]
    features["num_turns"] = len(conv)
    features["total_chars"] = sum(len(c["content"]) for c in conv)
    features["avg_turn_length"] = features["total_chars"] / features["num_turns"]
    features["has_tool_call"] = int(any("tool_call:" in c["content"] for c in conv))
    features["num_tool_calls"] = sum(c["content"].count("tool_call:") for c in conv)

    # === Text Features ===
    full_text = " ".join(c["content"] for c in conv)
    words = full_text.split()
    features["word_count"] = len(words)
    features["unique_word_ratio"] = len(set(words)) / (len(words) + 1)
    features["avg_word_length"] = np.mean([len(w) for w in words])

    # === Validation Features (from shared/validation/) ===
    features["schema_errors"] = count_schema_errors(example)
    features["xml_valid"] = int(validate_xml(example))
    features["json_valid"] = int(validate_json(example))

    return features
```

#### Training the Impact Meta-Model

```python
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import mlflow

def train_impact_predictor(features_df, influence_scores):
    """
    Train LightGBM to predict training data impact.

    features_df: DataFrame with per-example features
    influence_scores: Array of TracIn influence scores (target variable)
    """
    mlflow.set_experiment("impact-predictor")

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            device="gpu",
        )

        # Cross-validate
        cv_scores = cross_val_score(
            model, features_df, influence_scores,
            cv=5, scoring="neg_mean_squared_error"
        )
        print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} +/- {np.sqrt(-cv_scores).std():.4f}")

        # Fit final model
        model.fit(features_df, influence_scores)

        return model
```

#### Using the Meta-Model for Dataset Curation

```python
def score_new_dataset(impact_model, new_dataset_path):
    """Score a new JSONL dataset using the trained impact predictor."""
    scores = []
    examples = load_jsonl(new_dataset_path)

    for idx, example in enumerate(examples):
        features = extract_structural_features(example)  # No loss trajectory needed!
        predicted_impact = impact_model.predict([features])[0]
        scores.append({
            "index": idx,
            "predicted_impact": predicted_impact,
            "example_preview": example["conversations"][0]["content"][:100],
        })

    # Sort by predicted impact (highest first)
    scores.sort(key=lambda x: x["predicted_impact"], reverse=True)

    return scores
```

### 23.6 Recent Papers and Tools

#### Open-Source Implementations

| Tool | Type | Language | LLM Support | Link |
|------|------|----------|-------------|------|
| **Captum** (Meta) | TracIn, Influence Functions | PyTorch | Yes (any PyTorch model) | [github.com/meta-pytorch/captum](https://github.com/meta-pytorch/captum) |
| **TracIn-PyTorch** | TracIn reference implementation | PyTorch | Basic | [github.com/rollovd/TracIn-PyTorch](https://github.com/rollovd/TracIn-PyTorch) |
| **TracIn (frederick0329)** | TracIn NeurIPS 2020 reproduction | PyTorch | Basic | [github.com/frederick0329/TracIn](https://github.com/frederick0329/TracIn) |
| **DataComp** | Data-centric toolkit (OpenAI) | PyTorch | Pretraining focus | [github.com/mlfoundations/datacomp](https://github.com/mlfoundations/datacomp) |
| **DCAI Course** (MIT) | Data-centric AI toolkit | Python | Educational | [dcai.csail.mit.edu](https://dcai.csail.mit.edu/) |

#### Key Papers (2024-2025)

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| [Enhancing TDA for LLMs with Fitting Error (DDA)](https://aclanthology.org/2024.emnlp-main.782/) | EMNLP 2024 | Corrects influence function errors for LLMs; doubles Spearman correlation |
| [Do Influence Functions Work on LLMs?](https://aclanthology.org/2025.findings-emnlp.775.pdf) | EMNLP 2025 Findings | Evaluates IF reliability on modern LLMs |
| [Data Shapley in One Training Run](https://proceedings.iclr.cc/paper_files/paper/2025/file/20fdaf67581e6d7157376d1ed584040a-Paper-Conference.pdf) | ICLR 2025 | Computes Shapley during training with negligible overhead |
| [Scalable Influence and Fact Tracing](https://arxiv.org/html/2410.17413v1) | arXiv 2024 | Scales TDA to 8B model, 160B token corpus |
| [TracIn (original)](https://arxiv.org/abs/2002.08484) | NeurIPS 2020 | Foundational gradient-tracing method |
| [Training Data Attribution Survey](https://link.springer.com/article/10.1007/s10994-023-06495-7) | Machine Learning 2024 | Comprehensive survey of all TDA methods |
| [Efficient Ensembles for TDA](https://arxiv.org/html/2405.17293) | arXiv 2024 | 80% reduction in training cost for attribution |

### 23.7 Integration with Our Pipeline

#### Architecture

```
Trainers/
├── rtx3090_sft/
│   ├── train_sft.py                  # Add PerExampleLossCallback
│   └── sft_output_rtx3090/
│       └── YYYYMMDD_HHMMSS/
│           ├── checkpoints/           # Existing checkpoints (used by TracIn)
│           └── influence/             # NEW: Influence analysis output
│               ├── loss_trajectories.json
│               ├── tracin_scores.json
│               └── impact_model.joblib
│
├── ml/
│   └── configs/
│       └── impact_predictor.yaml     # Config for the meta-model
│
shared/
├── influence/                         # NEW: Shared influence estimation module
│   ├── __init__.py
│   ├── loss_tracking.py              # PerExampleLossCallback
│   ├── tracin.py                     # TracIn implementation (LoRA-aware)
│   ├── features.py                   # Influence feature extraction
│   └── meta_model.py                 # LightGBM impact predictor
│
SynthChat/
└── services/
    └── rubric_runner.py              # Enhanced: use impact scores for prioritization
```

#### Integration Points

| Component | Integration | How |
|-----------|-------------|-----|
| `Trainers/rtx3090_sft/train_sft.py` | Add loss callback | Custom TrainerCallback |
| `Trainers/rtx3090_kto/train_kto.py` | Add loss callback | Same callback pattern |
| `shared/influence/` | New module | TracIn + loss tracking + meta-model |
| `SynthChat/services/rubric_runner.py` | Prioritize by impact | Score examples, process high-impact first |
| `Trainers/ml/` | Meta-model training | Use standalone ML training config |
| `Evaluator/` | Influence-based evaluation | Identify which training data drives eval performance |
| MLflow | Track influence experiments | Log scores, meta-model metrics |

### 23.8 Concrete End-to-End Workflow

Here's the complete workflow from training to data-driven curation:

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: Train LLM with Instrumentation                     │
│                                                              │
│  python train_sft.py --config config.yaml                    │
│    └── PerExampleLossCallback logs loss trajectories         │
│    └── Checkpoints saved every N steps                       │
│                                                              │
│  Output: checkpoints/, influence/loss_trajectories.json      │
├──────────────────────────────────────────────────────────────┤
│  Phase 2: Compute Influence Scores                           │
│                                                              │
│  python -m shared.influence.tracin \                         │
│    --checkpoints sft_output/.../checkpoints/ \               │
│    --train-data Datasets/tools_datasets/.../tools_v1.7.jsonl │
│    --test-data Evaluator/prompts/tool_prompts.json \         │
│    --output influence/tracin_scores.json                     │
│                                                              │
│  Output: tracin_scores.json (per-example influence scores)   │
├──────────────────────────────────────────────────────────────┤
│  Phase 3: Extract Features + Train Impact Meta-Model         │
│                                                              │
│  python -m Trainers.ml.train \                               │
│    --config configs/impact_predictor.yaml                    │
│                                                              │
│  Features: structural + text + validation + loss trajectory  │
│  Target: TracIn influence scores                             │
│  Model: LightGBM regressor                                  │
│                                                              │
│  Output: impact_model.joblib, metrics.json, shap_summary.png │
├──────────────────────────────────────────────────────────────┤
│  Phase 4: Score & Curate Future Datasets                     │
│                                                              │
│  python -m shared.influence.meta_model \                     │
│    --model impact_model.joblib \                             │
│    --dataset Datasets/tools_datasets/new_synthetic.jsonl \   │
│    --output scored_dataset.jsonl                             │
│                                                              │
│  Result: Each example has a predicted impact score           │
│  Action: Filter top-K%, remove bottom-K%, or rank for review │
├──────────────────────────────────────────────────────────────┤
│  Phase 5: Retrain with Curated Dataset                       │
│                                                              │
│  python train_sft.py --dataset scored_top80pct.jsonl         │
│                                                              │
│  Expected: Better model from same/smaller data volume        │
│  Measure: Compare eval metrics vs uncurated baseline         │
└──────────────────────────────────────────────────────────────┘
```

### 23.9 Risk Assessment for Influence Scoring

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TracIn scores noisy for LoRA fine-tuning | Medium | Medium | Use multiple checkpoints; validate against LOO on small subset |
| Per-example loss logging slows training | Low | Low | Forward-only pass; ~5% overhead; can reduce evaluation frequency |
| Meta-model doesn't generalize to new data distributions | Medium | Medium | Retrain meta-model periodically; use SHAP to understand features |
| Circular reasoning: impact scores reflect training, not data quality | Medium | High | Use held-out evaluation data for TracIn test points; validate with human assessment |
| GPU memory pressure from gradient computation | Low | Medium | LoRA gradients are small; batch processing; can offload to CPU |

### 23.10 Smallest Useful First Step

A single Python script (`shared/influence/loss_tracking.py`) that:
1. Adds a `PerExampleLossCallback` to the existing SFT trainer
2. Logs per-example loss at each checkpoint save
3. Outputs `loss_trajectories.json` with per-example loss curves
4. Computes basic statistics: most/least learned examples, loss variance ranking

This requires **zero additional dependencies** and **minimal code changes** (adding one callback to the trainer). It immediately reveals which examples the model finds easy, hard, or confusing -- actionable intelligence for dataset curation even without TracIn or the meta-model.

---

## References

### Part I References

### Gradient Boosting Comparisons
- [XGBoost vs. LightGBM vs. CatBoost](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost)
- [When to Choose CatBoost Over XGBoost or LightGBM](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm)
- [Gradient Boosting Comparison - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/02/gradient-boosting-vs-adaboost-vs-xgboost-vs-catboost-vs-lightgbm/)
- [Benchmarking State-of-the-Art Gradient Boosting (arXiv)](https://arxiv.org/pdf/2305.17094)

### LLM + Traditional ML Integration
- [Integrating LLMs with Traditional ML - Iguazio](https://www.iguazio.com/blog/integrating-llms-with-traditional-ml-how-why-use-cases/)
- [Transitioning from MLOps to LLMOps (MDPI)](https://www.mdpi.com/2078-2489/16/2/87)
- [MLflow Alternatives for MLOps - ZenML](https://www.zenml.io/blog/mlflow-alternatives)

### MLflow and Experiment Tracking
- [MLflow Scikit-learn Integration](https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/)
- [End-to-End MLflow Guide (March 2026)](https://www.marktechpost.com/2026/03/01/a-complete-end-to-end-coding-guide-to-mlflow-experiment-tracking-hyperparameter-optimization-model-evaluation-and-live-model-deployment/)
- [MLflow Official Site](https://mlflow.org/)

### RAPIDS cuML
- [cuML Accelerated ML](https://rapids.ai/cuml-accel/)
- [NVIDIA cuML Zero Code Change Acceleration](https://developer.nvidia.com/blog/nvidia-cuml-brings-zero-code-change-acceleration-to-scikit-learn/)
- [cuML Documentation v26.02](https://docs.rapids.ai/api/cuml/stable/)
- [cuML GitHub](https://github.com/rapidsai/cuml)

### Data Quality and Routing
- [The Data-Quality Illusion (arXiv 2025)](https://arxiv.org/abs/2510.00866)
- [Evaluating Synthetic Data for Tool-Using LLMs (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.285.pdf)
- [RouteLLM - Cost-Effective LLM Routing](https://lmsys.org/blog/2024-07-01-routellm/)
- [NVIDIA LLM Router Blueprint](https://build.nvidia.com/nvidia/llm-router)
- [Routing on Random Forests](https://kleiber.me/blog/2025/08/10/llm-router-primer/)

### Feature Engineering
- [Feature Engineering for Text Data](https://apxml.com/courses/applied-data-science/chapter-2-practical-feature-engineering/text-feature-creation)
- [XGBoost with Word2Vec Framework](https://ijarcce.com/wp-content/uploads/2025/04/IJARCCE.2025.14459.pdf)

### Scikit-learn Evaluation
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Confusion Matrix Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Cross-Validation Documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [ROC with Cross-Validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html)

### Part II References

### AutoML Frameworks
- [AutoGluon TabularPredictor API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)
- [AutoGluon TabularPredictor.fit](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)
- [FLAML - Fast Lightweight AutoML](https://microsoft.github.io/FLAML/)
- [FLAML Getting Started](https://microsoft.github.io/FLAML/docs/Getting-Started/)
- [FLAML Task-Oriented AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/)
- [PyCaret 3.0 Documentation](https://pycaret.org/)
- [PyCaret Guide - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/02/pycaret-guide/)
- [AutoML Benchmarking Study (Nature, 2025)](https://www.nature.com/articles/s41598-025-02149-x)
- [Top AutoML Platforms 2026](https://www.devopsschool.com/blog/top-10-ai-automl-platforms-in-2025-features-pros-cons-comparison/)
- [AutoGluon vs PyCaret Comparison](https://www.analyticsinsight.net/tech-news/autogluon-vs-pycaret-which-framework-is-more-versatile)

### Scikit-learn Pipelines and Feature Engineering
- [ColumnTransformer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Column Transformer with Mixed Types Example](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- [Advanced Feature Engineering with Pipelines](https://machinelearningmastery.com/advanced-feature-engineering-using-scikit-learn-pipelines-with-pandas-columntransformer-and-numpy-arrays/)

### Model Deployment and ONNX
- [sklearn-onnx Documentation](https://onnx.ai/sklearn-onnx/)
- [Convert Pipeline with LightGBM to ONNX](https://onnx.ai/sklearn-onnx/auto_examples/plot_pipeline_lightgbm.html)
- [onnxmltools GitHub](https://github.com/onnx/onnxmltools)
- [Deploy Traditional ML with ONNX Runtime](https://onnxruntime.ai/docs/tutorials/traditional-ml.html)
- [ONNX Format Guide - DataCamp](https://www.datacamp.com/tutorial/onnx)

### Part III References (Training Data Influence)

### Training Data Attribution Methods
- [TracIn: Estimating Training Data Influence by Tracing Gradient Descent (NeurIPS 2020)](https://arxiv.org/abs/2002.08484)
- [TracIn - Google Research Blog](https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/)
- [Training Data Attribution Survey (Machine Learning, 2024)](https://link.springer.com/article/10.1007/s10994-023-06495-7)
- [Enhancing TDA for LLMs with Fitting Error (DDA) - EMNLP 2024](https://aclanthology.org/2024.emnlp-main.782/)
- [Do Influence Functions Work on Large Language Models? - EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.775.pdf)
- [Data Shapley in One Training Run - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/20fdaf67581e6d7157376d1ed584040a-Paper-Conference.pdf)
- [Scalable Influence and Fact Tracing for LLM Pretraining](https://arxiv.org/html/2410.17413v1)
- [Efficient Ensembles Improve Training Data Attribution](https://arxiv.org/html/2405.17293)
- [Data Shapley: Equitable Valuation of Data for ML](https://arxiv.org/abs/1904.02868)
- [Ian Tenney - Training Data Attribution Project](https://iftenney.github.io/projects/tda/)

### Tools and Implementations
- [Captum - Model Interpretability for PyTorch (Meta)](https://captum.ai/)
- [Captum TracInCP Tutorial](https://captum.ai/tutorials/TracInCP_Tutorial)
- [Captum Influence API](https://pytorch.org/captum/api/influence.html)
- [Captum GitHub](https://github.com/meta-pytorch/captum)
- [TracIn-PyTorch Implementation](https://github.com/rollovd/TracIn-PyTorch)
- [TracIn NeurIPS 2020 Reproduction](https://github.com/frederick0329/TracIn)

### Data-Centric AI
- [Introduction to Data-Centric AI (MIT CSAIL)](https://dcai.csail.mit.edu/)
- [Data-Centric AI Survey](https://arxiv.org/html/2212.11854v4)
- [Data-Centric Perspective on LLM Lifecycle](https://www.techrxiv.org/users/998473/articles/1370970/master/file/data/data-ai-arxiv/data-ai-arxiv.pdf)
- [HuggingFace SFT Trainer Documentation](https://huggingface.co/docs/trl/en/sft_trainer)
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
