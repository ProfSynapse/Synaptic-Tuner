"""
Config -> sentence-transformers loss mapping.

Location: Trainers/embedding/src/losses.py
Purpose:  Resolve a config `loss` string into a sentence-transformers loss
          instance, applying the Matryoshka wrapper when the model spec declares
          matryoshka_dims. No model-specific behavior — the dims come from the
          EmbeddingModelSpec (the registry SSOT), the loss name from config.
Used by:  Trainers/embedding/train_embedding.py.

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §6.1.

Loss map (§6.1):
    multiple_negatives_ranking        -> MultipleNegativesRankingLoss
    cached_multiple_negatives_ranking -> CachedMultipleNegativesRankingLoss
    triplet                           -> TripletLoss
    cosent                            -> CoSENTLoss
    cosine_similarity                 -> CosineSimilarityLoss
    (wrapper) matryoshka              -> MatryoshkaLoss over the base loss
                                         (auto-applied when spec.matryoshka_dims)

In-batch-negative losses (MNRL / cached MNRL) require BatchSamplers.NO_DUPLICATES
so a batch never contains a (query, positive) pair whose positive is another
query's hard-negative — see select_batch_sampler() (PREPARE §3.1, MNRL
correctness).
"""

from __future__ import annotations

from typing import Any

from registry import EmbeddingModelSpec

# Loss names that consume in-batch negatives and therefore require the
# NO_DUPLICATES batch sampler for correctness (§6.1 / PREPARE §3.1).
IN_BATCH_NEGATIVE_LOSSES = frozenset(
    {"multiple_negatives_ranking", "cached_multiple_negatives_ranking"}
)

# Loss names that require a graded (sentence1, sentence2, score) dataset shape.
# The v1 data loader (data_loader.load_embedding_dataset) only emits
# (anchor, positive[, negative]) columns from triplet/pairs JSONL — there is NO
# score column — so selecting one of these would crash at train time inside the
# loss. They are gated at config time (build_loss) with an actionable message
# rather than allowed to reach a deep train-time failure (§6.1).
SCORE_REQUIRING_LOSSES = frozenset({"cosent", "cosine_similarity"})


def build_loss(model: Any, loss_name: str, spec: EmbeddingModelSpec) -> Any:
    """Build the ST loss for `loss_name`, wrapping in MatryoshkaLoss if declared.

    Args:
        model:     the SentenceTransformer-compatible model (loss needs a handle).
        loss_name: config `training.loss` string (case-insensitive).
        spec:      the model's EmbeddingModelSpec; spec.matryoshka_dims (when
                   non-empty) triggers the MatryoshkaLoss wrapper.

    Returns:
        An ST loss instance ready to pass to SentenceTransformerTrainer.

    Raises:
        ValueError: unknown loss_name, OR a score-requiring loss (cosent /
                    cosine_similarity) that the triplet/pairs data loader cannot
                    feed — caught here at config time, not deep in train().
    """
    from sentence_transformers import losses

    key = (loss_name or "").strip().lower()
    base_loss_factories = {
        "multiple_negatives_ranking": lambda: losses.MultipleNegativesRankingLoss(model),
        "cached_multiple_negatives_ranking": lambda: losses.CachedMultipleNegativesRankingLoss(model),
        "triplet": lambda: losses.TripletLoss(model),
        "cosent": lambda: losses.CoSENTLoss(model),
        "cosine_similarity": lambda: losses.CosineSimilarityLoss(model),
    }

    if key not in base_loss_factories:
        raise ValueError(
            f"Unknown loss {loss_name!r}; supported: {sorted(base_loss_factories)}"
        )

    # Fail fast at config time: these losses need a (sentence1, sentence2, score)
    # dataset, but the v1 triplet/pairs loader emits no score column, so they would
    # otherwise crash mid-training with an opaque error.
    if key in SCORE_REQUIRING_LOSSES:
        raise ValueError(
            f"Loss {loss_name!r} requires a graded (sentence1, sentence2, score) "
            f"dataset, but the v1 embedding data loader only produces "
            f"(anchor, positive[, negative]) from triplet/pairs JSONL. "
            f"Use one of {sorted(set(base_loss_factories) - SCORE_REQUIRING_LOSSES)}, "
            f"or provide a scored-pairs data loader before selecting "
            f"{sorted(SCORE_REQUIRING_LOSSES)} (deferred in v1)."
        )

    base_loss = base_loss_factories[key]()

    # Matryoshka wrapper (§6.1): applied whenever the spec declares dims, so the
    # model is trained to be truncatable to the smaller MRL dimensions.
    if spec.matryoshka_dims:
        return losses.MatryoshkaLoss(
            model,
            base_loss,
            matryoshka_dims=list(spec.matryoshka_dims),
        )

    return base_loss


def select_batch_sampler(loss_name: str):
    """Return BatchSamplers.NO_DUPLICATES for in-batch-negative losses, else BATCH_SAMPLER.

    NO_DUPLICATES guarantees a batch holds no duplicate texts, which is required
    for MNRL-style in-batch negatives to be valid negatives (§6.1).
    """
    from sentence_transformers.training_args import BatchSamplers

    key = (loss_name or "").strip().lower()
    if key in IN_BATCH_NEGATIVE_LOSSES:
        return BatchSamplers.NO_DUPLICATES
    return BatchSamplers.BATCH_SAMPLER
