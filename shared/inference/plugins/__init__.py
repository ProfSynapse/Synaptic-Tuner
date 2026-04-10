"""vLLM inference plugins (layer-hook and logits-only)."""

from shared.inference.plugins.dola import DoLaPlugin
from shared.inference.plugins.min_p import MinPPlugin
from shared.inference.plugins.repetition import RepetitionPenaltyPlugin

__all__ = ["DoLaPlugin", "MinPPlugin", "RepetitionPenaltyPlugin"]
