"""
Natural Language Inference (NLI) detector for consistency checking.
Provides entailment probability between premise and hypothesis.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import pipeline
    _NLI_AVAILABLE = True
except ImportError:
    _NLI_AVAILABLE = False
    logger.warning("torch/transformers not available - NLIDetector will be disabled")


class NLIDetector:
    """
    Uses an NLI model to detect contradictions/hallucinations.
    Returns entailment probability (0 to 1) for a given premise‑hypothesis pair.
    
    If torch/transformers are not available, this class will be non-functional.
    """

    def __init__(self, model_name: str = "microsoft/deberta-base-mnli"):
        """
        Args:
            model_name: Hugging Face model identifier for NLI.
                       Default is a public model that does not require authentication.
        """
        if not _NLI_AVAILABLE:
            logger.warning("torch/transformers not available - NLIDetector will be non-functional")
            self.pipeline = None
            return
            
        try:
            # Use top_k=None to get all scores (equivalent to deprecated return_all_scores)
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=device,
                top_k=None
            )
            logger.info(f"NLI model {model_name} loaded with top_k=None.")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            self.pipeline = None

    def check(self, premise: str, hypothesis: str) -> Optional[float]:
        """
        Returns probability of entailment (higher means more consistent).
        Args:
            premise: The original input/context.
            hypothesis: The generated response.
        Returns:
            Float between 0 and 1, or None if model unavailable.
        """
        if self.pipeline is None:
            return None
        try:
            # For a single input, the pipeline returns a list of dicts (one per class)
            result = self.pipeline(f"{premise} </s></s> {hypothesis}")
            # result is a list of dicts with keys 'label', 'score'
            for item in result:
                if item['label'] == 'ENTAILMENT':
                    return item['score']
            logger.warning("ENTAILMENT label not found in NLI output; returning 0.0.")
            return 0.0
        except Exception as e:
            logger.error(f"NLI error: {e}")
            return None
