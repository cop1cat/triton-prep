import logging
from typing import Any

from pydantic import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.onnx import FeaturesManager

from triton_prep.exporters.base import TaskType

logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    """
    High-level information about a HuggingFace model.
    """

    model_id: str
    task_type: TaskType
    num_parameters: int
    embedding_dim: int | None
    vocab_size: int | None


class ModelInspector:
    """
    Extracts structural and configuration information from HuggingFace models.
    """

    def inspect(self, model_id: str) -> ModelInfo:
        """
        Loads a HuggingFace model and extracts task type, dimensionality,
        vocabulary size, and number of parameters.

        Args:
            model_id (str): HuggingFace model identifier or path.

        Returns:
            ModelInfo: Structured model information.
        """
        cfg = AutoConfig.from_pretrained(model_id)
        tokenizer = self._load_tokenizer(model_id)
        model = self._load_model(model_id)

        task_type = self._detect_task(cfg, model)
        num_params = self._count_parameters(model)
        embedding_dim = self._detect_embedding_dim(cfg)
        vocab_size = tokenizer.vocab_size if tokenizer else None

        return ModelInfo(
            model_id=model_id,
            task_type=task_type,
            num_parameters=num_params,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
        )

    def _load_tokenizer(self, model_id: str) -> Any | None:
        try:
            return AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        except Exception:
            logger.warning("Tokenizer not found for '%s'", model_id)
            return None

    def _load_model(self, model_id: str) -> Any:
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        return model

    def _count_parameters(self, model: Any) -> int:
        return sum(int(p.numel()) for p in model.parameters())

    def _detect_embedding_dim(self, cfg: Any) -> int | None:
        for attr in ("hidden_size", "d_model", "embed_dim"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        return None

    def _detect_task(self, cfg: Any, model: Any) -> TaskType:
        """
        Determines valid ONNX export task based on FeaturesManager metadata.

        This ensures that "feature-extraction" is only used when supported,
        avoiding KeyError in OnnxExporter.
        """
        model_type = cfg.model_type  # e.g. "bert", "gpt2", "t5"
        supported = FeaturesManager._SUPPORTED_MODEL_TYPE.get(model_type, {})

        # Prefer task from architecture if provided
        architectures = getattr(cfg, "architectures", None)
        arch_task = self._task_from_architectures(architectures)

        if arch_task and arch_task.value in supported:
            return arch_task

        # Fallback order: FE -> CLS -> TOKEN -> CAUSAL_LM -> SEQ2SEQ
        fallback_order = [
            TaskType.FEATURE_EXTRACTION,
            TaskType.TEXT_CLASSIFICATION,
            TaskType.TOKEN_CLASSIFICATION,
            TaskType.CAUSAL_LM,
            TaskType.SEQ2SEQ_LM,
        ]

        for t in fallback_order:
            if t.value in supported:
                return t

        logger.warning(
            "No supported export task for model '%s'. Using feature-extraction.",
            model_type,
        )
        return TaskType.FEATURE_EXTRACTION

    def _task_from_architectures(
        self, architectures: list[str] | None
    ) -> TaskType | None:
        """
        Maps HF architecture names to TaskType.
        """
        if not architectures:
            return None

        normalized = [arch.lower() for arch in architectures]

        for arch in normalized:
            if "tokenclassification" in arch:
                return TaskType.TOKEN_CLASSIFICATION
            if "sequenceclassification" in arch:
                return TaskType.TEXT_CLASSIFICATION
            if "causallm" in arch or "causal" in arch:
                return TaskType.CAUSAL_LM
            if "seq2seq" in arch:
                return TaskType.SEQ2SEQ_LM

        return None
