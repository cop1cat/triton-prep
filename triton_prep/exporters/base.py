import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskType(StrEnum):
    """
    Supported export task types for HuggingFace → ONNX.
    """

    FEATURE_EXTRACTION = "feature-extraction"
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    SEQ2SEQ_LM = "seq2seq-lm"
    CAUSAL_LM = "causal-lm"


MODEL_ONNX_FILENAME = "model.onnx"
MODEL_ONNX_FP16_FILENAME = "model_fp16.onnx"
DEFAULT_ONNX_VERSION = 16


class ExportConfig(BaseModel):
    """
    Configuration for exporting a model.

    Attributes:
        model_id (str): HuggingFace model identifier or path.
        output_dir (Path): Directory for exported artifacts.
        task_type (TaskType): Export task type.
        opset (int | None): ONNX opset version. Default to 16
        fp16 (bool): Indicates whether FP16 model is requested.
    """

    model_id: str
    output_dir: Path
    task_type: TaskType = Field(default=TaskType.FEATURE_EXTRACTION)
    opset: int = DEFAULT_ONNX_VERSION
    fp16: bool = False


class BaseExporter(ABC):
    """
    Abstract interface for model exporters.
    """

    @abstractmethod
    def export(self, cfg: ExportConfig) -> Path:
        """
        Executes the export process.

        Args:
            cfg (ExportConfig): Export configuration.

        Returns:
            Path: Path to the exported model file.
        """
        raise NotImplementedError
