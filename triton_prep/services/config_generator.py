import logging
from typing import Literal

from pydantic import BaseModel

from triton_prep.exporters.base import TaskType
from triton_prep.services.config_templates import (
    CONFIG_TEMPLATE,
    DYNAMIC_BATCHING_TEMPLATE,
    EMPTY_SECTION,
    INPUT_LM_TEMPLATE,
    INPUT_NLP_TEMPLATE,
    INSTANCE_GROUP_TEMPLATE,
    OUTPUT_CLASSIFICATION_TEMPLATE,
    OUTPUT_EMBED_TEMPLATE,
    OUTPUT_LM_TEMPLATE,
)
from triton_prep.services.inspector import ModelInfo

logger = logging.getLogger(__name__)


class TritonConfig(BaseModel):
    """
    Configuration parameters for generating a Triton config.pbtxt file.

    Attributes:
        model_name (str): Name of the model inside the Triton repository.
        backend (str): Triton backend identifier such as "onnxruntime_onnx".
        max_batch_size (int): Maximum allowed batch size.
        instance_count (int): Number of Triton execution instances.
        device (Literal["KIND_CPU", "KIND_GPU"]): Execution device type.
        output_name (str): Name of the output tensor.
        enable_dynamic_batching (bool): Enables Triton's dynamic batching block.
    """

    model_name: str
    backend: str
    max_batch_size: int
    instance_count: int
    device: Literal["KIND_CPU", "KIND_GPU"]
    output_name: str
    enable_dynamic_batching: bool = False


class TritonConfigGenerator:
    """
    Generates Triton config.pbtxt files from predefined templates.
    """

    def generate(self, cfg: TritonConfig, info: ModelInfo) -> str:
        """
        Generates a complete config.pbtxt content string.

        Args:
            cfg (TritonConfig): Triton configuration parameters.
            info (ModelInfo): Extracted HuggingFace model metadata.

        Returns:
            str: Fully rendered config.pbtxt content.
        """
        input_section = self._select_input_template(info.task_type)
        output_section = self._select_output_template(cfg, info)
        instance_group_section = self._render_instance_group(cfg)
        dynamic_section = self._render_dynamic_batching(cfg)

        text = (
            CONFIG_TEMPLATE.replace("%MODEL_NAME%", cfg.model_name)
            .replace("%BACKEND%", cfg.backend)
            .replace("%MAX_BATCH%", str(cfg.max_batch_size))
            .replace("%INPUT_SECTION%", input_section)
            .replace("%OUTPUT_SECTION%", output_section)
            .replace("%INSTANCE_GROUP_SECTION%", instance_group_section)
            .replace("%DYNAMIC_BATCHING_SECTION%", dynamic_section)
        )

        return text.strip()

    def _select_input_template(self, task: TaskType) -> str:
        """
        Selects input tensor template based on the model task.

        Args:
            task (TaskType): Inferred task type.

        Returns:
            str: Rendered input block.
        """
        match task:
            case TaskType.CAUSAL_LM | TaskType.SEQ2SEQ_LM:
                return INPUT_LM_TEMPLATE.strip()
            case _:
                return INPUT_NLP_TEMPLATE.strip()

    def _select_output_template(self, cfg: TritonConfig, info: ModelInfo) -> str:
        """
        Selects output template based on model task type.

        Args:
            cfg (TritonConfig): Triton configuration parameters.
            info (ModelInfo): Model metadata.

        Returns:
            str: Rendered output block.
        """
        match info.task_type:
            case TaskType.TEXT_CLASSIFICATION | TaskType.TOKEN_CLASSIFICATION:
                return OUTPUT_CLASSIFICATION_TEMPLATE.replace(
                    "%OUTPUT_NAME%", cfg.output_name
                ).strip()

            case TaskType.CAUSAL_LM | TaskType.SEQ2SEQ_LM:
                return OUTPUT_LM_TEMPLATE.replace(
                    "%OUTPUT_NAME%", cfg.output_name
                ).strip()

            case _:
                dim = info.embedding_dim or 768
                return (
                    OUTPUT_EMBED_TEMPLATE.replace("%OUTPUT_NAME%", cfg.output_name)
                    .replace("%EMBEDDING_DIM%", str(dim))
                    .strip()
                )

    def _render_instance_group(self, cfg: TritonConfig) -> str:
        """
        Renders Triton instance group section.

        Args:
            cfg (TritonConfig): Configuration parameters.

        Returns:
            str: Rendered instance group block.
        """
        return (
            INSTANCE_GROUP_TEMPLATE.replace("%DEVICE_KIND%", cfg.device)
            .replace("%INSTANCE_COUNT%", str(cfg.instance_count))
            .strip()
        )

    def _render_dynamic_batching(self, cfg: TritonConfig) -> str:
        """
        Renders dynamic batching section if enabled.

        Args:
            cfg (TritonConfig): Configuration parameters.

        Returns:
            str: Dynamic batching block or empty string.
        """
        if cfg.enable_dynamic_batching:
            return DYNAMIC_BATCHING_TEMPLATE.strip()

        return EMPTY_SECTION
