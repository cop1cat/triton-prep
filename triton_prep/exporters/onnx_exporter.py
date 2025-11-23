import logging
from pathlib import Path

import torch
import onnx
from onnxconverter_common import float16
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from triton_prep.exporters.base import (
    MODEL_ONNX_FILENAME,
    MODEL_ONNX_FP16_FILENAME,
    BaseExporter,
    ExportConfig,
    TaskType,
)

logger = logging.getLogger(__name__)


class OnnxExporter(BaseExporter):
    """
    ONNX exporter using torch.onnx.export with single-file output.
    """

    def export(self, cfg: ExportConfig) -> Path:
        model = self._load_model(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

        model.eval()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = cfg.output_dir / MODEL_ONNX_FILENAME

        dummy_inputs = self._dummy_inputs(tokenizer, cfg.task_type)
        input_names = list(dummy_inputs.keys())
        dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
        output_names = self._output_names(cfg.task_type)

        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                f=str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=cfg.opset,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
            )

        model_proto = onnx.load(str(output_path))
        onnx.save_model(model_proto, str(output_path), save_as_external_data=False)

        if cfg.fp16:
            fp32_model = onnx.load(str(output_path))
            fp16_model = float16.convert_float_to_float16(fp32_model)
            fp16_path = cfg.output_dir / MODEL_ONNX_FP16_FILENAME
            onnx.save(fp16_model, str(fp16_path))
            return fp16_path

        return output_path

    def _load_model(self, cfg: ExportConfig):
        factory = {
            TaskType.TEXT_CLASSIFICATION: AutoModelForSequenceClassification,
            TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
            TaskType.CAUSAL_LM: AutoModelForCausalLM,
            TaskType.SEQ2SEQ_LM: AutoModelForSeq2SeqLM,
        }
        constructor = factory.get(cfg.task_type, AutoModel)
        return constructor.from_pretrained(cfg.model_id)

    def _dummy_inputs(self, tokenizer, task: TaskType):
        encoder_inputs = tokenizer(
            ["hello world", "foo bar"],
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True,
        )
        inputs = dict(encoder_inputs)

        if task == TaskType.SEQ2SEQ_LM:
            decoder_inputs = tokenizer(
                ["response one", "response two"],
                return_tensors="pt",
                padding="max_length",
                max_length=16,
                truncation=True,
            )
            inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
            if "attention_mask" in decoder_inputs:
                inputs["decoder_attention_mask"] = decoder_inputs["attention_mask"]

        return inputs

    def _output_names(self, task: TaskType) -> list[str]:
        if task == TaskType.FEATURE_EXTRACTION:
            return ["last_hidden_state"]
        if task in (TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION):
            return ["logits"]
        if task in (TaskType.CAUSAL_LM, TaskType.SEQ2SEQ_LM):
            return ["logits"]
        return ["output"]
