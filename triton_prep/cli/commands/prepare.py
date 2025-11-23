import logging
from pathlib import Path
from typing import Literal

import typer

from triton_prep.exporters.base import DEFAULT_ONNX_VERSION, ExportConfig
from triton_prep.exporters.onnx_exporter import OnnxExporter
from triton_prep.services.config_generator import TritonConfig
from triton_prep.services.inspector import ModelInspector
from triton_prep.services.repository_manager import TritonRepositoryManager

cli = typer.Typer(help="Prepare a HuggingFace model for Triton.")
logger = logging.getLogger(__name__)


@cli.command()
def prepare(
    model_id: str = typer.Argument(..., help="HuggingFace model id or local path."),
    output_dir: Path = typer.Option(
        "models",
        "--output",
        "-o",
        help="Root directory for Triton Model Repository.",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        "-n",
        help="Override Triton model name. Defaults to sanitized model_id.",
    ),
    opset: int = typer.Option(
        DEFAULT_ONNX_VERSION,
        help=f"ONNX opset version. Default = {DEFAULT_ONNX_VERSION}.",
    ),
    fp16: bool = typer.Option(
        False,
        help="Enable FP16 conversion for ONNX model.",
    ),
    device: Literal["cpu", "gpu"] = typer.Option(
        "cpu",
        help="Execution device for Triton.",
    ),
    max_batch: int = typer.Option(
        128,
        "--max-batch",
        help="Triton max_batch_size.",
    ),
    instance_count: int = typer.Option(
        1,
        help="Number of Triton execution instances.",
    ),
    dynamic_batching: bool = typer.Option(
        False,
        "--dynamic",
        help="Enable Triton dynamic batching.",
    ),
    clean: bool = typer.Option(
        False,
        help="Clean existing Triton model directory.",
    ),
) -> None:
    """
    Prepares a HuggingFace model for Triton Inference Server.
    """

    resolved_name = (
        model_name if model_name else model_id.replace("/", "_").replace(".", "_")
    )

    inspector = ModelInspector()
    info = inspector.inspect(model_id)

    export_cfg = ExportConfig(
        model_id=model_id,
        output_dir=Path(".tmp_export") / resolved_name,
        task_type=info.task_type,
        opset=opset,
        fp16=fp16,
    )
    exporter = OnnxExporter()
    exported_path = exporter.export(export_cfg)

    triton_cfg = TritonConfig(
        model_name=resolved_name,
        backend="onnxruntime_onnx",
        max_batch_size=max_batch,
        instance_count=instance_count,
        device="KIND_GPU" if device == "gpu" else "KIND_CPU",
        output_name="output",
        enable_dynamic_batching=dynamic_batching,
    )

    repo_manager = TritonRepositoryManager()
    repo_paths = repo_manager.create(
        exported_model_path=exported_path,
        model_info=info,
        cfg=triton_cfg,
        repo_root=output_dir,
        clean=clean,
    )

    typer.echo("Model prepared successfully.")
    typer.echo(f"Repository: {repo_paths.model_dir}")
    typer.echo(f"config.pbtxt: {repo_paths.config_path}")
    typer.echo(f"Model file: {repo_paths.model_file_path}")
    typer.echo(f"Task type: {info.task_type}")
    typer.echo(f"Embedding dim: {info.embedding_dim}")
