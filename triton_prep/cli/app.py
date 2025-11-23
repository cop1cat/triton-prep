from typer import Typer

from triton_prep.cli.commands.check import cli as check_command
from triton_prep.cli.commands.info import cli as info_command
from triton_prep.cli.commands.prepare import cli as prepare_command


def _build_cli() -> Typer:
    app = Typer(
        name="triton-prep",
        help="Utilities for preparing HuggingFace models for Triton Inference Server.",
    )

    app.add_typer(
        prepare_command,
        help="Export HuggingFace model, build Triton model repository, generate config.pbtxt.",
    )

    app.add_typer(
        info_command, help="Display information about a HuggingFace model."
    )

    app.add_typer(
        check_command,
        help="Validate Triton model repository or run ONNX inference test.",
    )

    return app


cli = _build_cli()

__all__ = ["cli"]

if __name__ == "__main__":
    cli()
