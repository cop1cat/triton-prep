from pathlib import Path

import typer

from triton_prep.services.onnx_tester import OnnxTester
from triton_prep.services.repo_validator import TritonRepositoryValidator

cli = typer.Typer(name="check", help="Validate Triton model repository.")


@cli.command("repo")
def check_repo(model_dir: Path = typer.Argument(...)) -> None:
    """
    Validates Triton repository structure.
    """
    validator = TritonRepositoryValidator(model_path=model_dir)
    validator.check()
    typer.echo("Repository structure is valid.")


@cli.command("onnx")
def check_onnx(model_path: Path = typer.Argument(...)) -> None:
    """
    Runs minimal ONNX inference test.
    """
    tester = OnnxTester()
    tester.test(model_path)
    typer.echo("ONNX model executed successfully.")
