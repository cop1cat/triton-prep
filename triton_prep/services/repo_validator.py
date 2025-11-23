from pathlib import Path

from pydantic import BaseModel


class TritonRepositoryValidator(BaseModel):
    """
    Validates Triton Model Repository structure.
    """

    model_path: Path

    def check(self) -> None:
        if not self.model_path.exists():
            raise RuntimeError(f"Model directory not found: {self.model_path}")

        config = self.model_path / "config.pbtxt"
        if not config.exists():
            raise RuntimeError("Missing config.pbtxt")

        version_dir = self.model_path / "1"
        if not version_dir.exists():
            raise RuntimeError("Missing version directory '1'")

        model_files = list(version_dir.glob("*.onnx"))
        if not model_files:
            raise RuntimeError("No ONNX model found in version directory")
