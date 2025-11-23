import logging
from pathlib import Path
from shutil import copy2, rmtree

from pydantic import BaseModel

from triton_prep.services.config_generator import (
    TritonConfig,
    TritonConfigGenerator,
)
from triton_prep.services.inspector import ModelInfo

logger = logging.getLogger(__name__)


class RepositoryPaths(BaseModel):
    """
    Paths inside a Triton Model Repository.

    Attributes:
        model_dir (Path): Directory of the Triton model.
        version_dir (Path): Versioned subdirectory (e.g., model_dir/1).
        config_path (Path): Path to the config.pbtxt file.
        model_file_path (Path): Path where the model artifact will be copied.
    """

    model_dir: Path
    version_dir: Path
    config_path: Path
    model_file_path: Path


class TritonRepositoryManager:
    """
    Creates and manages a Triton Model Repository structure.
    """

    def __init__(self) -> None:
        self._config_gen = TritonConfigGenerator()

    def create(
        self,
        exported_model_path: Path,
        model_info: ModelInfo,
        cfg: TritonConfig,
        repo_root: Path,
        clean: bool = False,
    ) -> RepositoryPaths:
        """
        Creates a Triton model repository with config.pbtxt and model artifacts.

        Args:
            exported_model_path (Path): Path to exported model file.
            model_info (ModelInfo): Inspect information about the model.
            cfg (TritonConfig): Triton configuration.
            repo_root (Path): Root of Triton Model Repository.
            clean (bool): Whether to remove existing model directory before creating a new one.

        Returns:
            RepositoryPaths: Structured repository paths.
        """
        model_dir = repo_root / cfg.model_name
        version_dir = model_dir / "1"

        if clean and model_dir.exists():
            rmtree(model_dir)

        model_dir.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir(parents=True, exist_ok=True)

        model_file_path = version_dir / exported_model_path.name
        copy2(exported_model_path, model_file_path)

        config_text = self._config_gen.generate(cfg, model_info)
        config_path = model_dir / "config.pbtxt"
        config_path.write_text(config_text)

        self._copy_tokenizer(model_info.model_id, version_dir)

        return RepositoryPaths(
            model_dir=model_dir,
            version_dir=version_dir,
            config_path=config_path,
            model_file_path=model_file_path,
        )

    def _copy_tokenizer(self, model_id: str, dst_dir: Path) -> None:
        """
        Copies tokenizer files from HuggingFace, if available.

        Args:
            model_id (str): HuggingFace model identifier or path.
            dst_dir (Path): Destination directory for tokenizer files.
        """
        possible_files = [
            "tokenizer.json",
            "vocab.txt",
            "tokenizer.model",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ]

        for name in possible_files:
            try:
                src = Path(model_id) / name
                if src.exists():
                    copy2(src, dst_dir / name)
                    continue

                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call]
                if hasattr(tok, name.replace(".json", "")):
                    try:
                        tok.save_pretrained(dst_dir)
                        return
                    except Exception:
                        pass
            except Exception:
                continue
