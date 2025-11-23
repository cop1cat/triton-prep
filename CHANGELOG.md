# Changelog

## v0.1.0

### Added

- CLI entrypoint `triton-prep` based on Typer.
- Added commands:
  - Command `prepare` for exporting HuggingFace models to ONNX and generating Triton model repository structure.
  - Command `info` for inspecting HuggingFace model configuration.
  - Command `check` for validating Triton model repository and testing ONNX inference.
