# triton-prep

`triton-prep` is a lightweight CLI helper for turning HuggingFace *text* models into Triton Inference Server repositories. It focuses on the minimal flow required for encoder/decoder NLP models: inspect → export to ONNX → emit `config.pbtxt` → assemble the repo layout.

> **Note**  
> Only text models are supported in this MVP. Vision/audio architectures or custom preprocessing pipelines are out of scope for now.

## What you get

- Task-aware export to ONNX (feature extraction, classification, causal LM, seq2seq LM).
- Optional FP16 conversion of the exported graph.
- Automatic Triton repository layout creation with tokenizer artifacts.
- Config generator that tailors inputs/outputs/instance groups for text workloads.
- Quick sanity checks: repository validation and ONNX runtime smoke test.

## Requirements

- Python 3.11+
- PyTorch with CPU support (GPU optional)
- `onnx`, `onnxruntime`, `transformers`, `uv` (or your preferred runner)

Install the project in editable mode (example shown with `uv`; replace with `pip` if you prefer):

```bash
uv pip install -e .
```

## CLI overview

```plaintext
triton-prep
├── prepare  # inspect HF model, export ONNX, build Triton repo
├── info     # print high-level model metadata
└── check
    ├── repo  # validate Triton repository structure
    └── onnx  # run ONNXRuntime inference with dummy text inputs
```

### Prepare a model

```bash
triton-prep prepare hf-internal-testing/tiny-random-bert \
  --output models \
  --model-name tinybert \
  --opset 18 \
  --max-batch 8 \
  --dynamic \
  --clean
```

This will:

1. Inspect the HuggingFace model to determine the best task type, embeddings dimension, etc.
2. Export the model to ONNX (and FP16 if `--fp16` is set).
3. Generate a Triton config with the requested batch/device settings.
4. Assemble or refresh `models/tinybert` with `config.pbtxt`, versioned model file, and tokenizer assets.

### Inspect model info

```bash
triton-prep info hf-internal-testing/tiny-random-bert
```

Outputs the detected task, parameter count, embedding dimension, and vocabulary size so you can sanity check before export.

### Validate outputs

After preparing a repo you can run:

```bash
triton-prep check repo models/tinybert
triton-prep check onnx models/tinybert/1/model.onnx
```

The first command ensures the repository has the expected layout. The second spins up ONNX Runtime with auto-generated text tensors to make sure the exported graph executes.

## Development

```bash
uv run python -m triton_prep --help
uv run pytest          # if/when the project gains tests
```

Linting is configured through `ruff` and type checking via `mypy`. Feel free to extend the CLI or exporter for additional text tasks—just keep the MVP goal of text-only coverage in mind.
