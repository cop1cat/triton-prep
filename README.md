# triton-prep

A minimal CLI tool for preparing HuggingFace models for Triton Inference Server.

## Features

- Export HuggingFace models to ONNX.
- Generate a valid Triton model repository structure.
- Automatically create `config.pbtxt`.
- Inspect models and validate repositories.

## Usage

Export a model:

```bash
triton-prep prepare bert-base-uncased --output models --model-name bert
````

Show model info:

```bash
triton-prep info bert-base-uncased
```

Validate a repository:

```bash
triton-prep check models/bert
```
