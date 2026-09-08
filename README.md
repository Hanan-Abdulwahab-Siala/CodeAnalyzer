# Mamba Code Analyzer

A code-analysis system based on a Mamba fine-tuned language model.

## Components

- `analyzer.py` - command-line analysis entry point
- `model_service.py` - model loading and inference
- `input/` - submitted code samples
- `output/` - generated analysis results
- `.github/workflows/` - GitHub Actions automation

## Usage

```bash
python analyzer.py input/sample.py
