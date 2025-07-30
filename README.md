# amplify_onnx_inference

![Tests](https://github.com/WHOIGit/amplify_onnx_inference/workflows/Tests/badge.svg)
![Lint](https://github.com/WHOIGit/amplify_onnx_inference/workflows/Lint/badge.svg)

ONNX-based inference system for IFCB (Imaging FlowCytobot) data analysis. This tool performs automated plankton classification on IFCB bin files using pre-trained ONNX models.

## Features

- **Flexible model support**: Works with both static and dynamic batch size ONNX models
- **Multiple data loading backends**: Supports both PyTorch and non-PyTorch data loading
- **Configurable output organization**: Choose between run-date or model-name subfolder organization
- **Directory structure preservation**: Maintains input directory hierarchies in output
- **Containerized deployment**: Docker/Podman support for consistent environments
- **GPU acceleration**: CUDA support for faster inference

## Container Use

### Basic Usage
```bash
podman build . -t onnx:latest
podman run -it --rm -e CUDA_VISIBLE_DEVICES=1 \
       --device nvidia.com/gpu=all \
       -v $(pwd)/models:/app/models \ 
       -v $(pwd)/inputs/:/app/inputs \
       -v $(pwd)/outputs:/app/outputs \
       onnx:latest models/PathToYourModel.onnx inputs/PathToBinDirectory 
```

### Optional Flags:
```bash
--batch N                              # required for models without pre-set input sizes
--classes LISTFILE                     # to add class name headers to output score-matrix csv
--outdir DIRPATH                       # directory to write files to. Default is './outputs'
--outfile FILENAME                     # filename of output csv. Default is "{RUN_DATE}/{SUBPATH}.csv"
--subfolder-type {run-date,model-name} # organize outputs by run date (default) or model name
--force-notorch                        # use non-torch datasets and dataloaders. If torch is not installed, this flag is automatically set 
```

### Output Organization Examples:

**Run-date organization (default):**
```bash
# Outputs organized by date of inference run
outputs/
├── 2025-01-15/
│   ├── D20240301T123456_IFCB999.csv
│   └── D20240301T130000_IFCB999.csv
└── 2025-01-16/
    └── D20240302T090000_IFCB999.csv
```

**Model-name organization:**
```bash
podman run ... onnx:latest --subfolder-type model-name models/my_classifier.onnx inputs/
# Outputs organized by model name
outputs/
├── my_classifier/
│   ├── D20240301T123456_IFCB999.csv
│   └── D20240301T130000_IFCB999.csv
└── another_model/
    └── D20240302T090000_IFCB999.csv
```

## Direct Python Usage

You can also run the inference scripts directly:

```bash
# Using PyTorch data loaders (recommended)
python src/infer_ifcbbins_torch.py models/classifier.onnx data/bins/

# Using non-PyTorch data loaders
python src/infer_ifcbbins.py models/classifier.onnx data/bins/

# With custom output organization
python src/infer_ifcbbins_torch.py --subfolder-type model-name \
    --classes classes.txt models/classifier.onnx data/bins/
```

## Development

### Installing Dependencies

For production use:
```bash
pip install -r requirements.txt
```

For development (includes testing tools):
```bash
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_infer_functions.py
```

### Continuous Integration

The project includes GitHub Actions workflows that automatically:

- **Run tests** on Python 3.10, 3.11, and 3.12 when code is pushed or PRs are opened
- **Check code quality** with linting tools (flake8, black, isort)

Tests run automatically on pushes to `main` branch and on all pull requests.
