# S3Bert for [AncientGreekBERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)

## Setup

To get started, make the helper script executable, create a virtual environment with `uv`, and install the core dependencies:

```bash
# Make the main helper script executable
chmod +x run.sh

# Create and activate a virtual environment using uv
uv venv
source .venv/bin/activate

# Install the required Python packages (with CUDA 12.6 backend for PyTorch)
uv pip install \
  torch==2.9.1 \
  transformers==4.57.1 \
  sentence-transformers==5.1.2 \
  numpy==2.3.5 \
  scipy==1.16.3 \
  --torch-backend=cu126
```

You can then use the `run.sh` script as a simple entry point:

- `./run.sh prepare` – run the data preparation pipeline
- `./run.sh train` – train the model (requires `./run.sh prepare`)
- `./run.sh infer` – run inference using a trained model
