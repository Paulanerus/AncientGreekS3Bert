# S3Bert for [AncientGreekBERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)

## Setup

To get started, make the helper script executable, create a virtual environment, and install the core dependencies:

```bash
# Make the main helper script executable
chmod +x run.sh

# Create and activate a virtual environment using uv
python3 -m venv .venv
source .venv/bin/activate

# Install the required Python packages (with CUDA 12.9 backend for PyTorch)
pip install torch==2.9.1+cu126 \
  transformers==4.57.1 \
  sentence-transformers==5.1.2 \
  numpy==2.3.5 \
  pandas==2.3.3 \
  scipy==1.16.3 \
  datasets==4.4.1 \
  accelerate==1.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

You can then use the `run.sh` script as a simple entry point:

- `./run.sh prepare` – run the data preparation pipeline
- `./run.sh train` – train the model (requires `./run.sh prepare`)
- `./run.sh infer` – run inference using a trained model
