# Ancient Greek Variant S3BERT (Gender)

An S3BERT-style embedding model for **Ancient Greek biblical verse similarity**, with an explicit **"Gender" feature sub-vector**.

S3BERT aims to add **interpretability** to SBERT-style sentence embeddings by learning a **decomposition** of the embedding into
fixed-size, explainable **feature sub-embeddings** (plus a residual). In the original approach, these feature subspaces are trained
to approximate interpretable metrics (e.g., from meaning representations such as AMR), while a second objective enforces **consistency**
with the similarity behavior of an SBERT teacher model so overall embedding quality is preserved.

Paper: https://arxiv.org/abs/2206.07023

In this repo the only modeled feature is:

- `Gender` (one feature sub-vector; see `src/config.py`)

Base model: Ancient Greek Variant SBERT ([Paulanerus/AncientGreekVariantSBERT](https://huggingface.co/Paulanerus/AncientGreekVariantSBERT))

Most of the S3BERT-specific training code (custom losses/evaluators, prediction helpers, etc.) is adapted from the original implementation:

- https://github.com/flipz357/S3BERT

---

## Repository

This repo contains:

- a **data preparation pipeline** (`src/prepare.py`) that derives verse-level gender labels and generates train/dev/eval pair CSVs
- a **training script** (`src/train.py`) that fine-tunes an S3BERT-style model with distillation/consistency losses
- a small **inference script** (`src/infer.py`) that prints similarity scores for sentence pairs (global + `Gender`)
- a convenience entrypoint script `run.sh`

---

## Setup

Make the helper script executable, create a virtual environment, and install dependencies:

```bash
# Make the main helper script executable
chmod +x run.sh

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages (example CUDA build; adjust to your system)
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

You can then use `run.sh`:

- `./run.sh prepare` – data prep (required before training)
- `./run.sh train` – fine-tune the model
- `./run.sh infer` – run inference using the trained model in `model/`

---

## Data preparation (`src/prepare.py`)

### Dataset

Download the raw CSV files from [Zenodo](https://zenodo.org/records/15789063):

- `verses.csv`
- `occurrences.csv`
- `words.csv`

Place them in `data/`:

- `data/verses.csv`
- `data/occurrences.csv`
- `data/words.csv`

Then run:

```bash
./run.sh prepare
```

This creates:

- `data/temp_verses.csv` (intermediate verse table with derived `sentence_gender`)
- `data/pairs/{train,dev,eval}_pairs.csv` (training pairs with labels; see `src/config.py` for paths)

### What `prepare.py` does

TODO.

---

## Training (`src/train.py`)

Training follows the S3BERT idea: keep an SBERT backbone, but dedicate a small fixed-size subspace to an interpretable metric.

In this repo:

- the S3BERT initialization is `Paulanerus/AncientGreekVariantSBERT` (student + teacher)
- the student is partially unfrozen (see `src/utils/model_freeze.py` usage in `src/train.py`)
- the only modeled feature is `Gender` with `N=1` and `FEATURE_DIM=16` (see `src/config.py`)
- the model is optimized with two objectives on the same batches:
  - a **consistency loss** with a frozen teacher
  - a **distillation loss** that trains the metric sub-vector

Run:

```bash
./run.sh train
```

Outputs:

- model artifacts written to `model/` (see `SBERT_SAVE_PATH` in `src/config.py`)

---

## Inference (`src/infer.py`)

`infer.py` loads the trained model from `model/`, encodes example sentence lists, and uses `src/utils/prediction_helpers.py` to print:

- the **global** cosine similarity (full embedding)
- the **Gender** cosine similarity (the first feature sub-vector)

Run:

```bash
./run.sh infer
```

---

## Acknowledgments

This work builds on:

- the SentenceTransformers ecosystem: https://www.sbert.net
- the original S3BERT implementation: https://github.com/flipz357/S3BERT

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) 513300936.
