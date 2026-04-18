# PepBridge

**PepBridge for peptide-bridged, unified and structure-aware modeling of pMHC-TCR recognition**

## Overview

PepBridge is a peptide-bridged, pair-aware deep learning framework for **unified modeling of the pMHC-TCR recognition cascade**. PepBridge jointly models multiple related tasks and incorporates **structure-aware supervision** through residue-level contact and distance signals.

PepBridge supports:

- peptide-MHC binding prediction
- peptide-TCR binding prediction
- MHC-peptide-TCR binding prediction
- epitope immunogenicity prediction
- residue-level contact and distance map prediction for selected interfaces

This repository provides the model code, inference pipeline, and training utilities for applying PepBridge to pMHC-TCR related prediction tasks.

## Model framework

<p align="center">
  <img src="doc/figure1.png" alt="PepBridge framework" width="500">
</p>

---

## 🚀 Quick Start (End-to-End Test)

If you want to quickly test the model on a fresh system, follow these 4 steps:

**1. Install Git LFS & Clone**
```bash
# Ensure git-lfs is installed in your system (Ubuntu/Debian example)
sudo apt-get update && sudo apt-get install -y git-lfs

git lfs install
git clone https://github.com/aapupu/PepBridge.git
cd PepBridge
git lfs pull
```

**2. Setup Environment**
```bash
conda create -n pepbridge python=3.10 -y
conda activate pepbridge

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 scipy==1.13.1 \
    matplotlib==3.8 einops==0.8
```

**3. Prepare Example Data**
```bash
cat <<EOF > example.csv
MHC,peptide,v_gene,cdr3
HLA-A24:02,QLPRLFPLL,TRBV7-9,CASSLHHEQYF
HLA-A02:01,LLFGYPVYV,TRBV5-1,CASRPGLMSAQPEQYF
EOF
```

**4. Run Multi-task Inference**
```bash
python infer.py task=mp,pt,mpt,imm,mp_contact,pt_contact input_csv=example.csv out_dir=./results save_dist=true
```

Upon completion, predictions are saved in `./results/example_pred.csv`, with prediction probabilities (values [0, 1]) where closer to 1.0 indicates a higher likelihood of binding/immunogenicity.

---

## Download (Detailed)

You can clone the repository with:

```bash
git clone https://github.com/aapupu/PepBridge.git
cd PepBridge
```

### Important note about `esm_emb_HLAI.pkl`

The file `esm_emb_HLAI.pkl` is larger than 100 MB and may not be included in a normal GitHub clone.

You have **two options**:

**Option 1: Download with Git LFS (Recommended)**

Ensure the `git-lfs` system package is installed (e.g., `apt-get install git-lfs`). Install Git LFS inside git, clone the repo, and pull LFS files:

```bash
git lfs install
git clone https://github.com/aapupu/PepBridge.git
cd PepBridge
git lfs pull
```

**Option 2: Download manually from Zenodo**

If you do not use Git LFS, download `esm_emb_HLAI.pkl` separately from **Zenodo** and place it into the `doc/` folder manually:

```text
PepBridge/
└── doc/
    └── esm_emb_HLAI.pkl
```

> Zenodo: https://doi.org/10.5281/zenodo.19632906

---

## Environment setup

> The package list below is a suggested template and can be adjusted later according to your local environment.

We recommend using **Python 3.10** or **Python 3.11** with CUDA-enabled PyTorch. Please ensure your PyTorch build matches your local CUDA version (e.g. cu118 / cu121).

### Create a conda environment

```bash
conda create -n pepbridge python=3.10 -y
conda activate pepbridge
```

### Suggested packages and versions

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 scipy==1.13.1
pip install matplotlib==3.8 einops==0.8
```

If your local setup requires additional packages, such as ESM or other model-specific dependencies, please install them separately.

---

## Inference

PepBridge supports the following inference tasks:

- `mp`: peptide-MHC binding prediction
- `pt`: peptide-TCR binding prediction
- `mpt`: MHC-peptide-TCR binding prediction
- `imm`: epitope immunogenicity prediction
- `mp_contact`: peptide-MHC residue-level contact prediction
- `pt_contact`: peptide-TCR residue-level contact prediction

It supports running **one or multiple tasks** in a single command.

### Inference features

- Binding-related outputs are merged into **one CSV** (`<input_name>_pred.csv`).
- Output values (`pred_mp_binding`, etc.) output a float [0, 1]. Higher scores mean higher predicted likelihood of interaction.
- Contact predictions are saved as **per-sample matrices** in corresponding folders.

---

## Input CSV format

The inference script accepts flexible input column names and normalizes them automatically.

### Accepted aliases

- `MHC`, `HLA` → `MHC`
- `peptide`, `epitope` → `peptide`
- `cdr3`, `cdr3b` → `cdr3`
- `v_gene`, `trbv`, `bv`, `tcrbv` → `v_gene`

For HLA-I related tasks, `MHC` will be automatically mapped to pseudo-MHC sequences using `doc/pseudo_HLAI.csv` (Ensure your MHC alleles conform to standard nomenclatures like `HLA-A02:01`).

### Required columns by task

#### `mp` & `imm` & `mp_contact`
- `MHC`
- `peptide`

#### `pt` & `pt_contact`
- `peptide`
- `cdr3`

#### `mpt`
- `MHC`
- `peptide`
- `cdr3`
- `v_gene`

### Example input CSVs

#### For MP / IMM / MP contact

```csv
MHC,peptide
HLA-A24:02,QLPRLFPLL
HLA-A02:01,LLFGYPVYV
```

#### For PT / PT contact

```csv
peptide,cdr3
QLPRLFPLL,CASSLHHEQYF
LLFGYPVYV,CASRPGLMSAQPEQYF
```

#### For MPT or mixed multi-task input

A single CSV can contain all required columns:

```csv
MHC,peptide,v_gene,cdr3
HLA-A24:02,QLPRLFPLL,TRBV7-9,CASSLHHEQYF
HLA-A02:01,LLFGYPVYV,TRBV5-1,CASRPGLMSAQPEQYF
```

---

## Running inference

### Single-task inference

```bash
python infer.py task=mp input_csv=example.csv out_dir=./results
```

### Multi-task inference

```bash
python infer.py task=mp,pt,mpt,mp_contact,pt_contact,imm input_csv=example.csv out_dir=./results
```

---

## Optional arguments

### `batch_size` / `contact_batch_size`
Adjust the batch sizes depending on your GPU VRAM:
```bash
python infer.py task=mp,pt,mpt,imm input_csv=example.csv batch_size=32 contact_batch_size=8
```

### `save_dist`
Whether to save predicted distance matrices for contact tasks:
```bash
python infer.py task=mp_contact input_csv=example.csv save_dist=true
```

### `use_lora` / `path` / `paths`
LoRA adapter integration and manual checkpoint paths:
```bash
python infer.py task=mp input_csv=example.csv use_lora=true
# OR specify path manually
python infer.py task=mp input_csv=example.csv path=./doc/checkpoints_multi_lora_align3_ln
```

---

## Output structure

Assume running:

```bash
python infer.py task=mp,pt,mpt,imm,mp_contact,pt_contact input_csv=example.csv out_dir=./results save_dist=true
```

The outputs will look like:

```text
results/
├── example_pred.csv                     # Merged binding predictions for all binding tasks
├── mp_contact/
│   ├── <pseudoMHC>_<peptide>_site.csv   # Predicted contact probability matrix
│   ├── <pseudoMHC>_<peptide>_dist.csv   # Predicted distance matrix (if save_dist=true)
│   └── ...
└── pt_contact/
    ├── <peptide>_<cdr3>_site.csv
    ├── <peptide>_<cdr3>_dist.csv
    └── ...
```

### Binding output CSV

All binding-related tasks are merged into `example_pred.csv`, with added columns such as:
- `pred_mp_binding`
- `pred_pt_binding`
- `pred_mpt_binding`
- `pred_immunogenicity`

### Contact output matrices

#### `mp_contact`
Rows correspond to pseudo-MHC sequences. Columns correspond to peptide sequences.

#### `pt_contact`
Rows correspond to peptide residues. Columns correspond to CDR3 residues.

*Note: All contact matrices are cropped to the true sequence length.*

---

## Citation

PepBridge for peptide-bridged, unified and structure-aware modeling of pMHC-TCR recognition  
Wenpu Lai, Yang Li, Oscar Junhong Luo

---

## Contact

For issues, suggestions, or collaboration, please use the GitHub repository issue tracker or contact by e-mail: **kyzy850520@163.com**