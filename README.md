# Handwriting synthesis (PyTorch)

Recurrent-neural-network **handwriting prediction** and **conditional handwriting synthesis** in Python, following Alex Graves’s sequence-generation formulation ([*Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850)).

This fork keeps the original model and sampling behavior, adds a **JSON-driven run layout** (`runs/…`), optional **config files** for `prepare_data` / `train`, and a **Tkinter inference GUI** with animated drawing, scrolling, and stop/cancel controls.

---

## Features

| Area | What you get |
|------|----------------|
| **Training** | Synthesis network (text → strokes) or unconditional prediction network; resume from checkpoints; periodic samples under configurable biases. |
| **Data** | Built-in **IAM On-Line** provider; plug-in **custom** providers via `handwriting_synthesis.data_providers.custom`. |
| **CLI** | `prepare_data`, `train`, `synthesize`, `sample` (root shims call `scripts/`). |
| **GUI** | `inference_gui.py` — live stroke animation, colors, scrollable canvas, collapsible settings, **Stop** between lines (cannot interrupt a single `sample_means` forward pass without changing the model). |

---

## Repository layout

```
handwriting synthesis/
├── configs/                 # Example JSON for prepare_data / train
├── data/                    # Prepared HDF5 + charset (after prepare_data)
├── dataset/                 # Typical IAM-OnDB extract location (configure paths)
├── runs/                    # Training runs: checkpoints/, samples/, logs/, config
├── scripts/                 # Runnable entrypoints (prepare_data, train, synthesize, sample, inference_gui)
├── src/
│   ├── handwriting_synthesis/
│   │   ├── data_providers/  # iam, custom, registry
│   │   ├── inference/       # HandwritingSynthesizer, UnconditionalSampler
│   │   ├── model/           # SynthesisNetwork, MDN, sampling loops
│   │   ├── engine/          # Trainer / config integration
│   │   └── …
│   └── iam_ondb/            # IAM-OnDB helpers (when using `iam` provider)
├── train.py                 # → scripts/train.py
├── prepare_data.py          # → scripts/prepare_data.py
├── synthesize.py            # → scripts/synthesize.py
├── sample.py                # → scripts/sample.py
├── inference_gui.py         # → scripts/inference_gui.py
├── requirements.txt
└── README.md
```

---

## Requirements

- **Python** 3.10+ recommended (3.8+ may work; match your PyTorch build).
- **PyTorch** with CUDA optional (training and sampling use `auto` / `cpu` / `cuda` where supported).

Pinned-style constraints live in `requirements.txt` (e.g. `torch < 2`, `Pillow < 10`). Adjust in a venv if your stack needs newer majors.

---

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

All commands below assume the venv is active and the **current working directory is the repo root**. The root `*.py` files prepend `src/` to `sys.path` so you do **not** need `pip install -e .` for local runs.

---

## Data preparation

### Option A — JSON config

Example: `configs/prepare_default.json` (edit `iam_home` / `provider_args` to match your IAM-OnDB tree).

```bash
python prepare_data.py --config configs/prepare_default.json
```

This writes `train.h5`, `val.h5`, and `charset.txt` under the `prepared_data_dir` from the config (default `data/`).

### Option B — CLI arguments

```bash
python prepare_data.py data iam 9500 0 /path/to/iam_ondb_root -l 700
```

- **`data`**: output directory for HDF5 + charset.  
- **`iam`**: registered provider name.  
- **Provider args**: passed to the IAM factory (see `handwriting_synthesis.data_providers.iam_ondb`).  
- **`-l`**: max sequence length (points); `0` lets the toolkit estimate from the training split.

### IAM-OnDB layout

Download [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) and point the provider at a directory that contains the usual subtrees (`lineStrokes-all`, `ascii-all`, etc.), as expected by the `iam` provider.

### Custom datasets

Implement a provider in `src/handwriting_synthesis/data_providers/custom.py`, register it on the provider registry (`name` attribute + factory pattern). See inline patterns in that module and in `scripts/prepare_data.py` error text for requirements.

---

## Training

### Option A — JSON config (creates `runs/run_xxx/`)

```bash
python train.py --config configs/train_default.json
```

Optional overrides:

```bash
python train.py --config configs/train_default.json --device cuda --run_id run_001
```

The run directory typically contains:

- `checkpoints/Epoch_N/` — each with `model.pt` + `meta.json`  
- `samples/` — periodic PNGs during training  
- `logs/` — training logs  
- saved copy of the effective config  

### Option B — Legacy positional arguments

```bash
python train.py data runs/run_001/checkpoints -b 32 -e 50 -i 300
```

Use **either** `--config` **or** `data_dir model_dir` (see `python train.py --help`).

---

## Checkpoints for inference

A usable synthesis checkpoint is a **directory** (not a single file), for example:

```text
runs/run_001/checkpoints/Epoch_60/
├── model.pt
└── meta.json
```

`HandwritingSynthesizer.load(path, device, bias)` reads both files. Bias is applied in the mixture-density layer (see Graves / toolkit docs), not a separate “temperature.”

---

## CLI — synthesize (conditional)

```bash
python synthesize.py runs/run_001/checkpoints/Epoch_60 "Your line of text  " -b 0.9 --samples_dir samples --trials 1
```

Useful flags (see `--help`): `--bias`, `--trials`, `--thickness`, `--show_weights`, `--heatmap`, `--output_file_type` (`png` / `svg`).

---

## CLI — sample (unconditional)

```bash
python sample.py path/to/ucheckpoint_dir usamples --trials 3 -b 0.5
```

Uses the **prediction** network checkpoint layout from the upstream toolkit naming (`ucheckpoints/…` in older docs).

---

## Inference GUI (animated)

Tkinter app that loads checkpoints like `synthesize.py`, runs the same encode → `sample_means` → denormalize path, then draws strokes on a **scrollable** canvas (no scale-to-fit clipping of wide pages).

### Run

```bash
python inference_gui.py
# or
python inference_gui.py /path/to/Epoch_60 --device cuda
```

### Behaviour (high level)

- **Text**: multi-line input; long paragraphs are split into **multiple model calls** using **word-boundary–only** wrapping (`Max chars/line`). A word is never split across two lines; very long words get their own line.  
- **Suffix**: after splitting, **each segment** is sent to the model with **two trailing spaces** (`  `), similar to line/sentinel cues used elsewhere in handwriting pipelines.  
- **Settings**: collapsible panel beside the text field (values stay active when collapsed).  
- **Status**: `Generating…` during sampling; `Typing…` / `Drawing…` during animation; **Generate** is disabled until idle or **Stop**.  
- **Stop**: invalidates the current run; sampling stops between lines/trials; a single forward pass inside PyTorch may still finish before the thread exits.

---

## Configuration files

| File | Purpose |
|------|---------|
| `configs/prepare_default.json` | `prepare_data.py --config …` |
| `configs/train_default.json` | `train.py --config …` |

Copy and edit for your paths, epochs, batch size, and device.

---

## References

1. Alex Graves — *Generating Sequences With Recurrent Neural Networks* — [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)  
2. IAM-OnDB — [IAM On-Line English Sentence Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)  
3. Upstream toolkit and demo (related project): [pytorch-handwriting-synthesis-toolkit](https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit) and [demo site](https://x-raylaser.github.io/pytorch-handwriting-synthesis-toolkit/).

---

## License / attribution

Model and training code follow the lineage of the Graves formulation and community implementations. If you publish work built on this repo, cite Graves (2013) and the IAM-OnDB paper as appropriate for your data and experiments.
