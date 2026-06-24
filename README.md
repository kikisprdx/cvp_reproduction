# CVP Reproduction

Reproduction of Contrastive Visual Prompting (CVP) and related test-time adaptation methods on CIFAR-10-C.

## Installation

### Poetry

```bash
poetry install
```

### pip

```bash
pip install -r requirements.txt
```

## Usage

### Phase 1 — SSL pretraining (offline)

Trains the SSL MLP head on clean CIFAR-10 using contrastive loss. Saves weights to `results/ssl_weights.pth`.

```bash
# Poetry
poetry run python -m src.pipeline_phases.evaluate --mode training
# pip
python -m src.pipeline_phases.evaluate --mode training
```

### Phase 2 — Baseline evaluation

Evaluates the frozen ResNet-26 directly on corrupted CIFAR-10-C (no adaptation).

```bash
poetry run python -m src.pipeline_phases.evaluate --mode testing --model baseline
```

### Phase 3 — Test-time adaptation

```bash
poetry run python -m src.pipeline_phases.evaluate --mode testing --model CVP-F3
poetry run python -m src.pipeline_phases.evaluate --mode testing --model CVP-R3
poetry run python -m src.pipeline_phases.evaluate --mode testing --model SVP-Patch
poetry run python -m src.pipeline_phases.evaluate --mode testing --model SVP-Pad
poetry run python -m src.pipeline_phases.evaluate --mode testing --model FT
poetry run python -m src.pipeline_phases.evaluate --mode testing --model PFT
```

## Models

- `results/best_resnet26.pth` — pretrained ResNet-26 backbone (required, not included)
- `results/ssl_weights.pth` — SSL MLP head weights (produced by phase 1)

## Known Paper Discrepancies

**CVP λ range (CIFAR-10-C):** page 5 body text states λ ∈ [0.5, 3] for CIFAR-10-C and [0.5, 1] for ImageNet-C. Table 11 (page 17) shows the opposite. This reproduction follows the body text — λ clamped to [0.5, 3.0] — as it is the more explicit statement and consistent with `details.md`.
