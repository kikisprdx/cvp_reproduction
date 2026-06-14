# CVP Reproduction

Reproduction of Contrastive Visual Prompting (CVP) and related test-time adaptation methods on CIFAR-10-C.

## Setup

```bash
poetry install
```

## Usage

### Phase 1 — SSL pretraining (offline)

Trains the SSL MLP head on clean CIFAR-10 using contrastive loss. Saves weights to `models/ssl_weights.pth`.

```bash
poetry run python -m pipeline_phases.evaluate --mode training
```

### Phase 2 — Baseline evaluation

Evaluates the frozen ResNet-26 directly on corrupted CIFAR-10-C (no adaptation).

```bash
poetry run python -m pipeline_phases.evaluate --mode testing --model baseline
```

### Phase 3 — Test-time adaptation

```bash
# CVP
poetry run python -m pipeline_phases.evaluate --mode testing --model CVP

# SVP
poetry run python -m pipeline_phases.evaluate --mode testing --model SVP

# Fine-tuning
poetry run python -m pipeline_phases.evaluate --mode testing --model FT
```

SVP checkpoints to `models/svp/svp_entire.pth` after training — subsequent runs load it automatically.

## Models

- `models/best_resnet26.pth` — pretrained ResNet-26 backbone (required, not included)
- `models/ssl_weights.pth` — SSL MLP head weights (produced by phase 1)
- `models/svp/svp_entire.pth` — SVP model (produced by SVP test-time run)

## Known Paper Discrepancies

**CVP λ range (CIFAR-10-C):** page 5 body text states λ ∈ [0.5, 3] for CIFAR-10-C and [0.5, 1] for ImageNet-C. Table 11 (page 17) shows the opposite. This reproduction follows the body text — λ clamped to [0.5, 3.0] — as it is the more explicit statement and consistent with `details.md`.

**CVP pixel clamping:** the paper does not specify clamping the adapted image `x + λ·Conv(x,k)` to [0,1], likely because the original implementation uses mean/std normalised images where the sharpening kernel stays bounded. This repo trains the backbone on raw [0,1] images (no normalisation), so `CVPHead.forward` clamps output to [0,1] to keep inputs in-distribution for the backbone.
