# Neural methods

This package contains independent, from-scratch PyTorch implementations of five
published anomaly detectors. The code keeps the algorithms' model and scoring
semantics, while using AgentAD's common tensor convention:

```text
input series:   [batch, time, feature]
point scores:   [batch, time]
```

The methods are intentionally separate from dataset loading and metric evaluation.
Use `SeriesData` to obtain arrays, create model-specific training windows, and pass
the resulting tensors to a method. Install the optional dependency first:

```bash
uv sync --extra methods
```

| Package | Training signal | Score |
|---|---|---|
| `CrossAD` | cross-scale reconstruction plus context quantization | summed, upsampled reconstruction error |
| `DADA` | released masked reconstruction/MoE components | variance across complementary masked reconstructions |
| `KanAD` | next-value forecasting | absolute forecast error |
| `PaAno` | local-positive triplet and temporal pretext losses | top-k cosine distance to a normal patch bank |
| `ScatterAD` | scattering, temporal smoothness, cross-view consistency | inverse center distance plus temporal inconsistency |

Each package exports a frozen dataclass config and one `nn.Module`. `CrossAD`,
`DADA`, `PaAno`, and `ScatterAD` expose `score(...)`; KAN-AD exposes
`score(windows, targets)` because its published criterion is a one-step forecast
error rather than a reconstruction at every point in the input window.

DADA's public repository does not disclose its pretraining procedure. This package
does not invent a replacement objective. It implements the released architecture,
masked-copy inference, adaptive bottleneck, normal/adversarial decoders, gradient
reversal, and `load_reference_checkpoint(...)` for the published checkpoint.

Algorithm references and source material used for semantic verification:

- CrossAD, NeurIPS 2025, `decisionintelligence/CrossAD`.
- DADA, ICLR 2025, `iambowen/DADA`.
- KAN-AD, ICML 2025, `CSTCloudOps/KAN-AD`.
- PaAno, ICLR 2026, `jinnnju/PaAno`.
- ScatterAD, `jk-sounds/ScatterAD`.
