# SAM2-UNet (appendix extended baseline)

- Upstream: https://github.com/WZH0120/SAM2-UNet
- Vendored snapshot: `3rd/SAM2-UNet` (git HEAD `01598e5`)
- License: Apache-2.0 (`LICENSE` copied from upstream)
- Role: appendix / extended baseline, not one of the SACM paper's ten
  baselines and not in the main table.

## Files copied

| Upstream | Destination |
|---|---|
| `SAM2UNet.py` | `model.py` (device argument + import rewrite) |
| `sam2/__init__.py` | `sam2/__init__.py` (Hydra init removed) |
| `sam2/build_sam.py` | `sam2/build_sam.py` (rewritten) |
| `sam2/modeling/**` | `sam2/modeling/**` (imports rewritten) |
| `sam2/utils/misc.py` | `sam2/utils/misc.py` (trimmed to `get_sdpa_settings`) |
| `sam2_configs/sam2_hiera_l.yaml` | `sam2_configs/sam2_hiera_l.yaml` (targets rewritten) |
| `LICENSE` | `LICENSE` |

## Files intentionally omitted

- `sam2/automatic_mask_generator.py`, `sam2/sam2_image_predictor.py`,
  `sam2/sam2_video_predictor.py`, `sam2/utils/amg.py`,
  `sam2/utils/transforms.py` — not used by SAM2-UNet.
- `sam2/_C.pyd`, `sam2/csrc/` — Windows/CUDA extension not needed.
- Other Hiera configs (`t`, `s`, `b+`) — SAM2-UNet is hard-coded to Hiera-L.

## Modifications

1. **Package-internal imports.**  All `from sam2...` imports were rewritten
   to `baseline_zoo.models.sam2_unet.sam2...`; `model.py` uses
   `from .sam2.build_sam import build_sam2`.
2. **No global Hydra state.**  `sam2/__init__.py` no longer calls
   `initialize_config_module("sam2_configs")`.  `build_sam.py` loads
   `sam2_configs/sam2_hiera_l.yaml` with `OmegaConf.load` and uses the
   stateless `hydra.utils.instantiate`.
3. **Hydra targets.**  Every `_target_` in the YAML was prefixed with
   `baseline_zoo.models.sam2_unet.`.  The three
   `sam_mask_decoder_extra_args` overrides injected by upstream
   `build_sam2` are materialized in the YAML instead.
4. **CPU-safe construction.**  `SAM2UNet.__init__` accepts
   `device="cpu"` and passes it to `build_sam2`; the runtime moves the model
   to the requested device.  `torch.load` uses `map_location="cpu"`.
5. **Checkpoint loading.**  `torch.load(..., weights_only=False)` is explicit
   (SAM2 checkpoints are dictionaries), and key mismatches raise a readable
   error.  Use the original SAM2 Hiera-L checkpoint, not SAM2.1.
6. **Trimmed utilities.**  `sam2/utils/misc.py` keeps only
   `get_sdpa_settings`; the connected-component helper lazily imported the
   omitted compiled extension.
7. **No `model.train()` bug.**  Upstream `train.py` never called
   `model.train()`, leaving BatchNorm/dropout in eval mode.  The shared
   `baseline_zoo.runtime` calls `model.train()` each epoch; this is documented
   in the adapter deviations.

## Preserved logic

Frozen Hiera-L trunk, per-block `Adapter` with a trainable `prompt_learn`
MLP, RFB blocks, U-Net decoder, three logits (4x/8x/16x), weighted
BCE+IoU `structure_loss`, AdamW 1e-3 / 5e-4, cosine schedule, and 352x352
input.  The upstream `up4` module remains constructed but unused.

## Checkpoint

`checkpoints/sam2_hiera_large.pt` (original SAM2, not SAM2.1):

```text
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

Training requires it; `--random-init` is a code-smoke-only escape hatch.

## Runtime consumer

`baseline_zoo.adapters.sam2_unet.SAM2UNetAdapter`
