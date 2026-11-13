# Model packages

| Directory | Role | Change here when... |
|---|---|---|
| `sam_network/` | SAM network implementation: ViT encoder, prompt encoder, mask decoder, geometric adapters, Fusion v2, coarse-to-fine, multi-depth path | changing the network architecture |

Import example:

```python
from models.sam_network import build_sam_vit_l
```

`models.sam_network` is consumed directly by the baseline adapters (for
example `sam_vit_l_unet`). The former algorithm layer has been removed.

The official SACM baseline is self-contained at
`src/baseline_zoo/models/sacm_official/`; `baselines/sacm/` is only a
development reference snapshot and may be deleted. The proposed method
`src/baseline_zoo/models/sacm_gcd/` (registry id `sacm_gcd`) is the current
SACM-GCD implementation and does not replace the official entry.
