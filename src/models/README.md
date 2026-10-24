# Model packages

| Directory | Role | Change here when... |
|---|---|---|
| `sam_network/` | SAM network implementation: ViT encoder, prompt encoder, mask decoder, geometric adapters, Fusion v2, coarse-to-fine, multi-depth path | changing the network architecture |
| `sacm_x/` | SACM-X algorithm layer: registry (`specs.py`), freezing, parameter groups, preprocessing, presets (`configs.py`), diagnostics | changing training protocol, presets or registered model variants |

Import examples:

```python
from models.sam_network import build_sam_vit_l
from models.sacm_x.configs import PRESETS
```

`models/sacm_x` imports `models/sam_network`; the network package does not
import the algorithm layer. Model registry names remain `sam_l`, `sam_b`,
`sam_h`, and checkpoints are independent of these directory names.

The official SACM baseline is self-contained at
`src/baseline_zoo/models/sacm_official/`; `baselines/sacm/` is only a
development reference snapshot and may be deleted.
