# SACM (official)

- Upstream: https://github.com/kylechuuuuu/SACM (paper code)
- Vendored snapshot: `baselines/sacm` (development reference tree)
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied: `segment_anything/` (relative imports; the only change is
  `predictor.py`'s absolute `segment_anything.modeling` import).
- Runtime consumer: `baseline_zoo.adapters.sacm_official.SACMAdapter`
