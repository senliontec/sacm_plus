# CS2Net-2D (legacy model id `cs2net`)

- Upstream: https://github.com/iMED-Lab/CS-Net
- Vendored snapshot: `3rd/CS-Net`, HEAD
  `f1f8bbb011ebabc39b3ce6b552b8a0d881e587f6`.
- License: MIT (`LICENSE`, copied verbatim)
- Files copied: `model/csnet.py`, the official 2D network used by
  `train.py`.
- 2D/3D distinction:
  - `model/csnet.py` + `train.py`: 2D channel/spatial attention network.
    This is the branch used for the 2D protocol (DRIVE, CHASEDB1, DCA1,
    CORN, etc.).
  - `model/csnet_3d.py` + `train3d.py`: separate 3D CS2-Net extension with
    Conv3d and volume inputs. It is not applicable to 2D images without a
    private depth-1 adaptation.
  - `cs2net.md` is the CS2-Net 3D paper/README, not 2D code.
- Modifications: none. The model file only imports torch.
- Runtime consumer: `baseline_zoo.adapters.cs2net.CS2NetAdapter`
