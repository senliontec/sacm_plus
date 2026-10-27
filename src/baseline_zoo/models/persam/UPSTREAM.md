# PerSAM (Personalize-SAM) upstream record

## Source

- Repository: `ZrrSkywalker/Personalize-SAM`
  (https://github.com/ZrrSkywalker/Personalize-SAM)
- HEAD: `a7e87245156e0b8efcba10a15a745e699c679d9d` (2023-08-16).
- Local reference: `3rd/Personalize-SAM`.
- License: MIT (Copyright (c) 2022 Renrui Zhang), copied as `LICENSE.txt`.
  The SAM-derived modules retain Meta Platforms' source headers.

## Copied files

From `3rd/Personalize-SAM/per_segment_anything/`:

- `build_sam.py`, `predictor.py`
- `modeling/{common,image_encoder,mask_decoder,prompt_encoder,
  transformer,sam,tiny_vit_sam}.py`
- `utils/transforms.py`
- `LICENSE.txt`

The package is the authors' modified SAM fork: the predictor accepts a
reference mask in `set_image`, and the transformer/mask decoder consume
`attn_sim` and `target_embedding` for target-guided attention and
target-semantic prompting.  The official PerSAM driver is
`3rd/Personalize-SAM/persam.py`.

## Modifications

- `per_segment_anything/__init__.py` is reduced to the SAM builders and
  `SamPredictor`; the automatic mask generator and ONNX/predictor utility
  imports are omitted.
- `build_sam.py` loads checkpoints with
  `torch.load(..., map_location="cpu", weights_only=False)`.
- The official `persam.py` path/filename/plotting driver is replaced by
  `algorithm.py` + the baseline adapter.  The official inference sequence
  is preserved: reference encoding and mean object embedding, cosine
  similarity, top-1 foreground and least-similar background points,
  target-guided attention, the first mask, mask-input refinement, and the
  box/mask-input second refinement.

## Official recipe preserved by the adapter

- Training-free.  No optimizer, loss, epochs or method-specific weights.
- Reference (support) image and its binary mask are required.  The official
  driver reads the mask as a three-channel image; this port converts the
  binary GT mask to RGB 0/255 before SAM normalization, which is required
  for the foreground feature to be found.
- SAM ViT-H (`sam_vit_h_4b8939.pth`), original image resolution output,
  mask threshold logit 0.0 / probability 0.5.
- The mean object feature is L2-normalized for cosine similarity, while the
  raw mean is passed to target-semantic prompting, matching
  `3rd/Personalize-SAM/persam.py:90-92`.
- The internal positive/negative points and the refinement box are
  generated automatically from the support and target similarity map; no
  target GT or target prompts are used.

## Deviations in this port

- PerSAM is one-shot.  For the project's 3-shot protocol, each support image
  is run independently and the three binary masks are majority-voted
  (threshold 0.5); this is a 3-shot extension, not an official PerSAM mode.
  With `--support-shots 1` the result is the official one-shot prediction.
- Seen datasets use their three training shots as supports; unseen/novel
  datasets use a structurally matched donor domain from the training split
  (the shared structural donor map).
- The official driver assumes fixed `00.jpg`/`00.png` filenames and a data
  directory per object; the port consumes the shared `Sample` records.
- Empty first-pass masks skip the refinement stages and contribute an empty
  vote; the official code would fail on `x.min()` for an empty mask.
- The official PerSAM-F two-parameter fine-tuned variant is not used; this
  is the training-free PerSAM method.
- No official PerSAM method checkpoint exists; only the SAM ViT-H base is
  loaded.
- No test-time augmentation.
