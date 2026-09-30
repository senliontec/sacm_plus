"""Shared model I/O: the prompt-free forward protocol.

Centralizes the encoder call, the (image_embeddings, intermediate,
adapter_features) tuple unpacking and the decoder call, so the model's
output contract lives in exactly ONE place. Used by the Trainer, the
Evaluator and diagnostics.
"""


def predict_all(sam, images, sparse_embeddings, dense_embeddings, return_stage1=True):
    """Run the full prompt-free forward pass.

    Args:
        sam: the Sam model. `images` must already be preprocessed
            (spec.preprocess applied by the caller).
        images: [B, 3, 1024, 1024] normalized input.
        sparse_embeddings / dense_embeddings: prompt-free anchors
            [B, 0, 256] / [B, 256, 64, 64] (see get_prompt_embeddings).
        return_stage1: also return the gating-reordered Stage-1 masks.

    Returns:
        (masks_stage2 [B, 4, 256, 256], iou_pred [B, 4]) when
        return_stage1=False, otherwise (masks_stage2, iou_pred,
        masks_stage1 [B, 4, 256, 256]).
    """
    encoder_output = sam.image_encoder(images, return_features=True)
    if len(encoder_output) == 3:
        image_embeddings, intermediate_features, adapter_features = encoder_output
    else:
        image_embeddings, intermediate_features = encoder_output
        adapter_features = None

    return sam.mask_decoder.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        adapter_features=adapter_features,
        intermediate_features=intermediate_features,
        return_stage1=return_stage1,
    )


def get_prompt_embeddings(sam, batch_size):
    """Prompt-free anchors from the frozen prompt encoder: the learnable
    no_mask_embed dense prompt and the empty sparse prompt, expanded to
    the ACTUAL batch size (this also fixes the shape mismatch on a
    shorter last batch)."""
    sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
    sparse = sparse.expand(batch_size, 0, 256)
    dense = dense.expand(batch_size, -1, -1, -1)
    return sparse, dense
