"""Ported prompt-free decoder call (synthetic calibration case)."""
import torch

# Learned prompt parameters of the port; these do not exist upstream.
LEARNED_SPARSE = torch.nn.Parameter(torch.zeros(1, 4, 256))
LEARNED_DENSE = torch.nn.Parameter(torch.zeros(1, 256, 64, 64))


def decode(model, image_embeddings):
    batch = image_embeddings.shape[0]
    sparse = LEARNED_SPARSE.expand(batch, -1, -1)
    dense = LEARNED_DENSE.expand(batch, -1, 64, 64)
    return model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
