"""Official prompt-free decoder call (synthetic calibration case)."""
import torch


def decode(model, image_embeddings):
    batch = image_embeddings.shape[0]
    sparse = torch.zeros(
        (batch, 0, 256), dtype=torch.float,
        device=image_embeddings.device)
    dense = model.no_mask_embed.weight.reshape(
        1, -1, 1, 1).expand(batch, -1, 64, 64)
    return model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
