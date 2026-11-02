"""Stable type identity for the typed Stage-2 heads.

Slot i in MaskDecoder always uses this type, even when Stage-1 gating would
otherwise reorder the heads.  The names are consumed by the trainer's loss
dispatch; the model only needs the zero-initialized type embeddings.
"""

TYPED_HEAD_NAMES = ('thin', 'blob', 'boundary', 'recall')
