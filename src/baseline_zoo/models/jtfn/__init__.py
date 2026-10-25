"""Self-contained JTFN model package (no upstream license file)."""

from .jtfn import JTFN


def build_jtfn(backbone: str = 'base64', steps: int = 3,
               reduce_dim: bool = False) -> JTFN:
    """Build the official DRIVE configuration (base64, three refinements)."""
    return JTFN(
        backbone=backbone,
        use_gau=[True] * 5,
        use_fim=[True] * 4,
        up=[True] * 4,
        classes=1,
        steps=steps,
        reduce_dim=reduce_dim,
    )


__all__ = ['JTFN', 'build_jtfn']
