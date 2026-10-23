# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified: geometric strip-convolution adapters for curvilinear
# structure segmentation.
#   - GeoAdapterModule (Adapter-I_geo): block-internal bottleneck adapter
#     with an orthogonal strip-convolution branch (k x 1, 1 x k).
#   - AdapterEGlobal: block-external adapter with SE channel gating and
#     long strip convolutions, applied on window-attention layers only
#     (the global-attention layers already mix across the whole map).

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        use_adapter: bool = True,
        adapter_dim_ratio: float = 0.1,
        use_geo_i: bool = True,
        use_geo_e: bool = True,
        geo_e_indices: Tuple[int, ...] = None,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            global_attn_indexes (list): Indexes for blocks using global attention.
            use_adapter (bool): If True, use adapter modules in each block.
            adapter_dim_ratio (float): Ratio of adapter dim to embedding dim.
            use_geo_i (bool): If True, internal adapters get a strip-convolution
                branch (Adapter-I_geo); otherwise the plain token-wise adapter.
            use_geo_e (bool): If True, external adapters become AdapterEGlobal
                (SE gate + strip convs) and are applied on window-attention
                layers only; otherwise plain adapters on all layers.
            geo_e_indices (tuple): with use_geo_e, the exact layer indices
                that get AdapterEGlobal (default: ALL window-attention
                layers). Subsets enable the layer-placement ablation
                (shallow {0-4,6-10} vs deep {12-16,18-22}).
        """
        super().__init__()
        self.img_size = img_size
        self.use_adapter = use_adapter
        self.use_geo_i = use_geo_i
        self.use_geo_e = use_geo_e
        self.geo_e_indices = set(geo_e_indices) if use_geo_e and geo_e_indices is not None else None
        self.patch_size = patch_size
        self.depth = depth
        self.global_attn_indexes = tuple(global_attn_indexes)

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_adapter=use_adapter,
                adapter_dim_ratio=adapter_dim_ratio,
                geo_adapter=use_geo_i,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        # External adapters, one per block. With use_geo_e, global-attention
        # layers hold a placeholder AdapterModule that is skipped in forward
        # (their features are already globally mixed); the window layers get
        # AdapterEGlobal whose strip convolutions provide explicit
        # cross-window propagation.
        self.external_adapters = nn.ModuleList()
        self.adapter_norm = None
        if use_adapter:
            for i in range(depth):
                is_geo_e_layer = use_geo_e and (
                    self.geo_e_indices is None or i in self.geo_e_indices
                ) and i not in self.global_attn_indexes
                if is_geo_e_layer:
                    self.external_adapters.append(
                        AdapterEGlobal(embed_dim, adapter_dim_ratio, act_layer)
                    )
                else:
                    self.external_adapters.append(
                        AdapterModule(embed_dim, adapter_dim_ratio, act_layer)
                    )
            self.adapter_norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Intermediate features for the decoder multi-depth path (collected
        # shallow -> deep: block 0, depth//3, 2*depth//3, depth-1).
        intermediate_features = []
        # Outputs of the external adapters, consumed by the fusion module.
        adapter_features = []

        # Process through blocks
        for i, blk in enumerate(self.blocks):
            # Process through block
            x = blk(x)

            # Save intermediate features for the multi-depth decoder path
            if return_features and i in [0, self.depth // 3, 2 * self.depth // 3, self.depth - 1]:
                intermediate_features.append(x)

            # Apply external adapter and store output if adapters are enabled
            if self.use_adapter:
                # With the geometric variant, only the designated window
                # layers run the external adapter (global-attention layers
                # and non-designated layers skip it).
                if self.use_geo_e and (i in self.global_attn_indexes or
                                       (self.geo_e_indices is not None and i not in self.geo_e_indices)):
                    continue
                external_adapter_out = self.external_adapters[i](self.adapter_norm(x))
                adapter_features.append(external_adapter_out)

                # Directly add the adapter output to the main branch
                x = x + external_adapter_out

        # Apply neck to get final output
        output = self.neck(x.permute(0, 3, 1, 2))

        if return_features:
            if self.use_adapter:
                return output, intermediate_features, adapter_features
            else:
                return output, intermediate_features
        else:
            if self.use_adapter:
                return output, None, adapter_features
            else:
                return output


class AdapterModule(nn.Module):
    """Adapter module for efficient fine-tuning (original token-wise form)."""

    def __init__(self, dim: int, adapter_dim_ratio: float = 0.1, act_layer: Type[nn.Module] = nn.GELU):
        """
        Args:
            dim (int): Number of input channels.
            adapter_dim_ratio (float): Ratio of adapter dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()
        self.adapter_dim = int(dim * adapter_dim_ratio)

        self.down_proj = nn.Linear(dim, self.adapter_dim)
        self.act = act_layer()
        self.up_proj = nn.Linear(self.adapter_dim, dim)

        # Initialize weights to near zero to minimize initial impact
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.act(self.down_proj(x)))


class GeoAdapterModule(nn.Module):
    """Block-internal adapter with a strip-convolution branch (Adapter-I_geo).

    out = bottleneck(x) + gamma * strip_branch(x)

    The strip branch projects to a low-dimensional space, applies two
    orthogonal depthwise strip convolutions (k x 1 then 1 x k) and projects
    back. Both the bottleneck up-projection and gamma are zero-initialized,
    so at initialization the module behaves exactly like the plain
    Adapter-I and activates gradually during training.

    The k x 1 / 1 x k decomposition gives an anisotropic "cross" receptive
    field matching the elongated support of curvilinear structures, while
    keeping the incremental parameter cost tiny (~66K per layer at k=15).
    """

    def __init__(
        self,
        dim: int,
        adapter_dim_ratio: float = 0.1,
        act_layer: Type[nn.Module] = nn.GELU,
        strip_kernel: int = 15,
        strip_dim: int = 32,
    ):
        super().__init__()
        self.adapter_dim = int(dim * adapter_dim_ratio)

        # Original bottleneck path
        self.down_proj = nn.Linear(dim, self.adapter_dim)
        self.act = act_layer()
        self.up_proj = nn.Linear(self.adapter_dim, dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Strip branch (depthwise, orthogonal directions).
        # (k, 1) is a vertical strip along the height axis; (1, k) horizontal.
        self.strip_down = nn.Conv2d(dim, strip_dim, kernel_size=1)
        self.strip_v = nn.Conv2d(
            strip_dim, strip_dim, (strip_kernel, 1),
            groups=strip_dim, padding=(strip_kernel // 2, 0),
        )
        self.strip_h = nn.Conv2d(
            strip_dim, strip_dim, (1, strip_kernel),
            groups=strip_dim, padding=(0, strip_kernel // 2),
        )
        self.strip_up = nn.Conv2d(strip_dim, dim, kernel_size=1)

        # Branch coefficient, zero-initialized: initial output == Adapter-I
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        out = self.up_proj(self.act(self.down_proj(x)))

        xt = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        s = self.strip_down(xt)
        s = self.strip_v(s)
        s = self.strip_h(s)
        s = self.strip_up(s).permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        return out + self.gamma * s


class AdapterEGlobal(nn.Module):
    """Block-external adapter with SE gating and long strip convolutions.

    out = bottleneck(x + eps_se * SE(x) + eps_strip * Strip(x))

    - SE: GAP -> dim/se_ratio -> ReLU -> dim -> Sigmoid. A channel gate that
      selects which channels encode global structure, replacing the former
      unconditional (noisy) injection.
    - Strip: two orthogonal depthwise k x 1 / 1 x k convolutions at full
      channel count, providing explicit cross-window spatial propagation —
      windowed attention layers (window_size=14) have no such connection.
    - eps_se / eps_strip: learnable scalars, zero-initialized, so the extra
      branches activate gradually. The bottleneck up-projection is also
      zero-initialized, making the whole module a no-op at initialization.
    """

    def __init__(
        self,
        dim: int,
        adapter_dim_ratio: float = 0.1,
        act_layer: Type[nn.Module] = nn.GELU,
        strip_kernel: int = 31,
        se_ratio: int = 16,
    ):
        super().__init__()
        self.eps_se = nn.Parameter(torch.zeros(1))
        self.eps_strip = nn.Parameter(torch.zeros(1))

        hidden = max(dim // se_ratio, 8)
        self.se = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

        # (k, 1): vertical strip along the height axis; (1, k): horizontal
        self.strip_v = nn.Conv2d(
            dim, dim, (strip_kernel, 1), groups=dim, padding=(strip_kernel // 2, 0)
        )
        self.strip_h = nn.Conv2d(
            dim, dim, (1, strip_kernel), groups=dim, padding=(0, strip_kernel // 2)
        )

        self.adapter_dim = int(dim * adapter_dim_ratio)
        self.down_proj = nn.Linear(dim, self.adapter_dim)
        self.act = act_layer()
        self.up_proj = nn.Linear(self.adapter_dim, dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        gate = self.se(x.mean(dim=(1, 2)))  # [B, C]
        gated = self.eps_se * x * gate.unsqueeze(1).unsqueeze(2)

        xt = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        s = self.strip_v(xt)
        s = self.strip_h(s)
        s = self.eps_strip * s.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        x = x + gated + s
        return self.up_proj(self.act(self.down_proj(x)))


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_adapter: bool = False,
        adapter_dim_ratio: float = 0.1,
        geo_adapter: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            use_adapter (bool): If True, use adapter module after MLP in residual path.
            adapter_dim_ratio (float): Ratio of adapter dim to embedding dim.
            geo_adapter (bool): If True, the internal adapter gets a
                strip-convolution branch (Adapter-I_geo).
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        # Add adapter module in residual path after MLP
        self.use_adapter = use_adapter
        if use_adapter:
            if geo_adapter:
                self.adapter = GeoAdapterModule(dim, adapter_dim_ratio, act_layer)
            else:
                self.adapter = AdapterModule(dim, adapter_dim_ratio, act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        # MLP block with adapter in residual path
        mlp_output = self.mlp(self.norm2(x))

        if self.use_adapter:
            # Add adapter output to the MLP path
            adapter_output = self.adapter(self.norm2(x))
            x = x + mlp_output + adapter_output
        else:
            x = x + mlp_output

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
