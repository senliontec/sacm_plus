"""Joint image-mask augmentation, safe for curvilinear structures.

All geometric transforms are applied identically to image and mask so
the annotation stays pixel-aligned. Color augmentation (if enabled)
affects the image only and is automatically skipped for grayscale
images (R == G == B), where intensity itself is the signal
(X-ray / DSA / confocal microscopy).
"""

import random

import torch
import torch.nn.functional as F


def elastic_deform(image, mask, alpha=8.0, sigma=3.0):
    """Apply the same elastic deformation to image and mask.

    A random displacement field is smoothed with a Gaussian kernel
    (separable convolution), normalized to std `alpha` (pixels), and
    used to resample both tensors with grid_sample.

    Args:
        image: [C, H, W] float tensor.
        mask: [C, H, W] float tensor with the same spatial size.
        alpha: displacement magnitude in pixels (mild for thin curves).
        sigma: smoothness of the displacement field.
    Returns:
        Deformed (image, mask).
    """
    _, H, W = image.shape
    ksize = 2 * int(round(3 * sigma)) + 1
    ax = torch.arange(ksize, dtype=torch.float32, device=image.device) - ksize // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    def smooth(field):
        # Separable Gaussian smoothing: 1xK then Kx1
        field = F.conv2d(field, g.view(1, 1, 1, -1), padding=(0, ksize // 2))
        field = F.conv2d(field, g.view(1, 1, -1, 1), padding=(ksize // 2, 0))
        return field

    dx = smooth(torch.randn(1, 1, H, W, device=image.device))
    dy = smooth(torch.randn(1, 1, H, W, device=image.device))
    dx = dx / (dx.std() + 1e-8) * alpha
    dy = dy / (dy.std() + 1e-8) * alpha

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=image.device),
        torch.arange(W, dtype=torch.float32, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([xx + dx[0, 0], yy + dy[0, 0]], dim=-1).unsqueeze(0)
    grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0

    image = F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]
    # The mask is resampled bilinearly ON PURPOSE: near deformation
    # boundaries it yields soft labels in [0, 1], which act as boundary
    # label smoothing for the BCE/Dice/clDice targets. Nearest resampling
    # would keep the mask binary but can break 1-2px curvilinear
    # structures discontinuously (a displaced foreground pixel landing
    # between two background pixels is dropped) — worse for this task.
    # Evaluation never sees augmented masks, so the soft values only
    # affect training.
    mask = F.grid_sample(
        mask.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]
    return image, mask


class JointAugment:
    """Random augmentation shared by image and mask.

    Safe set for curvilinear structures: horizontal/vertical flips,
    90-degree rotations, mild elastic deformation. Color jittering is
    off by default (grayscale-safe).
    """

    def __init__(
        self,
        p_flip=0.5,
        p_rot90=0.5,
        p_elastic=0.3,
        elastic_alpha=8.0,
        elastic_sigma=3.0,
        p_color=0.0,
    ):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_color = p_color

    def __call__(self, image, mask):
        """Apply random augmentation.

        Args:
            image: [C, H, W] float tensor in [0, 1].
            mask: [C, H, W] float tensor in [0, 1].
        Returns:
            Augmented (image, mask).
        """
        if random.random() < self.p_flip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < self.p_flip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if random.random() < self.p_rot90:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        if self.p_elastic > 0 and random.random() < self.p_elastic:
            image, mask = elastic_deform(image, mask, self.elastic_alpha, self.elastic_sigma)
        if self.p_color > 0 and random.random() < self.p_color:
            # Grayscale images (R == G == B) skip color jitter: for
            # modalities like DSA/X-ray the intensity itself is the
            # diagnostic signal. This per-image self-detection handles the
            # mixed-modality training split (grayscale DCA1/CORN/CREMI +
            # RGB DRIVE/CHASEDB1/CrackTree in one folder) without any
            # per-dataset bookkeeping.
            is_gray = bool(
                (image[0] == image[1]).all() and (image[1] == image[2]).all()
            )
            if not is_gray:
                contrast = random.uniform(0.8, 1.2)
                brightness = random.uniform(0.8, 1.2)
                image = torch.clamp((image - 0.5) * contrast + 0.5, 0, 1)
                image = torch.clamp(image * brightness, 0, 1)
        return image, mask
