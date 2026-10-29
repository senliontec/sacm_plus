import kornia as K
import torch
import torch.nn as nn


def torch_norm(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-10)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Connect_Loss(nn.Module):
    """Connectivity loss from MaskVSC.

    ``mode='paper'`` follows the paper's Eq.1/2 intent.  Eq.2 normalizes the
    smoothness penalty by the total pixel count ``N`` while Eq.1 divides the
    component-count approximation by the number of foreground pixels; the
    printed equations omit the compensating ``N`` factor and become
    numerically inert.  Paper mode restores that factor:
    ``|S(Y) - S(Y_G)| * N / #foreground``.  It deliberately skips the
    released 3x3 morphological opening: that opening erases thin vessels,
    making an empty prediction have exactly zero L_C on CrackTree/CREMI/CORN.
    ``cap > 0`` bounds the paper-mode term; without a cap, sparse domains
    receive 10-100x the BCE weight and destabilize pooled training.
    ``mode='release'`` reproduces the official opening + ``count_nonzero``
    arithmetic for parity/ablation.
    """

    def __init__(self, mode: str = 'release', cap: float = 0.0):
        super().__init__()
        if mode not in ('paper', 'release'):
            raise ValueError(
                f'unknown Connect_Loss mode {mode!r}; use paper or release')
        if cap < 0:
            raise ValueError(f'Connect_Loss cap must be >= 0, got {cap}')
        self.mode = mode
        self.cap = float(cap)

    def smoothness_penalty(self, tensor):
        # Compute gradient differences for smooth connectivity
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)

        return smoothness_loss

    def forward(self, img, lab):
        foreground = torch.count_nonzero(lab > 0.5)
        # Upstream hardcodes .cuda(); use the input device so the identical
        # loss also runs on CPU during smoke tests.
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                              dtype=img.dtype, device=img.device)
        if self.mode == 'release':
            # Official arithmetic, including the 3x3 opening.
            img = K.morphology.opening(img, kernel)
            lab = K.morphology.opening(lab, kernel)

        # Apply a sigmoid to approximate binarization in a differentiable way
        img = torch.sigmoid(10 * (img - 0.5))  # Sharpens values close to 0 or 1
        lab = torch.sigmoid(10 * (lab - 0.5))  # Sharpens values close to 0 or 1

        smoothness_img = self.smoothness_penalty(img)
        smoothness_lab = self.smoothness_penalty(lab)
        difference = torch.abs(smoothness_img - smoothness_lab)

        if self.mode == 'release':
            return difference / torch.count_nonzero(lab)
        if foreground.item() == 0:
            return difference * 0.0
        value = difference * img.numel() / foreground
        if self.cap > 0:
            value = value.clamp(max=self.cap)
        return value
