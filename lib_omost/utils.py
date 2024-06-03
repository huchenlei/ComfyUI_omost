import torch
import numpy as np


@torch.inference_mode()
def numpy2pytorch(imgs: list[np.ndarray]):
    """Convert a list of numpy images to a pytorch tensor.
    Input: images in list[[H, W, C]] format.
    Output: images in [B, H, W, C] format.

    Note: ComfyUI expects [B, H, W, C] format instead of [B, C, H, W] format.
    """
    assert len(imgs) > 0
    assert all(img.ndim == 3 for img in imgs)
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 255.0
    return h
