import torch
import numpy as np


@torch.inference_mode()
def numpy2pytorch(imgs: list[np.ndarray]):
    """Convert a list of numpy images to a pytorch tensor."""
    assert len(imgs) > 0
    assert all(img.ndim == 3 for img in imgs)
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h
