import numpy as np
import torch


def make_compatible_with_torch(im: np.ndarray):
    """transform an images (W, H, C) to (N, C, H, W)
    Convert it to a Tensor with type float

    Parameters
    ----------
    im : Nd array
        images as an np.array object
    """

    # params
    w, h, c = im.shape
    # reshape then extend dims
    im_reshaped = im.reshape(c, w, h)
    im_reshaped = np.expand_dims(im_reshaped, 0)

    # convert to float tensor
    return torch.tensor(im_reshaped).float()
