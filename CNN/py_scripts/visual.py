import numpy as np
import torch

import plotly.express as px


def visualize_tensor(tensor_im: torch.Tensor):
    """Show a tensor image with dimension (N, C, W, H)

    Parameters
    ----------
    tensor_im : torch.Tensor
        images as tensor
    """

    # params
    _, c, h, w = tensor_im.shape

    # convert to nd array
    im_reshaped = tensor_im.detach().numpy()

    # reshape
    im_reshaped = im_reshaped.reshape(w, h, c)

    # convert to uint8
    im_reshaped = im_reshaped.astype("uint8")

    return px.imshow(im_reshaped)
