
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from nnunetv2.utilities.helpers import empty_cache

print("erosion_dilation is being used here!!")

def iterative_3x3_same_padding_pool3d(x, kernel_size=3, use_min_pool=False):
    assert kernel_size % 2 == 1, 'Only works with odd kernels'
    """
    Applies 3D max pooling with manual asymmetric padding such that
    the output shape is the same as the input shape.

    Args:
        x (Tensor): Input tensor of shape (N, C, D, H, W)
        kernel_size (int or tuple): Kernel size for the pooling.
            If int, the same kernel size is used for all three dimensions.

    Returns:
        Tensor: Output tensor with the same (D, H, W) dimensions as the input.
    """


    # Compute asymmetric padding for each dimension:
    pad_front = (kernel_size - 1) // 2
    pad_back = (kernel_size - 1) - pad_front


    # For 3D (input shape: [N, C, D, H, W]), F.pad expects the padding in the order:
    # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # Removed .pad

    iters = (kernel_size - 1) // 2

    # Handle edge replication padding manually:
    pad_width = iters
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)),
            mode='edge')

    # Apply max pooling with no additional padding.
    if not use_min_pool:
        for _ in range(iters):
            x_padded = maximum_filter(x_padded, size=(1, 1, 3, 3, 3))
        return x_padded
    else:
        for _ in range(iters):
            x_padded = minimum_filter(x_padded, size=(1, 1, 3, 3, 3))
        return x_padded
