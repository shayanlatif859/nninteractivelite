from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import sample_scalar, RandomScalar
from scipy.ndimage import distance_transform_edt, zoom
from skimage.morphology import disk, ball

print("point is being used here!!")

@lru_cache(maxsize=5)
def build_point(radii, use_distance_transform, binarize):
    max_radius = max(radii)
    ndim = len(radii)

    # Create a spherical (or circular) structuring element with max_radius
    if ndim == 2:
        coords = np.arange(-max_radius, max_radius + 1, dtype=np.float32)
        y, x = np.meshgrid(coords, coords, indexing='ij')
        dist = np.sqrt(x**2 + y**2)
    elif ndim == 3:
        coords = np.arange(-max_radius, max_radius + 1, dtype=np.float32)
        z, y, x = np.meshgrid(coords, coords, coords, indexing='ij')
        dist = np.sqrt(x**2 + y**2 + z**2)
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D and 3D are supported.")
    # Adjust distance to get binary structuring
    structuring_element = (dist <= radii[0]).astype(np.float32)


    # Create the target shape based on the sampled radii
    target_shape = [round(2 * r + 1) for r in radii]

    # Interpolation code
    if any([i != j for i, j in zip(target_shape, structuring_element.shape)]):
        scale_factors = [t / s for t, s in zip(target_shape, structuring_element.shape)]
        structuring_element_resized = zoom(structuring_element, scale_factors, order=1)
    else:
        structuring_element_resized = structuring_element

    if use_distance_transform:
        # Convert the structuring element to a binary mask for distance transform computation
        binary_structuring_element = (structuring_element_resized >= 0.5).astype(np.float32)

        # Compute the Euclidean distance transform of the binary structuring element
        structuring_element_resized = distance_transform_edt(binary_structuring_element)

        # Normalize the distance transform to have values between 0 and 1
        structuring_element_resized /= structuring_element_resized.max()

    if binarize and not use_distance_transform:
        # Normalize the resized structuring element to binary (values near 1 are treated as the point region)
        structuring_element_resized = (structuring_element_resized >= 0.5).astype(np.float32)
    return structuring_element_resized
    
def place_point(position, interaction_map, point_radius, use_distance_transform=False, binarize=False):
    """
    Uses NumPy to place a point on the interaction map at the specified position
    """
    ndim = interaction_map.ndim

    # Determine radius for each dimension
    if isinstance(point_radius, (list, tuple)):
        radius = tuple(point_radius)
    else:
        radius = (point_radius,) * ndim

    strel = build_point(radius, use_distance_transform, binarize)

    # Compute bounding box
    bbox = [[position[i] - strel.shape[i] // 2, position[i] + strel.shape[i] // 2 + strel.shape[i] % 2]
            for i in range(ndim)]

    # Check if completely outside
    if any(i[1] < 0 for i in bbox) or any(i[0] > s for i, s in zip(bbox, interaction_map.shape)):
        print('Point is outside the interaction map! Ignoring')
        return interaction_map

    slices = tuple(slice(max(0, bbox[i][0]), min(interaction_map.shape[i], bbox[i][1]))
                    for i in range(ndim))

    structuring_slices = tuple(slice(max(0, -bbox[i][0]),
                                     slices[i].stop - slices[i].start + max(0, -bbox[i][0]))
                               for i in range(ndim))

    # Place the structuring element
    interaction_map[slices] = np.maximum(interaction_map[slices], strel[structuring_slices])

    return interaction_map

class PointInteraction_stub():
    interaction_type = 'point'

    def __init__(self,
                 point_radius,
                 use_distance_transform=False):
        self.point_radius = point_radius
        self.use_distance_transform = use_distance_transform
    
    def place_point(self, position, interaction_map, binarize=False):
        # Use the numpy implementation!
        return place_point(position, interaction_map,
                                 self.point_radius,
                                 self.use_distance_transform,
                                 binarize)
    
