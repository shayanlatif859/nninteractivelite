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
        dist = np.sqrt(x ** 2 + y ** 2)
    elif ndim == 3:
        coords = np.arange(-max_radius, max_radius + 1, dtype=np.float32)
        z, y, x = np.meshgrid(coords, coords, coords, indexing='ij')
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D and 3D are supported.")

    # Adjust distance to get binary structuring
    dist = np.sqrt(sum((c / r) ** 2 for c, r in zip((z, y, x), radii)))
    structuring_element = (dist <= 1).astype(np.float32)

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

class PointInteraction_stub():
    interaction_type = 'points'

    def __init__(self,
                 point_radius,
                 use_etd: bool = True,
                 backend: str = "numpy"):

        self.point_radius = point_radius
        self.use_distance_transform = use_etd
        self.backend = backend

    def place_point(self, position, interaction_map, binarize: bool = False):
        ndim = interaction_map.ndim

        # Handle per-dimension radii
        if isinstance(self.point_radius, (int, float)):
            radii = (self.point_radius,) * ndim
        else:
            radii = tuple(
                sample_scalar(self.point_radius, d, interaction_map.shape)
                for d in range(ndim)
            )

        strel = build_point(radii, self.use_distance_transform, binarize)

        # Bounding box of the structuring element centered at position
        bbox = [[position[i] - strel.shape[i] // 2,
                 position[i] + strel.shape[i] // 2 + strel.shape[i] % 2]
                for i in range(ndim)]

        # Outside check
        if any(b[1] < 0 for b in bbox) or any(b[0] > s for b, s in zip(bbox, interaction_map.shape)):
            print("Point is outside the interaction map! Ignoring")
            print(f'Position: {position}')
            print(f'Interaction map shape: {interaction_map.shape}')
            print(f'Point bbox would have been {bbox}')
            return interaction_map

        # Clip to interaction_map bounds
        slices = tuple(
            slice(max(0, bbox[i][0]), min(interaction_map.shape[i], bbox[i][1]))
            for i in range(ndim)
        )
        structuring_slices = tuple(
            slice(max(0, -bbox[i][0]),
                  slices[i].stop - slices[i].start + max(0, -bbox[i][0]))
            for i in range(ndim)
        )

        # Apply with maximum overlay
        interaction_map[slices] = np.maximum(
            interaction_map[slices],
            strel[structuring_slices]
        )
        return interaction_map
