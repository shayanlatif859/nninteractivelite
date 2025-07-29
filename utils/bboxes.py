from time import time
from typing import List, Union, Tuple

import numpy as np

print("bboxes is being used here")

def generate_bounding_boxes(mask, bbox_size=(192, 192, 192), stride=(16, 16, 16), margin=(10, 10, 10), max_depth=5, current_depth=0):
    """
    Generate overlapping bounding boxes to cover a 3D binary segmentation mask using PyTorch tensors.

    Parameters:
    - mask: 3D PyTorch tensor with values 0 or 1 (binary mask)
    - bbox_size: Tuple or list of three integers specifying the size of bounding boxes per dimension (x, y, z)
    - stride: Tuple or list of three integers specifying the stride for subsampling centers per dimension
    - margin: Tuple or list of three integers specifying the margin to leave uncovered per dimension
    - max_depth: Maximum recursion depth to prevent infinite recursion
    - current_depth: Current recursion depth (used internally)

    Returns:
    - List of tuples [(min_coords, max_coords), ...], where min_coords and max_coords are lists [x, y, z] defining each box
      as a half-open interval [min_coords, max_coords).
    """
    # Prevent infinite recursion
    if current_depth > max_depth:
        # print('random fallback due to max recursion depth')
        return random_sampling_fallback(mask, bbox_size, margin, 25)

    # Ensure bbox_size, stride, and margin are lists
    bbox_size = list(bbox_size)
    margin = list(margin)

    # Compute half sizes for each dimension
    half_size = [bs // 2 for bs in bbox_size]
    # Adjust end offsets to ensure full bbox_size (handles odd sizes)
    end_offset = [bs - hs for bs, hs in zip(bbox_size, half_size)]  # e.g., 193 - 96 = 97

    # Find all object voxels
    object_voxels = np.argwhere(mask)
    if object_voxels.size == 0:
        return []

    # Compute the object's bounding box to limit potential centers
    min_coords = np.min(object_voxels, axis=0)
    max_coords = np.max(object_voxels, axis=0)

    if stride == 'auto':
        stride = [max(1, round((j - i) / 4)) for i, j in zip(min_coords, max_coords)]

    stride = list(stride)
    # print('stride', stride)
    # print('bbox', [[i, j] for i, j in zip(min_coords, max_coords)])

    # Generate potential centers within the object's bounding box
    potential_centers = []
    for x in range(max(0, min_coords[0]), min(mask.shape[0], max_coords[0] + 1), stride[0]):
        for y in range(max(0, min_coords[1]), min(mask.shape[1], max_coords[1] + 1), stride[1]):
            for z in range(max(0, min_coords[2]), min(mask.shape[2], max_coords[2] + 1), stride[2]):
                if mask[x, y, z]:
                    potential_centers.append([x, y, z])
    # print(f'got {len(potential_centers)} center candidates')

    if len(potential_centers) == 0:
        new_stride = [max(1, s // 2) for s in stride]
        return generate_bounding_boxes(mask, bbox_size, new_stride, margin, max_depth, current_depth + 1)

    # Set cover algorithm
    potential_centers = np.array(potential_centers)
    uncovered = mask.copy().astype(np.uint8)
    bboxes = []

    while len(potential_centers) > 0 and np.any(uncovered):
        best_center = None
        best_covered = 0
        best_bounds = None

        # Find the center that covers the most uncovered voxels
        for idx, center in enumerate(potential_centers):
            c_x, c_y, c_z = center
            x_start = max(0, c_x - half_size[0] + margin[0])
            x_end = min(mask.shape[0], c_x + end_offset[0] - margin[0])
            y_start = max(0, c_y - half_size[1] + margin[1])
            y_end = min(mask.shape[1], c_y + end_offset[1] - margin[1])
            z_start = max(0, c_z - half_size[2] + margin[2])
            z_end = min(mask.shape[2], c_z + end_offset[2] - margin[2])

            region = uncovered[x_start:x_end, y_start:y_end, z_start:z_end]
            num_covered = np.sum(region)
            if num_covered > best_covered:
                best_covered = num_covered
                best_center = center
                best_bounds = (x_start, x_end, y_start, y_end, z_start, z_end)

            # If no new voxels are covered, stop
            if best_covered == 0:
                break

            # Add the best bounding box
            c_x, c_y, c_z = best_center
            bboxes.append([
                [c_x - half_size[0], c_x + end_offset[0]],
                [c_y - half_size[1], c_y + end_offset[1]],
                [c_z - half_size[2], c_z + end_offset[2]],
            ])

            # Mark voxels as covered, respecting the margin
            x_s, x_e, y_s, y_e, z_s, z_e = best_bounds
            uncovered[
                x_s: x_e,
                y_s: y_e,
                z_s: z_e,
            ] = 0

            # Remove the used center from potential_centers
            potential_centers = np.array([c for c in potential_centers if uncovered[c[0], c[1], c[2]] > 0])

        # Step 5: Recursively cover remaining voxels using uncovered as the mask
        if np.any(uncovered):
            if np.sum(uncovered) < np.prod([i // 3 for i in bbox_size]):
                bboxes.extend(random_sampling_fallback(uncovered, bbox_size, margin, 25))
            else:
                new_stride = [max(1, s // 2) for s in stride]
                bboxes.extend(generate_bounding_boxes(uncovered, bbox_size, new_stride, margin, max_depth, current_depth + 1))

        return bboxes

def random_sampling_fallback(mask: np.array, bbox_size=(192, 192, 192), margin=(10, 10, 10), n_samples: int = 25):
    half_size = [bs // 2 for bs in bbox_size]
    # Adjust end offsets to ensure full bbox_size (handles odd sizes)
    end_offset = [bs - hs for bs, hs in zip(bbox_size, half_size)]  # e.g., 193 - 96 = 97

    bboxes = []

    while np.any(mask):
        indices = np.argwhere(mask) # nx3

        best_center = None
        best_covered = 0
        best_bounds = None

        # Find the center that covers the most uncovered voxels
        for _ in range(n_samples):
            idx = np.random.choice(len(indices))
            center = indices[idx]
            c_x, c_y, c_z = center
            x_start = max(0, c_x - half_size[0] + margin[0])
            x_end = min(mask.shape[0], c_x + end_offset[0] - margin[0])
            y_start = max(0, c_y - half_size[1] + margin[1])
            y_end = min(mask.shape[1], c_y + end_offset[1] - margin[1])
            z_start = max(0, c_z - half_size[2] + margin[2])
            z_end = min(mask.shape[2], c_z + end_offset[2] - margin[2])

            region = mask[x_start:x_end, y_start:y_end, z_start:z_end]
            num_covered = region.sum()
            if num_covered > best_covered:
                best_covered = num_covered
                best_center = center
                best_bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
            if num_covered > best_covered:
                best_covered = num_covered
                best_center = idx
                best_bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
        if best_center is None:
            break

        # Add the best bounding box
        c_x, c_y, c_z = best_center
        bboxes.append([
            [c_x - half_size[0], c_x + end_offset[0]],
            [c_y - half_size[1], c_y + end_offset[1]],
            [c_z - half_size[2], c_z + end_offset[2]],
        ])

        # Mark voxels as covered, respecting the margin
        x_s, x_e, y_s, y_e, z_s, z_e = best_bounds
        mask[
            x_s: x_e,
            y_s: y_e,
            z_s: z_e,
        ] = 0
    return bboxes


if __name__ == '__main__':
    times = []

    for _ in range(1):
        st = time()
        mask = np.zeros((256, 256, 256), dtype=np.uint8)
        mask[50:150, 50:150, 50:150] = 1  # A cubic object

        # Generate bounding boxes with an odd size to test
        bboxes = random_sampling_fallback(
            mask,
            bbox_size=(193, 193, 193),  # Odd size
            margin=(10, 10, 10),
            n_samples = 25
        )

        # Print results
        print(f"Number of bounding boxes: {len(bboxes)}")
        end = time()
        times.append(end - st)
    print(times)
