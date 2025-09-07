from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import binary_opening, binary_dilation
from time import time
from typing import Union, List, Tuple, Optional
import numpy as np
from scipy.ndimage import zoom
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


from torch import nn
from torch._dynamo import OptimizedModule
import SimpleITK as sitk
import onnxruntime as ort

import nnInteractive
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.rounding import round_to_nearest_odd
import torch

print("Inference_sessionnew is being used here!!")

class nnInteractiveInferenceSessionnew():
    def __init__(self, model_path, device="cpu", do_autozoom=False, verbose=False):
        print("nnInteractiveInferenceSessionnew has run __init__")

        self.predict_entire_image = True # THIS IS EXPERIMENTAL! set to false if something breaks.
        self.verbose = verbose
        self.do_autozoom: bool = do_autozoom
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_torch_compile = False
        self.preferred_scribble_thickness = [2, 2, 2]
        print(self.preferred_scribble_thickness)
        self.configuration_manager = SimpleONNXConfig(patch_size=(192, 192, 192))
        assert self.use_torch_compile is False, (
            "This implementation places the preprocessed image and the interactions "
            "into pinned memory for speed reasons..."
        )
        #providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.network = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.load_network(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.network.get_inputs()[0].name
        self.output_name = self.network.get_outputs()[0].name
        print(f"{self.input_name} -> {self.output_name}")
        print(f"self.network.get_inputs()[0].shape:", self.network.get_inputs()[0].shape)
        print(f"self.network.get_outputs()[0].shape:", self.network.get_outputs()[0].shape)

        """
        Only intended to work with nnInteractiveTrainerV2 and its derivatives
        """
        # set as part of initialization
        assert self.use_torch_compile is False, ('This implementation places the preprocessed image and the interactions '
                                            'into pinned memory for speed reasons. This is incompatible with '
                                            'torch.compile because of inconsistent strides in the memory layout. '
                                            'Note to self: .contiguous() on GPU could be a solution. Unclear whether '
                                            'that will yield a benefit though.')

        self.model_path = model_path


    def load_network(self, model_path, providers):
        assert model_path.exists(), f"Model file does not exist: {model_path}"
        assert model_path.suffix == ".onnx", f"Expected .onnx file, got: {model_path.suffix}"
        print(f"[debug] Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.network.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[debug] Model loaded successfully. Input name: {self.input_name}")
        print("[debug] input shape expected by model:", self.session.get_inputs()[0].shape)

        def get_providers():
            available_providers = ort.get_all_providers()
            print(f"[debug] Available providers: {available_providers}")  # print procviders
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider"]  # cuda
            elif "CoreMLExecutionProvider" in available_providers:
                return ["CoreMLExecutionProvider"]  # coreml
            else:
                return ["CPUExecutionProvider"]  # cpu (worst choice)

        providers = get_providers()

        if not providers:
            print("No provider found... have you checked your installed drivers?")

        #self.session = ort.InferenceSession("/Users/shayanlatif/PycharmProjects/AIModelConversionKit/interactive_nnunet_fp16.onnx", providers=providers)
        self.label_manager = None
        self.dataset_json = None
        self.trainer_name = None
        self.plans_manager = None
        #self.use_pinned_memory = use_pinned_memory
        #self.use_torch_compile = use_torch_compile

        # Set shape and decay manually
        #self.source_shape = (128,128,128)
        self.interaction_decay = 0.9
        # Define enabled interaction types:
        self.enabled_interactions = {
            "point": True,
            "bbox": False,
            "scribble": False,
            "lasso": False,
            "initial_segmentation": False
        }

        CHANNELS_PER_TYPE = {
            "point": 2,
            "bbox": 2,
            "scribble": 2,
            "lasso": 2,
            "initial_segmentation": 1
        }

        # Count channels needed:
        self.num_interaction_channels = sum([
            2 if enabled else 0
            for enabled in self.enabled_interactions.values()
        ])

        # these are preset values. change them!!
        self.input_shape = (79, 95, 79)
        image_channels = 1 # assume grayscale input
        expected_input_shape = self.session.get_inputs()[0].shape

        expected_total_channels = expected_input_shape[1]  # ONNX uses NCHWD

        self.num_interaction_channels = expected_total_channels - image_channels
        self.interactions = np.zeros((self.num_interaction_channels, *self.input_shape), dtype=np.float32)

        # Determine total expected input channels
        expected_total_channels = expected_input_shape[1]

        #print("[debug] source shape", self.source_shape)
        print("[debug] Session ID (init):", id(self))

        #self.output_name = self.session.get_outputs()[0].name

        # image specific
        #self.interactions: torch.Tensor = None I commented this out cause it was causing problems. I hope this doesnt do anything too bad...
        self.preprocessed_image: np.array = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None

        # this will be set when loading the model (initialize_from_trained_model_folder), commented out cause it was also problem
        #self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None
        self.pad_mode_data = "constant"
        #self.point_interaction = None

        # ill be surprised if this works!!
        self.point_interaction = PointInteraction_stub(
            4,
            True
        )

        self.original_image_shape = None


        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers = []
        self.has_positive_bbox = False

        # Create a thread pool executor for background tasks.
        # this only takes care of preprocessing and interaction memory initialization so there is no need to give it
        # more than 2 workers
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

    def set_image(self, image: np.ndarray, image_properties: dict = None):
        """
        Image must be 4D to satisfy nnU-Net needs: [c, x, y, z]
        Offload the processing to a background thread.
        """
        if image_properties is None:
            image_properties = {}
        self._reset_session()
        if image.ndim == 3:
            print("[debug] Input was 3D; adding channel dimension to make it 4D.")
            image = image[None, ...]
        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape} data type {image.dtype}' #CHANGE
        print("[debug] input image channels =", image.shape[0])
        if self.verbose:
            print(f'Initialize with raw image shape {image.shape}')

        # Offload all image preprocessing to a background thread.
        self.preprocess_future = self.executor.submit(self._background_set_image, image, image_properties)
        self.original_image_shape = image.shape

    def _finish_preprocessing_and_initialize_interactions(self):
        print("[debug] _finish_preprocessing_and_initialize_interactions() was called")

        if self.original_image_shape is not None:
            self.preprocessed_props = {
                'bbox_used_for_cropping': [(0, d) for d in self.original_image_shape]
            }

        elif self.preprocessed_props is None:
            raise RuntimeError("Preprocessing still not done after waiting! Check _background_set_image in inference_sessionnew.py")

        """
        Block until both the image preprocessing and the interactions tensor initialization
        are finished.
        """
        if self.preprocess_future is not None:
            # Wait for image preprocessing to complete.
            self.preprocess_future.result()
            del self.preprocess_future
            self.preprocess_future = None

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        print("[debug] set_target_buffer() was called")
        """
        Must be 3d numpy array or torch.Tensor
        """
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_propagation: bool, max_num_patches: Optional[int] = None):
        print("[debug] set_do_autozoom() was called")
        self.do_autozoom = do_propagation

    def _reset_session(self):
        print("[debug] _reset_session() was called")
        self.interactions_future = None
        self.preprocess_future = None

        del self.preprocessed_image
        del self.target_buffer
        del self.interactions
        del self.preprocessed_props
        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.preprocessed_props = None
        empty_cache(self.device)
        self.original_image_shape = None
        self.has_positive_bbox = False

    def _initialize_interactions(self, image: np.ndarray):
        print("[debug] _intialize_interactions was called")
        if self.verbose:
            print(f'Initialize interactions. Pinned: {self.use_pinned_memory}')
        # Create the interaction tensor based on the target shape.
        print("[debug] given image shape = ", image.shape)

        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape} data type {image.dtype}' #CHANGE
        _, d, h, w = image.shape

        # Extract shape data from image with 4d tuple
        self.input_shape = (d, h, w)
        self.interactions = np.zeros((4, d, h, w), dtype=np.float32)
        self.target_buffer = np.zeros_like(self.interactions[0], dtype=np.uint8)

        print("[debug] initialized interactions shape =", self.interactions.shape)
        print("[debug] model expects input shape:", self.session.get_inputs()[0].shape)

        #self.interactions = np.zeros(
        #    (self.num_interaction_channels, *self.source_shape),
        #    dtype=np.float32
        #)


        if self.interactions is None:
            print("[debug] ⚠ interactions is still None! ⚠")
            return

    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        # Convert and clone the image tensor.
        print("[debug] _background_set_image was called")
        image_torch = (image - np.mean(image)) / np.std(image)
        # Crop to nonzero region.
        if self.verbose:
            print('Cropping input image to nonzero region')
        nonzero_idx = np.nonzero(image_torch)
        # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
        bbox = [[int(np.min(i)), int(np.max(i)) + 1] for i in nonzero_idx]
        del nonzero_idx
        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        image_torch = image_torch[slicer].astype(np.float32)
        if self.verbose:
            print(f'Cropped image shape: {image_torch.shape}')

        # As soon as we have the target shape, start initializing the interaction tensor in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image_torch)

        # Normalize the cropped image.
        if self.verbose:
            print('Normalizing cropped image')
        image_torch -= image_torch.mean()
        image_torch /= image_torch.std()

        self.preprocessed_image = image_torch

        # No pinned memory in ONNX files... but if u change it to torch, dont forget this!!!
        #if self.use_pinned_memory and self.device.type == 'cuda':
        if self.verbose:
            print('Pin memory: image')
            # Note: pin_memory() in PyTorch typically returns a new tensor.
        #self.preprocessed_image = self.preprocessed_image.pin_memory()

        self.preprocessed_props = {'bbox_used_for_cropping': bbox[1:]}

        self.original_image_shape = image.shape

        # we need to wait for this here I believe
        self.interactions_future.result()
        del self.interactions_future
        self.interactions_future = None

        return self.preprocessed_props, bbox

    def reset_interactions(self):
        """
        Use this to reset all interactions and start from scratch for the current image. This includes the initial
        segmentation!
        """

        print("[debug] reset_interactions was called")

        if self.interactions is not None:
            self.interactions.fill(0)

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        empty_cache(self.device)
        self.has_positive_bbox = False

    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True) -> np.ndarray:
        print("[debug] add_bbox_interaction was called")
        if include_interaction:
            self.has_positive_bbox = True

        lbs_transformed = [round(i) for i in transform_coordinates_noresampling([i[0] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        ubs_transformed = [round(i) for i in transform_coordinates_noresampling([i[1] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        transformed_bbox_coordinates = [[i, j] for i, j in zip(lbs_transformed, ubs_transformed)]

        if self.verbose:
            print(f'Added bounding box coordinates.\n'
                  f'Raw: {bbox_coords}\n'
                  f'Transformed: {transformed_bbox_coordinates}\n'
                  f"Crop Bbox: {self.preprocessed_props['bbox_used_for_cropping']}")

        # Prevent collapsed bounding boxes and clip to image shape
        image_shape = self.preprocessed_image.shape  # Assuming shape is (C, H, W, D) or similar

        for dim in range(len(transformed_bbox_coordinates)):
            transformed_start, transformed_end = transformed_bbox_coordinates[dim]

            # Clip to image boundaries
            transformed_start = max(0, transformed_start)
            transformed_end = min(image_shape[dim + 1], transformed_end)  # +1 to skip channel dim

            # Ensure the bounding box does not collapse to a single point
            if transformed_end <= transformed_start:
                if transformed_start == 0:
                    transformed_end = min(1, image_shape[dim + 1])
                else:
                    transformed_start = max(transformed_start - 1, 0)

            transformed_bbox_coordinates[dim] = [transformed_start, transformed_end]

        if self.verbose:
            print(f'Bbox coordinates after clip to image boundaries and preventing dim collapse:\n'
                  f'Bbox: {transformed_bbox_coordinates}\n'
                  f'Internal image shape: {self.preprocessed_image.shape}')

        self._add_patch_for_bbox_interaction(transformed_bbox_coordinates)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # place bbox
        slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
        channel = -6 if include_interaction else -5
        self.interactions[(channel, *slicer)] = 1

        # forward pass
        if run_prediction:
            self._predict()

    def add_point_interaction(self, coordinates: Tuple[int, ...], include_interaction: bool, run_prediction: bool = True):

        print("[debug] add_point_interaction was called")

        # print("[debug] interactions before calling _finish_preprocessing_and_initialize_interactions:", self.interactions.shape)

        self._finish_preprocessing_and_initialize_interactions()
        assert self.interactions.dtype == np.float32 #Interactions MUST be float 32

        transformed_coordinates = [round(i) for i in transform_coordinates_noresampling(coordinates,
                                                             self.preprocessed_props['bbox_used_for_cropping'])]

        self._add_patch_for_point_interaction(transformed_coordinates)

        # Debug to print interactions
        print("[debug] interactions =", self.interactions.shape)
        print("[debug] decay =", self.interaction_decay)
        # print("[debug] Session ID (add_point_interaction):", id(self))

        # decay old interactions
        self.interactions[-4:-2] *= self.interaction_decay

        interaction_channel = -4 if include_interaction else -3
        self.interactions[interaction_channel] = self.point_interaction.place_point(
            transformed_coordinates, self.interactions[interaction_channel])
        if run_prediction:
            self._predict()
        print("[debug] interactions after add_point_interaction:", np.unique(self.interactions))

    def add_scribble_interaction(self, scribble_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.original_image_shape[1:], scribble_image.shape)]), f'Given scribble image must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {scribble_image.shape}'
        self._finish_preprocessing_and_initialize_interactions()

        scribble_image = torch.from_numpy(scribble_image)

        # crop (as in preprocessing)
        scribble_image = crop_and_pad_nd(scribble_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_scribble_interaction(scribble_image)

        # decay old interactions
        self.interactions[-2:] *= self.interaction_decay

        interaction_channel = -2 if include_interaction else -1
        torch.maximum(self.interactions[interaction_channel], scribble_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del scribble_image
        empty_cache(self.device)
        if run_prediction:
            self._predict()

        if self.verbose:
            print("Waiting for preprocess_future to finish...")
        self.preprocess_future.result()  # wait until preprocessing done
        self.preprocess_future = None

    def add_lasso_interaction(self, lasso_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.original_image_shape[1:], lasso_image.shape)]), f'Given lasso image must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {lasso_image.shape}'
        print("[debug] add_lasso_interaction was called")
        # Stop program if image shape isnt changed
        if self.original_image_shape is None:
            raise RuntimeError(
                "You must call add_image() or run() before using add_lasso_interaction(). original_image_shape is not set.")

        # remove initial channel input
        if lasso_image.shape[0] == 1:
            print("removing channel input of lasso_image...")
            lasso_image = lasso_image[1:]

        self._finish_preprocessing_and_initialize_interactions()

        lasso_image = torch.from_numpy(lasso_image)

        # crop (as in preprocessing)
        lasso_image = crop_and_pad_nd(lasso_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_lasso_interaction(lasso_image)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # lasso is written into bbox channel
        interaction_channel = -6 if include_interaction else -5
        torch.maximum(self.interactions[interaction_channel], lasso_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del lasso_image
        # commented out cause its acting up
        #empty_cache(self.device)
        if run_prediction:
            self._predict()

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False):
        """
        WARNING THIS WILL RESET INTERACTIONS!
        """

        print("[debug] add_initial_seg_interaction was called")

        assert self.original_image_shape is not None, "Original image shape not set. Did you call set_image first?"

        assert initial_seg.shape == self.original_image_shape[1:], (
            f"Initial seg must match input image shape. "
            f"Expected {self.original_image_shape[1:]}, got {initial_seg.shape}"
        )
        self._finish_preprocessing_and_initialize_interactions()

        self.reset_interactions()

        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer[:] = initial_seg

        elif isinstance(self.target_buffer, torch.Tensor):
            self.target_buffer[:] = torch.from_numpy(initial_seg)

        # crop (as in preprocessing)
        # commented out cause torch function... idk if this is a bad idea...
        #initial_seg = torch.from_numpy(initial_seg)
        initial_seg = crop_and_pad_nd(initial_seg, self.preprocessed_props['bbox_used_for_cropping'])

        # initial seg is written into initial seg buffer
        interaction_channel = -4
        self.interactions[interaction_channel] = initial_seg
        empty_cache(self.device)
        if run_prediction:
            self._add_patch_for_initial_seg_interaction(initial_seg)
            del initial_seg
            self._predict()
        else:
            del initial_seg

    # def _predict_full_image(self):
    #     print("[debug] Running full image prediction")
    #     start_time = time()
    #     patch_size = self.configuration_manager.patch_size
    #     stride = [s // 2 for s in patch_size]  # 50% overlap
    #     image = self.preprocessed_image[:4]  # First 4 modalities only
    #     seg_output = np.zeros_like(self.target_buffer, dtype=np.uint8)
    #     count_output = np.zeros_like(self.target_buffer, dtype=np.uint8)
    #
    #     D, H, W = image.shape[1:]
    #
    #     for z in range(0, D, stride[0]):
    #         for y in range(0, H, stride[1]):
    #             for x in range(0, W, stride[2]):
    #                 patch_bbox = [
    #                     [z, min(z + patch_size[0], D)],
    #                     [y, min(y + patch_size[1], H)],
    #                     [x, min(x + patch_size[2], W)],
    #                 ]
    #
    #                 patch = image[:,
    #                         patch_bbox[0][0]:patch_bbox[0][1],
    #                         patch_bbox[1][0]:patch_bbox[1][1],
    #                         patch_bbox[2][0]:patch_bbox[2][1]]
    #
    #                 pad_width = [(0, 0)]
    #                 for axis in range(3):
    #                     pad_amount = patch_size[axis] - patch.shape[axis + 1]
    #                     before = 0
    #                     after = pad_amount if pad_amount > 0 else 0
    #                     pad_width.append((before, after))
    #                 patch = np.pad(patch, pad_width, mode='constant')
    #
    #                 dummy_int = np.zeros((4, *patch_size), dtype=np.float32)
    #                 input_data = np.concatenate([patch, dummy_int], axis=0)[None].astype(np.float32)
    #
    #                 pred_logits = self.network.run(None, {self.input_name: input_data})[0]
    #                 pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)
    #
    #                 dz = patch_bbox[0][1] - patch_bbox[0][0]
    #                 dy = patch_bbox[1][1] - patch_bbox[1][0]
    #                 dx = patch_bbox[2][1] - patch_bbox[2][0]
    #                 pred_seg = pred_seg[:dz, :dy, :dx]
    #
    #                 seg_output[
    #                 patch_bbox[0][0]:patch_bbox[0][1],
    #                 patch_bbox[1][0]:patch_bbox[1][1],
    #                 patch_bbox[2][0]:patch_bbox[2][1]
    #                 ] += pred_seg
    #
    #                 count_output[
    #                 patch_bbox[0][0]:patch_bbox[0][1],
    #                 patch_bbox[1][0]:patch_bbox[1][1],
    #                 patch_bbox[2][0]:patch_bbox[2][1]
    #                 ] += 1
    #
    #     count_output[count_output == 0] = 1
    #     final_seg = (seg_output / count_output).round().astype(np.uint8)
    #
    #     self.target_buffer[:] = final_seg
    #     print("[debug] full image prediction complete")
    #
    #     if self._viewer is not None:
    #         if "Prediction" not in self._viewer.layers:
    #             self._viewer.add_labels(final_seg, name="Prediction")
    #         else:
    #             self._viewer.layers["Prediction"].data = final_seg
    #
    #     print(f"Prediction complete in {round(time() - start_time, 3)} seconds.")
    def _predict_full_image(self, region_center: Optional[Tuple[int, int, int]] = None):

        # Assume: self.preprocessed_image shape is (D, H, W)
        #          self.interactions shape is (C_interaction, D, H, W)
        #          self.patch_size = (pD, pH, pW)

        seg_prediction = np.zeros_like(self.interactions[0])  # Shape: (D, H, W)
        count_map = np.zeros_like(seg_prediction)

        print("[debug] Running full image prediction")
        print(f"[debug] count_map is {count_map}")
        patch_size = self.configuration_manager.patch_size
        stride = [s // 2 for s in patch_size]  # 50% overlap
        image = self.preprocessed_image[:4]  # First 4 modalities only
        interaction = self.interactions[:4]  # First 4 interaction channels
        bbox_offset = self.preprocessed_props["bbox_used_for_cropping"]
        print(f"[debug] bbox_offset is {(bbox_offset)}")

        if len(bbox_offset) == 4:
            _, z_range, y_range, x_range = bbox_offset
        elif len(bbox_offset) == 3:
            z_range, y_range, x_range = bbox_offset

        z0, y0, x0 = z_range[0], y_range[0], x_range[0]

        print("z0, y0, x0", z0, y0, x0)

        start_time = time()

        print("image.shape =", image.shape)

        orig_shape = (
            image.shape[1] + z0,
            image.shape[2] + y0,
            image.shape[3] + x0,
        )

        seg_output = np.zeros(orig_shape, dtype=np.uint8)
        count_output = np.zeros(orig_shape, dtype=np.uint8)


        D, H, W = image.shape[1:]

        print("[debug] self.new_interaction_centers=", self.new_interaction_centers)

        if self.new_interaction_centers is not None:
            print('it worked!! self.new_interaction_centers =', self.new_interaction_centers)

            # Use most recent center
            cz, cy, cx = self.new_interaction_centers[-1]
            pD, pH, pW = patch_size
            D, H, W = image.shape[1:]
            print ("[debug] self.new_interaction_centers =", self.new_interaction_centers)
            print ("[debug] patch_size =", patch_size)
            print ("image.shape", {image.shape})


            # Calculate patch bounds
            z = max(0, min(D - pD, cz - pD // 2))
            y = max(0, min(H - pH, cy - pH // 2))
            x = max(0, min(W - pW, cx - pW // 2))

            scaled_bbox = []

            for c, p, dim in zip(self.new_interaction_centers[-1], patch_size, self.preprocessed_image.shape[1:]):
                half = p // 2
                start = max(0, c - half)
                end = min(dim, c + half + (p % 2))
                scaled_bbox.append([start, end])
            print("[debug] scaled_patch_size:", patch_size)
            print("[debug] scaled_bbox:", scaled_bbox)

            print(f"patch bounds z,y,x:{z,y,x}")
            # Extract patch
            patch_image = image[:, z:z + pD, y:y + pH, x:x + pW]
            patch_interaction = interaction[:, z:z + pD, y:y + pH, x:x + pW]

            print("[debug] patch_interaction.shape =", patch_interaction.shape)
            print("[debug] patch_image.shape =", patch_image.shape)

            if patch_image.shape[0] == 1:
                print("padding to patch_image")
                patch_image = np.repeat(patch_image, 4, axis=0)
                print("[debug] patch_image.shape now after we padded it up =", patch_image.shape)
                print("[debug] patch_interaction.shape =", patch_interaction.shape)

            # Run inference
            print("[debug] patch_image.shape =", patch_image.shape)

            if patch_image.shape[0] < 4:
                # Replicate the single channel into 4
                patch_image = np.repeat(patch_image, 4, axis=0)

            model_input = np.concatenate([patch_image, patch_interaction], axis=0)[np.newaxis]

            print(f"self.input_name: {self.input_name}")
            print(f"model_input.shape: {model_input.shape}")
            print(f"model_input.dtype: {model_input.dtype}")

            pred = self.onnx_session.run(None, {self.input_name: model_input})[0][0]
            pred_seg = np.argmax(pred, axis=0).astype(np.uint8)  # [D, H, W]

            # Overlay prediction into correct place in the target buffer
            # We may need to uncrop using z0/y0/x0 offsets!
            z0, y0, x0 = z_range[0], y_range[0], x_range[0]
            abs_z = z + z0
            abs_y = y + y0
            abs_x = x + x0

            # Flip axes if needed (match for full image)
            pred_seg = np.flip(pred_seg, axis=(1, 2))

            target_slice = self.target_buffer[
                           abs_z:abs_z + pred_seg.shape[0],
                           abs_y:abs_y + pred_seg.shape[1],
                           abs_x:abs_x + pred_seg.shape[2]
                           ]

            print("before:", np.unique(target_slice))

            if not np.array_equal(pred_seg, target_slice):
                print("This patch is changing something!")

            # Update target buffer directly (like overlaying a window of prediction)
            self.target_buffer[
            abs_z:abs_z + pred_seg.shape[0],
            abs_y:abs_y + pred_seg.shape[1],
            abs_x:abs_x + pred_seg.shape[2]
            ] = pred_seg


        for z in range(0, D, stride[0]):
            for y in range(0, H, stride[1]):
                for x in range(0, W, stride[2]):
                    patch_bbox = [
                        [z, min(z + patch_size[0], D)],
                        [y, min(y + patch_size[1], H)],
                        [x, min(x + patch_size[2], W)],
                    ]

                    patch = image[:,
                            patch_bbox[0][0]:patch_bbox[0][1],
                            patch_bbox[1][0]:patch_bbox[1][1],
                            patch_bbox[2][0]:patch_bbox[2][1]]

                    interaction_patch = interaction[:,
                                        patch_bbox[0][0]:patch_bbox[0][1],
                                        patch_bbox[1][0]:patch_bbox[1][1],
                                        patch_bbox[2][0]:patch_bbox[2][1]]

                    pad_width = [(0, 0)]
                    for axis in range(3):
                        pad_amount = patch_size[axis] - patch.shape[axis + 1]
                        before = 0
                        after = pad_amount if pad_amount > 0 else 0
                        pad_width.append((before, after))

                    patch = np.pad(patch, pad_width, mode='constant')
                    interaction_patch = np.pad(interaction_patch, pad_width, mode='constant')

                    input_data = np.concatenate([patch, interaction_patch], axis=0)[None].astype(np.float32)
                    print("[debug] input_data.shape =", input_data.shape)
                    required_channels = 8
                    current_channels = input_data.shape[1]
                    if current_channels < required_channels:
                        pad_channels = required_channels - current_channels
                        padding = np.zeros((1, pad_channels, *input_data.shape[2:]), dtype=input_data.dtype)
                        input_data = np.concatenate([input_data, padding], axis=1)
                        print(f"[debug] Zero-padded input_data to shape {input_data.shape}")

                    pred_logits = self.onnx_session.run(None, {self.input_name: input_data})[0]
                    pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)

                    dz = patch_bbox[0][1] - patch_bbox[0][0]
                    dy = patch_bbox[1][1] - patch_bbox[1][0]
                    dx = patch_bbox[2][1] - patch_bbox[2][0]
                    pred_seg = pred_seg[:dz, :dy, :dx]

                    z_start = patch_bbox[0][0] + z0
                    y_start = patch_bbox[1][0] + y0
                    x_start = patch_bbox[2][0] + x0

                    z_end = z_start + pred_seg.shape[0]
                    y_end = y_start + pred_seg.shape[1]
                    x_end = x_start + pred_seg.shape[2]

                    seg_output[z_start:z_end, y_start:y_end, x_start:x_end] += pred_seg
                    count_output[z_start:z_end, y_start:y_end, x_start:x_end] += 1
        print("[debug] count_output =", count_output)
        count_output[count_output == 0] = 1
        final_seg = (seg_output / count_output).round().astype(np.uint8)

        final_seg = (1 - final_seg)
        final_seg = np.flip(final_seg, axis=(1, 2))
        spacing = [2, 2, 2]

        z0 = max(0, min(z0, final_seg.shape[0] - self.target_buffer.shape[0]))
        y0 = max(0, min(y0, final_seg.shape[1] - self.target_buffer.shape[1]))
        x0 = max(0, min(x0, final_seg.shape[2] - self.target_buffer.shape[2]))

        print(f"final_seg.shape: {final_seg.shape}")
        print(f"z0, y0, x0: {z0}, {y0}, {x0}")
        print(f"target_buffer.shape: {self.target_buffer.shape}")

        self.target_buffer[:] = final_seg[
                                z0:z0 + self.target_buffer.shape[0],
                                y0:y0 + self.target_buffer.shape[1],
                                x0:x0 + self.target_buffer.shape[2]
                                ]

        print("[debug] full image prediction complete")

        if self._viewer is not None:
            if "Prediction" not in self._viewer.layers:
                image_layer = next((layer for layer in self._viewer.layers if layer.__class__.__name__ == "Image"),
                                   None)
                if image_layer is not None:
                    seg_layer = self._viewer.add_labels(final_seg, name="Prediction")
                    if len(image_layer.scale) >= 3:
                        seg_layer.scale = spacing
                        seg_layer.translate = tuple(-o * s for o, s in zip((z0, y0, x0), [1,1,1]))
                else:
                    print("[warning] No image layer found! Could not align prediction.")
            else:
                self._viewer.layers["Prediction"].data = final_seg

            print(f"Prediction complete in {round(time() - start_time, 3)} seconds.")
            print("[debug] pred_seg unique:", np.unique(pred_seg))
            print("[debug] pred_logits.shape =", pred_logits.shape)
            print("[debug] pred_seg.shape =", pred_seg.shape)
            print("[debug] target_buffer.shape =", self.target_buffer.shape)
            return None
        return None

    @torch.inference_mode
    def _predict(self):
        print("[debug] _predict was called")
        """
        This function is a smoking mess to read. This is deliberate. Initially it was super pretty and easy to
        understand. Then the run time optimization began.
        If it feels like we are excessively transferring tensors between CPU and GPU, this is deliberate as well.
        Our goal is to keep this tool usable even for people with smaller GPUs (8-10GB VRAM). In an ideal world
        everyone would have 24GB+ of VRAM and all tensors would like on GPU all the time.
        The amount of hours spent optimizing this function is substantial. Almost every line was turned and twisted
        multiple times. If something appears odd, it is probably so for a reason. Don't change things all willy nilly
        without first understanding what is going on. And don't make changes without verifying that the run time or
        VRAM consumption is not adversely affected.


        Returns:

        """
        print("[debug] interaction centers:", self.new_interaction_centers)
        print("[debug] zoom factors:", self.new_interaction_zoom_out_factors)

        if self.predict_entire_image:
            self._predict_full_image()
            return
        assert self.pad_mode_data == 'constant', 'Only constant padding is supported.'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)

        if len(self.new_interaction_centers) > 1:
            print("Warning: Multiple new interactions. This may cause inefficiencies.")

        start_time = time()

        patch_size = self.configuration_manager.patch_size
        bbox_offset = self.preprocessed_props["bbox_used_for_cropping"]
        # below are start coords in full image

        for center, zoom_factor in zip(self.new_interaction_centers, self.new_interaction_zoom_out_factors):
            # Set center to 3d (not sure why it must be 3d... wasn't 4d just working??
            if len(center) > 3:
                center = center[-3:]
            zoom_factor = max(1, min(zoom_factor, 4)) if self.do_autozoom else 1
            scaled_patch_size = [round(dim * zoom_factor) for dim in patch_size]
            scaled_bbox = []

            for c, p, dim in zip(center, scaled_patch_size, self.preprocessed_image.shape[1:]):
                half = p // 2
                start = max(0, c - half)
                end = min(dim, c + half + (p % 2))
                scaled_bbox.append([start, end])
            print("[debug] scaled_patch_size:", scaled_patch_size)
            print("[debug] scaled_bbox:", scaled_bbox)

            # Crop image and interaction tensor
            crop_img, pad_img = crop_to_valid(self.preprocessed_image, scaled_bbox)
            crop_int, pad_int = crop_to_valid(self.interactions, scaled_bbox)

            # Pad if needed
            if any(pad_img):
                crop_img = pad_cropped(crop_img, pad_img)
            if any(pad_int):
                crop_int = pad_cropped(crop_int, pad_int)

            print("[debug] crop_img.shape =", crop_img.shape)  # Should be (C1, D, H, W)
            print("[debug] crop_int.shape =", crop_int.shape)  # Should be (C2, D, H, W)
            print("[debug] Total channels:", crop_img.shape[0] + crop_int.shape[0])

            # Prepare input for ONNX
            crop_img = crop_img[:4]  # KEEP FIRST 4 MODALITIES
            crop_int = crop_int[:4]  # KEEP FIRST 4 INTERACTION CHANNELS
            print("[debug] crop_img shape after change =", crop_img.shape)
            print("[debug] crop_int shape before change =", crop_int.shape)

            # Resize to patch size if zooming
            if list(crop_img.shape[1:]) != patch_size:
                print("Warning: Patch size mismatch Calling resize_np.")
                crop_img = resize_np(crop_img, patch_size, order=1)  # trilinear
                crop_int = resize_np(crop_int, patch_size, order=0)  # nearest

            input_data = np.concatenate((crop_img, crop_int), axis=0)[None].astype(np.float32)  # (1, C, D, H, W)

            # ONNX Inference
            assert self.network is not None, "ONNX network is not loaded!"
            pred_logits = self.network.run(None, {self.input_name: input_data})[0]  # (1, C, D, H, W)
            pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)  # (D, H, W)
            print("[debug] pred_seg unique:", np.unique(pred_seg))

            print("[debug] crop_int unique values:", np.unique(crop_int))
            print("[debug] input_data.shape =", input_data.shape)
            print("[debug] pred_logits.shape =", pred_logits.shape)
            print("[debug] pred_seg =", pred_seg.shape)
            print("[debug] target_buffer.shape =", self.target_buffer.shape)

            global_bbox = [
    [start + offset_start, end + offset_start]
    for (start, end), (offset_start, _) in zip(scaled_bbox, bbox_offset)
]
            print("[debug] global_bbox =", global_bbox)
            print("[debug] bbox_offset", bbox_offset)
            print("Will paste to bbox:", global_bbox)

            clamped_bbox = []
            valid_crop = []
            for i in range(3):
                g0, g1 = global_bbox[i]
                t0 = max(g0, 0)
                t1 = min(g1, self.target_buffer.shape[i])
                if t1 <= t0:
                    print(f"[warning] BBox out of bounds on axis {i}, skipping paste.")
                    return  # Skip if nothing valid
                s0 = t0 - g0
                s1 = s0 + (t1 - t0)
                clamped_bbox.append([t0, t1])
                valid_crop.append([s0, s1])

            # Crop pred_seg to valid region
            pred_seg_cropped = pred_seg[
                               valid_crop[0][0]:valid_crop[0][1],
                               valid_crop[1][0]:valid_crop[1][1],
                               valid_crop[2][0]:valid_crop[2][1]
                               ]

            # Paste cropped prediction
            self.target_buffer[
            clamped_bbox[0][0]:clamped_bbox[0][1],
            clamped_bbox[1][0]:clamped_bbox[1][1],
            clamped_bbox[2][0]:clamped_bbox[2][1]
            ] = pred_seg_cropped

            print("[debug] final pasted shape:", pred_seg_cropped.shape)

            paste_tensor(self.interactions[0], pred_seg, scaled_bbox)

            # Check if Napari viewer opened prediction
            if self._viewer is not None:
                if "Prediction" not in self._viewer.layers:
                    self._viewer.add_labels(pred_seg, name="Prediction")
                else:
                    self._viewer.layers["Prediction"].data = pred_seg

        self.new_interaction_centers.clear()
        self.new_interaction_zoom_out_factors.clear()

        print(f"Prediction complete in {round(time() - start_time, 3)} seconds.")

    # def _predict(self):
    #     print("[debug] _predict was called")
    #     assert self.pad_mode_data == 'constant', 'Only constant padding is supported.'
    #     assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
    #
    #     if len(self.new_interaction_centers) > 1:
    #         print("Warning: Multiple new interactions. This may cause inefficiencies.")
    #
    #     start_time = time()
    #
    #     patch_size = self.configuration_manager.patch_size
    #     bbox_offset = self.preprocessed_props["bbox_used_for_cropping"]
    #
    #     previous_prediction = self.interactions[0].copy()  # Save current state
    #
    #      for center, zoom_factor in zip(self.new_interaction_centers, self.new_interaction_zoom_out_factors):
    #         print("[debug] center =", center)
    #         if len(center) > 3:
    #             center = center[-3:]
    #         zoom_factor = max(1, min(zoom_factor, 4)) if self.do_autozoom else 1
    #         scaled_patch_size = [round(dim * zoom_factor) for dim in patch_size]
    #
    #         scaled_bbox = []
    #         for c, p, dim in zip(center, scaled_patch_size, self.preprocessed_image.shape[1:]):
    #             half = p // 2
    #             start = max(0, c - half)
    #             end = min(dim, c + half + (p % 2))
    #             scaled_bbox.append([start, end])
    #         print("[debug] scaled_patch_size:", scaled_patch_size)
    #         print("[debug] scaled_bbox:", scaled_bbox)
    #
    #         # Crop image and interaction tensor
    #         crop_img, pad_img = crop_to_valid(self.preprocessed_image, scaled_bbox)
    #         crop_int, pad_int = crop_to_valid(self.interactions, scaled_bbox)
    #         print("[debug] crop_int unique values:", np.unique(crop_int))
    #
    #         if any(pad_img):
    #             crop_img = pad_cropped(crop_img, pad_img)
    #         if any(pad_int):
    #             crop_int = pad_cropped(crop_int, pad_int)
    #
    #         crop_img = crop_img[:4]
    #         crop_int = crop_int[:4]
    #
    #         if list(crop_img.shape[1:]) != patch_size:
    #             print("Warning: Patch size mismatch Calling resize_np.")
    #             crop_img = resize_np(crop_img, patch_size, order=1)
    #             crop_int = resize_np(crop_int, patch_size, order=0)
    #
    #         input_data = np.concatenate((crop_img, crop_int), axis=0)[None].astype(np.float32)
    #         assert self.network is not None, "ONNX network is not loaded!"
    #         pred_logits = self.network.run(None, {self.input_name: input_data})[0]
    #         pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)
    #
    #         global_bbox = []
    #         for (start, end), (offset_start, _) in zip(scaled_bbox, bbox_offset):
    #             global_bbox.append([start + offset_start, end + offset_start])
    #
    #         print("[debug] global_bbox =", global_bbox)
    #         print("[debug] bbox_offset", bbox_offset)
    #         print("Will paste to bbox:", global_bbox)
    #
    #         # Clamp and crop
    #         clamped_bbox = []
    #         valid_crop = []
    #         for i in range(3):
    #             g0, g1 = global_bbox[i]
    #             t0 = max(g0, 0)
    #             t1 = min(g1, self.target_buffer.shape[i])
    #             if t1 <= t0:
    #                 print(f"[warning] BBox out of bounds on axis {i}, skipping paste.")
    #                 return
    #             s0 = t0 - g0
    #             s1 = s0 + (t1 - t0)
    #             clamped_bbox.append([t0, t1])
    #             valid_crop.append([s0, s1])
    #
    #         pred_seg_cropped = pred_seg[
    #                            valid_crop[0][0]:valid_crop[0][1],
    #                            valid_crop[1][0]:valid_crop[1][1],
    #                            valid_crop[2][0]:valid_crop[2][1]
    #                            ]
    #
    #         self.target_buffer[
    #         clamped_bbox[0][0]:clamped_bbox[0][1],
    #         clamped_bbox[1][0]:clamped_bbox[1][1],
    #         clamped_bbox[2][0]:clamped_bbox[2][1]
    #         ] = pred_seg_cropped
    #
    #         paste_tensor(self.interactions[0], pred_seg, scaled_bbox)
    #         paste_tensor(self.target_buffer, pred_seg, global_bbox)
    #
    #         # Show in napari if available
    #         if self._viewer is not None:
    #             if "Prediction" not in self._viewer.layers:
    #                 self._viewer.add_labels(pred_seg, name="Prediction")
    #             else:
    #                 self._viewer.layers["Prediction"].data = pred_seg
    #
    #         # --- BEGIN REFINEMENT ---
    #         print("[debug] Starting refinement...")
    #         diff_map = (self.interactions[0] != previous_prediction).astype(np.uint8)
    #
    #         # Smooth the difference map
    #         diff_map = binary_opening(diff_map, structure=np.ones((3, 3, 3)))
    #         diff_map = binary_dilation(diff_map, structure=np.ones((3, 3, 3))).astype(np.uint8)
    #
    #         # Get centers of changed regions
    #         all_boxes = generate_bounding_boxes(diff_map, bbox_size=patch_size, stride='auto',
    #                                                    margin=(10, 10, 10))
    #         print(f"[debug] {len(all_boxes)} refinement boxes created")
    #
    #         all_boxes.sort(key=lambda box: np.prod([b[1] - b[0] for b in box]), reverse=True)
    #
    #         refinement_boxes = all_boxes[:20]
    #         print(f"[debug] Cut down to {len(refinement_boxes)} boxes")
    #
    #         for box in refinement_boxes:
    #             # Compute center of box
    #             center = [(b[0] + b[1]) // 2 for b in box]
    #
    #             # Define fixed-size patch around the center
    #             fixed_box = [[c - s // 2, c + s // 2] for c, s in zip(center, patch_size)]
    #
    #             crop_img, pad_img = crop_to_valid(self.preprocessed_image, fixed_box)
    #             crop_int, pad_int = crop_to_valid(self.interactions, fixed_box)
    #
    #             if any(pad_img):
    #                 crop_img = pad_cropped(crop_img, pad_img)
    #             if any(pad_int):
    #                 crop_int = pad_cropped(crop_int, pad_int)
    #
    #             crop_img = crop_img[:4]
    #             crop_int = crop_int[:4]
    #
    #             input_data = np.concatenate((crop_img, crop_int), axis=0)[None].astype(np.float32)
    #             pred_logits = self.network.run(None, {self.input_name: input_data})[0]
    #             pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)
    #
    #             paste_tensor(self.interactions[0], pred_seg, fixed_box)
    #
    #             global_box = fixed_box
    #             paste_tensor(self.target_buffer, pred_seg, global_box)
    #         # --- END REFINEMENT ---
    #
    #     self.new_interaction_centers.clear()
    #     self.new_interaction_zoom_out_factors.clear()
    #
    #     del previous_prediction
    #
    #     print(f"Prediction complete in {round(time() - start_time, 3)} seconds.")

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f'Added new point interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1]-i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(bbox_center)
        # maybe im stupid... but why is it printing zoom out factors?
        # this is old code: print(f'Added new bbox interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')
        print(f'Added new bbox interaction: center {self.bbox_center}, scale {self.new_interaction_centers}')



    def _add_patch_for_scribble_interaction(self, scribble_image):
        return self._generic_add_patch_from_image(scribble_image)

    def _add_patch_for_lasso_interaction(self, lasso_image):
        return self._generic_add_patch_from_image(lasso_image)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def _generic_add_patch_from_image(self, image: torch.Tensor):
        if not torch.any(image):
            print('Received empty image prompt. Cannot add patches for prediction')
            return
        nonzero_indices = np.array(np.nonzero(image)).T
        mn = np.min(nonzero_indices, dim=0)[0]
        mx = np.max(nonzero_indices, dim=0)[0]
        roi = [[i.item(), x.item() + 1] for i, x in zip(mn, mx)]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1]- i[0] for i in roi]
        requested_size = [i + j // 3 for i, j in zip(roi_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(roi_center)
        print(f'Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {self.new_interaction_centers}')

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_fold: Union[int, str] = None,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        print("[debug] initialize_from_trained_model_folder called. Model is: ", model_training_output_dir)
        model_training_output_dir = str(model_training_output_dir)
        """
        This is used when making predictions with a trained model
        """
        # load trainer specific settings

        if model_training_output_dir.endswith(".onnx"):
            self.initialize_from_onnx_model(model_training_output_dir)
            return

        expected_json_file = join(model_training_output_dir, 'inference_session_class.json')
        json_content = load_json(expected_json_file)
        if isinstance(json_content, str):
            point_interaction_radius = json_content['point_radius']
            self.preferred_scribble_thickness = json_content['preferred_scribble_thickness']
            if not isinstance(self.preferred_scribble_thickness, (tuple, list)):
                self.preferred_scribble_thickness = [self.preferred_scribble_thickness] * 3
            self.interaction_decay = json_content['interaction_decay'] if 'interaction_decay' in json_content.keys() else 0.9
            point_interaction_use_etd = True # so far this is not defined in that file so we stick with default
            self.point_interaction = PointInteraction_stub(point_interaction_radius, point_interaction_use_etd)

        else:
            # padding mode for data. See nnInteractiveTrainerV2_nodelete_reflectpad
            self.pad_mode_data = json_content['pad_mode_image'] if 'pad_mode_image' in json_content.keys() else "constant"
            # ... you are probably gonna have to change this
            # old convention where we only specified the inference class in this file. Set defaults for stuff
            point_interaction_radius = 4
            point_interaction_use_etd = True
            self.point_interaction = PointInteraction_stub(
                point_interaction_radius,
                point_interaction_use_etd)
            self.pad_mode_data = "constant"
            self.interaction_decay = 0.9
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if use_fold is not None:
            use_fold = int(use_fold) if use_fold != 'all' else use_fold
            fold_folder = f'fold_{use_fold}'
        else:
            fldrs = subdirs(model_training_output_dir, prefix='fold_', join=False)
            assert len(fldrs) == 1, f'Attempted to infer fold but there is != 1 fold_ folders: {fldrs}'
            fold_folder = fldrs[0]

        checkpoint = torch.load(join(model_training_output_dir, fold_folder, checkpoint_name),
                                map_location=self.device, weights_only=False)
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']

        parameters = checkpoint['network_weights']

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnInteractive.__path__[0], "trainer"),
                                                    trainer_name, 'nnInteractive.trainer')
        if trainer_class is None:
            print(f'Unable to locate trainer class {trainer_name} in nnInteractive.trainer. '
                               f'Please place it there (in any .py file)!')
            print('Attempting to use default nnInteractiveTrainer_stub. If you encounter errors, this is where you need to look!')
            trainer_class = nnInteractiveTrainer_stub

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        network.load_state_dict(parameters)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def initialize_from_onnx_model(self, model_path: str):
        # stub values for onnx

        global onnx_session_initialized

        print("[debug] initialize_from_onnx_model HAS BEEN CALLED!")
        point_radius = 4
        self.pad_mode_data = "constant"
        self.point_interaction = PointInteraction_stub(
            point_radius,
            use_etd=True
        )
        self.interaction_decay = 0.9
        # Dummy or default configuration
        # wahh! i commented this all out! i am too lazy to find the actual dummies for these.
        # self.configuration_manager = configuration_manager
        # dummy_plans = PlansManager.default_plans()
        # dummy_labels = dummy_plans.get_label_manager({})

        # self.plans_manager = dummy_plans
        # self.label_manager = dummy_labels
        # self.dataset_json = {}

        # Run ONNX session
        import onnxruntime as ort
        self.onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


        for inp in self.onnx_session.get_inputs():
            print("Input name:", inp.name)
            print("Shape:", inp.shape)
            print("Type:", inp.type)

        self.network = ONNXWrapper(self.onnx_session)
        print("ONNX model loaded successfully.")


        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.network = ONNXWrapper(self.onnx_session)
        print("ONNX model loaded successfully.")



    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager,
                              dataset_json: dict, trainer_name: str):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        if not self.use_torch_compile and isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

        self.network = self.network.to(self.device)

def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    print("[debug] transform_coordinates_noresampling has been called")
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple([coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0] for d in range(len(coords_orig))])

def resize_np(arr: np.ndarray, new_shape, order=1):
    print("[debug] resize_np was called")
    """
    Resize a 3D or 4D NumPy array to the given shape using scipy.ndimage.zoom.

    Args:
        arr (np.ndarray): The array to resize. Shape (C, D, H, W) or (D, H, W)
        new_shape (tuple): The desired spatial shape (D, H, W)
        order (int): Interpolation order: 0 = nearest, 1 r= trilinear/linear

    Returns:
        np.ndarray: Resized array of shape (C, D, H, W) or (D, H, W)
    """
    if arr.ndim == 4:
        # Multi-channel (C, D, H, W)
        channels = []
        for c in arr:
            zoom_factors = [n / o for n, o in zip(new_shape, c.shape)]
            resized = zoom(c, zoom_factors, order=order)
            channels.append(resized)
        return np.stack(channels, axis=0)
    elif arr.ndim == 3:
        # Single-channel (D, H, W)
        zoom_factors = [n / o for n, o in zip(new_shape, arr.shape)]
        return zoom(arr, zoom_factors, order=order)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


class SimpleONNXConfig:
    print("[debug] Class SimpleONNXConfig has been called")
    def __init__(self, patch_size=(192, 192, 192)):
        self.patch_size = list(patch_size)

class ONNXWrapper:
    def __init__(self, session):
        self.session = session

    def __call__(self, input_tensor: torch.Tensor):
        input_np = input_tensor.cpu().numpy()
        outputs = self.session.run(None, {"input": input_np})  # you may need to check the input name
        return torch.from_numpy(outputs[0])  # Adjust based on actual output shape

if __name__ == "__main__":
    print("Running the file now!!")

    session = nnInteractiveInferenceSessionnew()
    print(f"Session type: {type(session)}")

    print(hasattr(session, 'add_point_interaction'))
