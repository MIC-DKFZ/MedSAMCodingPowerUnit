import time
from typing import List, Tuple
import argparse
import numpy as np
from pathlib import Path
import random
import torch
from tqdm import tqdm
#from numba import jit
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


#@jit(nopython=True)
def _get_min_max_crop(d_patch_size, context_fraction, d_min, d_max, image_max) -> Tuple[int, int]:
    """
    Return the min and max values to crop the image, i.e. the part of the image we consider for prediction.
    """
    d_min_res = max(0, d_min - int(context_fraction * d_patch_size))
    d_max_res = min(image_max, d_max + int(context_fraction * d_patch_size))

    # d_max_res - d_min_res can be smaller than patch size -> enlarge crop to patch size
    if d_max_res - d_min_res < d_patch_size:
        current_size = d_max_res - d_min_res
        free_space = d_patch_size - current_size
        d_max_res = d_max_res + int(np.ceil(free_space / 2))  # .astype(np.int32)
        optional_context = 0
        if d_max_res > image_max:
            optional_context += d_max_res - image_max
            d_max_res = image_max
        # d_min_res = d_min_res - np.floor(free_space / 2).astype(np.int32) - optional_context
        d_min_res = d_min_res - int(np.floor(free_space / 2)) - optional_context
        if d_min_res < 0:
            optional_context += -d_min_res
            d_min_res = 0
            d_max_res = min(image_max, d_max_res + optional_context)
    return d_min_res, d_max_res

class CVPRPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = False,
                 use_mirroring: bool = False,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 is_openvino: bool = False):
        super().__init__(perform_everything_on_device=perform_everything_on_device, device=device, verbose=verbose,
                         verbose_preprocessing=verbose_preprocessing, allow_tqdm=allow_tqdm, use_gaussian = use_gaussian,
                         use_mirroring = use_mirroring)
        self.is_openvino = is_openvino

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        self.network.eval()
        for params in self.list_of_parameters:
            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)
        if self.is_openvino:
            print('OpenVino')
            import openvino as ov
            core = ov.Core()
            # input_tensor = torch.randn(1, 4, 224, 224, requires_grad=False)
            ov_model = ov.convert_model(self.network)  # , example_input=input_tensor)
            self.network = core.compile_model(ov_model, "CPU")

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        if self.is_openvino:
            prediction = torch.from_numpy(self.network(x)[0])
        else:
            prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
            ]
            for axes in axes_combinations:
                if not self.is_openvino:
                    prediction += torch.flip(self.network(torch.flip(x, (*axes,))), (*axes,))
                else:
                    temp_pred = torch.from_numpy(self.network(torch.flip(x, (*axes,)))[0])
                    prediction += torch.flip(temp_pred, (*axes,))
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def predict_case_with_bbox(self, npz_file, output_dir):
        npz_file = Path(npz_file)
        npz_name = npz_file.name
        #try:
        #    np.load(output_dir/npz_name)
        #    return
        #except:
        #    (output_dir/npz_name).unlink(missing_ok=True)
        with np.load(npz_file) as f:
            imgs = f['imgs'].astype(np.float32)
            bboxs = f["boxes"]
        segs = np.zeros_like(imgs, dtype=np.uint16)
        if npz_name.startswith("3D"):
            for z in range(imgs.shape[0]):
                bboxs_2d = {idx: [bbox[0], bbox[1], bbox[3], bbox[4]] for idx, bbox in enumerate(bboxs, 1) if bbox[2] <= z < bbox[5]}
                if len(bboxs_2d) == 0:
                    continue
                imgs_2d = imgs[z]
                if imgs_2d.ndim == 2:
                    imgs_2d = np.repeat(imgs_2d[None, None], 3, axis=0)
                segs[z] = self.predict_2d_npy_array_with_bbox(imgs_2d, {'spacing': (999, 1, 1)}, bboxs_2d)
        else:
            bboxs = {idx: bbox for idx, bbox in enumerate(bboxs, 1)}
            imgs = imgs.transpose((2, 0, 1))[:, None] # (3, 1, H, W)
            segs = self.predict_2d_npy_array_with_bbox(imgs, {'spacing': (999, 1, 1)}, bboxs)
        np.savez_compressed(output_dir/npz_name, segs=segs)

    def predict_2d_npy_array_with_bbox(self, input_image: np.ndarray, image_properties: dict, input_bboxs: np.ndarray,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False,
                                 crop_to_bbox_mask: bool = True,
                                 context_fraction: float = 0.25):
        """
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        dct = next(ppa)
        data = dct["data"]
        original_shape = dct["data_properties"]["shape_before_cropping"]
        crop_bbox = dct["data_properties"]["bbox_used_for_cropping"]
        to_pad = np.array([[crop_bbox[i][0], original_shape[i] - crop_bbox[i][1]] for i in [2, 1, 0]] + [[0, 0]]).flatten().tolist()
        data = torch.nn.functional.pad(data, to_pad, mode='constant')
        seg = np.zeros(data.shape[2:], dtype=np.uint16)

        for idx, bbox in input_bboxs.items():
            bbox_mask = torch.zeros((1, 1, *data.shape[2:]), dtype=data.dtype)
            bbox_mask[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            net_input = torch.cat([data, bbox_mask], dim=0)
            if crop_to_bbox_mask:
                net_input, to_pad_input = self.crop_to_bbox_mask(net_input, bbox, context_fraction)
            predicted_logits = self.predict_logits_from_preprocessed_data(net_input).cpu()
            if crop_to_bbox_mask:
                predicted_logits = torch.nn.functional.pad(predicted_logits, to_pad_input, mode='constant')
            prediction = np.argmax(predicted_logits.numpy(), 0)[0]
            prediction[bbox_mask[0, 0] == 0] = 0
            seg[prediction == 1] = idx

        return seg

    def crop_to_bbox_mask(self, net_input: torch.Tensor, bbox: np.ndarray, context_fraction: int = 0.25) -> Tuple[torch.Tensor, List]:
        '''
        Return the cropped image and the padding to recover the original shape.
        '''
        global _get_min_max_crop
        patch_size = self.configuration_manager.patch_size
        x_min, y_min, x_max, y_max = bbox
        x_image_max = net_input.shape[3]
        y_image_max = net_input.shape[2]
        # def _get_min_max_crop(d_patch_size, context_fraction, d_min, d_max, image_max) -> Tuple[int, int]:
        #     """
        #     Return the min and max values to crop the image, i.e. the part of the image we consider for prediction.
        #     """
        #     d_min_res = max(0, d_min - int(context_fraction * d_patch_size))
        #     d_max_res = min(image_max, d_max + int(context_fraction * d_patch_size))
        #
        #     # d_max_res - d_min_res can be smaller than patch size -> enlarge crop to patch size
        #     if d_max_res - d_min_res < d_patch_size:
        #         current_size = d_max_res - d_min_res
        #         free_space = d_patch_size - current_size
        #         d_max_res = d_max_res + np.ceil(free_space / 2).astype(int)
        #         optional_context = 0
        #         if d_max_res > image_max:
        #             optional_context += d_max_res - image_max
        #             d_max_res = image_max
        #         d_min_res = d_min_res - np.floor(free_space / 2).astype(int) - optional_context
        #         if d_min_res < 0:
        #             optional_context += -d_min_res
        #             d_min_res = 0
        #             d_max_res = min(image_max, d_max_res + optional_context)
        #     return d_min_res, d_max_res

        # TODO: check which patch size goes in which image dim...
        x_min_res, x_max_res = _get_min_max_crop(patch_size[1], context_fraction, x_min, x_max, x_image_max)
        y_min_res, y_max_res = _get_min_max_crop(patch_size[0], context_fraction, y_min, y_max, y_image_max)

        # TODO: understand why padding can be negative here... -> seemingly I used the wrong {x,y}_image_max wrt {x,y}_{min,max}_res
        return net_input[:, :, y_min_res:y_max_res, x_min_res:x_max_res], [x_min_res, x_image_max - x_max_res, y_min_res, y_image_max - y_max_res, 0, 0, 0, 0]
        # return net_input[:, :, y_min_res:y_max_res, x_min_res:x_max_res], [y_min_res, max(0, y_image_max - y_max_res), x_min_res, max(0, x_image_max - x_max_res), 0, 0, 0, 0]


def main(args):
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    if args.device == 'cuda':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(f"Device {args.device} not supported")

    args.fold = int(args.fold) if args.fold != 'all' else 'all'
    predictor = CVPRPredictor(allow_tqdm=False, device=device)
    predictor.initialize_from_trained_model_folder(args.model_path, (args.fold,), args.checkpointname)
    if hasattr(predictor.network, 'return_unet_head'):
        predictor.network.return_unet_head = False
    random.seed(42)
    files_to_predict = sorted(list(input_dir.glob("*.npz")))
    if args.modality is not None:
        files_to_predict = [f for f in files_to_predict if args.modality in f.name]
    random.shuffle(files_to_predict)
    if args.num_gpus > 1:
        files_to_predict = files_to_predict[args.gpu_id::args.num_gpus]
    for npz_file in tqdm(files_to_predict):
        predictor.predict_case_with_bbox(npz_file, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input_dir',
        type=str,
        required=False,
        default="/home/y033f/DataDrive/CVPR_challenge/CVPR_MedSAM/CVPR24-MedSAMLaptopData/imgs",
        help='root directory of the data',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        required=False,
        help='directory to save the prediction',
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        default='/home/y033f/DataDrive/CVPR_challenge/trained_models/Dataset987_CodingPowerUnit/nnUNetTrainerCPU__nnUNetResEncPlans__2d',
        required=False,
        help='directory to the model checkpoint',
    )
    parser.add_argument(
        '-c',
        '--checkpointname',
        type=str,
        default='checkpoint_final.pth',
        required=False,
        help='checkpoint file name',
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
        required=False,
        help='number of GPUs to use'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        required=False,
        help='GPU id to use'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default=None,
        required=False,
        help='If set only the modality provided will be predicted'
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if args.output_dir is None:
        output_dir = input_dir.parent / "predictions"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    predictor = CVPRPredictor(allow_tqdm=False, device = torch.device('cpu'), is_openvino = True)
    predictor.initialize_from_trained_model_folder(args.model_path, (0,), args.checkpointname)
    random.seed(42)
    files_to_predict = sorted(list(input_dir.glob("*.npz")))
    if args.modality is not None:
        files_to_predict = [f for f in files_to_predict if args.modality in f.name]
    random.shuffle(files_to_predict)
    if args.num_gpus > 1:
        files_to_predict = files_to_predict[args.gpu_id::args.num_gpus]
    for npz_file in tqdm(files_to_predict):
        predictor.predict_case_with_bbox(npz_file, output_dir)
