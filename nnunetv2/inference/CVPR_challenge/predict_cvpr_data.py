import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class CVPRPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        super().__init__(perform_everything_on_device=perform_everything_on_device, device=device, verbose=verbose,
                         verbose_preprocessing=verbose_preprocessing, allow_tqdm=allow_tqdm)

    def predict_case_with_bbox(self, npz_file, output_dir):
        npz_file = Path(npz_file)
        npz_name = npz_file.name
        try:
            np.load(output_dir/npz_name)
            return
        except:
            (output_dir/npz_name).unlink(missing_ok=True)
        with np.load(npz_file) as f:
            imgs = f['imgs'].astype(np.float16)
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
                                 save_or_return_probabilities: bool = False):
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
            predicted_logits = self.predict_logits_from_preprocessed_data(net_input).cpu().numpy()
            prediction = np.argmax(predicted_logits, 0)[0]
            prediction[bbox_mask[0, 0] == 0] = 0
            seg[prediction == 1] = idx

        return seg


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
        default='',
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
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if args.output_dir is None:
        output_dir = input_dir.parent / "predictions"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    predictor = CVPRPredictor(allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args.model_path, (0,), args.checkpointname)
    for npz_file in tqdm(list(input_dir.glob("*.npz"))):
        predictor.predict_case_with_bbox(npz_file, output_dir)
