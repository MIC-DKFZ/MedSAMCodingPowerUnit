from batchgenerators.utilities.file_and_folder_operations import List
from skimage.morphology import dilation
from skimage.measure import label
from typing import Tuple
import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoader2DSAM(nnUNetDataLoaderBase):
    def __init__(self, data: nnUNetDataset, batch_size: int, patch_size: List[int] | Tuple[int] | np.ndarray, 
                 final_patch_size: List[int] | Tuple[int] | np.ndarray, label_manager: LabelManager, 
                 oversample_foreground_percent: float = 0, sampling_probabilities: List[int] | Tuple[int] | np.ndarray = None, 
                 pad_sides: List[int] | Tuple[int] | np.ndarray = None, probabilistic_oversampling: bool = False,
                 num_bbox: int = 1, bbox_dilation: int = 0):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, sampling_probabilities, pad_sides, probabilistic_oversampling)
        self.num_bbox = num_bbox
        self.bbox_dilation = bbox_dilation

    def get_bbox(self, seg: np.ndarray, properties: dict):
        targets = np.random.choice(list(properties["mask_dict"].keys()), self.num_bbox, replace=True)
        labels = [properties["mask_dict"][target] for target in targets]
        bboxs = [properties["bboxs"][target-1] for target in targets]
        bboxs = [list(map(int, [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]])) for bbox in bboxs]
        seg_bin = np.zeros((self.num_bbox, *seg.shape[1:]), dtype=seg.dtype)
        for i, label in enumerate(labels):
            seg_bin[i] = np.where(np.isin(seg, list(label)), 1, 0)

        need_to_pad = self.need_to_pad.copy()
        data_shape = seg.shape[1:]
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        bbox_lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        # bbox_lbs = [np.random.randint(bbox[0]-max(0, self.patch_size[0]-size[0]), bbox[0]+1),
                    # np.random.randint(bbox[2]-max(0, self.patch_size[1]-size[1]), bbox[2]+1)]
        # bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(2)]
        bbox_mask = np.zeros_like(seg_bin)
        jitter = np.random.randint(0, self.bbox_dilation+1, size=(self.num_bbox, 4))
        for i, bbox in enumerate(bboxs):
            bbox_mask[i, bbox[0]-jitter[i][0]:bbox[1]+jitter[i][1], bbox[2]-jitter[i][2]:bbox[3]+jitter[i][3]] = 1

        return bbox_lbs, bbox_ubs, np.concatenate([seg_bin.astype(seg.dtype), bbox_mask.astype(seg.dtype)], axis=0)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros((self.seg_shape[0], 2*self.num_bbox, *self.seg_shape[2:]), dtype=np.int16)
        case_properties = []

        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, seg, properties = self._data.load_case(current_key)
            case_properties.append(properties)

            selected_slice = np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]

            # print(properties)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs, seg = self.get_bbox(seg, properties)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}