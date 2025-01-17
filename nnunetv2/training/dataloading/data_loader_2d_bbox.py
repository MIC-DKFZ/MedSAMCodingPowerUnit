from batchgenerators.utilities.file_and_folder_operations import List
from skimage.morphology import dilation
from skimage.measure import label
from typing import Tuple
import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoader2DBBOX(nnUNetDataLoaderBase):
    def __init__(self, data: nnUNetDataset, batch_size: int, patch_size: List[int] | Tuple[int] | np.ndarray, 
                 final_patch_size: List[int] | Tuple[int] | np.ndarray, label_manager: LabelManager, 
                 oversample_foreground_percent: float = 0, sampling_probabilities: List[int] | Tuple[int] | np.ndarray = None, 
                 pad_sides: List[int] | Tuple[int] | np.ndarray = None, probabilistic_oversampling: bool = False,
                 footprint: int = None, bbox_dilation: int = 0):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, sampling_probabilities, pad_sides, probabilistic_oversampling)
        self.footprint = footprint
        self.bbox_dilation = bbox_dilation

    def get_bbox(self, seg: np.ndarray):
        uniques = np.unique(seg)
        if np.any(uniques>0):
            cls = np.random.choice(uniques[uniques>0])
            if self.footprint is not None:
                struct = np.ones((self.footprint,)*2, dtype=np.uint8)
                dilated_seg = dilation((seg[0]==cls).astype(np.uint8), struct)
            else:
                dilated_seg = (seg[0] == cls).astype(np.uint8)
            seg_inst, seg_num = label(dilated_seg, return_num=True)
            if seg_num > 1:
                rand_inst = np.random.randint(1, seg_num+1)
                seg_inst = np.where(seg_inst == rand_inst, 1, 0).astype(np.uint8)
            seg_inst[seg[0]==0] = 0
            w = np.where(seg_inst == 1)
            bbox = [np.min(w[0]), np.max(w[0])+1, np.min(w[1]), np.max(w[1])+1]
            size = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
            bbox_lbs = [np.random.randint(bbox[0]-max(0, self.patch_size[0]-size[0]), bbox[0]+1),
                        np.random.randint(bbox[2]-max(0, self.patch_size[1]-size[1]), bbox[2]+1)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(2)]
            bbox_mask = np.zeros_like(seg)
            jitter = np.random.randint(0, self.bbox_dilation+1, size=4)
            bbox_mask[:, bbox[0]-jitter[0]:bbox[1]+jitter[1], bbox[2]-jitter[2]:bbox[3]+jitter[3]] = 1

            return bbox_lbs, bbox_ubs, np.concatenate([seg_inst[None].astype(seg.dtype), bbox_mask.astype(seg.dtype)], axis=0)
        else:
            need_to_pad = self.need_to_pad.copy()
            shape = seg.shape[1:]

            for d in range(2):
                if need_to_pad[d] + shape[d] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - shape[d]

            lbs = [- need_to_pad[i] // 2 for i in range(2)]
            ubs = [shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(2)]

            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(2)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(2)]

            return bbox_lbs, bbox_ubs, np.concatenate([seg, np.zeros_like(seg)], axis=0)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros((self.seg_shape[0], 2, *self.seg_shape[2:]), dtype=np.int16)
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
            bbox_lbs, bbox_ubs, seg = self.get_bbox(seg)

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


class nnUNetDataLoader2DBBOX2HEAD(nnUNetDataLoader2DBBOX):
    def get_bbox(self, seg: np.ndarray):
        uniques = np.unique(seg)
        if np.any(uniques>0):
            cls = np.random.choice(uniques[uniques>0])
            if self.footprint is not None:
                struct = np.ones((self.footprint,)*2, dtype=np.uint8)
                dilated_seg = dilation((seg[0]==cls).astype(np.uint8), struct)
            else:
                dilated_seg = (seg[0] == cls).astype(np.uint8)
            seg_inst, seg_num = label(dilated_seg, return_num=True)
            if seg_num > 1:
                rand_inst = np.random.randint(1, seg_num+1)
                seg_inst = np.where(seg_inst == rand_inst, 1, 0).astype(np.uint8)
            seg_inst[seg[0]==0] = 0
            w = np.where(seg_inst == 1)
            bbox = [np.min(w[0]), np.max(w[0])+1, np.min(w[1]), np.max(w[1])+1]
            size = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
            bbox_lbs = [np.random.randint(bbox[0]-max(0, self.patch_size[0]-size[0]), bbox[0]+1),
                        np.random.randint(bbox[2]-max(0, self.patch_size[1]-size[1]), bbox[2]+1)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(2)]
            bbox_mask = np.zeros_like(seg)
            jitter = np.random.randint(0, self.bbox_dilation+1, size=4)
            bbox_mask[:, bbox[0]-jitter[0]:bbox[1]+jitter[1], bbox[2]-jitter[2]:bbox[3]+jitter[3]] = 1

            return bbox_lbs, bbox_ubs, np.concatenate([(seg>0).astype(seg.dtype), seg_inst[None].astype(seg.dtype), bbox_mask.astype(seg.dtype)], axis=0)
        else:
            need_to_pad = self.need_to_pad.copy()
            shape = seg.shape[1:]

            for d in range(2):
                if need_to_pad[d] + shape[d] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - shape[d]

            lbs = [- need_to_pad[i] // 2 for i in range(2)]
            ubs = [shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(2)]

            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(2)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(2)]

            return bbox_lbs, bbox_ubs, np.concatenate([seg, seg, np.zeros_like(seg)], axis=0)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros((self.seg_shape[0], 3, *self.seg_shape[2:]), dtype=np.int16)
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
            bbox_lbs, bbox_ubs, seg = self.get_bbox(seg)

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


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset004_Hippocampus/2d'
    ds = nnUNetDataset(folder, None, 1000)  # this should not load the properties!
    dl = nnUNetDataLoader2DBBOX(ds, 366, (65, 65), (56, 40), 0.33, None, None)
    a = next(dl)
