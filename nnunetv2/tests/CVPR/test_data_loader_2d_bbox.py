import json
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_2d_bbox import nnUNetDataLoader2DBBOX
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def visualize_batch(data):
    # find nearest square number of batch size
    nrows_cols = int(np.ceil(np.sqrt(data['data'].shape[0])))
    fig, axes = plt.subplots(nrows=nrows_cols, ncols=nrows_cols, figsize=(7*nrows_cols, 7*nrows_cols))

    def _add_subplot(ax, data, i):
        # image
        ax.imshow(data['data'][i, 0], cmap='gray')

        # segmentation
        seg = np.where(data['seg'][i, 0] > 0, 1, 0)
        ax.imshow(seg, cmap=ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.5)]), norm=Normalize(vmin=0, vmax=1))

        # bounding box
        box = np.where(data['seg'][i, 1] == 1)
        x_min, y_min, width, height = box[1].min(), box[0].min(), box[1].max()-box[1].min(), box[0].max()-box[0].min()
        rect = Rectangle(xy=(x_min, y_min), width=width, height=height, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.axis('off')

    if data['data'].shape[0] == 1:
        _add_subplot(axes, data, 0)
    else:
        # iterate over batch
        for i in range(data['data'].shape[0]):
            _add_subplot(axes[i // nrows_cols, i % nrows_cols], data, i)
        # delte empty subplots
        axes = axes.flatten()
        for i in range(data['data'].shape[0], nrows_cols**2):
            fig.delaxes(axes[i])
    fig.tight_layout()
    fig.show()
    print('done')


if __name__ == "__main__":
    dataset = nnUNetDataset('/home/r403k/Data/CVPR/sandbox/test_data', None, 0)
    with open('/home/r403k/Data/CVPR/sandbox/dataset.json', 'r') as f:
        label_dict = json.load(f)
    label_manager = LabelManager(label_dict['labels'], None)
    dataloader = nnUNetDataLoader2DBBOX(dataset, 16, (224, 288), (224, 288), label_manager=label_manager, bbox_dilation=20)
    data = next(iter(dataloader))
    visualize_batch(data)