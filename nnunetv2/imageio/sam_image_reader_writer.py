#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import json
import numpy as np
from skimage import io


from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class SAMImage2DIO(BaseReaderWriter):
    """
    ONLY SUPPORTS 2D IMAGES!!!
    """

    # there are surely more we could add here. Everything that can be read by skimage.io should be supported
    supported_file_endings = [
        '.jpg'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        for f in image_fnames:
            npy_img = io.imread(f)
            if npy_img.ndim == 3:
                # rgb image, last dimension should be the color channel and the size of that channel should be 3
                # (or 4 if we have alpha)
                assert npy_img.shape[-1] == 3 or npy_img.shape[-1] == 4, "If image has three dimensions then the last " \
                                                                         "dimension must have shape 3 or 4 " \
                                                                         f"(RGB or RGBA). Image shape here is {npy_img.shape}"
                # move RGB(A) to front, add additional dim so that we have shape (1, c, X, Y), where c is either 3 or 4
                images.append(npy_img.transpose((2, 0, 1))[:, None])
            elif npy_img.ndim == 2:
                # grayscale image
                images.append(npy_img[None, None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images).astype(np.float32), {'spacing': (999, 1, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        if seg_fname.endswith(".jpg"):
            seg_fname = seg_fname[:-4] + ".json"
        with open(seg_fname, "r") as f:
            annotations = json.load(f)["annotations"]
        return annotations, {"spacing": (999, 1, 1)}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        raise NotImplementedError("Write Seg is not implemented for this the SAM Reader/Writer")