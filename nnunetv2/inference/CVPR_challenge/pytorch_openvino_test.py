import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import time

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)

model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
torch_model.eval()
# torch_model.half()
n_iter = 1#000
input_tensor = torch.randn(batch_size, 1, 224, 224, requires_grad=False)
start = time.time()
for _ in range(n_iter):
    torch_out = torch_model(input_tensor)
print('pytorch elapsed', time.time() - start)


import openvino as ov
core = ov.Core()
ov_model = ov.convert_model(torch_model)
compiled_model = core.compile_model(ov_model, "CPU")

start = time.time()
for _ in range(n_iter):
    result = compiled_model(input_tensor)[0]
print('OpenVINO compiled model elapsed', time.time() - start)


ov_model_2 = ov.convert_model(torch_model, example_input= input_tensor)
ov_static_model = core.compile_model(ov_model_2, "CPU")

start = time.time()
for _ in range(n_iter):
    result = ov_static_model(input_tensor)[0]
print('OpenVINO converted model elapsed', time.time() - start)