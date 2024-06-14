"""Tests on the data set."""

import torch
import torchvision
import numpy as np
from util.tensors import count_unique_values
import torch.nn.functional as F

def test_data_items():
     mask = np.load("./tests/let-late-young-result_00_00-00-256-3264_853_mask.npz")
     assert(np.max(mask) == 2)

     mask = np.load("./tests/look-final-human-air_00_00-00-256-3264_854_mask.npz")
     assert(np.max(mask) == 1)

     target = torch.from_numpy(mask).to(dtype=torch.int64)
     target = F.one_hot(target, num_classes=3)
     target = target.permute(3, 0, 1, 2)
     target = target.to(dtype=float)

     squashed = torch.argmax(target, dim=0)
     squashed = squashed.max(dim=0).values
     print("Squashed Target", squashed[35, 5], squashed.shape)
 
     # Apply the color_map transform to the input tensor
     # Three RGB values, one for each class
     colour_palette = torch.tensor([ [80, 70, 90], [255, 66, 0], [12, 205, 68]], dtype=torch.uint8)
     colour_palette = colour_palette.permute(1,0)
     output_tensor = colour_palette[:, squashed]

     colour_map_img = torchvision.transforms.ToPILImage()(output_tensor)
     colour_map_img.save("./tests/test_data_target.png")

     values, counts = count_unique_values(target)
     assert(values[1] == 1) # Background will always be the first one.

     # Fake up a prediction
     fake = torch.rand((3,16,816,256))
     fake = torch.argmax(fake, dim=0)
     fake = fake.max(dim=0).values

     output_tensor = colour_palette[:, fake]

     colour_map_img = torchvision.transforms.ToPILImage()(output_tensor)
     colour_map_img.save("./tests/test_data_pred.png")

