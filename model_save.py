import torch
import torch.nn as nn
from base_model import LinearModel

# set up model
channels, height, width = 3, 224, 224
input_dim = channels * height * width
num_classes = 290
model = LinearModel(input_dim, num_classes)

# load in saved model weights
model.load_state_dict(torch.load('base_model_weights.pth'))

model.half()  # Convert model parameters to half precision


torch.save(model.state_dict(), 'base_model_weights_half.pth')
