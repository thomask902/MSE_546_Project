import torch
import torch.nn as nn
from base_model import LinearModel
from torch.utils.data import DataLoader, random_split

# load in data with same splitting seed as training
dataset = torch.load('dataset.pt')
print("Dataset loaded")
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size
torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print("Dataset split into test and train")

# set up model
channels, height, width = 3, 224, 224
input_dim = channels * height * width
num_classes = 290
model = LinearModel(input_dim, num_classes)
print("model created")

# load in saved model weights
state_dict_half = torch.load('base_model_weights_half.pth')

# Convert each tensor in the state dictionary to FP32
state_dict_fp32 = {k: v.float() for k, v in state_dict_half.items()}

# Load the converted state dictionary
model.load_state_dict(state_dict_fp32)
print("model weights set up")

# EVALUATE HERE USING TEST DATASET

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# basic accuracy measure (replace with MAP @ 20)
def evaluate_accuracy(model, loader, threshold=0.5):
    model.eval()
    correct = 0  
    total_labels = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            # converts output to probability then checks if >= threshold
            preds = torch.sigmoid(outputs) >= threshold
            # sum number of correct labels
            correct += (preds == labels.byte()).sum().item()
            total_labels += labels.numel()
    # divide number of correct labels by total number of labels
    accuracy = correct / total_labels
    return accuracy

print("evaluating accuracy")
test_accuracy = evaluate_accuracy(model, test_loader)
print(f"Test per-label accuracy: {test_accuracy:.4f}")

