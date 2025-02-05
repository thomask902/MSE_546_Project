import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from image_dataset import MultiLabelImageDataset

# hyperparameters
batch_size = 32
channels, height, width = 3, 224, 224 
lr = 1e-3
num_epochs = 3
threshold = 0.5

# define basic logistic regression model (just linear regression for now, will become logistic due to loss used)
class LinearModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearModel, self).__init__()

        # this is a linear layer from input to classes, we will train coefficents for this
        self.linear = nn.Linear(input_dim, num_classes)
    
    # defines how data is passed through model for inference
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten each image into a vector
        return self.linear(x) # pass into linear layer to get output

def train():
    # load in dataset and do test train split (80/20)
    dataset = torch.load('dataset.pt')
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")

    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # create dataloaders (for now no batching so these are kind of unneeded)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize the model
    input_dim = channels * height * width
    model = LinearModel(input_dim, 290)

    # define loss and optimizer
    criterion = nn.BCEWithLogitsLoss() # this applies sigmoid to each of our regression values, making it logistic regression
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch = 0
        for images, labels in train_loader:
            batch += 1
            print(f"Batch #{batch}")
            # forward pass to get outputs and compute loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # compute total loss over number of samples in batch
            running_loss += loss.item() * images.size(0)

        # find average loss over the epoch    
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'base_model_weights.pth')

if __name__ == "__main__":
    train()


'''
# basic accuracy measure (replace with MAP @ 20)
def evaluate_accuracy(model, loader, threshold=threshold):
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


test_accuracy = evaluate_accuracy(model, test_loader)
print(f"Test per-label accuracy: {test_accuracy:.4f}")

#train_accuracy = evaluate_accuracy(model, train_loader)
#print(f"Train per-label accuracy: {train_accuracy:.4f}")
'''
