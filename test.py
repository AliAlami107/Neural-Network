# Import packages we need
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

raw_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())


data_loader = torch.utils.data.DataLoader(
    raw_dataset, batch_size=len(raw_dataset)
)
data = next(iter(data_loader))
mean = data[0].mean()  
std = data[0].std()

print(f"The mean of the FashionMNIST is {mean} and the standard deviation is {std}")


plt.hist(data[0].flatten())
plt.axvline(data[0].mean(), label="Mean", color="r")
plt.title("Figure 1: Histogram of unnormalized data")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

train_set_normalized = datasets.FashionMNIST(root="data", 
                           download=True, 
                           train=True, 
                           transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                           ]))
test_set_normalized = datasets.FashionMNIST(root="data", 
                           download=True, 
                           train=False, 
                           transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                           ]))


train_loader = torch.utils.data.DataLoader(
    train_set_normalized, batch_size=len(train_set_normalized), num_workers=1
)
data = next(iter(train_loader))
mean_normalized = data[0].mean(), 
std_normalized = data[0].std()
print(f"The mean of the FashionMNIST (standarized) is {mean_normalized} and the standard deviation is {std_normalized}")


plt.hist(data[0].flatten())
plt.axvline(data[0].mean(), label="Mean", color="r")
plt.title("Figure 2: Histogram of normalized data")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.flatten = nn.Flatten() # To flatten (28x28) -> 784
        self.input_layer = nn.Linear(in_features=784, out_features=hidden_units)
        self.output_layer = nn.Linear(in_features=hidden_units, out_features=10) # 10 classes
    
    def forward(self, x):
        # We dont need activation function for flattening
        x = self.flatten(x)
        
        # We choose relu activation function
        x = nn.functional.relu(self.input_layer(x))
        
        x = self.output_layer(x)
        
        # We need an output activation function, we choose softmax because of multiclass classification.
        return nn.functional.softmax(x, dim=1)
    

# Set seed to allow for reproducability
torch.manual_seed(42)


batch_size = 32
hidden_units = 5
learning_rate = 2
epochs = 3
loss_func = torch.nn.CrossEntropyLoss()
model = NeuralNetwork(hidden_units=hidden_units) # DO NOT CHANGE ME!
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)


train_loader = torch.utils.data.DataLoader(
    train_set_normalized, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_set_normalized, batch_size=batch_size, shuffle=False
)


training_loss_epochs = []
test_loss_epochs = []
for epoch in range(epochs):
    print(f"---\nEpoch: {epoch}")
    train_loss = 0
    test_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        # Set model in training mode
        model.train()
        
        # 1. Forward pass
        y_pred = model(X)
        
        # 2. Calculate loss
        loss = loss_func(y_pred, y)
        train_loss += loss 
        
        # 3. Optimizer zero grad (reset all gradients)
        optimizer.zero_grad()
        
        # 4. Back propagation
        loss.backward()
        
        # 5. Step the optimizer, ie. update parameters
        optimizer.step()
    
    # Now we calculate the test loss per epoch
    model.eval()
    with torch.inference_mode():
        for X, y in test_loader:
            # 1. Forward pass
            test_pred = model(X)
           
            # 2. Calculate loss (accumulatively)
            test_loss += loss_func(test_pred, y) # accumulatively add up the loss per epoch
        
        # Divide total test loss by the batch
        test_loss /= len(test_loader)
        test_loss_epochs.append(test_loss.detach().numpy())
        
    train_loss = train_loss / len(train_loader)
    training_loss_epochs.append(train_loss.detach().numpy())
    
    print(f"Train loss: {train_loss}, test loss: {test_loss}")


epochs_list = np.arange(stop=epochs, step=1)
plt.plot(epochs_list, training_loss_epochs, label="Train", marker="o")
plt.plot(epochs_list, test_loss_epochs, label="Test", marker="o")
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs_list)
plt.legend()
plt.show()

model.eval()
correct = 0
# Inference mode - do no gradient calculation
with torch.inference_mode():
    for batch, (X, y) in enumerate(test_loader):
        # Get the prediction of the model
        y_pred = model(X)
        
        # Retrieve the label of the prediction
        pred_labels = y_pred.argmax(dim=1)
        
        # Get the correct labels (ground_truth)
        ground_labels = y
        
        # Find out if the predictions are the same as ground truth - equality
        # torch.eq gives boolean tensor, we convert it to 1/0
        correct_labels = pred_labels.eq(ground_labels).long()
        
        # Sum up the correct count for this batch
        correct += correct_labels.sum()


print(f"The accuracy of the model is {correct/len(test_loader.dataset)*100}.")