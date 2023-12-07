# THIS MODEL IS TAKEN FROM GEEKS FOR GEEKS
# 
# ENCODER STRUCTURE
# Feed-forward neural network given by hz = fe(ge(xi))
# ge = hidden layer 1
# fe = hidden layer 2
# xi = input of the autoencoder
# hz = low-dimensional data space of the input
#
# DECODER STRUCTURE
# Feed-forward neural network but the dimension increases -- xbar = gd(fd(hz))
# fd = hidden layer 1
# gd = hidden layer 2
# hz = low-dimensional data space (output of encoder)
# xbar = reconstructed input

import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Transforms images into PyTorch tensors
tensor_transform = transforms.ToTensor()

#download MNIST dataset
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)

# Load dataset for training
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 32,
                                     shuffle = True)


# Dimensions: 28*28 = 784 --> 128 --> 64 --> 36 --> 18 --> 9
# Vice versa for Decoder
# create PyTorch class
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Linear encoder construction with Relu activation function
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )
        
        # Linear decoder with same construction as above
        # Additional Sigmoid activation function to output values between 0 and 1
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Model initialzation
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Use Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

# run the model
epochs = 20
outputs = []
losses = []
for epoch in tqdm(range(epochs)):
    for (image, x) in tqdm(loader, leave=False):
        # Reshape image to (-1, 784)
        image = image.reshape(-1, 28*28)
        
        # Output of Autoencoder
        reconstructed = model(image)
        # Calculate loss
        loss = loss_function(reconstructed, image)
        
        # Set gradients to zero,
        # compute gradients & store
        # .step() to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store the losses for plotting -- 
        losses.append(loss.detach().numpy())
    outputs.append((epochs, image, reconstructed))
    
# Plot
plt.figure(1)
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
# Last 100 values only
plt.plot(losses)

# Show image and AE outputs
fig1 = plt.figure(2, figsize=(8,8))
rows = 4
columns = 8
for i, item in enumerate(image):
    #Reshape the array for plotting
    item = item.reshape(-1, 28, 28)
    fig1.add_subplot(rows, columns, i+1)
    plt.imshow(item[0])

fig2 = plt.figure(3, figsize=(8,8))
rows = 4
columns = 8
print('These are now the reconstructed versions!')
for i, item in enumerate(reconstructed):
    item = item.reshape(-1, 28, 28).detach().numpy()
    print(item)
    fig2.add_subplot(rows, columns, i+1)
    plt.imshow(item[0])
plt.show()
