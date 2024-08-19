# %%
import numpy as np
import torch
import torch.nn as nn

# create a custom layer
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLayer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        x = np.random.randn(x.shape[0], self.in_features)
        x = torch.tensor(x, dtype=torch.float32)
        return torch.matmul(x, self.weight) + self.bias

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.custom = CustomLayer(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.custom(x)
        x = self.fc2(x)
        return x
    
# Create a simple dataset
X = torch.randn(100, 2)
y = torch.randn(100, 1)

# dataset
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# train the model
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    for i, (X_batch, y_batch) in enumerate(dataloader):
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')



