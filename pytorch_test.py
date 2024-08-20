# %%

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm


class QuantumConv2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(QuantumConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size))
        self.simulator = AerSimulator()
        self.circ = self.build_circuit()

    def build_circuit(self):
        lens = self.kernel_size**2
        qc = QuantumCircuit(lens)
        for i in range(lens):
            qc.h(i)

        for i in range(lens):
            angle = Parameter('enc_0_'+str(i))
            qc.rz(angle,i)
        
        for i in range(lens-1):
            for j in range(i+1, lens):
                angle = Parameter('enc_1_'+str(i)+'_'+str(j))
                qc.rzz(angle, i, j)

        for i in range(lens):
            angle = Parameter('conv_'+str(i))
            qc.rx(angle, i)

        for i in range(lens):
            qc.cx(i, (i+1) % lens)

        qc.measure_all()
        qc = generate_preset_pass_manager(optimization_level=3).run(qc)

        return qc
    
    def bin_to_num(self, list_x: list):
        return [int(x, 2) for x in list_x]
    
    def str_to_tensor(self, str_x: str):
        res = []
        for x in str_x:
            res.append([int(y) for y in x])
        return torch.tensor(res, dtype=torch.float32)
    
    def assign_params(self, x):
        params = {}
        lens = self.kernel_size**2
        x_flat = x.flatten()
        w_flat = self.weight.flatten()
        for i in range(lens):
            params['enc_0_'+str(i)] = x_flat[i].item()

        for i in range(lens-1):
            for j in range(i+1, lens):
                params['enc_1_'+str(i)+'_'+str(j)] = x_flat[i].item() * x_flat[j].item()

        for i in range(lens):
            params['conv_'+str(i)] = w_flat[i].item()

        return self.circ.assign_parameters(params)
    
    def sub_forward(self, x):
        circ = self.assign_params(x)
        # Run and get counts
        result = self.simulator.run(circ).result()
        counts = result.get_counts(circ)

        count_bins = self.str_to_tensor(list(counts.keys()))
        count_vals = torch.tensor(list(counts.values()), dtype=torch.float32)
        count_vals = count_vals / count_vals.sum()
        return torch.matmul(count_vals, count_bins)
    
    def forward(self, x):
        out = torch.zeros(x.shape[0], self.kernel_size**2, x.shape[1]//self.stride, x.shape[2]//self.stride)
        for b in range(x.shape[0]):
            for i in range(0, x.shape[1], self.stride):
                for j in range(0, x.shape[2], self.stride):
                    out[b, :, i//self.stride, j//self.stride] = self.sub_forward(x[b, i:i+self.kernel_size, j:j+self.kernel_size])
        return out


class QuantumConvNet(nn.Module):
    def __init__(self):
        super(QuantumConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, 2, 2)
        self.conv2 = nn.Conv2d(1, 1, 2, 2)
        self.qconv2 = QuantumConv2d(2, 2)
        self.fc1 = nn.Linear(64, 10)
    
    def forward(self, x):  
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.squeeze(1)
        x = torch.relu(self.qconv2(x))
        # 4 * 4 * 4 = 64
        x = x.flatten(1)
        x = torch.softmax(self.fc1(x), dim=-1)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, 2)
        self.conv2 = nn.Conv2d(1, 1, 2, 2)
        self.fc1 = nn.Linear(49, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.softmax(self.fc1(x), dim=-1)
        return x

#%%


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('./', download=True, train=True, transform=transform)
# cut dataset to 1000 samples
trainset.data = trainset.data[:100]
valset = datasets.MNIST('./', download=True, train=False, transform=transform)
# cut dataset to 100 samples
valset.data = valset.data[:10]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=True)

net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in tqdm(range(10)):
    running_loss = []
    for i, (X_batch, y_batch) in enumerate(trainloader):
        
        optimizer.zero_grad()

        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    print(f'Epoch: {epoch}, Loss: {np.mean(running_loss)}')
    running_loss = []

print('Finished Training')

#%%

qnet = QuantumConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(qnet.parameters(), lr=0.001)

for epoch in tqdm(range(10)):
    running_loss = []
    for i, (X_batch, y_batch) in enumerate(trainloader):
        
        optimizer.zero_grad()

        outputs = qnet(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    print(f'Epoch: {epoch}, Loss: {np.mean(running_loss)}')
    running_loss = []

print('Finished Training')