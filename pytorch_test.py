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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


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
        self.conv1 = nn.Conv2d(1, 1, 2, 2, 2)
        self.conv2 = nn.Conv2d(1, 1, 2, 2)
        self.conv3 = nn.Conv2d(1, 4, 2, 2)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.flatten(1)
        x = torch.softmax(self.fc1(x), dim=-1)
        return x

#%%

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('./', download=True, train=True, transform=transform)
# cut dataset to 1000 samples
trainset.data = trainset.data[:10000]
valset = datasets.MNIST('./', download=True, train=False, transform=transform)
# cut dataset to 100 samples
valset.data = valset.data[:1000]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=True)

net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

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

    # Validation
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in valloader:
            outputs = net(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f'Accuracy: {100 * correct / total}')

print('Finished Training')

#%%

qnet = QuantumConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(qnet.parameters(), lr=0.01)

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

    # Validation
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in valloader:
            outputs = net(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f'Accuracy: {100 * correct / total}')

print('Finished Training')


# %%

lenet = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters(), lr=0.01)

for epoch in tqdm(range(10)):
    running_loss = []
    for i, (X_batch, y_batch) in enumerate(trainloader):
        
        optimizer.zero_grad()

        outputs = lenet(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    print(f'Epoch: {epoch}, Loss: {np.mean(running_loss)}')
    running_loss = []

    # Validation
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in valloader:
            outputs = net(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f'Accuracy: {100 * correct / total}')


# %%
    
import qiskit.quantum_info as qi
import plotly.express as px

lens = 4
qc = QuantumCircuit(lens)

#for i in range(lens):
#    qc.h(i)

'''

for i in range(lens):
    angle = Parameter('enc_0_'+str(i))
    angle = 0.7
    qc.rzz(angle,1,2)
'''
'''
for i in range(lens-1):
    for j in range(i+1, lens):
        angle = Parameter('enc_1_'+str(i)+'_'+str(j))
        angle = 0.9
        qc.rzz(angle, i, j)

for i in range(lens):
    angle = Parameter('conv_'+str(i))
    angle = 0.5
    qc.rx(angle, i)


for i in range(1):
    qc.cx(2, 0)
'''

qc.rzz(1, 0, 1)
qc.rzz(1, 0, 2)
qc.rzz(3, 0, 3)
qc.rzz(4, 1, 2)
qc.rzz(5, 1, 3)
qc.rzz(6, 2, 3)

#qc.rz(0.7, 1)
#qc.measure_all()

# plot the circuit
qc.draw('mpl')

op = qi.Operator(qc)

test = op.data
px.imshow(test.real).show()
px.imshow(test.imag).show()

simulator = AerSimulator()
result = simulator.run(qc, shots=1000).result()
counts = result.get_counts(qc)


#%% build unitary

gates = {
    'H': qi.Operator.from_label('H').data,
    'X': qi.Operator.from_label('X').data,
    'Y': qi.Operator.from_label('Y').data,
    'Z': qi.Operator.from_label('Z').data,
    'I': qi.Operator.from_label('I').data,
    'CNOT': np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]]),
    'RZ_S': np.array([[-1j/2, 0], [0, 1j/2]], dtype=np.complex128),
    'RX': np.array([[np.cos(np.pi/4), -1j*np.sin(np.pi/4)], [-1j*np.sin(np.pi/4), np.cos(np.pi/4)]], dtype=np.complex128)
}

def get_CNOT(control, target, qubits):
    """
    qubit index from 1 to N qubits
    """
    swap = True
    if control > target:
        swap = False
        control, target = target, control
    diff = target - control
    if diff > 1:
        scaler = np.eye(2**(diff-1))
        upper = np.kron(scaler, gates['I'])
        lower = np.kron(scaler, gates['X'])
    else:
        upper = gates['I']
        lower = gates['X']
    
    unitary = np.kron(np.array([[1, 0], [0, 0]]), upper) + np.kron(np.array([[0, 0], [0, 1]]), lower)

    if swap:
        swap_matrix = gates['H']
        for _ in range(1,diff+1):
            swap_matrix = np.kron(swap_matrix, gates['H'])
        unitary = swap_matrix @ unitary @ swap_matrix

    if qubits > diff + 1:
        bits_before = int(control - 1)
        bits_after = int(qubits - target)
        unitary = np.kron(np.eye(2**bits_after), np.kron(unitary, np.eye(2**bits_before)))

    return unitary

def get_RZ_static(qubits):
    """
    qubit index from 1 to N qubits
    """
    unitary = gates['RZ']
    for i in range(1, qubits):
        unitary = np.kron(unitary, gates['RZ'])
    return unitary

def get_RX(rotations):
    """
    qubit index from 1 to N qubits
    """
    unitary = np.array([[np.cos(rotations[0]/2), -1j*np.sin(rotations[0]/2)], 
                        [-1j*np.sin(rotations[0]/2), np.cos(rotations[0]/2)]], dtype=np.complex128)
    for i in range(1, qubits):
        unitary = np.kron(unitary, np.array([[np.cos(rotations[i]/2), -1j*np.sin(rotations[i]/2)], 
                                            [-1j*np.sin(rotations[i]/2), np.cos(rotations[i]/2)]], dtype=np.complex128))
    return unitary

def get_RZZ(qubits, rotation):
    """
    TODO qubit index from 1 to N qubits
    """
    pass

def get_all_H(num_qubits):
    unitary = gates['H']
    for _ in range(1, num_qubits):
        unitary = np.kron(unitary, gates['H'])
    return unitary

def get_CNOT_ring(num_qubits):
    unitary = get_CNOT(1, 2, num_qubits)
    for i in range(2, num_qubits):
        unitary = get_CNOT(i, i+1, num_qubits) @ unitary
    unitary = get_CNOT(num_qubits, 1, num_qubits) @ unitary
    return unitary

test = get_CNOT(4, 1, 4)

qubits = 4

matrix = []
op_list = ['I' for i in range(qubits)]
for i in range(qubits):
    op_list_t = op_list.copy()
    op_list_t[i] = 'H'
    unitary = gates[op_list_t[0]]
    for j in range(1,qubits):
        unitary = np.kron(unitary, gates[op_list_t[j]])
    matrix.append(unitary)

final = matrix[0]
for i in range(1, qubits):
    final = final @ matrix[i]

#%%
    
# to measure the state probabilities
# first row of the matrix, than abs() and square
# then state to binary and calculate per qubit probability of 1