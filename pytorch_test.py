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

gates = {
    'H': torch.tensor([[1, 1], [1, -1]]) / torch.sqrt(torch.tensor([2.0])),
    'X': torch.tensor([[0, 1], [1, 0]]),
    'I': torch.eye(2),
}

class QuantumConv2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(QuantumConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.qubits = kernel_size**2

        # static layers
        self.h_static = self.get_all_H(self.qubits)
        self.rz_static = self.get_static_RZ(self.qubits)
        self.cnot_static = self.get_CNOT_ring(self.qubits)

        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size))

    def get_all_H(self, num_qubits):
        unitary = gates['H']
        for _ in range(1, num_qubits):
            unitary = np.kron(unitary, gates['H'])
        return unitary

    def get_static_RZ(self, qubits):
        def bin_list(num, length):
            formating = '{0:0' + str(length) + 'b}'
            binary = f'{formating}'.format(num)
            return [int(x) for x in binary]
        binary_array = [bin_list(x, qubits) for x in range(2**qubits)]
        binary_array = np.flip(np.array(binary_array), axis=1)
        sign_matrix = -np.ones((2**qubits, qubits)) + 2*np.array(binary_array)
        return sign_matrix
    
    def get_RZZ(self, qubits:list, rotation:float, num_qubits:int):
        """
        qubit index from 1 to N qubits
        """
        control = min(qubits)
        target = max(qubits)
        diff = target - control
        upper_diff = num_qubits - target
        b1 = np.exp(1j*rotation/2)
        b2 = np.exp(-1j*rotation/2)
        operator_core = np.diag([b2, b1, b1, b2])
        operator = np.kron(operator_core, np.eye(int(2**(control-1))))

        if diff > 1:
            operator_upper = operator[:len(operator)//2, :len(operator)//2]
            operator_lower = operator[len(operator)//2:, len(operator)//2:]
            scaler = np.eye(2**(diff-1))
            upper = np.kron(scaler, operator_upper)
            lower = np.kron(scaler, operator_lower)
            operator = np.kron(np.array([[1, 0], [0, 0]]), upper) + np.kron(np.array([[0, 0], [0, 1]]), lower)
        
        if upper_diff > 0:
            operator = np.kron(np.eye(2**upper_diff), operator)

        return operator

    def get_all_RZ(self, rotations, sign_matrix):
        rots = np.array(rotations) / 2 * 1j
        unitary = np.sum(sign_matrix * rots, axis=1)
        unitary = np.exp(unitary)
        unitary = np.diag(unitary)
        return unitary

    def get_RZZ_interconection(self, rotations, qubits):
        ops = []
        for i in range(qubits-1):
            for j in range(i+1, qubits):
                ops.append(self.get_RZZ([i+1,j+1], rotations[i]*rotations[j], qubits))
        unitary = ops[-1]
        for i in range(0, len(ops)-1):
            unitary = unitary @ ops[len(ops)-2-i]
        return unitary

    def get_RX(self, rotations, qubits):
        """
        qubit index from 1 to N qubits
        """
        unitary = np.array([[np.cos(rotations[0]/2), -1j*np.sin(rotations[0]/2)], 
                            [-1j*np.sin(rotations[0]/2), np.cos(rotations[0]/2)]], dtype=np.complex128)
        for i in range(1, qubits):
            unitary = np.kron(np.array([[np.cos(rotations[i]/2), -1j*np.sin(rotations[i]/2)], 
                                                [-1j*np.sin(rotations[i]/2), np.cos(rotations[i]/2)]], dtype=np.complex128), unitary)
        return unitary

    def get_CNOT(self, control, target, qubits):
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

    def get_CNOT_ring(self, num_qubits):
        unitary = self.get_CNOT(1, 2, num_qubits)
        for i in range(2, num_qubits):
            unitary = self.get_CNOT(i, i+1, num_qubits) @ unitary
        unitary = self.get_CNOT(num_qubits, 1, num_qubits) @ unitary
        return unitary

    def bin_to_num(self, list_x: list):
        return [int(x, 2) for x in list_x]
    
    def str_to_tensor(self, str_x: str):
        res = []
        for x in str_x:
            res.append([int(y) for y in x])
        return torch.tensor(res, dtype=torch.float32)
    
    def sub_forward(self, x):
        operations = []
        operations.append(self.h_static)
        operations.append(self.get_all_RZ(x.flatten().tolist(), self.rz_static))
        operations.append(self.get_RZZ_interconection(x.flatten().tolist(), self.qubits))
        operations.append(self.get_RX(self.weight, self.qubits))
        operations.append(self.cnot_static)

        final = operations[-1]
        for i in range(0, len(operations)-1):
            final = final @ operations[len(operations)-2-i]

        return np.abs(final[:, 0])**2
    
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

rotations = [1,2,3,4]
rotations_params = [3,4,5,6]

for i in range(lens):
    qc.h(i)

for i in range(lens):
    angle = Parameter('enc_0_'+str(i))
    angle = rotations[i]
    qc.rz(angle,i)

for i in range(lens-1):
    for j in range(i+1, lens):
        angle = Parameter('enc_1_'+str(i)+'_'+str(j))
        angle = rotations[i] * rotations[j]
        qc.rzz(angle, i, j)

for i in range(lens):
    angle = Parameter('conv_'+str(i))
    angle = rotations_params[i]
    qc.rx(angle, i)

for i in range(lens):
    qc.cx(i, (i+1) % lens)

#qc.measure_all()

# plot the circuit
#qc.draw('mpl')

op = qi.Operator(qc)

test = op.data
px.imshow(test.real).show()
px.imshow(test.imag).show()

#%%

simulator = AerSimulator()
result = simulator.run(qc, shots=100000).result()
counts = result.get_counts(qc)
counts = {int(k, 2): v for k, v in counts.items()}
counts = np.array([counts.get(i, 0) for i in range(2**lens)])
counts = counts / np.sum(counts)

px.bar(x=[str(i) for i in range(2**lens)], y=counts).show()


#%% build unitary

gates = {
    'H': torch.tensor([[1, 1], [1, -1]]) / torch.sqrt(torch.tensor([2.0])),
    'X': torch.tensor([[0, 1], [1, 0]]),
    'I': torch.eye(2),
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
        scaler = torch.eye(2**(diff-1))
        upper = torch.kron(scaler, gates['I'])
        lower = torch.kron(scaler, gates['X'])
    else:
        upper = gates['I']
        lower = gates['X']
    
    unitary = torch.kron(torch.tensor([[1, 0], [0, 0]]), upper) + torch.kron(torch.tensor([[0, 0], [0, 1]]), lower)

    if swap:
        swap_matrix = gates['H']
        for _ in range(1,diff+1):
            swap_matrix = torch.kron(swap_matrix, gates['H'])
        unitary = swap_matrix @ unitary @ swap_matrix

    if qubits > diff + 1:
        bits_before = int(control - 1)
        bits_after = int(qubits - target)
        unitary = torch.kron(torch.eye(2**bits_after), torch.kron(unitary, torch.eye(2**bits_before)))

    return unitary

def get_RX(rotations, qubits):
    """
    qubit index from 1 to N qubits
    """
    unitary = torch.tensor([[torch.cos(rotations[0]/2), -1j * torch.sin(rotations[0]/2)], 
                    [-1j*torch.sin(rotations[0]/2), torch.cos(rotations[0]/2)]], dtype=torch.cfloat)
    for i in range(1, qubits):
        unitary = torch.kron(torch.tensor([[torch.cos(rotations[i]/2), -1j * torch.sin(rotations[i]/2)], 
                    [-1j * torch.sin(rotations[i]/2), torch.cos(rotations[i]/2)]], dtype=torch.cfloat), unitary)
    return unitary

def get_RZZ(qubits:list, rotation:float, num_qubits:int):
    """
    qubit index from 1 to N qubits
    """
    control = min(qubits)
    target = max(qubits)
    diff = target - control
    upper_diff = num_qubits - target
    b1 = torch.exp(1j*rotation/2)
    b2 = torch.exp(-1j*rotation/2)
    operator_core = torch.diag(torch.tensor([b2, b1, b1, b2]))
    operator = torch.kron(operator_core, torch.eye(2**(control-1)))

    if diff > 1:
        operator_upper = operator[:len(operator)//2, :len(operator)//2]
        operator_lower = operator[len(operator)//2:, len(operator)//2:]
        scaler = torch.eye(2**(diff-1))
        upper = torch.kron(scaler, operator_upper)
        lower = torch.kron(scaler, operator_lower)
        operator = torch.kron(torch.tensor([[1, 0], [0, 0]]), upper) + torch.kron(torch.tensor([[0, 0], [0, 1]]), lower)
    
    if upper_diff > 0:
        operator = torch.kron(torch.eye(2**upper_diff), operator)

    return operator

def get_all_H(num_qubits):
    unitary = gates['H']
    for _ in range(1, num_qubits):
        unitary = torch.kron(unitary, gates['H'])
    return unitary

def get_CNOT_ring(num_qubits):
    unitary = get_CNOT(1, 2, num_qubits)
    for i in range(2, num_qubits):
        unitary = get_CNOT(i, i+1, num_qubits) @ unitary
    unitary = get_CNOT(num_qubits, 1, num_qubits) @ unitary
    return unitary

def get_static_RZ(qubits):
    def bin_list(num, length):
        formating = '{0:0' + str(length) + 'b}'
        binary = f'{formating}'.format(num)
        return [int(x) for x in binary]

    binary_array = [bin_list(x, qubits) for x in range(2**qubits)]
    binary_array = torch.flip(torch.tensor(binary_array), dims=(-1,))
    sign_matrix = -torch.ones((2**qubits, qubits)) + 2 * binary_array

    return sign_matrix

def get_all_RZ(rotations, sign_matrix):
    rots = torch.tensor(rotations, dtype=torch.cfloat) / 2 * 1j
    unitary = torch.sum(sign_matrix * rots, dims=(-1,))
    unitary = torch.exp(unitary)
    unitary = torch.diag(unitary)
    return unitary

def get_RZZ_interconection(rotations, qubits):
    a = get_RZZ([1,2], rotations[0]*rotations[1], qubits)
    b = get_RZZ([1,3], rotations[0]*rotations[2], qubits)
    c = get_RZZ([1,4], rotations[0]*rotations[3], qubits)
    d = get_RZZ([2,3], rotations[1]*rotations[2], qubits)
    e = get_RZZ([2,4], rotations[1]*rotations[3], qubits)
    f = get_RZZ([3,4], rotations[2]*rotations[3], qubits)

    return f @ e @ d @ c @ b @ a

#%%
qubits = 4
matrix = []
rotations = [1,2,3,4]
rotations_params = [3,4,5,6]

matrix.append(get_all_H(qubits))
signs = get_static_RZ(qubits)
matrix.append(get_all_RZ(rotations, signs))
matrix.append(get_RZZ_interconection(rotations, qubits))
matrix.append(get_RX(rotations_params, qubits))
matrix.append(get_CNOT_ring(qubits))

final = matrix[-1]
if len(matrix) > 1:
    for i in range(0, len(matrix)-1):
        final = final @ matrix[len(matrix)-2-i]

px.imshow(final.real).show()
px.imshow(final.imag).show()

#%%
    
# to measure the state probabilities
# first row of the matrix, than abs() and square
# then state to binary and calculate per qubit probability of 1