# %%

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Create a new circuit with two qubits
qc = QuantumCircuit(4)

# Add a Hadamard gate to qubit 0
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)

# Add rotation gates to qubit 1
qc.rz(0.5, 0)
qc.rz(0.5, 1)
qc.rz(0.5, 2)
qc.rz(0.5, 3)

# add rzz rotations to qubit 1
qc.rzz(0.5, 0, 1)
qc.rzz(0.5, 0, 2)
qc.rzz(0.5, 0, 3)
qc.rzz(0.5, 1, 2)
qc.rzz(0.5, 1, 3)
qc.rzz(0.5, 2, 3)

# Add rotation gates to qubit 1
qc.rx(0.5, 0)
qc.rx(0.5, 1)
qc.rx(0.5, 2)
qc.rx(0.5, 3)

# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.cx(3, 0)

# add measurement gates
qc.measure_all()

# Return a drawing of the circuit using MatPlotLib ("mpl"). This is the
# last line of the cell, so the drawing appears in the cell output.
# Remove the "mpl" argument to get a text drawing.
qc.draw("mpl")


from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi

simulator = AerSimulator()
circ = transpile(qc, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Bell-State counts')

# %%
import numpy as np
import torch
import torch.nn as nn

"""
conv2d network idea

conv2d 28x28x1, stride 2 -> 14x14x (num_qchannels + num_cchannels)
conv2d 14x14x(num_qchannels + num_cchannels), stride 2 -> 7x7x (num_qchannels + num_cchannels)
conv2d 7x7x(num_qchannels + num_cchannels), stride 2 -> 3x3x (num_qchannels + num_cchannels)
flatten 3x3x(num_qchannels + num_cchannels) -> 9(num_qchannels + num_cchannels)
fc 9(num_qchannels + num_cchannels) -> output_size

"""
class QuantumConv2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(QuantumConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size))
        self.simulator = AerSimulator()

    def build_circuit(self, x: torch.Tensor):
        x = x.flatten().numpy()
        qc = QuantumCircuit(len(x))
        for i in range(len(x)):
            qc.h(i)

        for i, val in enumerate(x):
            qc.rz(val,i)
        
        for i in range(len(x)-1):
            for j in range(i+1, len(x)):
                qc.rzz(x[i]*x[j], i, j)

        for i in range(len(self.weight)):
            qc.rx(self.weight.flatten()[i].item(), i)

        for i in range(len(x)):
            qc.cx(i, (i+1)%len(x))

        qc.measure_all()
        return qc
    
    def bin_to_num(self, list_x: list):
        return [int(x, 2) for x in list_x]
    
    def sub_forward(self, x):
        x = self.build_circuit(x)
        circ = transpile(qc, simulator)
        # Run and get counts
        result = simulator.run(circ).result()
        counts = result.get_counts(circ)

        count_bins = torch.tensor(self.bin_to_num(list(counts.keys())), dtype=torch.float32)
        count_vals = torch.tensor(list(counts.values()), dtype=torch.float32)
        count_vals = count_vals / count_vals.sum()
        return torch.matmul(count_bins, count_vals)
    
    def forward(self, x):
        out = torch.zeros(x.shape[0], x.shape[1]//self.stride, x.shape[2]//self.stride)
        for b in range(x.shape[0]):
            for i in range(0, x.shape[1], self.stride):
                for j in range(0, x.shape[2], self.stride):
                    out[b, i//self.stride, j//self.stride] = self.sub_forward(x[b, i:i+self.kernel_size, j:j+self.kernel_size])
        return out

# Create a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.custom = QuantumConv2d(2, 2)
        self.fc2 = nn.Linear(25, 1)
    
    def forward(self, x):
        x = torch.relu(x)
        x = self.custom(x)
        x = x.flatten(1)
        x = self.fc2(x)
        return x
    
# Create a simple dataset
X = torch.randn(100, 10, 10)
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



