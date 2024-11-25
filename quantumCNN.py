import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm

gates = {
    'H': torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / torch.sqrt(torch.tensor([2.0], dtype=torch.cfloat)),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat),
    'I': torch.eye(2, dtype=torch.cfloat),
}

class QuantumConv2d(nn.Module):
    """
    A quantum convolutional layer for 2D inputs.
    """
    def __init__(self, kernel_size, stride, size):
        """
        Initialize the QuantumConv2d layer.

        Args:
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            size (int): Size of the input.
        """
        super(QuantumConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.qubits = kernel_size**2
        self.size = size

        self.sub_part_permutation = ((torch.arange(0,(size**2)/kernel_size) % size)* (size/kernel_size) 
                                     + torch.arange(0,(size**2)/kernel_size) // size).int()
        self.upper_triag = torch.triu(torch.arange(self.qubits**2).reshape(self.qubits, self.qubits), diagonal=1)
        self.upper_triag = self.upper_triag[self.upper_triag != 0]
        self.qubit_tuples = torch.tensor([[i, j] for i in range(1, self.qubits+1) for j in range(i+1, self.qubits+1)]).int()
        divisor = self.size // self.kernel_size
        self.permute_back = (torch.arange(0, self.size**2//self.kernel_size) % divisor * self.size
                             + torch.arange(0, self.size**2//self.kernel_size) // divisor).int()


        # static layers
        self.h_static = self.get_all_H(self.qubits)
        self.rz_static = self.get_static_RZ(self.qubits)
        self.cnot_static = self.get_CNOT_ring(self.qubits)
        self.states = self.get_static_state_list(self.qubits)

        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size, requires_grad=True))

    def get_all_H(self, num_qubits):
        """
        Get the tensor product of Hadamard gates for all qubits.

        Args:
            num_qubits (int): Number of qubits.

        Returns:
            torch.Tensor: Unitary matrix representing the Hadamard gates.
        """
        unitary = gates['H']
        for _ in range(1, num_qubits):
            unitary = torch.kron(unitary, gates['H'])
        return unitary
    
    def bin_list(self, num, length):
        """
        Convert a number to a binary list of a given length.

        Args:
            num (int): The number to convert.
            length (int): The length of the binary list.

        Returns:
            list: Binary representation of the number.
        """
        formating = '{0:0' + str(length) + 'b}'
        binary = f'{formating}'.format(num)
        return [int(x) for x in binary]
    
    def get_static_RZ(self, qubits):
        """
        Get the static RZ rotation matrix for all qubits.

        Args:
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: Static RZ rotation matrix.
        """
        binary_array = [self.bin_list(x, qubits) for x in range(2**qubits)]
        binary_array = torch.flip(torch.tensor(binary_array), dims=(-1,))
        sign_matrix = -torch.ones((2**qubits, qubits)) + 2 * binary_array
        return sign_matrix
    
    def get_static_state_list(self, qubits):
        """
        Get the static state list for all qubits.

        Args:
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: Static state list.
        """
        states = [self.bin_list(x, qubits) for x in range(2**qubits)]
        states = torch.tensor(states, dtype=torch.float32)
        return states
    
    def get_RZZ(self, qubits, rotation):
        """
        Get the RZZ gate for given qubits and rotation.

        Args:
            qubits (int): Number of qubits.
            rotation (torch.Tensor): Rotation angle.

        Returns:
            torch.Tensor: RZZ gate matrix.
        """
        qubits = self.qubit_tuples
        control = torch.min(qubits)
        target = torch.max(qubits)
        diff = target - control
        upper_diff = self.qubits - target
        b1 = torch.exp(1j*rotation/2)
        b2 = torch.exp(-1j*rotation/2)
        b12 = torch.concat([b2, b1, b1, b2])
        
        operator_core = torch.diag(b12)
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

    def get_all_RZ(self, rotations:torch.Tensor, sign_matrix):
        """
        Get the RZ gate for all qubits.

        Args:
            rotations (torch.Tensor): Rotation angles.
            sign_matrix (torch.Tensor): Sign matrix for the rotations.

        Returns:
            torch.Tensor: RZ gate matrix.
        """
        rots = rotations.div(torch.tensor([2 * 1j], dtype=torch.cfloat))
        unitary = torch.sum(sign_matrix * rots, dim=(-1,))
        unitary = torch.exp(unitary)
        unitary = torch.diag(unitary)
        return unitary

    def get_RZZ_interconnection(self, rotations:torch.Tensor):
        """
        Get the RZZ interconnection for all qubits.

        Args:
            rotations (torch.Tensor): Rotation angles.

        Returns:
            torch.Tensor: RZZ interconnection matrix.
        """
        rot_mul = rotations[:,None].matmul(rotations[None,:])
        rot_mul = rot_mul.flatten()[self.upper_triag]
        ops:torch.Tensor = torch.vmap(self.get_RZZ)(self.qubit_tuples, rot_mul[:,None])
        ops = ops.reshape(len(self.upper_triag), ops.shape[1], ops.shape[2])
        # 6, 16, 16
        unitary = ops[-1,:,:]
        for i in range(0, ops.shape[0]-1):
            unitary = unitary.matmul(ops[ops.shape[0]-2-i,:,:])
        return unitary

    def get_RX(self, rotations, qubits):
        """
        Get the RX gate for given qubits and rotations.

        Args:
            rotations (torch.Tensor): Rotation angles.
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: RX gate matrix.
        """
        unitary = torch.tensor([[torch.cos(rotations[0]/2), -1j * torch.sin(rotations[0]/2)], 
                        [-1j*torch.sin(rotations[0]/2), torch.cos(rotations[0]/2)]], dtype=torch.cfloat)
        for i in range(1, qubits):
            unitary = torch.kron(torch.tensor([[torch.cos(rotations[i]/2), -1j * torch.sin(rotations[i]/2)], 
                        [-1j * torch.sin(rotations[i]/2), torch.cos(rotations[i]/2)]], dtype=torch.cfloat), unitary)
        return unitary

    def get_CNOT(self, control, target, qubits):
        """
        Get the CNOT gate for given control and target qubits.

        Args:
            control (int): Control qubit index.
            target (int): Target qubit index.
            qubits (int): Number of qubits.

        Returns:
            torch.Tensor: CNOT gate matrix.
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

    def get_CNOT_ring(self, num_qubits):
        """
        Get the CNOT ring for all qubits.

        Args:
            num_qubits (int): Number of qubits.

        Returns:
            torch.Tensor: CNOT ring matrix.
        """
        unitary = self.get_CNOT(1, 2, num_qubits)
        for i in range(2, num_qubits):
            unitary = self.get_CNOT(i, i+1, num_qubits) @ unitary
        unitary = self.get_CNOT(num_qubits, 1, num_qubits) @ unitary
        return unitary

    def bin_to_num(self, list_x: list):
        """
        Convert a list of binary strings to a list of integers.

        Args:
            list_x (list): List of binary strings.

        Returns:
            list: List of integers.
        """
        return [int(x, 2) for x in list_x]
    
    def str_to_tensor(self, str_x: str):
        """
        Convert a string of binary digits to a tensor.

        Args:
            str_x (str): String of binary digits.

        Returns:
            torch.Tensor: Tensor representation of the binary string.
        """
        res = []
        for x in str_x:
            res.append([int(y) for y in x])
        return torch.tensor(res, dtype=torch.float32)
    
    def sub_forward(self, x: torch.Tensor):
        """
        NOTE This function simulates the quantum circuit for a sub-part of the input.
        Perform the forward pass for a sub-part of the input.

        Args:
            x (torch.Tensor): Input tensor. shape:(batch*(pixelanzahl/kernel_size), kernel_size, kernel_size)

        Returns:
            torch.Tensor: Output tensor. shape:(batch*(pixelanzahl/kernel_size), kernel_size, kernel_size)
        """
        operations = []
        operations.append(self.h_static)
        operations.append(self.get_all_RZ(x.flatten(), self.rz_static))
        operations.append(self.get_RZZ_interconnection(x.flatten()))
        operations.append(self.get_RX(self.weight.flatten(), self.qubits))
        operations.append(self.cnot_static)

        final:torch.Tensor = operations[-1]
        for i in range(0, len(operations)-1):
            final = final.matmul(operations[len(operations)-2-i])

        state_probs = (torch.abs(final[:, 0])**2)[None,:]
        state_probs = state_probs.matmul(self.states)

        return state_probs
    
    def forward(self, x):
        """
        Perform the forward pass of the QuantumConv2d layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_dim = x.shape[0]
        out = x.reshape(batch_dim,(self.size**2)//2,1,2)
        out = out[:,self.sub_part_permutation].reshape(batch_dim,(self.size**2)//self.kernel_size**2,2,2)
        out = out.flatten(end_dim=-3)
        # NOTE Quantum Circuit
        out = torch.vmap(self.sub_forward)(out).reshape(
            batch_dim,out.shape[0]//batch_dim*self.kernel_size,self.kernel_size)
        out = out[:,self.permute_back,:].reshape(batch_dim,self.size,self.size)
        return out


class QuantumConvNet(nn.Module):
    """
    A quantum convolutional neural network.
    """
    def __init__(self):
        """
        Initialize the QuantumConvNet.
        """
        super(QuantumConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 2, 2, 2)
        self.conv2 = nn.Conv2d(1, 1, 2, 2)
        self.qconv2 = QuantumConv2d(2, 2, 8)
        self.fc1 = nn.Linear(64, 10)
    
    def forward(self, x):  
        """
        Perform the forward pass of the QuantumConvNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.squeeze(1)
        x = torch.relu(self.qconv2(x))
        # 4 * 4 * 4 = 64
        x = x.flatten(1)
        x = torch.softmax(self.fc1(x), dim=-1)
        return x

if __name__ == '__main__':

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
                outputs = qnet(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        print(f'Accuracy: {100 * correct / total}')

    print('Finished Training')