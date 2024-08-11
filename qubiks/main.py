import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from cube import RubiksCube, Action, get_initial_state, get_next_state, is_solved, compute_target_value
import time

NUM_QUBITS = 10
NUM_LAYERS = 3

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    for layer in range(NUM_LAYERS):
        for i in range(NUM_QUBITS):
            qml.RY(weights[layer][i][0], wires=i)
            qml.RZ(weights[layer][i][1], wires=i)
        
        for i in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[NUM_QUBITS - 1, 0])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (NUM_LAYERS, NUM_QUBITS, 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        return self.qlayer(x)

class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.quantum_layer = QuantumLayer()
        self.fc2 = nn.Linear(NUM_QUBITS, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x_quantum = self.quantum_layer(x[:NUM_QUBITS])
        x_quantum = torch.relu(self.fc2(x_quantum))
        x = x + x_quantum  # Skip connection
        x = self.fc3(x)
        return x

def state_to_tensor(state: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(state.flatten()).to(torch.float32)

def action_to_index(action: Action) -> int:
    return action.value

def index_to_action(index: int) -> Action:
    return Action(index)

def visualize_cube(state: np.ndarray):
    cube = RubiksCube()
    cube.state = state.copy()
    print(cube)

def print_progress(episode: int, steps: int, is_solved: bool):
    print(f"Episode {episode}, Steps: {steps}, Solved: {is_solved}")

class RubiksSolver:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train_step(self, state: np.ndarray, epsilon: float = 0.1) -> Tuple[Action, float]:
        state_tensor = state_to_tensor(state)
        
        if np.random.random() < epsilon:
            action = np.random.choice(list(Action))
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = index_to_action(q_values.argmax().item())
        
        next_state = get_next_state(state, action)
        next_state_tensor = state_to_tensor(next_state)
        
        target = compute_target_value(next_state)
        
        self.optimizer.zero_grad()
        q_values = self.model(state_tensor)
        loss = self.criterion(q_values[action_to_index(action)], torch.tensor(target, dtype=torch.float32))
        loss.backward()
        self.optimizer.step()
        
        return action, loss.item()
    
    def solve(self, initial_state: np.ndarray, max_steps: int = 1000, visualize: bool = False) -> List[Action]:
        state = initial_state
        actions = []
        
        if visualize:
            print("Initial state:")
            visualize_cube(state)
        
        for step in range(max_steps):
            if is_solved(state):
                if visualize:
                    print(f"\nSolved in {step} steps!")
                break
            
            action, _ = self.train_step(state, epsilon=0.1)
            state = get_next_state(state, action)
            actions.append(action)
            
            if visualize and step % 10 == 0:
                print(f"\nStep {step}:")
                visualize_cube(state)
                time.sleep(0.5)
        
        return actions

INPUT_SIZE = 6 * 3 * 3  # 6 faces, 3x3 grid
HIDDEN_SIZE = 64
OUTPUT_SIZE = len(Action)
LEARNING_RATE = 0.001

model = HybridModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

for param in model.parameters():
    param.data = param.data.to(torch.float32)

solver = RubiksSolver(model, optimizer, criterion)

NUM_EPISODES = 1000
for episode in range(NUM_EPISODES):
    initial_state = get_initial_state()
    actions = solver.solve(initial_state, visualize=(episode % 100 == 0))
    
    print_progress(episode, len(actions), is_solved(get_next_state(initial_state, actions[-1]) if actions else initial_state))

test_cube = RubiksCube()
test_cube.scramble()
print("\nScrambled cube:")
visualize_cube(test_cube.get_state())

initial_state = test_cube.get_state()
solution = solver.solve(initial_state, visualize=True)

print(f"\nSolution found in {len(solution)} steps:")
for action in solution:
    print(action)

for action in solution:
    test_cube.apply_action(action)

print("\nCube after applying solution:")
visualize_cube(test_cube.get_state())
print(f"Is cube solved? {test_cube.is_solved()}")