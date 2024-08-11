import numpy as np
from typing import List, Tuple
from enum import Enum
import random

class Color(Enum):
    WHITE = 0
    RED = 1
    BLUE = 2
    ORANGE = 3
    GREEN = 4
    YELLOW = 5

class Action(Enum):
    F = 0           # Front clockwise
    F_PRIME = 1     # Front counterclockwise
    B = 2           # Back clockwise
    B_PRIME = 3     # Back counterclockwise
    U = 4           # Up clockwise
    U_PRIME = 5     # Up counterclockwise
    D = 6           # Down clockwise
    D_PRIME = 7     # Down counterclockwise
    L = 8           # Left clockwise
    L_PRIME = 9     # Left counterclockwise
    R = 10          # Right clockwise
    R_PRIME = 11    # Right counterclockwise

class RubiksCube:
    def __init__(self, size: int = 3):
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.array([
            [[Color.WHITE.value] * self.size for _ in range(self.size)],  # Up
            [[Color.RED.value] * self.size for _ in range(self.size)],    # Front
            [[Color.BLUE.value] * self.size for _ in range(self.size)],   # Right
            [[Color.ORANGE.value] * self.size for _ in range(self.size)], # Back
            [[Color.GREEN.value] * self.size for _ in range(self.size)],  # Left
            [[Color.YELLOW.value] * self.size for _ in range(self.size)]  # Down
        ])

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def is_solved(self) -> bool:
        return all(np.all(face == color.value) for face, color in zip(self.state, Color))

    def apply_action(self, action: Action):
        if action in [Action.F, Action.F_PRIME]:
            self._rotate_face(1, action == Action.F)
        elif action in [Action.B, Action.B_PRIME]:
            self._rotate_face(3, action == Action.B)
        elif action in [Action.U, Action.U_PRIME]:
            self._rotate_face(0, action == Action.U)
        elif action in [Action.D, Action.D_PRIME]:
            self._rotate_face(5, action == Action.D)
        elif action in [Action.L, Action.L_PRIME]:
            self._rotate_face(4, action == Action.L)
        elif action in [Action.R, Action.R_PRIME]:
            self._rotate_face(2, action == Action.R)

    def _rotate_face(self, face: int, clockwise: bool):
        self.state[face] = np.rot90(self.state[face], k=3 if clockwise else 1)
        
        if face == 0:       # Up
            self._rotate_adjacent(1, 2, 3, 4, 0, clockwise)
        elif face == 1:     # Front
            self._rotate_adjacent(0, 2, 5, 4, 2, clockwise)
        elif face == 2:     # Right
            self._rotate_adjacent(0, 3, 5, 1, 1, clockwise)
        elif face == 3:     # Back
            self._rotate_adjacent(0, 4, 5, 2, 0, clockwise)
        elif face == 4:     # Left
            self._rotate_adjacent(0, 1, 5, 3, 3, clockwise)
        elif face == 5:     # Down
            self._rotate_adjacent(1, 4, 3, 2, 2, clockwise)

    def _rotate_adjacent(self, f1: int, f2: int, f3: int, f4: int, row: int, clockwise: bool):
        row = row % self.size
        
        temp = self.state[f1][row].copy()
        if clockwise:
            self.state[f1][row] = self.state[f4][:, self.size-1-row]
            self.state[f4][:, self.size-1-row] = self.state[f3][self.size-1-row][::-1]
            self.state[f3][self.size-1-row] = self.state[f2][:, row][::-1]
            self.state[f2][:, row] = temp
        else:
            self.state[f1][row] = self.state[f2][:, row]
            self.state[f2][:, row] = self.state[f3][self.size-1-row][::-1]
            self.state[f3][self.size-1-row] = self.state[f4][:, self.size-1-row][::-1]
            self.state[f4][:, self.size-1-row] = temp[::-1]

    def scramble(self, num_moves: int = 20):
        for _ in range(num_moves):
            action = random.choice(list(Action))
            self.apply_action(action)

    def __str__(self):
        face_symbols = ['U', 'F', 'R', 'B', 'L', 'D']
        color_symbols = ['.', 'R', 'B', 'O', 'G', 'Y']
        result = []
        for i, face in enumerate(self.state):
            result.append(f"{face_symbols[i]}:")
            for row in face:
                result.append(' '.join(color_symbols[cell] for cell in row))
            result.append('')
        return '\n'.join(result)

def get_initial_state() -> np.ndarray:
    cube = RubiksCube()
    cube.scramble()
    return cube.get_state()

def get_next_state(state: np.ndarray, action: Action) -> np.ndarray:
    cube = RubiksCube()
    cube.state = state.copy()
    cube.apply_action(action)
    return cube.get_state()

def is_solved(state: np.ndarray) -> bool:
    cube = RubiksCube()
    cube.state = state.copy()
    return cube.is_solved()

def compute_target_value(state: np.ndarray) -> float:
    cube = RubiksCube()
    cube.state = state.copy()
    solved_faces = sum(np.all(face == color.value) for face, color in zip(cube.state, Color))
    return solved_faces / 6.0

if __name__ == "__main__":
    cube = RubiksCube()
    print("Solved cube:")
    print(cube)
    
    cube.scramble()
    print("\nScrambled cube:")
    print(cube)
    
    print("\nApplying action F (Front clockwise):")
    cube.apply_action(Action.F)
    print(cube)
    
    print(f"\nIs cube solved? {cube.is_solved()}")
    print(f"Target value: {compute_target_value(cube.get_state())}")