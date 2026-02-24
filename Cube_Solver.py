import cv2
import numpy as np
import kociemba
import logging
from Cube_Scanner import RubiksCubeDetector

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

number_to_letter = {
    1: 'U',  # (UP)    White
    2: 'D',  # (DOWN)  Yellow
    3: 'L',  # (LEFT)  Red
    4: 'B',  # (BACK)  Green
    5: 'F',  # (FRONT) Blue
    6: 'R',  # (RIGHT) Orange
}

number_to_color = {
    1: 'W',  # White
    2: 'Y',  # Yellow
    3: 'R',  # Red
    4: 'G',  # Green
    5: 'B',  # Blue
    6: 'O',  # Orange
}

class RubiksCubeSolver:
    # Main class for the Rubik's Cube solver
    def __init__(self):
        self.detector = RubiksCubeDetector()

    def run(self):
        cap, writer = self.detector.initialize_capture()
        if self.detector._scan_cube(cap, writer) == 'quit':
            return ('quit', None)
        solution = self.solve_cube()

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if solution == None:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "No solution found! Try again.",
                (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.imshow("Invalid Scan Error", error_frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            return (None, None)
        
        # Accurately orient the scanned faces before mapping them to 3D cube
        self.detector.state.up_face = np.flipud(self.detector.state.up_face)
        self.detector.state.right_face = np.rot90(np.flipud(self.detector.state.right_face),k=2)
        self.detector.state.down_face = np.fliplr(self.detector.state.down_face)

        face_configs = {
            'U': [number_to_color[int(i)] for i in self.detector.state.up_face.flatten()],
            'F': [number_to_color[int(i)] for i in self.detector.state.front_face.flatten()],
            'R': [number_to_color[int(i)] for i in self.detector.state.right_face.flatten()],
            'B': [number_to_color[int(i)] for i in self.detector.state.back_face.flatten()],
            'L': [number_to_color[int(i)] for i in self.detector.state.left_face.flatten()],
            'D': [number_to_color[int(i)] for i in self.detector.state.down_face.flatten()],
        }

        return solution.split(' '), face_configs

    def solve_cube(self):
        try:
            cube_string = self._state_to_kociemba_string()
            logger.info(f"Cubestring: {cube_string}")
            solution = kociemba.solve(cube_string)
            logger.info(f"Solution found: {solution}")
            return solution
        except Exception as e:
            logger.error(f"Could not solve cube: {e}")
            return None
            
    def _explain_move(self, move):
        face_map = {
            'U': "WHITE face",
            'D': "YELLOW face",
            'F': "BLUE face",
            'B': "GREEN face",
            'L': "RED face",
            'R': "ORANGE face"
        }
        
        if len(move) == 1:
            return f"Turn {face_map[move]} clockwise"
        elif len(move) == 2:
            if move[1] == '2':
                return f"Turn {face_map[move[0]]} 180 degrees"
            else:
                return f"Turn {face_map[move[0]]} counter-clockwise"
        
        return "Unknown move"

    def _state_to_kociemba_string(self) -> str:
        # Mapping faces of the scanned cube to Kociemba's convention
        face_order = [
            (self.detector.state.up_face, "up"),       # U
            (self.detector.state.right_face, "right"), # R
            (self.detector.state.front_face, "front"), # F
            (self.detector.state.down_face, "down"),   # D
            (self.detector.state.left_face, "left"),   # L
            (self.detector.state.back_face, "back"),   # B
        ]

        cube_str = ''
        for face, face_name in face_order:
            face = self.reorganise_face(face, face_name)
            flat_face = face.flatten()
            for color_val in flat_face:
                cube_letter = number_to_letter.get(int(color_val), 'X')
                cube_str += cube_letter
        if len(cube_str) != 54:
            logger.error(f"Cube string length error: {len(cube_str)}")
            raise ValueError("Invalid cube configuration detected.")

        return cube_str
    
    def reorganise_face(self, face, face_name):
        # Reorganising cube faces to ensure correct mapping to Kociemba's notation
        face_prime = face.copy()
        if face_name in ["up", "front"]:
            face[0] = face_prime[2]
            face[2] = face_prime[0]
            return np.array(face)
        elif face_name == "left":
            return np.fliplr(np.flipud(face_prime)).T
        elif face_name == "right":
            return np.fliplr(np.rot90(face_prime, k=-1))
        elif face_name in ["back", "down"]:
            return np.fliplr(face_prime)