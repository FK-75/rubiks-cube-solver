import sys
import math
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CUBE_SIZE = 3.0
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
YELLOW = (1, 1, 0)
ORANGE = (1, 0.5, 0)

COLOR_MAP = {
    'W': WHITE,
    'R': RED,
    'G': GREEN,
    'B': BLUE,
    'Y': YELLOW,
    'O': ORANGE
}

# Default starting state of the cube from user's perspective in order: front, back, left, right, top, bottom
DEFAULT_CUBE_COLORS = [
    BLUE,   # Front
    GREEN,  # Back
    RED,    # Left
    ORANGE, # Right
    WHITE,  # Top
    YELLOW, # Bottom
]

def normalize(v):
    norm = math.sqrt(sum(x*x for x in v))
    return tuple(x/norm for x in v) if norm > 0 else v

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def axisangle_to_q(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    angle /= 2
    sinA = math.sin(angle)
    cosA = math.cos(angle)
    return (cosA, x*sinA, y*sinA, z*sinA)

def q_to_mat4(q):
    w, x, y, z = q
    return [
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0,
        2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0,
        0, 0, 0, 1
    ]

def x_rot(vertex, angle):
    x, y, z = vertex
    new_y = y * math.cos(angle) - z * math.sin(angle)
    new_z = y * math.sin(angle) + z * math.cos(angle)
    return (x, new_y, new_z)

def y_rot(vertex, angle):
    x, y, z = vertex
    new_x = x * math.cos(angle) - z * math.sin(angle)
    new_z = x * math.sin(angle) + z * math.cos(angle)
    return (new_x, y, new_z)

def z_rot(vertex, angle):
    x, y, z = vertex
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return (new_x, new_y, z)

class Cubie:
    def __init__(self, position, colors):
        self.position = position
        self.colors = colors
        self.original_position = position
        
        x, y, z = position
        size = CUBE_SIZE / 6
        self.vertices = [
            (x + size, y + size, z + size),
            (x - size, y + size, z + size),  
            (x - size, y - size, z + size),  
            (x + size, y - size, z + size),  
            (x + size, y + size, z - size),  
            (x - size, y + size, z - size),  
            (x - size, y - size, z - size),  
            (x + size, y - size, z - size),  
        ]
        
        self.faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],  
            [1, 5, 6, 2],  
            [0, 4, 7, 3],  
            [0, 1, 5, 4],  
            [3, 2, 6, 7],  
        ]
        
    def draw(self):
        glBegin(GL_QUADS)
        for face_idx, face in enumerate(self.faces):
            color = self.colors[face_idx] if self.colors[face_idx] else BLACK
            glColor3fv(color)
            for vertex_idx in face:
                glVertex3fv(self.vertices[vertex_idx])
        glEnd()
        
        glColor3fv(BLACK)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  
            (4, 5), (5, 6), (6, 7), (7, 4),  
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for edge in edges:
            for vertex_idx in edge:
                glVertex3fv(self.vertices[vertex_idx])
        glEnd()
    
    def rotate_x(self, angle):
        self.vertices = [x_rot(v, angle) for v in self.vertices]
        self.position = x_rot(self.position, angle)
    
    def rotate_y(self, angle):
        self.vertices = [y_rot(v, angle) for v in self.vertices]
        self.position = y_rot(self.position, angle)
    
    def rotate_z(self, angle):
        self.vertices = [z_rot(v, angle) for v in self.vertices]
        self.position = z_rot(self.position, angle)

class RubiksCube:
    def __init__(self, face_configurations=None):
        self.cubies = []
        if face_configurations is None:
            self._create_solved_cube()
        else:
            self._create_custom_cube(face_configurations)
        
        self.animating = False
        self.animation_frames = 0
        self.animation_face = None
        self.animation_direction = 1
        self.animation_progress = 0

        self.move_info = {
            'U': 'Turn UP face 90 degrees clockwise',
            'U\'': 'Turn UP face 90 degrees counterclockwise',
            'D': 'Turn DOWN face 90 degrees clockwise',
            'D\'': 'Turn DOWN face 90 degrees counterclockwise',
            'L': 'Turn LEFT face 90 degrees clockwise',
            'L\'': 'Turn LEFT face 90 degrees counterclockwise',
            'R': 'Turn RIGHT face 90 degrees clockwise',
            'R\'': 'Turn RIGHT face 90 degrees counterclockwise',
            'F': 'Turn FRONT face 90 degrees clockwise',
            'F\'': 'Turn FRONT face 90 degrees counterclockwise',
            'B': 'Turn BACK face 90 degrees clockwise',
            'B\'': 'Turn BACK face 90 degrees counterclockwise',
        }
    
    def _create_solved_cube(self):
        positions = [-CUBE_SIZE/3, 0, CUBE_SIZE/3]
        
        for x in positions:
            for y in positions:
                for z in positions:
                    colors = [None] * 6
                    
                    if z > 0:
                        colors[0] = DEFAULT_CUBE_COLORS[0]  # Front
                    if z < 0:
                        colors[1] = DEFAULT_CUBE_COLORS[1]  # Back
                    if x < 0:
                        colors[2] = DEFAULT_CUBE_COLORS[2]  # Left
                    if x > 0:
                        colors[3] = DEFAULT_CUBE_COLORS[3]  # Right
                    if y > 0:
                        colors[4] = DEFAULT_CUBE_COLORS[4]  # Top
                    if y < 0:
                        colors[5] = DEFAULT_CUBE_COLORS[5]  # Bottom
                    
                    self.cubies.append(Cubie((x, y, z), colors))
    
    def _create_custom_cube(self, face_configurations):
        positions = [-CUBE_SIZE/3, 0, CUBE_SIZE/3]
        
        cube_array = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]
        
        if 'U' in face_configurations:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(face_configurations['U']):
                        color = self._get_color_from_code(face_configurations['U'][idx])
                        if cube_array[i][2][j] is None:
                            cube_array[i][2][j] = [None] * 6
                        cube_array[i][2][j][4] = color
        
        if 'D' in face_configurations:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(face_configurations['D']):
                        color = self._get_color_from_code(face_configurations['D'][idx])
                        if cube_array[i][0][j] is None:
                            cube_array[i][0][j] = [None] * 6
                        cube_array[i][0][j][5] = color  
        
        if 'F' in face_configurations:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(face_configurations['F']):
                        color = self._get_color_from_code(face_configurations['F'][idx])
                        if cube_array[i][j][2] is None:
                            cube_array[i][j][2] = [None] * 6
                        cube_array[i][j][2][0] = color 
        
        if 'B' in face_configurations:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(face_configurations['B']):
                        color = self._get_color_from_code(face_configurations['B'][idx])
                        if cube_array[i][j][0] is None:
                            cube_array[i][j][0] = [None] * 6
                        cube_array[i][j][0][1] = color 
        
        if 'L' in face_configurations:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(face_configurations['L']):
                        color = self._get_color_from_code(face_configurations['L'][idx])
                        if cube_array[0][i][j] is None:
                            cube_array[0][i][j] = [None] * 6
                        cube_array[0][i][j][2] = color 
        
        if 'R' in face_configurations:
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if idx < len(face_configurations['R']):
                        color = self._get_color_from_code(face_configurations['R'][idx])
                        if cube_array[2][i][j] is None:
                            cube_array[2][i][j] = [None] * 6
                        cube_array[2][i][j][3] = color

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = positions[i]
                    y = positions[j]
                    z = positions[k]
                    
                    colors = cube_array[i][j][k] if cube_array[i][j][k] is not None else [None] * 6
                    
                    self.cubies.append(Cubie((x, y, z), colors))
    
    def _get_color_from_code(self, code):
        if code in COLOR_MAP:
            return COLOR_MAP[code]
        else:
            return WHITE
    
    def draw(self):
        for cubie in self.cubies:
            cubie.draw()
    
    def rotate_cube(self, axis, angle):
        for cubie in self.cubies:
            if axis == 'x':
                cubie.rotate_x(angle)
            elif axis == 'y':
                cubie.rotate_y(angle)
            elif axis == 'z':
                cubie.rotate_z(angle)
    
    def get_face_cubies(self, face):
        face_cubies = []
        tolerance = 0.01
        if face == 'U':
            face_cubies = [c for c in self.cubies if abs(c.position[1] - CUBE_SIZE/3) < tolerance]
        elif face == 'D':  
            face_cubies = [c for c in self.cubies if abs(c.position[1] + CUBE_SIZE/3) < tolerance]
        elif face == 'L': 
            face_cubies = [c for c in self.cubies if abs(c.position[0] + CUBE_SIZE/3) < tolerance]
        elif face == 'R':  
            face_cubies = [c for c in self.cubies if abs(c.position[0] - CUBE_SIZE/3) < tolerance]
        elif face == 'F':  
            face_cubies = [c for c in self.cubies if abs(c.position[2] - CUBE_SIZE/3) < tolerance]
        elif face == 'B':
            face_cubies = [c for c in self.cubies if abs(c.position[2] + CUBE_SIZE/3) < tolerance]
        return face_cubies
    
    def start_animation(self, face, direction):
        self.animating = True
        self.animation_frames = 10
        self.animation_face = face
        self.animation_direction = direction
        self.animation_progress = 0
    
    def update_animation(self):
        if not self.animating:
            return False
        
        self.animation_progress += 1
        angle = (math.pi/2) / self.animation_frames * self.animation_direction
        
        face_cubies = self.get_face_cubies(self.animation_face)
        
        if self.animation_face == 'U':
            for cubie in face_cubies:
                cubie.rotate_y(angle)
        elif self.animation_face == 'D':
            for cubie in face_cubies:
                cubie.rotate_y(-angle)
        elif self.animation_face == 'L':
            for cubie in face_cubies:
                cubie.rotate_x(angle)
        elif self.animation_face == 'R':
            for cubie in face_cubies:
                cubie.rotate_x(-angle)
        elif self.animation_face == 'F':
            for cubie in face_cubies:
                cubie.rotate_z(-angle)
        elif self.animation_face == 'B':
            for cubie in face_cubies:
                cubie.rotate_z(angle)
        
        if self.animation_progress >= self.animation_frames:
            self.animating = False
            return False
        
        return True
    
    def make_move(self, move):
        if self.animating:
            return
        
        face = move[0]
        direction = -1 if "'" in move else 1
        
        self.start_animation(face, direction)


def visualise_solution(custom_face_configs=None, solving_instructions=None):
    # This function maps the scanned faces to the 3D Cube
    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Rubik's Cube")
    
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -10)
    
    if custom_face_configs:
        rotated_face_configs = {}
        for face, colors in custom_face_configs.items():
            
            face_grid = np.array(colors).reshape(3, 3)
            rotated_grid = np.rot90(face_grid, k=-1)
            rotated_face_configs[face] = rotated_grid.flatten().tolist()
        
        cube = RubiksCube(rotated_face_configs)
    else:
        cube = RubiksCube()
    
    rotation_x = 0
    rotation_y = 0
    last_mouse_pos = None
    
    current_instruction_index = -1
    history = []  
    
    def get_opposite_move(move):
        if len(move) == 1:
            return move + "'"
        elif move.endswith("'"):
            return move[0]
        elif move.endswith("2"):
            return move
        else:
            return move + "'"
    
    def execute_move(move):
        """Execute a move on the cube with proper animation for all move types."""
        face = move[0]

        if len(move) == 1 or move.endswith("'"):
            cube.make_move(move)
            while cube.animating:
                cube.update_animation()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
                glTranslatef(0, 0, -10)
                glRotatef(rotation_x * 20, 1, 0, 0)
                glRotatef(rotation_y * 20, 0, 1, 0)
                cube.draw()
                pygame.display.flip()
                clock.tick(60)
        
        elif move.endswith("2"):

            cube.make_move(face)

            while cube.animating:
                cube.update_animation()

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
                glTranslatef(0, 0, -10)
                glRotatef(rotation_x * 20, 1, 0, 0)
                glRotatef(rotation_y * 20, 0, 1, 0)
                cube.draw()
                pygame.display.flip()
                clock.tick(60)
            
            cube.make_move(face)
            while cube.animating:
                cube.update_animation()
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
                glTranslatef(0, 0, -10)
                glRotatef(rotation_x * 20, 1, 0, 0)
                glRotatef(rotation_y * 20, 0, 1, 0)
                cube.draw()
                pygame.display.flip()
                clock.tick(60)
    
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
 
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  
                    last_mouse_pos = pygame.mouse.get_pos()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  
                    last_mouse_pos = None
            
            elif event.type == pygame.MOUSEMOTION:
                if last_mouse_pos:
                    x, y = event.pos
                    last_x, last_y = last_mouse_pos
                    
                    rotation_y += (x - last_x) / 100.0
                    rotation_x += (y - last_y) / 100.0
                    
                    last_mouse_pos = (x, y)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                
                if not cube.animating and solving_instructions:
                    if event.key == K_RIGHT: 
                        if current_instruction_index < len(solving_instructions) - 1:
                            current_instruction_index += 1
                            move = solving_instructions[current_instruction_index]
                            execute_move(move)
                            history.append(move)
                            print(f"Executing move: {move}, Move {current_instruction_index + 1}/{len(solving_instructions)}")
                    
                    elif event.key == K_LEFT:
                        if history:
                            last_move = history.pop()
                            opposite_move = get_opposite_move(last_move)
                            execute_move(opposite_move)
                            current_instruction_index -= 1
                            print(f"Undoing move: {last_move} with {opposite_move}, Back to move {current_instruction_index + 1}/{len(solving_instructions)}")
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0, 0, -10)
        glRotatef(rotation_x * 20, 1, 0, 0)
        glRotatef(rotation_y * 20, 0, 1, 0)
        
        cube.update_animation()
        
        cube.draw()
        
        pygame.display.flip()
        
        clock.tick(60)
    
    pygame.quit()
    sys.exit()