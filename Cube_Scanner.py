import cv2
import numpy as np
import logging
from typing import Tuple, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CubeState:
    up_face: np.ndarray
    right_face: np.ndarray
    front_face: np.ndarray
    down_face: np.ndarray
    left_face: np.ndarray
    back_face: np.ndarray

class RubiksCubeDetector:
    # This class handles detection and processing of the Rubik's Cube faces

    # Pre-defined hsv color ranges used for classifying the sampled colors
    color_ranges_hsv = {
        'white':  ((0, 0, 150), (180, 60, 255)),
        'yellow': ((20, 100, 100), (40, 255, 255)),
        'blue':   ((90, 70, 40), (120, 255, 255)),
        'red':    ((160, 75, 75), (180, 255, 255)),
        'orange': ((5, 75, 75), (15, 255, 255)),
        'green':  ((55, 80, 80), (85, 255, 255)),
    }

    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        self.debug_mode = True
        self.state = CubeState(
            *[np.zeros((3, 3)) for _ in range(6)]
        )
    
    def process_image(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        color_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
        processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, self.kernel)
        
        return processed, color_frame

    def detect_face_from_grid(self, frame: np.ndarray, grid_coords: Tuple[int, int, int]) -> Tuple[np.ndarray, List]:
        # Scanning faces of the cube face by directly sampling colors from the grid
        x, y, size = grid_coords
        cell_size = size // 3
        blob_colors = []
        
        flipped_frame = cv2.flip(frame, 1)
        debug_frame = flipped_frame.copy()
        
        w = frame.shape[1]
        
        # Creating predetermined grid cells to sample colors directly from them
        for i in range(3):
            for j in range(3):
                
                display_cell_x = x + j * cell_size
                display_cell_y = y + i * cell_size
                
                orig_cell_x = w - display_cell_x - cell_size
                orig_cell_y = display_cell_y 
                
                roi = frame[orig_cell_y:orig_cell_y+cell_size, orig_cell_x:orig_cell_x+cell_size]
                if roi.size == 0:
                    continue
                
                center_region_size = max(cell_size // 3, 4)
                center_x = cell_size // 2 - center_region_size // 2
                center_y = cell_size // 2 - center_region_size // 2
                
                if (center_x < 0 or center_y < 0 or 
                    center_x + center_region_size > roi.shape[1] or 
                    center_y + center_region_size > roi.shape[0]):
                    center_roi = roi
                else:
                    center_roi = roi[center_y:center_y+center_region_size, 
                                    center_x:center_x+center_region_size]
                
                # Calculating the mean color value
                color = np.array(cv2.mean(center_roi)).astype(int)[:3]
                
                position_val = (i * 3) + j
                
                blob_info = [*color, position_val, orig_cell_x, orig_cell_y, cell_size, cell_size]
                blob_colors.append(blob_info)
                
                cv2.rectangle(debug_frame, (display_cell_x, display_cell_y), 
                            (display_cell_x+cell_size, display_cell_y+cell_size), (0, 0, 0), 2)
                
                cv2.rectangle(debug_frame, 
                            (display_cell_x+cell_size//4, display_cell_y+cell_size//4), 
                            (display_cell_x+3*cell_size//4, display_cell_y+3*cell_size//4), 
                            (int(color[0]), int(color[1]), int(color[2])), -1)
                
                b, g, r = color
                cv2.putText(debug_frame, f"{r},{g},{b}", 
                        (display_cell_x, display_cell_y+cell_size-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Process the face of the cube at the end
        return self._process_face(blob_colors)

    def detect_face(self, frame: np.ndarray, grid_coords=None) -> Tuple[np.ndarray, List]:
        if grid_coords is not None:
            return self.detect_face_from_grid(frame, grid_coords)
        
        processed, color_frame = self.process_image(frame)
        
        debug_frame = frame.copy()
        
        contours = cv2.findContours(processed, 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        blob_colors = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.7 < aspect_ratio < 1.3 and self._is_valid_square(contour):
                    roi = color_frame[y:y+h, x:x+w]
                    color = self._analyze_color(roi)
                    position_val = (50*y) + (10*x)
                    blob_info = [*color, position_val, x, y, w, h]
                    blob_colors.append(blob_info)
                    
                    cv2.drawContours(debug_frame, [contour], -1, (0, 0, 0), 2)
                    cv2.rectangle(debug_frame, (x,y), (x+w,y+h), (0, 0, 0), 1)
        
        return self._process_face(blob_colors)

    def _is_valid_square(self, contour: np.ndarray) -> bool:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) != 4:
            return False
        return True

    def _analyze_color(self, roi: np.ndarray) -> List[int]:
        h, w = roi.shape[:2]
        center_roi = roi[h//4:3*h//4, w//4:3*w//4]
        
        if center_roi.size > 0:
            return np.array(cv2.mean(center_roi)).astype(int)[:3]
        return np.array(cv2.mean(roi)).astype(int)[:3]

    def _process_face(self, blob_colors: List) -> Tuple[np.ndarray, List]:
        if len(blob_colors) != 9:
            logger.warning(f"Expected 9 squares, found {len(blob_colors)}")
            return np.array([0, 0, 0]), blob_colors
            
        blob_colors = sorted(blob_colors, key=lambda x: x[3])
        face = np.zeros(9)
        face = face.astype(int)
        
        for i, color in enumerate(blob_colors):
            b, g, r = color[:3]
            logger.info(f"Square {i}: B={b}, G={g}, R={r}")
            face[i] = self._classify_color(color[:3])
            
        logger.info(f"Classified face: {face}")
            
        return (face.reshape(3, 3), blob_colors) if np.count_nonzero(face) == 9 \
               else (np.array([0, 0]), blob_colors)

    def _classify_color(self, rgb_color: np.ndarray) -> int:
        # Classifying the RGB colour using HSV ranges
        rgb_pixel = np.uint8([[rgb_color]])
        hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_pixel

        for color_name, (lower, upper) in self.color_ranges_hsv.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return self._color_name_to_code(color_name)

        logger.warning(f"Unclassified HSV color: H={h}, S={s}, V={v}")
        return 0

    def _color_name_to_code(self, color_name: str) -> int:
        mapping = {
            'white': 1,
            'yellow': 2,
            'red': 3,
            'green': 4,
            'blue': 5,
            'orange': 6,
        }
        return mapping.get(color_name, 0)
    
    def initialize_capture(self) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "ERROR: Cannot access camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, "Please connect camera or check permissions", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(error_frame, "Press any key to exit", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera Error", error_frame)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            raise RuntimeError("Cannot access camera")
            
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception as e:
            logger.warning(f"Could not set camera properties: {e}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(
            'cube_solution.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            20.0,
            (width, height)
        )
        return cap, writer

    def draw_unfolded_cube_grid(self, frame: np.ndarray, show_alignment_text=True) -> Tuple[np.ndarray, dict]:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cell_size = min(w, h) // 12
        grid_overlay = frame.copy()
        center_x = w // 2
        center_y = h // 2
        
        # Defining face positions in the grid
        face_positions = {
            'up': (0,0),        # White (middle center)
            'front': (-3,0),    # Blue (top center)
            'left': (0, -3),    # Red (middle left)
            'right': (0, 3),    # Orange (middle right)
            'back': (3,0),      # Green (bottom center)
            'down': (0,6),      # Yellow (middle far right)
        }
        
        grid_coords = {}
        
        # Defining letters to represent each face
        face_letters = {
            'up': 'W',
            'front': 'B',
            'left': 'R',      
            'right': 'O',
            'back': 'G',
            'down': 'Y'
        }
        
        orig_w = cv2.flip(frame, 1).shape[1]
        
        # Drawing each 3x3 face
        for face, (row_offset, col_offset) in face_positions.items():
            face_center_x = center_x + col_offset * cell_size
            face_center_y = center_y + row_offset * cell_size
            
            start_x = face_center_x - (cell_size * 3) // 2
            start_y = face_center_y - (cell_size * 3) // 2
            
            grid_coords[face] = (start_x, start_y, cell_size * 3)
            
            for i in range(3):
                for j in range(3):
                    cell_x = start_x + (j * cell_size)
                    cell_y = start_y + (i * cell_size)
                    
                    cv2.rectangle(
                        grid_overlay, 
                        (cell_x, cell_y), 
                        (cell_x + cell_size, cell_y + cell_size),
                        (0, 0, 0), 
                        2
                    )
                    
                    orig_frame = cv2.flip(frame, 1)
                    orig_cell_x = orig_w - cell_x - cell_size
                    orig_cell_y = cell_y
                    
                    roi = orig_frame[orig_cell_y:orig_cell_y+cell_size, orig_cell_x:orig_cell_x+cell_size]
                    if roi.size > 0: 
                        center_region_size = max(cell_size // 3, 4)
                        center_x_offset = cell_size // 2 - center_region_size // 2
                        center_y_offset = cell_size // 2 - center_region_size // 2
                        
                        if (center_x_offset >= 0 and center_y_offset >= 0 and
                            center_x_offset + center_region_size <= roi.shape[1] and
                            center_y_offset + center_region_size <= roi.shape[0]):
                            center_roi = roi[center_y_offset:center_y_offset+center_region_size,
                                            center_x_offset:center_x_offset+center_region_size]
                            # Get mean color value
                            color = np.array(cv2.mean(center_roi)).astype(int)[:3]
                            
                            cv2.rectangle(
                                grid_overlay,
                                (cell_x + cell_size//4, cell_y + cell_size//4),
                                (cell_x + 3*cell_size//4, cell_y + 3*cell_size//4),
                                (int(color[0]), int(color[1]), int(color[2])),
                                -1
                            )
                            
                    # Adding the letters of each face to the center cell
                    if i == 1 and j == 1:
                        if face_letters[face] == 'W':
                            cv2.putText(
                            grid_overlay,
                            face_letters[face],
                            (cell_x + cell_size//2 - 8, cell_y + cell_size//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )
                        elif face_letters[face] == 'B':
                            cv2.putText(
                            grid_overlay,
                            face_letters[face],
                            (cell_x + cell_size//2 - 8, cell_y + cell_size//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0),
                            2
                        )
                        elif face_letters[face] == 'R':
                            cv2.putText(
                            grid_overlay,
                            face_letters[face],
                            (cell_x + cell_size//2 - 8, cell_y + cell_size//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )
                        elif face_letters[face] == 'O':
                            cv2.putText(
                            grid_overlay,
                            face_letters[face],
                            (cell_x + cell_size//2 - 8, cell_y + cell_size//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 165, 255),
                            2
                        )
                        elif face_letters[face] == 'G':
                            cv2.putText(
                            grid_overlay,
                            face_letters[face],
                            (cell_x + cell_size//2 - 8, cell_y + cell_size//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )
                        else: # Yellow
                            cv2.putText(
                            grid_overlay,
                            face_letters[face],
                            (cell_x + cell_size//2 - 8, cell_y + cell_size//2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                            (0, 255, 255),
                            2
                        )

        if show_alignment_text:
            text_bg = np.zeros((60, w, 3), dtype=np.uint8)
            cv2.putText(
                text_bg,
                "Align cube face with the highlighted grid",
                (50, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            top_overlay = cv2.addWeighted(frame[:60, :], 0.3, text_bg, 0.7, 0)
            grid_overlay[:60, :] = top_overlay
        
        instruction_bg = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.putText(
            instruction_bg,
            "Press SPACE to capture or 'q' to quit",
            (w//2 - 200, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        bottom_overlay = cv2.addWeighted(frame[h-40:, :], 0.3, instruction_bg, 0.7, 0)
        grid_overlay[h-40:, :] = bottom_overlay
        
        return grid_overlay, grid_coords

    def _scan_cube(self, cap: cv2.VideoCapture, writer: cv2.VideoWriter):
        # Scan all faces of the cube with user confirmation
        faces = ['up', 'front', 'left', 'right', 'back', 'down']
        face_colors = {
            'up': 'WHITE',
            'front': 'BLUE', 
            'left': 'RED',
            'right': 'ORANGE',
            'back': 'GREEN',
            'down': 'YELLOW'
        }
        
        color_to_rgb = {
            1: (255, 255, 255),  # White
            2: (0, 255, 255),    # Yellow
            3: (0, 0, 255),      # Red
            4: (0, 255, 0),      # Green
            5: (255, 0, 0),      # Blue
            6: (0, 165, 255)     # Orange
        }
        
        scanned_faces = {}
        
        current_face = 0
        
        while current_face < len(faces):
            face_name = faces[current_face]
            face_captured = False
            face_confirmed = False
            temp_face = None
            
            while not face_confirmed:
                while not face_captured:
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("Failed to capture frame")
                    
                    frame_with_grid, grid_coords = self.draw_unfolded_cube_grid(frame)
                    
                    grid_with_scanned = frame_with_grid.copy()
                    for scanned_face_name, face_data in scanned_faces.items():
                        scanned_face_coords = grid_coords[scanned_face_name]
                        x_scanned, y_scanned, size_scanned = scanned_face_coords
                        cell_size_scanned = size_scanned // 3
                        
                        for i in range(3):
                            for j in range(3):
                                cell_x = x_scanned + j * cell_size_scanned
                                cell_y = y_scanned + i * cell_size_scanned
                                
                                color_val = int(face_data[i, j])
                                if color_val in color_to_rgb:
                                    bg_color = color_to_rgb[color_val]
                                else:
                                    bg_color = (128, 128, 128)
                                
                                cv2.rectangle(
                                    grid_with_scanned,
                                    (cell_x, cell_y),
                                    (cell_x + cell_size_scanned, cell_y + cell_size_scanned),
                                    bg_color,
                                    -1
                                )
                                
                                cv2.rectangle(
                                    grid_with_scanned,
                                    (cell_x, cell_y),
                                    (cell_x + cell_size_scanned, cell_y + cell_size_scanned),
                                    (0, 0, 0),
                                    2
                                )

                    highlight_frame = grid_with_scanned.copy()
                    face_coord = grid_coords[face_name]
                    x, y, size = face_coord
                  
                    cv2.rectangle(
                        highlight_frame,
                        (x, y),
                        (x + size, y + size),
                        (0, 255, 255), 3  # Yellow highlight to distinguish the cube face currently being captured
                    )
                    
                    faces = ['up', 'front', 'left', 'right', 'back', 'down']
                    color_to_rgb = {
                        1: (255, 255, 255),  
                        2: (0, 255, 255),    
                        3: (0, 0, 255),      
                        4: (0, 255, 0),
                        5: (255, 0, 0),     
                        6: (0, 165, 255)    
                    }

                    color_name_to_key = {
                        'WHITE': 1,
                        'YELLOW': 2,
                        'RED': 3,
                        'GREEN': 4,
                        'BLUE': 5,
                        'ORANGE': 6
                    }

                    def draw_colored_text(img, text, position, font, font_scale, thickness):
                        x, y = position
                        for word in text.split():

                            if word in color_name_to_key:
                                color_key = color_name_to_key[word]
                                color = color_to_rgb[color_key]
                            else:
                                color = (255, 255, 255)
                            
                            (word_width, word_height), baseline = cv2.getTextSize(word + " ", font, font_scale, thickness)
                            
                            cv2.putText(img, word + " ", (x, y), font, font_scale, color, thickness)
                            
                            x += word_width

                    if face_name == 'up':
                        draw_colored_text(highlight_frame, f"Show the WHITE centered face with the BLUE center facing upwards", (50, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    elif face_name == 'front':
                        draw_colored_text(highlight_frame, f"Show the BLUE centered face with the YELLOW center facing upwards", (50, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    elif face_name == 'left':
                        draw_colored_text(highlight_frame, f"Show the RED centered face with the BLUE center facing upwards", (50, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    elif face_name == 'right':
                        draw_colored_text(highlight_frame, f"Show the ORANGE centered face with the BLUE center facing upwards", (50, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    elif face_name == 'back':
                        draw_colored_text(highlight_frame, f"Show the GREEN centered face with the WHITE center facing upwards", (50, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    else:
                        draw_colored_text(highlight_frame, f"Show the YELLOW centered face with the BLUE center facing upwards", (50, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                    writer.write(highlight_frame)
                    cv2.imshow("Rubik's Cube Scanner", highlight_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Spacebar
                        face, colors = self.detect_face_from_grid(frame, face_coord)
                        
                        if len(colors) == 9:
                            if face.size == 9 and face.shape != (3, 3):
                                face = face.reshape(3, 3)
                                
                            if np.count_nonzero(face) == 9:
                                temp_face = face.copy()
                                face_captured = True
                                logger.info(f"Captured {face_name} face")
                            else:
                                h, w = highlight_frame.shape[:2]
                                center_x, center_y = w // 2, h // 2
                                error_frame = highlight_frame.copy()
                                cv2.putText(error_frame, 
                                        "Could not classify all colors! Try again.", 
                                        (center_x - 250, center_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.imshow("Rubik's Cube Scanner", error_frame)
                                cv2.waitKey(1000)
                        else:
                            h, w = highlight_frame.shape[:2]
                            center_x, center_y = w // 2, h // 2
                            error_frame = highlight_frame.copy()
                            cv2.putText(error_frame, 
                                    "Could not detect all 9 squares! Try again.", 
                                    (center_x - 250, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow("Rubik's Cube Scanner", error_frame)
                            cv2.waitKey(1000) 
                    elif key == ord('q'):
                        return 'quit'
                
                # After face is captured, mpa the results to the main grid and ask for confirmation
                if temp_face is not None:
                    ret, fresh_frame = cap.read()
                    if not ret:
                        raise RuntimeError("Failed to capture frame")

                    frame_with_grid, grid_coords = self.draw_unfolded_cube_grid(fresh_frame, show_alignment_text=False)
                    face_coord = grid_coords[face_name]
                    x, y, size = face_coord
                    cell_size = size // 3
                    
                    confirm_frame = frame_with_grid.copy()

                    for i in range(3):
                        for j in range(3):
                            cell_x = x + j * cell_size
                            cell_y = y + i * cell_size
                            
                            color_val = int(temp_face[i, j])
                            if color_val in color_to_rgb:
                                bg_color = color_to_rgb[color_val]
                            else:
                                bg_color = (128, 128, 128)

                            cv2.rectangle(
                                confirm_frame,
                                (cell_x, cell_y),
                                (cell_x + cell_size, cell_y + cell_size),
                                bg_color,
                                -1 
                            )
                            
                            cv2.rectangle(
                                confirm_frame,
                                (cell_x, cell_y),
                                (cell_x + cell_size, cell_y + cell_size),
                                (0, 0, 0), 
                                2
                            )
                    
                    cv2.rectangle(
                        confirm_frame,
                        (x, y),
                        (x + size, y + size),
                        (0, 255, 255), 3
                    )

                    draw_colored_text(confirm_frame,
                            f"Detected {face_colors[face_name]} face. Is this correct?",
                            (50, 51),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    cv2.putText(confirm_frame,
                            "Press 'y' to confirm or 'n' to retry.",
                            (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.imshow("Rubik's Cube Scanner", confirm_frame)
                    writer.write(confirm_frame)
                    
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('y'):
                            setattr(self.state, f"{face_name}_face", temp_face)
                            
                            scanned_faces[face_name] = temp_face.copy()
                            
                            face_confirmed = True
                            
                            success_frame = confirm_frame.copy()
                            cv2.putText(success_frame,
                                    "Face confirmed! Proceeding...",
                                    (50, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.imshow("Rubik's Cube Scanner", success_frame)
                            writer.write(success_frame)
                            cv2.waitKey(1500)
                            break
                        elif key == ord('n'):
                            face_captured = False
                            retry_frame = confirm_frame.copy()
                            cv2.putText(retry_frame,
                                    "Retrying face capture...",
                                    (50, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                            cv2.imshow("Rubik's Cube Scanner", retry_frame)
                            writer.write(retry_frame)
                            cv2.waitKey(1500)
                            break
                        elif key == ord('q'):
                            return 'quit'

            current_face += 1