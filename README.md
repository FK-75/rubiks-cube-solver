# Rubik's Cube Solver

A computer vision-powered Rubik's Cube solver that scans a physical cube via webcam, computes an optimal solution using the Kociemba two-phase algorithm, and visualises the step-by-step solution on an interactive 3D cube.

---

## Features

- **Webcam face scanning** — hold each face up to your camera and press spacebar to capture it
- **HSV color detection** — samples a fixed grid overlay to classify each cubie's color robustly
- **Kociemba solver** — finds a near-optimal solution (typically under 20 moves)
- **Interactive 3D visualisation** — step forward and backward through the solution on a rendered OpenGL cube
- **Confirmation step** — review each scanned face before committing, with the option to retry

---

## Performance

> Run `python benchmark_performance.py` to generate these stats for your own hardware.

```
Average solve time : 8.1 ms  (min 0.01 ms / max 50.1 ms)
Average move count : 18 moves  (range 6–22)
Algorithm          : Kociemba two-phase
Hardware           : Apple Silicon (M-series), macOS
```

Scan processing time per face is dominated by webcam frame capture and is typically under 50 ms on any modern machine.

---

## Requirements

- Python 3.9+
- A webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** On some systems you may need to install `PyOpenGL-accelerate` separately or skip it if it fails to build. The solver will still work without it.

---

## Usage

```bash
python main.py
```

### Scanning your cube

The scanner will guide you through all 6 faces one at a time. For each face:

1. Hold the face up to your webcam so it fits inside the on-screen grid
2. Follow the on-screen instruction for which face and orientation to show
3. Press **`Space`** to capture
4. Press **`Y`** to confirm or **`N`** to retry
5. Press **`Q`** at any time to quit

**Scanning order and orientations:**

| Face  | Center colour | Top of the cube when scanning |
|-------|--------------|-------------------------------|
| Up    | White        | Blue centre facing up         |
| Front | Blue         | Yellow centre facing up       |
| Left  | Red          | Blue centre facing up         |
| Right | Orange       | Blue centre facing up         |
| Back  | Green        | White centre facing up        |
| Down  | Yellow       | Blue centre facing up         |

### 3D Visualiser controls

| Key / Input     | Action                   |
|-----------------|--------------------------|
| `→` Right Arrow | Apply next solution move |
| `←` Left Arrow  | Undo last move           |
| Click + Drag    | Rotate the cube view     |
| `Escape`        | Quit                     |

---

## Project Structure

```
├── main.py                  # Entry point — orchestrates scanning and visualisation
├── Cube_Scanner.py          # Webcam capture, color detection, and face scanning UI
├── Cube_Solver.py           # Kociemba string construction and solution retrieval
├── Cube_3D.py               # Pygame/OpenGL 3D cube renderer and animation
├── tests/
│   ├── test_cube_logic.py       # Unit tests (20 tests, no camera required)
│   └── benchmark_performance.py # Automated solve-speed benchmarking tool
└── requirements.txt
```

---

## How It Works

1. **Scanning** (`Cube_Scanner.py`): A fixed 3×3 grid overlay is drawn on the webcam feed. When the user captures a face, the mean BGR color of each cell's centre region is sampled and classified against pre-defined HSV ranges for all 6 colors.

2. **Solving** (`Cube_Solver.py`): The 6 scanned face arrays are mapped to the face order and orientation expected by the [Kociemba algorithm](https://github.com/muodov/kociemba) (`U R F D L B`). The resulting 54-character string is passed to the solver, which returns a move sequence.

3. **Visualisation** (`Cube_3D.py`): The scanned face colors are mapped onto 27 individual cubie objects rendered with PyOpenGL. Each solution move is applied as a smooth animation. The user steps through moves manually and can undo them.

---

## Design Decisions

**HSV over RGB for color classification**
RGB values change dramatically with lighting — the same orange sticker can look red or yellow under different lamps. HSV separates hue from brightness, making the color ranges far more robust to lighting variation. The scanner defines per-color HSV bands that tolerate a wide range of exposures.

**Fixed grid overlay over contour detection**
An early version attempted to detect the cube's squares via edge detection and contour filtering. This proved unreliable — shadows, sticker wear, and cube angle all produced false positives or missed squares. Switching to a fixed-position overlay and asking the user to align the cube removes that uncertainty entirely and makes classification deterministic.

**Manual step-through over auto-playback**
Automatically running through all moves at speed makes it easy to lose track of where you are, especially for a beginner following along with a physical cube. Requiring a keypress per move keeps the user in control and lets them pause, rotate the view, and undo mistakes — which is far more useful in practice.

**Orientation normalisation in software**
Each face is scanned from a slightly different camera angle, so the raw grid arrays don't map directly to Kociemba's expected layout. Rather than forcing the user to hold the cube identically for every face, the face arrays are rotated and flipped in `reorganise_face` before building the cube string. This keeps the UX simple while keeping the mapping correct.

---

## Tests

Tests cover cube string generation, face orientation transforms, move explanations, color mappings, and invalid state handling. All tests are mocked — no camera or physical cube required.

```bash
python tests/test_cube_logic.py
```

```
Ran 20 tests in 0.08s

OK
```

Or with pytest for a richer output:

```bash
pip install pytest
pytest tests/test_cube_logic.py -v
```

---

## Future Improvements

- **CNN-based color classification** — replace the HSV range lookup with a small trained classifier to handle unusual sticker colors and poor lighting more gracefully
- **Automatic face detection** — detect the cube's position in the frame using contour analysis so the user doesn't need to manually align it to the grid
- **Auto-solve playback** — add an optional mode that steps through the solution automatically at a configurable speed
- **Mobile deployment** — port the scanner to a phone camera using OpenCV on Android/iOS or a lightweight web interface
- **Edge device optimisation** — reduce dependency footprint to run on a Raspberry Pi or similar hardware

---

## Troubleshooting

**Colors are misclassified** — Improve the lighting around your cube. Avoid strong shadows across the face. The HSV ranges in `Cube_Scanner.py` (`color_ranges_hsv`) can be tuned if your cube's sticker colors are unusual.

**"No solution found"** — One or more faces were likely scanned in the wrong orientation. Retry the full scan, paying close attention to the orientation instructions for each face.

**Camera not detected** — Ensure no other application is using your webcam. If you have multiple cameras, you can change the device index in `Cube_Scanner.py` inside `initialize_capture`.

---

## Dependencies

| Package           | Purpose                                    |
|-------------------|--------------------------------------------|
| `opencv-python`   | Webcam capture and image processing        |
| `numpy`           | Face array manipulation                    |
| `kociemba`        | Two-phase Rubik's Cube solving algorithm   |
| `pygame`          | Window and event management for 3D viewer  |
| `PyOpenGL`        | 3D cube rendering                          |
