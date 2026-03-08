"""
test_cube_logic.py

Unit tests for cube state handling, face mapping, and solver logic.

Usage:
    python -m pytest test_cube_logic.py -v
    python test_cube_logic.py
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Patch cv2 and kociemba so tests run without a camera or the solver installed
import sys
sys.modules['cv2'] = MagicMock()
sys.modules['kociemba'] = MagicMock()

from Cube_Solver import RubiksCubeSolver, number_to_letter, number_to_color


def make_solved_solver():
    """Return a solver pre-loaded with a valid solved cube state."""
    solver = RubiksCubeSolver()
    solver.detector.state.up_face    = np.full((3, 3), 1)  # White
    solver.detector.state.right_face = np.full((3, 3), 6)  # Orange
    solver.detector.state.front_face = np.full((3, 3), 5)  # Blue
    solver.detector.state.down_face  = np.full((3, 3), 2)  # Yellow
    solver.detector.state.left_face  = np.full((3, 3), 3)  # Red
    solver.detector.state.back_face  = np.full((3, 3), 4)  # Green
    return solver


class TestKociembaStringGeneration(unittest.TestCase):

    def test_solved_string_length(self):
        """Kociemba string must always be exactly 54 characters."""
        solver = make_solved_solver()
        cube_str = solver._state_to_kociemba_string()
        self.assertEqual(len(cube_str), 54,
                         f"Expected 54 chars, got {len(cube_str)}")

    def test_solved_string_starts_with_up_face(self):
        """First 9 characters must correspond to the Up (White) face."""
        solver = make_solved_solver()
        cube_str = solver._state_to_kociemba_string()
        self.assertTrue(cube_str.startswith('UUUUUUUUU'),
                        f"Expected UUUUUUUUU at start, got: {cube_str[:9]}")

    def test_solved_string_ends_with_back_face(self):
        """Last 9 characters must correspond to the Back (Green) face."""
        solver = make_solved_solver()
        cube_str = solver._state_to_kociemba_string()
        self.assertTrue(cube_str.endswith('BBBBBBBBB'),
                        f"Expected BBBBBBBBB at end, got: {cube_str[-9:]}")

    def test_solved_string_contains_all_faces(self):
        """A solved cube string must contain exactly 9 of each face letter."""
        solver = make_solved_solver()
        cube_str = solver._state_to_kociemba_string()
        for letter in ['U', 'R', 'F', 'D', 'L', 'B']:
            count = cube_str.count(letter)
            self.assertEqual(count, 9,
                             f"Expected 9 '{letter}', found {count} in: {cube_str}")

    def test_full_string_face_order(self):
        """Kociemba expects faces in order U R F D L B — verify the layout."""
        solver = make_solved_solver()
        cube_str = solver._state_to_kociemba_string()
        expected_order = 'UUUUUUUUU' + 'RRRRRRRRR' + 'FFFFFFFFF' + \
                         'DDDDDDDDD' + 'LLLLLLLLL' + 'BBBBBBBBB'
        self.assertEqual(cube_str, expected_order,
                         f"Face order mismatch.\nExpected: {expected_order}\nGot:      {cube_str}")


class TestFaceOrientationTransforms(unittest.TestCase):

    def test_reorganise_up_face_swaps_rows(self):
        """Up face reorganisation should swap the first and last rows."""
        solver = make_solved_solver()
        face = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        result = solver.reorganise_face(face.copy(), "up")
        np.testing.assert_array_equal(result[0], [7, 8, 9])
        np.testing.assert_array_equal(result[2], [1, 2, 3])

    def test_reorganise_front_face_swaps_rows(self):
        """Front face reorganisation should behave identically to up."""
        solver = make_solved_solver()
        face = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        result = solver.reorganise_face(face.copy(), "front")
        np.testing.assert_array_equal(result[0], [7, 8, 9])
        np.testing.assert_array_equal(result[2], [1, 2, 3])

    def test_reorganise_back_face_flips_horizontally(self):
        """Back face reorganisation should flip the face left-right."""
        solver = make_solved_solver()
        face = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        result = solver.reorganise_face(face.copy(), "back")
        np.testing.assert_array_equal(result[0], [3, 2, 1])

    def test_reorganise_down_face_flips_horizontally(self):
        """Down face reorganisation should flip the face left-right."""
        solver = make_solved_solver()
        face = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        result = solver.reorganise_face(face.copy(), "down")
        np.testing.assert_array_equal(result[0], [3, 2, 1])

    def test_reorganise_output_shape_preserved(self):
        """All reorganised faces must remain (3, 3)."""
        solver = make_solved_solver()
        face = np.arange(1, 10).reshape(3, 3).astype(float)
        for face_name in ["up", "front", "left", "right", "back", "down"]:
            result = solver.reorganise_face(face.copy(), face_name)
            self.assertEqual(result.shape, (3, 3),
                             f"Shape mismatch for face '{face_name}': {result.shape}")


class TestMoveExplanations(unittest.TestCase):

    def test_clockwise_move(self):
        """Single-letter moves should return a clockwise description."""
        solver = make_solved_solver()
        result = solver._explain_move('U')
        self.assertIn('clockwise', result.lower())
        self.assertIn('WHITE', result)

    def test_counter_clockwise_move(self):
        """Prime moves should return a counter-clockwise description."""
        solver = make_solved_solver()
        result = solver._explain_move("U'")
        self.assertIn('counter-clockwise', result.lower())

    def test_double_move(self):
        """Double moves (e.g. U2) should mention 180 degrees."""
        solver = make_solved_solver()
        result = solver._explain_move('U2')
        self.assertIn('180', result)

    def test_all_faces_explained(self):
        """Every face letter should map to a non-empty explanation."""
        solver = make_solved_solver()
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            result = solver._explain_move(face)
            self.assertTrue(len(result) > 0,
                            f"Empty explanation for face '{face}'")


class TestColorMappings(unittest.TestCase):

    def test_number_to_letter_has_six_entries(self):
        self.assertEqual(len(number_to_letter), 6)

    def test_number_to_color_has_six_entries(self):
        self.assertEqual(len(number_to_color), 6)

    def test_all_kociemba_face_letters_present(self):
        """All six Kociemba face letters must be in the mapping."""
        letters = set(number_to_letter.values())
        self.assertEqual(letters, {'U', 'D', 'L', 'R', 'F', 'B'})

    def test_all_color_codes_present(self):
        """All six colour codes must be in the mapping."""
        colors = set(number_to_color.values())
        self.assertEqual(colors, {'W', 'Y', 'R', 'G', 'B', 'O'})


class TestInvalidStateHandling(unittest.TestCase):

    def test_impossible_state_returns_none(self):
        """Solver must return None when kociemba raises an exception."""
        solver = make_solved_solver()
        # Force kociemba.solve to raise (simulates an invalid cube string)
        import kociemba as kc
        kc.solve.side_effect = Exception("Invalid cube")
        result = solver.solve_cube()
        self.assertIsNone(result)

    def test_unknown_color_maps_to_X(self):
        """Color value 0 (unknown) should map to 'X' as a safe fallback."""
        solver = make_solved_solver()
        solver.detector.state.up_face = np.zeros((3, 3))
        cube_str = solver._state_to_kociemba_string()
        self.assertEqual(len(cube_str), 54)
        self.assertIn('X', cube_str,
                      "Unknown color values should produce 'X' in the cube string")


if __name__ == "__main__":
    unittest.main(verbosity=2)
