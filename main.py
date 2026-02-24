from Cube_Solver import RubiksCubeSolver
from Cube_3D import visualise_solution

if __name__ == "__main__":
    solution_found = False

    while not solution_found: 
        solver = RubiksCubeSolver()
        solution, face_configs = solver.run()

        solution_found = solution != None
        if solution == 'quit':
            break

        if solution_found:
            visualise_solution(face_configs, solving_instructions=solution)
            break