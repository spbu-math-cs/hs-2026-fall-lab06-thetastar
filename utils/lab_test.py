import random
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, Union

from utils.Map import Map
from utils.Node import Node
from utils.utils import make_path, convert_string_to_cells, draw, draw_anyangle


def simple_test(search_function: Callable, task: Union[int, None], draw_result=True, *args):
    """
    Function `simple_test` runs `search_function` on one task. Use a number from 0 to 24 to choose a specific debug task
    on a simple map, or use None to select a random task from this pool. The function displays:
     - 'Path found!' and some statistics if a path was found.
     - 'Path not found!' if a path was not discovered.
     - 'Execution error' if an error occurred during the execution of the search_function.
    In the first case, the function also provides a visualization of the task.

    Parameters
    ----------
    search_function : Callable
        Implementation of the search method.
    task : int | None
        A number from 0 to 24 to choose a specific debug task on a simple map,
        or None to select a random task from this pool.
    *args
        Additional arguments passed to the search function.
    """

    def get_map_data():
        map_str = dedent(
            """
            . . . . . . . . . . . . . . . . . . . . . # # . . . . . . .  
            . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . 
            . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . 
            . . . # # . . . . . . . . . . . . . . . . # # . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . # # . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . # # # # # . . . . 
            . . . # # . . . . . . . . # # . . . . . . # # # # # . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . . . . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
        """
        )
        cells = convert_string_to_cells(map_str)
        return Map(cells)

    task_map = get_map_data()
    starts = [(9, 0), (13, 0), (7, 28), (14, 29), (4, 1), (0, 17), (5, 6), (5, 20), (12, 2), (7, 28), (11, 9), (3, 2), (3, 17), (13, 20), (1, 1), (9, 10), (14, 6), (2, 0), (9, 28), (8, 6), (11, 6), (3, 0), (8, 9), (14, 7), (12, 4)]
    goals = [(11, 20), (2, 19), (6, 5), (4, 18), (9, 20), (7, 0), (2, 25), (12, 4), (3, 25), (0, 12), (4, 23), (2, 24), (9, 2), (1, 6), (13, 29), (14, 29), (2, 28), (14, 16), (13, 0), (1, 27), (14, 25), (10, 20), (12, 28), (2, 29), (1, 29)]
    theta_lenths = [28.784, 22.934, 24.549, 14.866, 22.359, 19.973, 25.846, 20.008, 32.162, 18.402, 27.443, 30.408, 20.964, 19.908, 31.181, 26.149, 31.718, 25.514, 32.991, 27.964, 27.008, 23.393, 23.904, 31.782, 32.788]
    astar_lenghts = [29.456, 23.556, 25.899, 15.142, 22.728, 21.071, 27.314, 20.556, 33.213, 18.899, 29.142, 31.728, 21.142, 21.314, 32.971, 27.799, 33.799, 26.243, 34.627, 29.556, 27.456, 24.314, 25.385, 33.799, 34.042]
    
    if (task is None) or not (0 <= task < 25):
        task = random.randint(0, 24)

    start = Node(*starts[task])
    goal = Node(*goals[task])
    try:
        (
            found,
            end_node,
            number_of_steps,
            search_tree_size,
            open_set,
            closed_set,
        ) = search_function(task_map, start.i, start.j, goal.i, goal.j, *args)

        if found:
            path, path_length = make_path(end_node)
            if search_function.__name__ == "astar":
                if draw_result:
                    draw(task_map, start, goal, path, open_set, closed_set)
                correct = abs(float(path_length) - float(astar_lenghts[task])) < 0.01
                comparison = f"Correct: {correct}."
            else:
                if draw_result:
                    draw_anyangle(task_map, start, goal, path, open_set, closed_set)
                comparison = (
                    f"Reference run length: {theta_lenths[task]} "
                    "(for comparison only; Theta* does not guarantee the shortest path)."
                )
            print(
                f"Path found! Length: {round(path_length, 3)}. "
                f"Tree size: {search_tree_size}. "
                f"Number of steps: {number_of_steps}. "
                + comparison
            )
        else:
            print("Path not found!")
        return

    except Exception as e:
        print(f"Execution error: {e}")
        traceback.print_exc()

def massive_test(search_function: Callable, *args):
    for i in range(25):
        simple_test(search_function, i, False, *args)
