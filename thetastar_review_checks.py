"""Behavior checks without reference Theta* paths or rasterization-specific LOS tests."""

import io
import math
from contextlib import redirect_stdout

import numpy as np


def _check_path(task_map, start, goal, end, line_of_sight):
    path = []
    seen = set()
    while end is not None:
        point = (end.i, end.j)
        assert point not in seen, "Parent pointers contain a cycle."
        assert task_map.in_bounds(*point) and task_map.traversable(*point)
        assert math.isfinite(end.g), "Path costs must be finite."
        seen.add(point)
        path.append(end)
        end = end.parent
    path.reverse()
    assert path and (path[0].i, path[0].j) == start
    assert (path[-1].i, path[-1].j) == goal
    assert math.isclose(path[0].g, 0, abs_tol=1e-9)
    length = 0.0
    for first, second in zip(path, path[1:]):
        step = math.hypot(second.i - first.i, second.j - first.j)
        assert step > 0
        # Neighbor moves may follow a different corner policy from LOS.
        # Non-neighbor shortcuts must be visible under the student's LOS.
        if (second.i, second.j) not in task_map.get_neighbors(first.i, first.j):
            assert line_of_sight(task_map, first, second), "A shortcut is not visible."
        assert math.isclose(second.g, first.g + step, rel_tol=1e-9, abs_tol=1e-9), (
            "A g-value does not match the Euclidean segment cost."
        )
        length += step
    assert math.isclose(path[-1].g, length, rel_tol=1e-9, abs_tol=1e-9)
    lower_bound = math.hypot(goal[0] - start[0], goal[1] - start[1])
    assert length + 1e-9 >= lower_bound, "A path cannot be shorter than straight-line distance."
    return path


def review_thetastar(Map, Node, SearchTreePQD, euclidean_distance, compute_cost,
                     line_of_sight, thetastar):
    for start, goal, expected in (((0, 0), (0, 0), 0), ((1, 2), (4, 6), 5),
                                  ((0, 0), (1, 1), math.sqrt(2))):
        assert math.isclose(euclidean_distance(*start, *goal), expected)
        assert math.isclose(euclidean_distance(*goal, *start), expected)
        assert math.isclose(compute_cost(*start, *goal), expected)

    empty = Map(np.zeros((9, 12), dtype=np.int8))
    for start, goal in (((0, 0), (0, 0)), ((1, 1), (1, 10)),
                        ((1, 2), (7, 2)), ((1, 1), (7, 7)), ((1, 1), (6, 9))):
        assert line_of_sight(empty, Node(*start), Node(*goal)), "Free space should be visible."
        assert line_of_sight(empty, Node(*goal), Node(*start))

    # All obstacles here intersect the segment away from cell boundaries.
    # No expected answer is imposed for grazing edges or corner contacts.
    for start, goal, obstacle in (((3, 0), (3, 8), (3, 4)),
                                  ((0, 4), (8, 4), (4, 4)),
                                  ((0, 0), (8, 8), (4, 4))):
        cells = np.zeros((9, 9), dtype=np.int8)
        cells[obstacle] = 1
        task_map = Map(cells)
        assert not line_of_sight(task_map, Node(*start), Node(*goal)), (
            "LOS must reject a segment through an obstacle's center."
        )
        assert not line_of_sight(task_map, Node(*goal), Node(*start))
        assert not line_of_sight(task_map, Node(*obstacle), Node(*goal))
        assert not line_of_sight(task_map, Node(*start), Node(*obstacle))

    wall = np.zeros((9, 12), dtype=np.int8)
    wall[:6, 5:7] = 1  # A thick wall with a wide passage below it.
    disconnected = np.zeros((9, 12), dtype=np.int8)
    disconnected[:, 5:7] = 1
    cases = (
        (np.zeros((1, 1), dtype=np.int8), (0, 0), (0, 0), True),
        (np.zeros((9, 12), dtype=np.int8), (1, 1), (6, 9), True),
        (wall, (2, 2), (2, 9), True),
        (wall, (2, 9), (2, 2), True),
        (disconnected, (2, 2), (2, 9), False),
    )
    for index, (cells, start, goal, reachable) in enumerate(cases):
        task_map = Map(cells)
        with redirect_stdout(io.StringIO()):
            result = thetastar(task_map, *start, *goal, euclidean_distance, SearchTreePQD)
        assert isinstance(result, tuple) and len(result) == 6, "Search must return six values."
        found, end, steps, tree_size, opened, expanded = result
        assert isinstance(found, bool) and found == reachable
        assert isinstance(steps, int) and steps >= 0
        assert isinstance(tree_size, int) and tree_size >= 0
        assert expanded is not None
        if not reachable:
            assert end is None
            continue  # OPEN may be None or an empty iterable on failure.
        assert end is not None and opened is not None
        path = _check_path(task_map, start, goal, end, line_of_sight)
        if index == 1:
            assert any(max(abs(b.i - a.i), abs(b.j - a.j)) > 1
                       for a, b in zip(path, path[1:])), (
                "Theta* should use a non-neighbor shortcut in unobstructed space."
            )
    print("Theta* assertion checks passed (path validity, not global optimality).")
