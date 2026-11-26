"""Pure-Python-compatible LAPJV adapter backed by SciPy.

This shadows the blocked compiled `lap` wheel when commands run from the project
root and preserves the subset of lapjv's API used by Ultralytics trackers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "0.5.14-scipy"


def lapjv(
    cost_matrix: Any,
    extend_cost: bool = False,
    cost_limit: float = np.inf,
    return_cost: bool = True,
):
    """Solve a rectangular assignment and leave over-limit matches unassigned."""
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be a two-dimensional array")
    if not np.isfinite(costs).all():
        raise ValueError("cost_matrix must contain only finite values")

    rows, columns = costs.shape
    row_assignment = np.full(rows, -1, dtype=np.int32)
    column_assignment = np.full(columns, -1, dtype=np.int32)
    if rows == 0 or columns == 0:
        if return_cost:
            return 0.0, row_assignment, column_assignment
        return row_assignment, column_assignment
    if not extend_cost and rows != columns:
        raise ValueError("cost_matrix must be square when extend_cost is False")

    if np.isfinite(cost_limit):
        # Dummy rows/columns let the solver prefer any valid match over leaving
        # both endpoints unmatched, while still excluding over-limit edges.
        size = rows + columns
        scale = max(1.0, float(np.max(np.abs(costs))), abs(float(cost_limit)))
        unmatched_cost = scale * (size + 1) * 4.0
        augmented = np.full((size, size), unmatched_cost, dtype=np.float64)
        augmented[:rows, :columns] = np.where(costs <= cost_limit, costs, unmatched_cost * 2.0)
        augmented[rows:, columns:] = 0.0
        assigned_rows, assigned_columns = linear_sum_assignment(augmented)
        assignments = [
            (row, column)
            for row, column in zip(assigned_rows.tolist(), assigned_columns.tolist())
            if row < rows and column < columns and costs[row, column] <= cost_limit
        ]
    else:
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        assignments = list(zip(assigned_rows.tolist(), assigned_columns.tolist()))

    total_cost = 0.0
    for row, column in assignments:
        match_cost = float(costs[row, column])
        row_assignment[row] = column
        column_assignment[column] = row
        total_cost += match_cost

    if return_cost:
        return total_cost, row_assignment, column_assignment
    return row_assignment, column_assignment
