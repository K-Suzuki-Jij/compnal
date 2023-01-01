from enum import Enum

from compnal.base_compnal import base_solver


class Algorithm(Enum):
    """Update algorithm."""

    METROPOLIS = 0
    HEAT_BATH = 1
    IRKMR = 2
    RKMR = 3
    SWENDSEN_WANG = 4
    WOLFF = 5


def cast_base_algorithm(algorithm: base_solver.Algorithm) -> Algorithm:
    """Cast the algorithm from base_solver.Algorithm to solver.Algorithm.

    Args:
        algorithm (base_solver.Algorithm): The algorithm as base_solver.Algorithm.

    Raises:
        RuntimeError: When the unknown algorithm is input.

    Returns:
        Algorithm: The algorithm as solver.Algorithm.
    """
    if algorithm == base_solver.Algorithm.METROPOLIS:
        return Algorithm.METROPOLIS
    elif algorithm == base_solver.Algorithm.HEAT_BATH:
        return Algorithm.HEAT_BATH
    elif algorithm == base_solver.Algorithm.IRKMR:
        return Algorithm.IRKMR
    elif algorithm == base_solver.Algorithm.RKMR:
        return Algorithm.RKMR
    elif algorithm == base_solver.Algorithm.SWENDSEN_WANG:
        return Algorithm.SWENDSEN_WANG
    elif algorithm == base_solver.Algorithm.WOLFF:
        return Algorithm.WOLFF
    else:
        raise RuntimeError("Unknown algorithm")


def cast_algorithm(algorithm: Algorithm):
    """Cast the algorithm from solver.Algorithm to base_solver.Algorithm.

    Args:
        algorithm (Algorithm): The algorithm as solver.Algorithm.

    Raises:
        RuntimeError: When the unknown algorithm is input.

    Returns:
        base_solver.Algorithm: The algorithm as base_solver.Algorithm.
    """
    if algorithm == Algorithm.METROPOLIS:
        return base_solver.Algorithm.METROPOLIS
    elif algorithm == Algorithm.HEAT_BATH:
        return base_solver.Algorithm.HEAT_BATH
    elif algorithm == Algorithm.IRKMR:
        return base_solver.Algorithm.IRKMR
    elif algorithm == Algorithm.RKMR:
        return base_solver.Algorithm.RKMR
    elif algorithm == Algorithm.SWENDSEN_WANG:
        return base_solver.Algorithm.SWENDSEN_WANG
    elif algorithm == Algorithm.WOLFF:
        return base_solver.Algorithm.WOLFF
    else:
        raise RuntimeError("Unknown algorithm")
