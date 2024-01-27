#  Copyright 2024 Kohei Suzuki
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from enum import Enum
from typing import Union

from compnal.base_compnal import base_solver


class StateUpdateMethod(str, Enum):
    """State update method.

    Args:
        METROPOLIS: Metropolis algorithm.
        HEAT_BATH: Heat bath algorithm.
    """

    METROPOLIS = "METROPOLIS"
    HEAT_BATH = "HEAT_BATH"


class RandomNumberEngine(str, Enum):
    """Random number engine.

    Args:
        MT: Mersenne Twister.
        MT_64: Mersenne Twister 64-bit.
        XORSHIFT: XORSHIFT.
    """

    MT = "MT"
    MT_64 = "MT_64"
    XORSHIFT = "XORSHIFT"


class SpinSelectionMethod(str, Enum):
    """Spin selection method.

    Args:
        RANDOM: Random selection.
        SEQUENTIAL: Sequential selection.
    """

    RANDOM = "RANDOM"
    SEQUENTIAL = "SEQUENTIAL"


class TemperatureDistribution(str, Enum):
    """Temperature distribution.

    Args:
        ARITHMETIC: Arithmetic (linear) distribution.
        GEOMETRIC: Geometric (exponential) distribution.
    """

    ARITHMETIC = "ARITHMETIC"
    GEOMETRIC = "GEOMETRIC"


class CMCAlgorithm(str, Enum):
    """CMC algorithm.

    Args:
        SINGLE_FLIP: Single-flip algorithm.
        PARALLEL_TEMPERING: Parallel tempering algorithm, also known as replica exchange method.
    """

    SINGLE_FLIP = "SINGLE_FLIP"
    PARALLEL_TEMPERING = "PARALLEL_TEMPERING"


def _to_state_update_method(
    base_state_update_method: Union[
        base_solver.StateUpdateMethod, StateUpdateMethod, str
    ]
) -> StateUpdateMethod:
    """Cast base state update method to state update method.

    Args:
        base_state_update_method (Union[base_solver.StateUpdateMethod, StateUpdateMethod, str]): State update method.

    Raises:
        ValueError: When unknown state update method is input.

    Returns:
        StateUpdateMethod: State update method in compnal.
    """

    if base_state_update_method in (
        base_solver.StateUpdateMethod.METROPOLIS,
        StateUpdateMethod.METROPOLIS,
        "METROPOLIS",
    ):
        return StateUpdateMethod.METROPOLIS
    elif base_state_update_method in (
        base_solver.StateUpdateMethod.HEAT_BATH,
        StateUpdateMethod.HEAT_BATH,
        "HEAT_BATH",
    ):
        return StateUpdateMethod.HEAT_BATH
    else:
        raise ValueError("Invalid state update method.")


def _to_base_state_update_method(
    state_update_method: Union[str, StateUpdateMethod, base_solver.StateUpdateMethod]
) -> base_solver.StateUpdateMethod:
    """Cast state update method to base state update method.

    Args:
        state_update_method (Union[str, StateUpdateMethod, base_solver.StateUpdateMethod]): State update method.

    Raises:
        ValueError: When unknown state update method is input.

    Returns:
        base_solver.StateUpdateMethod: State update method in base_compnal.
    """

    if state_update_method in (
        "METROPOLIS",
        StateUpdateMethod.METROPOLIS,
        base_solver.StateUpdateMethod.METROPOLIS,
    ):
        return base_solver.StateUpdateMethod.METROPOLIS
    elif state_update_method in (
        "HEAT_BATH",
        StateUpdateMethod.HEAT_BATH,
        base_solver.StateUpdateMethod.HEAT_BATH,
    ):
        return base_solver.StateUpdateMethod.HEAT_BATH
    else:
        raise ValueError("Invalid state update method.")


def _to_random_number_engine(
    base_random_number_engine: Union[
        base_solver.RandomNumberEngine, RandomNumberEngine, str
    ]
) -> RandomNumberEngine:
    """Cast base random number engine to random number engine.

    Args:
        base_random_number_engine (Union[base_solver.RandomNumberEngine, RandomNumberEngine, str]): Random number engine.

    Raises:
        ValueError: When unknown random number engine is input.

    Returns:
        RandomNumberEngine: Random number engine in compnal.
    """

    if base_random_number_engine in (
        base_solver.RandomNumberEngine.MT,
        RandomNumberEngine.MT,
        "MT",
    ):
        return RandomNumberEngine.MT
    elif base_random_number_engine in (
        base_solver.RandomNumberEngine.MT_64,
        RandomNumberEngine.MT_64,
        "MT_64",
    ):
        return RandomNumberEngine.MT_64
    elif base_random_number_engine in (
        base_solver.RandomNumberEngine.XORSHIFT,
        RandomNumberEngine.XORSHIFT,
        "XORSHIFT",
    ):
        return RandomNumberEngine.XORSHIFT
    else:
        raise ValueError("Invalid random number engine.")


def _to_base_random_number_engine(
    random_number_engine: Union[str, RandomNumberEngine, base_solver.RandomNumberEngine]
) -> base_solver.RandomNumberEngine:
    """Cast random number engine to base random number engine.

    Args:
        random_number_engine (Union[str, RandomNumberEngine, base_solver.RandomNumberEngine]): Random number engine.

    Raises:
        ValueError: When unknown random number engine is input.

    Returns:
        base_solver.RandomNumberEngine: Random number engine in base_compnal.
    """

    if random_number_engine in (
        "MT",
        RandomNumberEngine.MT,
        base_solver.RandomNumberEngine.MT,
    ):
        return base_solver.RandomNumberEngine.MT
    elif random_number_engine in (
        "MT_64",
        RandomNumberEngine.MT_64,
        base_solver.RandomNumberEngine.MT_64,
    ):
        return base_solver.RandomNumberEngine.MT_64
    elif random_number_engine in (
        "XORSHIFT",
        RandomNumberEngine.XORSHIFT,
        base_solver.RandomNumberEngine.XORSHIFT,
    ):
        return base_solver.RandomNumberEngine.XORSHIFT
    else:
        raise ValueError("Invalid random number engine.")


def _to_spin_selection_method(
    base_spin_selection_method: Union[
        base_solver.SpinSelectionMethod, SpinSelectionMethod, str
    ]
) -> SpinSelectionMethod:
    """Cast base spin selection method to spin selection method.

    Args:
        base_spin_selection_method (Union[base_solver.SpinSelectionMethod, SpinSelectionMethod, str]): Spin selection method.

    Raises:
        ValueError: When unknown spin selection method is input.

    Returns:
        SpinSelectionMethod: Spin selection method in compnal.
    """

    if base_spin_selection_method in (
        base_solver.SpinSelectionMethod.RANDOM,
        SpinSelectionMethod.RANDOM,
        "RANDOM",
    ):
        return SpinSelectionMethod.RANDOM
    elif base_spin_selection_method in (
        base_solver.SpinSelectionMethod.SEQUENTIAL,
        SpinSelectionMethod.SEQUENTIAL,
        "SEQUENTIAL",
    ):
        return SpinSelectionMethod.SEQUENTIAL
    else:
        raise ValueError("Invalid spin selection method.")


def _to_base_spin_selection_method(
    spin_selection_method: Union[
        str, SpinSelectionMethod, base_solver.SpinSelectionMethod
    ]
) -> base_solver.SpinSelectionMethod:
    """Cast spin selection method to base spin selection method.

    Args:
        spin_selection_method (Union[str, SpinSelectionMethod, base_solver.SpinSelectionMethod]): Spin selection method.

    Raises:
        ValueError: When unknown spin selection method is input.

    Returns:
        base_solver.SpinSelectionMethod: Spin selection method in base_compnal.
    """

    if spin_selection_method in (
        "RANDOM",
        SpinSelectionMethod.RANDOM,
        base_solver.SpinSelectionMethod.RANDOM,
    ):
        return base_solver.SpinSelectionMethod.RANDOM
    elif spin_selection_method in (
        "SEQUENTIAL",
        SpinSelectionMethod.SEQUENTIAL,
        base_solver.SpinSelectionMethod.SEQUENTIAL,
    ):
        return base_solver.SpinSelectionMethod.SEQUENTIAL
    else:
        raise ValueError("Invalid spin selection method.")


def _to_temperature_distribution(
    base_temperature_distribution: Union[TemperatureDistribution, str]
) -> TemperatureDistribution:
    """Cast base temperature distribution to temperature distribution.

    Args:
        base_temperature_distribution (Union[TemperatureDistribution, str]): Temperature distribution.

    Raises:
        ValueError: When unknown temperature distribution is input.

    Returns:
        TemperatureDistribution: Temperature distribution in compnal.
    """

    if base_temperature_distribution in (
        TemperatureDistribution.ARITHMETIC,
        "ARITHMETIC",
    ):
        return TemperatureDistribution.ARITHMETIC
    elif base_temperature_distribution in (
        TemperatureDistribution.GEOMETRIC,
        "GEOMETRIC",
    ):
        return TemperatureDistribution.GEOMETRIC
    else:
        raise ValueError("Invalid temperature distribution.")
