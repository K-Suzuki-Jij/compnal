import numpy as np


def determine_value_type(v: int) -> type:
    """Determine the type of a value.

    Args:
        v (int): Value.

    Raises:
        ValueError: When the value is too large.

    Returns:
        type: Type of the value.
    """
    if v <= np.iinfo(np.int8).max:
        return np.int8
    elif v <= np.iinfo(np.int16).max:
        return np.int16
    elif v <= np.iinfo(np.int32).max:
        return np.int32
    elif v <= np.iinfo(np.int64).max:
        return np.int64
    else:
        raise ValueError("Value is too large.")
