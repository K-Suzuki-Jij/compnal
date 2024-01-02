from compnal import base_compnal
import numpy as np

def calculate_fft_magnitude_list(
    array_list: np.array,
    norm: str = "backward",
    power: int = 1,
    num_threads: int = 1,
) -> np.array:
    """Calculate FFT magnitude of a list of arrays, which is the absolute value of the Fourier component.

    Args:
        array_list (np.array): List of arrays.
        n (int): Number of points in each array in array_list.
        norm (str, optional): Normalization of the FFT. Defaults to "backward".
        power (int, optional): The exponent to be applied to the calculated Fourier intensities. Defaults to 1.
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Returns:
        np.array: FFT magnitude of the array list.
    """
    if len(array_list.shape) != 2:
        raise ValueError("array_list must be a 2D array.")
    
    return base_compnal.base_utility.calculate_fft_magnitude_list(
        array_list=array_list,
        n=array_list.shape[1],
        norm=norm,
        power=power,
        num_threads=num_threads,
    )

def calculate_fft2_magnitude_list(
    array_list: np.array,
    norm: str = "backward",
    power: int = 1,
    num_threads: int = 1,
) -> np.array:
    """Calculate 2D FFT magnitude of a list of arrays, which is the absolute value of the Fourier component.

    Args:
        array_list (np.array): List of arrays.
        n_x (int): Number of points in the x direction of each array in array_list.
        n_y (int): Number of points in the y direction of each array in array_list.
        norm (str, optional): Normalization of the FFT. Defaults to "backward".
        power (int, optional): The exponent to be applied to the calculated Fourier intensities. Defaults to 1.
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Returns:
        np.array: 2D FFT magnitude of the array list.
    """
    if len(array_list.shape) != 3:
        raise ValueError("array_list must be a 3D array.")
    
    num_arrays = array_list.shape[0]
    n_x = array_list.shape[1]
    n_y = array_list.shape[2]
    magnitude = base_compnal.base_utility.calculate_fft2_magnitude_list(
        array_list=array_list.reshape(num_arrays, n_x*n_y),
        n_x=n_x,
        n_y=n_y,
        norm=norm,
        power=power,
        num_threads=num_threads,
    )

    return magnitude.reshape(num_arrays, n_x, n_y)
