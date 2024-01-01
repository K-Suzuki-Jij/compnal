from compnal import base_compnal
import numpy as np

def calculate_fft_magnitude_list(
    array_list: np.array,
    n: int,
    norm: str = "backward",
    num_threads: int = 1,
) -> np.array:
    """Calculate FFT magnitude of a list of arrays, which is the absolute value of the Fourier component.

    Args:
        array_list (np.array): List of arrays.
        n (int): Number of points in each array in array_list.
        norm (str, optional): Normalization of the FFT. Defaults to "backward".
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Returns:
        np.array: FFT of the array list.
    """
    if len(array_list.shape) != 2:
        raise ValueError("array_list must be a 2D array.")
    if array_list.shape[1] != n:
        raise ValueError("The second dimension of array_list must be n.")
    
    return base_compnal.base_utility.calculate_fft_magnitude_list(
        array_list=array_list,
        n=n,
        norm=norm,
        num_threads=num_threads,
    )

def calculate_fft2_magnitude_list(
    array_list: np.array,
    n_x: int,
    n_y: int,
    norm: str = "backward",
    num_threads: int = 1,
) -> np.array:
    """Calculate 2D FFT magnitude of a list of arrays, which is the absolute value of the Fourier component.

    Args:
        array_list (np.array): List of arrays.
        n_x (int): Number of points in the x direction of each array in array_list.
        n_y (int): Number of points in the y direction of each array in array_list.
        norm (str, optional): Normalization of the FFT. Defaults to "backward".
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Returns:
        np.array: 2D FFT of the array list.
    """
    if len(array_list.shape) != 3:
        raise ValueError("array_list must be a 3D array.")
    if array_list.shape[1] != n_x:
        raise ValueError("The second dimension of array_list must be n_x.")
    if array_list.shape[2] != n_y:
        raise ValueError("The third dimension of array_list must be n_y.")
    
    num_arrays = array_list.shape[0]
    
    magnitude = base_compnal.base_utility.calculate_fft2_magnitude_list(
        array_list=array_list.reshape(num_arrays, -1),
        n_x=n_x,
        n_y=n_y,
        norm=norm,
        num_threads=num_threads,
    )
    
    return magnitude.reshape(num_arrays, n_x, n_y)
