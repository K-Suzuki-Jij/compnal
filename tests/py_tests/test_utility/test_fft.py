import numpy as np
import pytest

from compnal.utility.fft import calculate_fft_magnitude_list, calculate_fft2_magnitude_list

def test_fft_magnitude():
    array = np.random.rand(3, 4)
    numpy_fft = np.array([np.abs(np.fft.fft(array[i], norm="ortho")) for i in range(3)])
    compnal_fft = calculate_fft_magnitude_list(array, norm="ortho", num_threads=2)
    assert np.allclose(numpy_fft, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft(array[i], norm="forward")) for i in range(3)])
    compnal_fft = calculate_fft_magnitude_list(array, norm="forward", num_threads=2)
    assert np.allclose(numpy_fft, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft(array[i], norm="backward")) for i in range(3)])
    compnal_fft = calculate_fft_magnitude_list(array, norm="backward", num_threads=2)
    assert np.allclose(numpy_fft, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft(array[i], norm="ortho")) for i in range(3)])
    compnal_fft = calculate_fft_magnitude_list(array, norm="ortho", power=2, num_threads=2)
    assert np.allclose(numpy_fft**2, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft(array[i], norm="forward")) for i in range(3)])
    compnal_fft = calculate_fft_magnitude_list(array, norm="forward", power=2, num_threads=2)
    assert np.allclose(numpy_fft**2, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft(array[i], norm="backward")) for i in range(3)])
    compnal_fft = calculate_fft_magnitude_list(array, norm="backward", power=2, num_threads=2)
    assert np.allclose(numpy_fft**2, compnal_fft)

    with pytest.raises(ValueError):
        calculate_fft_magnitude_list(np.random.rand(3, 4, 2), num_threads=2)

def test_fft2_magnitude():
    array = np.random.rand(3, 4, 7)
    numpy_fft = np.array([np.abs(np.fft.fft2(array[i], norm="ortho")) for i in range(3)])
    compnal_fft = calculate_fft2_magnitude_list(array, norm="ortho", num_threads=2)
    assert np.allclose(numpy_fft, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft2(array[i], norm="forward")) for i in range(3)])
    compnal_fft = calculate_fft2_magnitude_list(array, norm="forward", num_threads=2)
    assert np.allclose(numpy_fft, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft2(array[i], norm="backward")) for i in range(3)])
    compnal_fft = calculate_fft2_magnitude_list(array, norm="backward", num_threads=2)
    assert np.allclose(numpy_fft, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft2(array[i], norm="ortho")) for i in range(3)])
    compnal_fft = calculate_fft2_magnitude_list(array, norm="ortho", power=2, num_threads=2)
    assert np.allclose(numpy_fft**2, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft2(array[i], norm="forward")) for i in range(3)])
    compnal_fft = calculate_fft2_magnitude_list(array, norm="forward", power=2, num_threads=2)
    assert np.allclose(numpy_fft**2, compnal_fft)

    numpy_fft = np.array([np.abs(np.fft.fft2(array[i], norm="backward")) for i in range(3)])
    compnal_fft = calculate_fft2_magnitude_list(array, norm="backward", power=2, num_threads=2)
    assert np.allclose(numpy_fft**2, compnal_fft)

    with pytest.raises(ValueError):
        calculate_fft2_magnitude_list(np.random.rand(3, 4), num_threads=2)

