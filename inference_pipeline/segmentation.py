"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np
import scipy
from scipy.signal import medfilt
from matplotlib import pyplot as plt


def squareWave(signal: np.ndarray) -> np.ndarray:
    """
    Convert a signal to a square wave signal.

    Parameters
    ----------
    `signal` : np.ndarray
        The signal to convert.

    Returns
    -------
    The square wave signal.
    """

    # Last 100 samples are removed to avoid noise
    signal = signal[:-100]
    threshold = np.mean(signal)

    square_signal = []
    for value in signal:
        if value >= threshold:
            square_signal.append(1)
        else:
            square_signal.append(-1)
    return square_signal


def medianFilter(square_signal: np.ndarray,
                 kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to a square wave signal.
    The precess is needed to remove noise from the signal.

    Parameters
    ----------
    `square_signal` : np.ndarray
        The square wave signal.
    `kernel_size` : int
        The kernel size of the median filter.

    Returns
    -------
    The square wave signal after applying the median filter.
    """
    signal = medfilt(square_signal, kernel_size=kernel_size)

    for i in range(0, len(signal)-1):
        if signal[i] == 0 and signal[i+1] == 1:
            signal[i] = -1
        elif signal[i] == 0 and signal[i+1] == -1:
            signal[i] = 1

    return signal


def getStartCOs(square_signal: np.ndarray,
                min_distance: int,
                noise_applications: bool) -> np.ndarray:
    """
    Get the starting sample for each COs present in the square wave signal.

    Parameters
    ----------
    `square_signal` : np.ndarray
        The square wave signal.
    `min_distance` : int
        The minimum distance between two COs.
    `noise_applications` : bool
        True if the COs has noisy application between them, False if the COs are consecutive.

    Returns
    -------
    The starting sample for each COs present in the square wave signal.
    """

    edges = _extractEdges(square_signal, min_distance, noise_applications)
    return np.asarray(edges)[:-1]


def plotStartCOs(startCOs: np.ndarray,
                 classifications: np.ndarray,
                 n_trace: int = 0):
    """
    Plot the start COs samples over the sliding window classification output.
    Useful to check if the COs are correctly located.

    Parameters
    ----------
    `startCOs` : np.ndarray
        The start COs samples.
    `classifications` : np.ndarray
        The sliding window classification output.
    `n_trace` : int, optional
        The trace number to plot (default is 0).
    """

    fig, ax = plt.subplots(1, figsize=(13, 5))
    plt.rcParams.update({'font.size': 18})
    fig.tight_layout(pad=2.0)

    ax.set_xlim((-10, 10_000))
    ax.set_title("Start COs samples")
    for sample in startCOs[n_trace]:
        ax.axvline(x=sample, color='r', linestyle='--')
    ax.plot(classifications[n_trace, 0:-100])

    del fig
    del ax


def _extractEdges(square_signal, min_distance, noise):
    falling_edge_indices = []

    if not noise:
        prev_falling_edge_index = - min_distance
        threshold = -1
    else:
        threshold = 1
        prev_falling_edge_index = - min_distance - 1
        if square_signal[0] == 1:
            prev_falling_edge_index = 0
            falling_edge_indices.append(0)

    for i in range(1, len(square_signal)):
        if square_signal[i] == threshold and square_signal[i - 1] == -threshold and i - prev_falling_edge_index > min_distance:
            falling_edge_indices.append(i)
            prev_falling_edge_index = i

    return falling_edge_indices
