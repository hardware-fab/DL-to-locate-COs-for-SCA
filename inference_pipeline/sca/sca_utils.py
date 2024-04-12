"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
    
Other contributor(s):  
    Giuseppe Diceglie
"""

import numpy as np
import pickle
from typing import Union

__hw = [bin(x).count("1") for x in range(256)]

hw = np.array(__hw)

_aes_sbox = (
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16)

aes_sbox = np.array(_aes_sbox, dtype=np.uint8)

def rankKey(data: np.ndarray,
            key: int) -> int:
    """
    Computes the rank of `key` in `data`, i.e.,
    the position of `key` in the sorted array `data`.

    Parameters
    ----------
    `data` :  array_like
        The data to rank.
    `key`  : int
        The key value to rank.

    Returns
    ----------
    The rank of `key` in `data`.
    """

    sort = np.argsort(data)
    rank = np.argwhere(sort == key)
    if rank.shape[0] > 1:
        rank = rank[0]

    return 256 - int(rank[0])


def kahanSum(sum_: np.ndarray,
             c: np.ndarray,
             element: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the sum of `element` and `sum_`, using the Kahan summation algorithm.

    Parameters
    ----------
    `sum_`    : array_like
        The current running sum.
    `c`       : array_like
        The previous correction term.
    `element` : array_like
        The next element to add.

    Returns
    ----------
    The new running sum and the new correction term.

    Raises
    ------
    `AssertionError`
        If parameters shapes are not equal.
    """

    assert sum_.shape == c.shape, \
        f"sum_ and c shape must be equal: sum_ {sum_.shape}, c {c.shape}"
    assert element.shape == c.shape, \
        f"element and c shape must be equal: element {element.shape}, c {c.shape}"

    y = element - c
    t = sum_ + y
    c = (t - sum_) - y

    sum_ = t

    return sum_, c


def intToBytes(integer: int,
               num_bytes: int = None) -> np.ndarray:
    """
    Convert an integer into a numpy array of bytes.

    Parameters
    ----------
    `integer` : int
        The integer to be converted.
    `num_bytes` : int, optional
        The number of bytes to use for the conversion (default is the minimum number of bytes).

    Returns
    ----------
    A numpy array of unsigned 8-bit integers (dtype=np.uint8) that represents
    the byte arrays of the input integer.
    """

    if num_bytes is None:
        # Compute the number of bytes required
        num_bytes = (integer.bit_length() + 7) // 8
    bytes = list(integer.to_bytes(num_bytes, 'big'))
    return np.array(bytes, dtype=np.uint8)


def bytesToInt(bytes_: Union[bytes, list[int]]) -> int:
    """
    Convert a list of bytes into an integer.

    Parameters
    ----------
    `bytes_` : bytes | list[int8]
        A list of bytes to be converted.

    Returns
    ----------
    An integer that represents the concatenation of the input bytes.
    """

    integer = int.from_bytes(bytes_, byteorder='big')
    return integer


def saveObject(obj: object,
               filename: str):
    """
    Save a Python objet into a pickle file.
    Overwrites any existing file.

    Parameters
    ----------
    `obj` : object
        Object to save.
    `filename` : str
        File path where to save `obj`.
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def loadObject(filename: str) -> object:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    `filename` : str
        File path where to load the object from.

    Returns
    -------
    The loaded object.
    """
    with open(filename, 'rb') as obj:
        ret_obj = pickle.load(obj)
    return ret_obj
