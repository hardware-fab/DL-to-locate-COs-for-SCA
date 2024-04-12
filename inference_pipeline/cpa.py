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

import os
import numpy as np
from matplotlib import pyplot as plt

from inference_pipeline.sca.sca_utils import kahanSum, saveObject, intToBytes, rankKey, hw, aes_sbox
from inference_pipeline.sca.preprocess import aggregate, highpass

class Cpa():
    """
    Correaltion Power Analysis class for AES cipher.
       
    Methods
    ----------
    `rankKey(key)`:
        Computes the rank of `key` for each byte.
    `update(traces, plains)`:
        Updates CPA's intermediate values with new `traces` and corresponding `plains`.
    `computeCoefficient()`:
        Computes Pearson's correlation cofficient.
    `dump(cpa_folder, key, ovveride)`:
        Save in `cpa_folder` a .txt with the principal results and dump the cpa object.
    `plot(key, byte, *kargs)`:
        Plot Pearson's correlation cofficient for the indicated byte.
    `plotPearsonOverTraces(key, byte, *kargs)`:
        Plot Pearson's correlation coefficient peaks for the indicated byte at inreasing number of traces.
    """

    def __init__(self,
                 filter: bool = False,
                 aggregate_n_samples: int = 1
                 ):
        """
        Implementation of an online CPA attack for all byte.
        Attack intermediate is SBOX(plain ^ key).
        All 16 bytes are attacked.

        Parameters
        ----------
        `filter`  : bool, optional
            Apply a highpass filter to traces as preprocessing (default is False).
        `aggregate_n_samples` : int, optional
            How many consecutive samples avarage together as preprocessing (default is 1).
        """
        
        self._nBytes = 16
        
        self._filter = filter
        self._aggregate_n_samples = aggregate_n_samples

        self._D = 0
        self._I = 256

    def _getPowerEstimation(self, plains, byte, keys_byte):
        plain_byte = plains[:, byte][:, None]
        return hw[aes_sbox[np.repeat(plain_byte, self._I, axis=1) ^ keys_byte]]
      
    def rankKey(self,
                key: list[int]
                ) -> list[int]:
        """
        Computes the rank of `key` for each byte.

        Parameters
        ----------
        `key`: list[int]
            The key value to rank.

        Returns
        ----------
        The rank of `key` for each byte.
        """
        r = self.computeCoefficient()
        
        k = [key[i] for i in range(self._nBytes)]
        return [rankKey(np.max(abs(cpa), axis=1), kb) for cpa, kb in zip(r, k)]

    def getPearsonsPeacks(self) -> np.ndarray:
        """
        Get Pearson's correlation cofficient.
        It is a 3D array with shape (n_steps, n_bytes, n_guesses).
        
        Returns
        ----------
        Peacks of Pearson's correlation cofficient for each byte as the number of processed traces increases.
        """
        return np.array(self._r_peack, dtype=np.float16)
    
    def update(self,
               traces: np.ndarray,
               plains: np.ndarray):
        """
        Updates CPA's intermediate values with new `traces` and corresponding `plains`.

        Parameters
        ----------
        `traces`: array-like
            Traces to attack with CPA
        `plains`: array-like
            Plaintexts corresponding to traces
        """
        # Preprocess traces
        t = self.__preprocess(traces)

        keys_byte = np.arange(0, self._I)[None, :]
        if self._D == 0:
            self.__reset(t.shape[1])
        # Compute intermediate values
        for byte in range(self._nBytes):
            # Estimate power consumption of SBOX(plain ^ key)
            h = self._getPowerEstimation(plains, byte, keys_byte)

            self._sum_h[byte], self._c_h[byte] = kahanSum(
                self._sum_h[byte], self._c_h[byte], np.sum(h, 0, keepdims=True).T)
            self._sum_t[byte], self._c_t[byte] = kahanSum(
                self._sum_t[byte], self._c_t[byte], np.sum(t, 0, keepdims=True))
            self._sum_ht[byte], self._c_ht[byte] = kahanSum(
                self._sum_ht[byte], self._c_ht[byte], h.T @ t)
            self._sum_h2[byte], self._c_h2[byte] = kahanSum(
                self._sum_h2[byte], self._c_h2[byte], np.sum(h**2, 0, keepdims=True).T)
            self._sum_t2[byte], self._c_t2[byte] = kahanSum(
                self._sum_t2[byte], self._c_t2[byte], np.sum(t**2, 0, keepdims=True))

        self._D += traces.shape[0]
        r = self.computeCoefficient()
        self._r_peack.append(np.max(np.abs(r, dtype=np.float16), axis=2))

    def computeCoefficient(self) -> list[float]:
        """
        Computes Pearson's correlation cofficient.

        .. math::
            r = \\frac{D \\sum_{i=1}^{D} h_i t_i - \\sum_{i=1}^{D} h_i \\sum_{i=1}^{D} t_i}\
            {\\sqrt{(D \\sum_{i=1}^{D} h_i^2 - (\\sum_{i=1}^{D} h_i)^2)(D \\sum_{i=1}^{D} t_i^2 - (\\sum_{i=1}^{D} t_i)^2)}}

        Returns
        ----------
        Pearson's correlation cofficient `r`.
        """
        r = [(self._D*self._sum_ht[byte] - (self._sum_h[byte] @ self._sum_t[byte]))
             / np.sqrt((self._sum_h[byte]**2 - self._D*self._sum_h2[byte]) @ (self._sum_t[byte]**2 - self._D*self._sum_t2[byte]))
             for byte in range(0, self._nBytes)]

        return r

    def dump(self,
             cpa_folder: str,
             key: int,
             *,
             ovveride: bool = False):
        """
        Save in `cpa_folder` a .txt with the principal results and dump the cpa object.
        Reults are: N. traces and rank for each byte.

        Parameters
        ----------
        `cpa_folder`: str
            Folder path where to dump the cpa object and results.
        `key`: int
            Real key of the processed traces.
        `ovveride`: bool, optional
            Override existing files (default is False).
        """

        if not os.path.exists(cpa_folder):
            os.mkdir(cpa_folder)
        saveObject(self, cpa_folder+'cpa.pkl')
        rank = self.rankKey(intToBytes(key))
        
        mode = 'w' if ovveride else 'a'
        with open(cpa_folder+"results.txt", mode) as file:
            if os.stat(cpa_folder+"results.txt").st_size == 0:
                file.write(f'Aggregation: {self._aggregate_n_samples}\n')
                file.write(f'Filter: {self._filter}\n')
                file.write(f'Key: {hex(key)}\n')
                file.write(
                    '-----------------------------------------------------------------------\n\n')
            file.write(f'N. traces: {self._D}\n')
            file.write(f'Ranks: {rank}\n\n')

    def plot(self,
             key: list[int],
             byte: int,
             *, # Force to use keyword arguments
             figsize: tuple[int, int] = (13, 5),
             fontsize: int = 18,
             xlim: tuple[int, int] = None,
             save: bool = False,
             path: str = ...):
        """
        Plot Pearson's correlation cofficient for the indicated byte.
        The correct key byte is plotted in black.

        Parameters
        ----------
        `key`: list[int]
            Real key of the processed traces.
        `byte`: int
            Pearson's byte to plot.
        `figsize`: tuple[int, int], optional
            Figure size (default is (13, 5)).
        `fontsize`: int, optional
            Font size (default is 18).
        `xlim`: tuple[int, int], optional
            X axis limits (default plots all samples).
        `save`: bool, optional
            Save the plot (default is False).
        `path`: str, optional
            Path where to save the plot.
        """

        r = self.computeCoefficient()
        keyByte = key[byte]
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': fontsize*0.8})
        for i in range(len(r[byte])):
            if save:
                plt.plot(r[byte][i], alpha=0.6, color='grey')
            else:
                plt.plot(r[byte][i], alpha=0.6)
        plt.plot(r[byte][keyByte], color='black')
        plt.xlabel('Sample', fontsize=fontsize)
        plt.ylabel('Pearson\'s coeff.', fontsize=fontsize)
        plt.xlim(xlim)
        plt.grid(True)
        plt.show()
        if save:
            plt.savefig(path, dpi=1200, bbox_inches='tight')
        plt.close()

    def plotPearsonOverTraces(self,
                key: list[int],
                byte: int,
                *, # Force to use keyword arguments
                figsize: tuple[int, int] = (13, 5),
                fontsize: int = 18,
                xlim: tuple[int, int] = None,
                save: bool = False,
                path: str = ...):
        """
        Plot Pearson's correlation coefficient peaks for the indicated byte at increasing number of traces.
        The peak values for correct key is plotted in black.

        Parameters
        ----------
        `key`: list[int]
            Real key of the processed traces.
        `byte`: int
            Pearson's byte to plot.
        `figsize`: tuple[int, int], optional
            Figure size (default is (13, 5)).
        `fontsize`: int, optional
            Font size (default is 12).
        `xlim`: tuple[int, int], optional
            X axis limits (default plots all samples).
        `save`: bool, optional
            Save the plot (default is False).
        `path`: str, optional
            Path where to save the plot.
        """

        if len(self._r_peack) < 2:
            print('Not enough Pearson\'s coefficient over traces to plot.')
            return
        
        keyByte = key[byte]
        _r_peack = np.array(self._r_peack)
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': fontsize*0.8})
        for i in range(_r_peack.shape[2]):
            plt.plot(_r_peack[:, byte, i], alpha=0.6, color='grey')
        plt.plot(_r_peack[:, byte, keyByte], color='black')
        plt.xlim(xlim)
        plt.xlabel('N. Traces', fontsize=fontsize)
        plt.ylabel('Peak Pearson\'s Coeff.', fontsize=fontsize)
        plt.grid(True)
        plt.show()
        
        if save:
            plt.savefig(path, dpi=1200, bbox_inches='tight')
        plt.close()

    def __preprocess(self, traces):
        if self._aggregate_n_samples > 1:
            traces = aggregate(traces, self._aggregate_n_samples)
        if self._filter:
            traces = highpass(traces)
        return traces

    def __reset(self, shape):
        self._T = shape

        self._sum_h = [np.zeros((self._I, 1))]*self._nBytes
        self._sum_t = [np.zeros((1, self._T))]*self._nBytes
        self._sum_ht = [np.zeros((self._I, self._T))]*self._nBytes
        self._sum_h2 = [np.zeros((self._I, 1))]*self._nBytes
        self._sum_t2 = [np.zeros((1, self._T))]*self._nBytes
        self._c_h = [np.zeros(self._sum_h[0].shape)]*self._nBytes
        self._c_t = [np.zeros(self._sum_t[0].shape)]*self._nBytes
        self._c_ht = [np.zeros(self._sum_ht[0].shape)]*self._nBytes
        self._c_h2 = [np.zeros(self._sum_h2[0].shape)]*self._nBytes
        self._c_t2 = [np.zeros(self._sum_t2[0].shape)]*self._nBytes
        
        self._r_peack = []
