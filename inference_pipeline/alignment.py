"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np

def alignCos(trace: np.ndarray,
             startCOs: list[np.ndarray],
             stride: int):
    """
    Locate the COs in the trace and align them in time.
    Trace and stride are the same used for the sliding window classification.
    
    Parameters
    ----------
    `trace` : np.ndarray
        The trace to align. To be set as the trace used for the sliding window classification.
    `startCOs` : list[np.ndarray]
        The starting sample for each COs present in the trace.
    `stride` : int
        The stride used to align the COs. To be set as the stride used for the sliding window.
    
    Returns
    -------
    The aligned COs, i.e., a matrix with a row for each CO.
    """
    algned_cos = []

    for n_trace, peaks in enumerate(startCOs):
        cos = []
        for p_id in range(len(peaks)):
            start = peaks[p_id] * stride
            end = trace.shape[1] - 1 if p_id == len(peaks)-1 else peaks[p_id + 1] * stride
            cos.append(trace[n_trace, start:end])
        algned_cos.extend(np.asarray(cos, dtype=object))
        
    return algned_cos

def saveCos(cos: list,
            output_file: str):
    """
    Save the aligned COs to a file.
    
    Parameters
    ----------
    `cos` : list
        The aligned COs.
    `output_file` : str
        The file to save the aligned COs.
    """
    
    # COs have diffent length, so we need to save them as object
    np.save(output_file, np.array(cos, dtype=object))
    
def padCos(cos_to_pad):
    """
    Pad the COs to the same length.
    Since the COs have different length, we need to padd them to the same length.
    We choose the minimum length among all the COs.
    
    Parameters
    ----------
    `cos_to_pad` : np.ndarray
        The aligned COs.
    
    Returns
    -------
    The aligned COs with the same length.
    """
    
    min_len = len(min(cos_to_pad, key = lambda x: len(x)))
    new_seg = [co[:min_len] for co in cos_to_pad]
    return np.array(new_seg, dtype=np.float32)