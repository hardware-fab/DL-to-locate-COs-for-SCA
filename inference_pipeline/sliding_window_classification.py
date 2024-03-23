"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

from tqdm.auto import tqdm
import torch
import numpy as np
from math import ceil
from multiprocessing import Pool
from CNN.utils import (parse_arguments, get_neptune_run,
                       get_experiment_config_dir)
import CNN.modules as modules
from .sca.preprocess import highpass


def _predict(traces, module, window_size,
             device, stride):

    classified_points = []
    n_iters = ceil(traces.shape[0] / stride)

    for i in tqdm(range(n_iters), leave=False):

        start = i * stride
        end = min(traces.shape[0], window_size + i*stride)
        curr_traces = traces[start: end]

        curr_traces = curr_traces.reshape(
            1, min(window_size, len(curr_traces)))
        curr_traces = torch.from_numpy(curr_traces)
        curr_traces = curr_traces.to(device)

        y_hat = module(curr_traces).cpu().detach().numpy()

        classified_points.append(y_hat[0])

    return classified_points


def _slidingWindowClassify(module, traces, device, window_size, stride):

    classified_points = []
    for batch in tqdm(traces, colour='green'):
        batch = highpass(batch, Wn=0.001)
        ret = _predict(batch, module, window_size, device, stride)
        classified_points.append(ret)

    classified_points = np.asarray(classified_points)

    return classified_points


def getModule(SID: str,
              neptune_config: str = 'CNN/configs/common/neptune_configs.yaml') -> torch.nn.Module:
    """
    Get the best model from a Neptune Run and return the corresponding module.

    Parameters
    ----------
    `SID` : str
        The Neptune Run ID.
    `neptune_config` : str, optional
        The path to the Neptune configuration file (default is 'CNN/configs/common/neptune_configs.yaml').

    Returns
    -------
    The best model from the Neptune Run.
    """

    # Get Neptune Run (by SID)
    df = get_neptune_run(neptune_config, SID)

    # Get experiment name
    exp_name = df['sys/name'].iloc[0]

    # Get best model path
    best_model_ckpt = df['experiment/model/best_model_path'].iloc[0]

    # Get config dir
    config_dir = get_experiment_config_dir(best_model_ckpt, exp_name)

    _, module_config, __ = parse_arguments(config_dir)

    # Build Model
    # -----------
    module_name = module_config['module']['name']
    module_config = module_config['module']['config']
    module_class = getattr(modules, module_name)
    module = module_class.load_from_checkpoint(best_model_ckpt,
                                               module_config=module_config)

    return module


def classifyTrace(trace_file: str,
                  module: torch.nn.Module,
                  stride: int,
                  window_size: int,
                  gpu: int = 0) -> np.ndarray:
    """
    Classify a side-channel trace using a sliding-windows approach.
    Stride and window size are configurable.

    Parameters
    ----------
    `trace_file` : str
        The path to the trace file to classify.
    `module` : torch.nn.Module
        The CNN module to use for classification.
    `stride` : int
        The stride to use for the sliding window.
    `window_size` : int
        The size of the sliding window.
    `gpu` : int, optional
        The GPU to use for classification (default is 0).
        0 means the first GPU, 1 means the second GPU, and so on.

    Returns
    -------
    A classification score for each winodw in the trace.
    """
    # Get Device
    device = torch.device(
        f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # Set Module
    module.to(device)
    module.eval()

    traces = np.load(trace_file, mmap_mode='r')[:3]

    segmentation = _slidingWindowClassify(module, traces, device,
                                          window_size, stride)
    return segmentation


def saveClassification(segmentation: np.ndarray,
                       output_file: str) -> None:
    """
    Save the segmentation to a file.

    Parameters
    ----------
    `segmentation` : np.ndarray
        The segmentation to save.
    `output_file` : str
        The file where the segmentation will be saved.
    """

    np.save(output_file, segmentation)
