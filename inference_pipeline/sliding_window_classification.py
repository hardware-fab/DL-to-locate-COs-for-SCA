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
from CNN.utils import (parse_arguments, get_neptune_run,
                       get_experiment_config_dir)
import CNN.modules as modules
from .sca.preprocess import highpass


def _cutSubWindows(traces, window_size, stride):
    windows = []
    n_iters = traces.shape[0] // stride
    for i in range(n_iters):
      start = i * stride
      end = min(window_size + start, traces.shape[0])
      windows.append(traces[start:end])
      if end==traces.shape[0]:
        break
    return windows
    
def _predict(traces, module, window_size,
             device, stride, batch_size=1024):

    # Split traces into windows with stride
    windows = _cutSubWindows(traces, window_size, stride)
    # Last window can have different size, so we need to predict it separately
    windows_np = np.array(windows[0:-1])
    window_np = np.array(windows[-1])
    del windows
    
    classified_points = []
    n_iters = ceil(len(windows_np) / batch_size)
    for i in tqdm(range(n_iters), leave=False):
        start = i*batch_size
        end = min((i+1)*batch_size, windows_np.shape[0])
        real_batch_size = end - start

        curr_windows = windows_np[start : end]
        curr_windows = curr_windows.reshape(real_batch_size, windows_np.shape[-1])

        curr_windows = torch.from_numpy(curr_windows)
        curr_windows = curr_windows.to(device)
        
        with torch.no_grad():
            y_hat = module(curr_windows)
            classified_points.append(y_hat.detach())
    
    curr_window = window_np.reshape(1, window_np.shape[-1])
    curr_window = torch.from_numpy(curr_window)
    curr_window = curr_window.to(device)
    with torch.no_grad():
        y_hat = module(curr_window)
        classified_points.append(y_hat.detach())
    

    return torch.cat(classified_points, dim=0).cpu().data.numpy()


def _slidingWindowClassify(module, traces, device, window_size, stride):

    classified_points = []
    for batch in tqdm(traces, colour='green'):
        batch = highpass(batch, 0.001)
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

    traces = np.load(trace_file, mmap_mode='r')

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
