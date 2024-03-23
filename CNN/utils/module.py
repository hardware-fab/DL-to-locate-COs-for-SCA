"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

from omegaconf import OmegaConf

import CNN.modules


def build_module(module_config, gpu):
    module_name = module_config['module']['name']
    module_config = module_config['module']['config']
    module_class = getattr(CNN.modules, module_name)
    module = module_class(module_config, gpu)
    return module
