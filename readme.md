# A Deep-Learning Technique to Locate Cryptographic Operations in Side-Channel Traces

To run an experiment, follow the Jupyter Notebook `DL_to_locate_COs.ipynb`.

## Organization

The repository is organized as follows:

- `/CNN`: Contains modules and configuration files for the Convolutional Neural Network use for classify the start of cryptographic operations.
- `/inference_pipeline`: Contains functions for classify, segment, align, and attack a side-channel trace.
- `DL_to_locate_COs`: It is a Jupyter Notebook for run a demo.

```txt
.
├── CNN
│   ├── configs
│   │   ├── common
│   │   │   └── neptune_configs.yaml
│   │   └── exp
│   │       ├── data.yaml
│   │       ├── experiment.yaml
│   │       └── module.yaml
│   ├── datasets
│   │   └── co_class_dataset.py
│   ├── models
│   │   ├── custom_layers.py
│   │   ├── resnet.py
│   │   └── resnet_time_series_classifier.py
│   ├── modules
│   │   ├── co_class_datamodule.py
│   │   └── co_class_module.py
│   ├── train.py
│   └── utils
│       ├── data.py
│       ├── logging.py
│       ├── module.py
│       ├── trainer.py
│       └── utils.py
├── inference_pipeline
│   ├── alignment.py
│   ├── cpa.py
│   ├── sca
│   │   ├── preprocess.py
│   │   └── sca_utils.py
│   ├── segmentation.py
│   └── sliding_window_classification.py
└── DL_to_locate_COs.ipynb
```

## Dataset

The dataset for the AES cryptosystem is avaible [here](https://doi.org/10.5281/zenodo.10955733).

The dataset is organized as follows:

- `\training`: contains three subsets, i.e., __train__, __valid__, and __test__.  
    Each subset consist into two .npy files:
  - _set_: it contains the side-channel traces accordingly preprocessed.
  - _labels_: it contains the target labels for training the CNN, labelling each data as _cipher start_, _cipher rest_, or _noise_.
- `\inference`: contains two file as demo of the inference pipeline.
    One file is the side-channel trace containing an undefine number of AES encryptions. The other file is a list of plaintexts matching the AES encryptions to test a CPA attack.

## Note

This work is part of [1] available [online](https://arxiv.org/abs/2402.19037).

This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/DL-to-locate-COs-for-SCA/blob/main/LICENSE) file.

© 2024 hardware-fab

> [1] Chiari, G., Galli, D., Lattari, F., Matteucci, M., & Zoni, D. (2024). A Deep-Learning Technique to Locate Cryptographic Operations in Side-Channel Traces. arXiv preprint arXiv:2402.19037.
