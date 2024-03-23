# A Deep-Learning Technique to Locate Cryptographic Operation in Side-Channel Traces

To run an experiment, follow the Jupyter Notebook `DL_to_locate_COs.ipynb`.

## Organization

The repository is organized as follows:

- `/CNN`: Contains modules and configuration files for the Convolutional Neural Network use for classifying the start of cryptographic operations.
- `/inference_pipeline`: Contains functions for classifying, segmenting, and aligning a side-channel trace.
- `DL_to_locate_COs.ipynb`: It is a Jupyter Notebook for running a demo.

```txt
.
├── CNN
│   ├── configs/
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
│   ├── sca
│   │   └── preprocess.py
│   ├── segmentation.py
│   └── sliding_window_classification.py
└── DL_to_locate_COs.ipynb
```

## Dataset

The dataset for the AES cryptosystem is avaible [here]().

It is organized as follows:

- `\training`: contains three subsets, i.e., __train__, __valid__, and __test__.  
    Each subset consist into two .npy files:
  - _set_: it contains the side-channel traces accordingly preprocessed.
  - _labels_: it contains the target labels for trainig the CNN, labelling each data as _cipher start_, _cipher rest_, or _noise_.
- `\inference`: contains four file as demo of the inference pipeline.  
    Two files are the side-channels traces containg an undeifne number of AES encryptions.
    The last two files are a list of plaintexts matching the AES encryptions to test a CPA attack.

## Note

This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/DL-to-locate-COs-for-SCA/blob/main/LICENSE) file.

© 2024 hardware-fab
