# RFCIL-PTC

Class-incremental learning on pre-trained encrypted traffic classifiers. RFCIL-PTC builds on a BERT-style pre-trained traffic model (ET-Bert) and updates it continually with lightweight adapter modules, so that the classifier can learn new traffic classes over time without catastrophic forgetting and without retraining the full backbone.

## Overview

Traffic-classification models deployed in the wild must periodically absorb new application/protocol classes. Re-training the whole model from scratch is expensive and destroys previously acquired knowledge. RFCIL-PTC addresses this with a **rehearsal-based, adapter-style class-incremental learning** scheme on top of a frozen pre-trained traffic Transformer.

## Repository structure

```
├── main.py                     # entry point: parses args, loads JSON configs, launches training
├── trainer.py                  # incremental learning loop (task sequence, evaluation)
├── models/
│   ├── base.py                 # BaseLearner: shared logic for all CIL learners
│   └── adapter_one_feature.py  # adapter-based CIL learner (the RFCIL-PTC method)
├── exps/
│   └── adapter_one_feature.json# default configuration
├── convs/
│   ├── traffic_transformer.py  # ET-Bert-style traffic Transformer backbone
│   ├── linears.py              # linear / adapter layer definitions
│   ├── models/bert/            # BERT model configuration
│   ├── uer/                    # UER-py components (encoders, layers, targets, trainer)
│   └── vocab_gen/              # traffic vocabulary generation (packet bytes -> tokens)
├── utils/
│   ├── inc_net.py              # IncrementalNet: backbone + adapter groups + head management
│   ├── data_manager.py         # dataset splitting for incremental tasks
│   ├── data.py / factory.py    # dataset / model factories
│   └── toolkit.py, ops.py, autoaugment.py
└── re.txt                      # Python dependencies
```

## Requirements

Install the dependencies with:

```bash
pip install -r re.txt
```


## Data preparation

The traffic datasets follow the ET-Bert preprocessing pipeline (raw PCAP -> per-flow packet-byte token sequences). The traffic vocabulary used by the backbone is provided in `convs/vocab_gen/vocab_packet_4chars_all.txt`.

## Usage

1. **Configure the experiment.** Edit `exps/adapter_one_feature.json` (or add your own JSON file) to set the dataset, task schedule (`init_cls`, `increment`), memory size, and optimization hyper-parameters.
2. **Select configs in `main.py`.** Add the JSON file names to the `model_list` dict.
3. **Adjust training epochs.** Epochs and optimizer settings are defined at the top of `models/adapter_one_feature.py` (`epochs`, `epochs_old`, `batch_size`, `lrate`, ...).
4. **Run.**

```bash
python main.py
```

The trainer walks through the task sequence: for each task it expands the network with a new adapter group and classification head, trains the new head, builds the rehearsal memory for old classes, then briefly re-trains each previous head (`epochs_old`) before evaluating on all classes seen so far.

## Key hyper-parameters (`exps/adapter_one_feature.json`)

| Parameter      | Default | Description                                        |
| -------------- | ------- | -------------------------------------------------- |
| `dataset`      | —       | dataset to run                                     |
| `init_cls`     | 50      | number of classes in the first task                |
| `increment`    | 10      | number of new classes per incremental task         |
| `memory_size`  | 0       | rehearsal memory budget (samples kept from old classes) |
| `seed`         | 1993    | random seed(s)                                     |
| `device`       | ["0"]   | GPU device id(s)                                   |

Method-specific settings (adapter count, loss weights, learning rates) are defined in `models/adapter_one_feature.py` and `utils/inc_net.py`.

## Acknowledgements

This code is based on [ET-Bert](https://github.com/linwhitehat/ET-BERT), [UER-py](https://github.com/dbiir/UER-py) and [PyCIL](https://github.com/LAMDA-CL/PyCIL). Many thanks to the authors.