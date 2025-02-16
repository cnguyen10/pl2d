[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Probabilistic learning to defer

This repository provides an implementation of **Probabilistic Learning to Defer: Handling Missing Expert Annotations and Controlling Workload Distribution** presented at ICLR 2025 in Singapore.

![visualisation](img/visualisation.png)
*The visualisation of the proposed Prob-L2D in which the EM algorithm is used to infer the missing annotations of each sample.*


### Contributions

 - relax the strong assumption requiring: for any sample in the training set, **every human expert must annotate** to **some do annotate, while the others do not**, and apply the EM algorithm to optimise a L2D system in that setting, and
 - integrate a workload constraint into the E-step to distribute workload across all human experts, make the system closely aligned with the one in practice.


### Requirements

 - implementation is written in Jax,
 - data loading is handled by Grain, and
 - configuration is specified in Hydra.

Please refer to `apptainer.def` for the detailed setup of the python environment in Apptainer.


### Data

The training and evaluation data is specified through json files. Each json file has a similar structure as follows:

```bash
[
    {
        "file": "train/19/bos_taurus_s_000507.png",
        "label": 19
    },
    {
        "file": "train/29/stegosaurus_s_000125.png",
        "label": 29
    },
    {
        "file": "train/0/mcintosh_s_000643.png",
        "label": 0
    },
    {
        "file": "train/11/altar_boy_s_001435.png",
        "label": 11
    }
]
```
Note that the key `file` in each json file represents the relative link of each sample. It is concatenated with the `root` specified in the `<dataset_name>.yaml` under the `dataset` to form the absolute path to each sample.


### Running

The experiment is setup through `hydra`. Thus, please make approriate changes in the `conf.yaml` and `conf/<dataset_name>.yaml` files before executing `run.sh`.


### Tracking, monitoring and managing experiments

The experiment management is handled by MLFlow.


### Citation

```bibtex
@inproceedings{nguyen2025probabilistic,
    title={Probabilistic Learning to Defer: Handling Missing Expert Annotations and Controlling Workload Distribution},
    author={Nguyen, Cuong and Do, Toan and Carneiro, Gustavo},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=zl0HLZOJC9}
}
```