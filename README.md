[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Probabilistic learning to defer

This repository provides an implementation of the *probabilistic learning to defer*. The implementation is written in Jax, and in particular, Flax. The data loading is handled by Apple MLX. Please see the `requirements.txt` for the list of packages used to run the code.

### Data loading

The data loading is heavily relied on `mlx-data`. In particular, the data loading only require a JSON file consisting a list of all samples with the structure as follows:

```bash
[
    {
        "file": "train/19/bos_taurus_s_000507.png",
        "label": 19,
        "superclass": 11
    },
    {
        "file": "train/29/stegosaurus_s_000125.png",
        "label": 29,
        "superclass": 15
    },
    {
        "file": "train/0/mcintosh_s_000643.png",
        "label": 0,
        "superclass": 4
    },
    {
        "file": "train/11/altar_boy_s_001435.png",
        "label": 11,
        "superclass": 14
    }
]
```
Note that the `file` in the JSON file represents the relative link to each sample. It is concatenated with the `root` specified in the `conf.yaml` (or `<dataset_name>.ymal`) under the `dataset` dictionary to form the absolute path to each sample.

### Running

The experiment is setup through `hydra`. Thus, please make approriate changes in the `conf.yaml` file before executing `run.sh`.

### Tracking, monitoring and managing experiments

The experiment management is handled by MLFlow.
