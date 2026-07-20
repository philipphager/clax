# CLAX: Fast and Flexible Neural Click Models in JAX

[CLAX](https://arxiv.org/pdf/2511.03620) is a modular framework to build click models with gradient-based optimization in [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html).
CLAX is built to be fast, providing orders of magnitudes speed-up compared to classic EM-based frameworks, such as [PyClick](https://github.com/markovi/PyClick), by leveraging auto-diff and vectorized computations on GPUs.

The current documentation is available [here](https://philipphager.github.io/clax/). For more context, read our [SIGIR paper](https://dl.acm.org/doi/epdf/10.1145/3805712.3808584) or our [longer pre-print](https://arxiv.org/pdf/2511.03620).

## Installation
CLAX requires JAX. For installing JAX with CUDA support, please refer to the [JAX documentation](https://github.com/jax-ml/jax?tab=readme-ov-file#installation). CLAX itself is available via pypi:
```
pip install clax-models
```

## Basic Usage
CLAX is designed with sensible defaults, while also allowing for a high-level of customization. E.g., training a [User Browsing Model](https://dl.acm.org/doi/abs/10.1145/1390334.1390392) in CLAX is as simple as:
```Python
from clax import Trainer, UserBrowsingModel
from flax import nnx
from optax import adamw

model = UserBrowsingModel(
    query_doc_pairs=100_000_000, # Number of query-document pairs in the dataset
    positions=10, # Number of ranks per result page
    rngs=nnx.Rngs(42), # NNX random number generator
)
trainer = Trainer(
    optimizer=adamw(0.003),
    epochs=50,
)
train_df = trainer.train(model, train_loader, val_loader)
test_df = trainer.test(model, test_loader)
```

However, the modular design of CLAX also allows for more complex models from [two-tower models](https://dl.acm.org/doi/abs/10.1145/3477495.3531837), [mixture models](https://dl.acm.org/doi/abs/10.1145/3477495.3531837), or plugging-in custom FLAX modules as model parameters. We provide usage examples for getting started under [examples/](https://github.com/philipphager/clax/tree/main/examples).

## Reproducibility & Development
In the following, we cover how to reproduce the experiments from our paper or how to set up a fork of CLAX for development.

### Initial Setup
1. **Install the UV package manager**  
   UV is a fast Python dependency manager. Install it from: https://github.com/astral-sh/uv

2. **Clone the CLAX repository**
```bash
   git clone git@github.com:philipphager/clax.git
   cd clax/
```

3. **Install dependencies**
```bash
   uv sync
```
   This creates a virtual environment and installs all required dependencies.

### Running Experiments
Our paper's experiments are located in the `experiments/` directory. Each experiment contains:
- A Python script with the experiment logic: `main.py`
- A [Hydra](https://hydra.cc/) config file for configuration management: `config.yaml`
- A bash script with all experimental configurations: `main.sh`

To run an experiment, follow these steps.

1. **Install experiment dependencies**
Installs additional packages for SLURM support and data analysis/plotting.
```bash
   uv sync --group experiments
```

2. **Download datasets**  
   Clone the Yandex and Baidu-ULTR datasets from HuggingFace. If you have GIT LFS installed, clone the datasets using:
```bash
   git lfs install
   git clone https://huggingface.co/datasets/philipphager/clax-datasets
```
   Otherwise, download the datasets manually from [HuggingFace](https://huggingface.co/datasets/philipphager/clax-datasets).
   **Note:** The full datasets require 85GB of disk space. By default, CLAX expects datasets at `./clax-datasets/` relative to the project root. To use a custom path, update the `dataset_dir` parameter in each experiment's `config.yaml`:
```yaml
   dataset_dir: /my/custom/path/to/datasets/
```

3. **Run an experiment script**  
   Navigate to your experiment of interest and run the bash script, e.g.:
```bash
   cd experiments/1-yandex-baseline/
   chmod +x ./main.sh
   ./main.sh
```
   Optionally, you can run the script directly on a SLURM cluster using:
```bash
   sbatch ./main.sh +launcher=slurm
```
You can adjust the SLURM configuration to your cluster under: `experiments/config/slurm.yaml`

### PyClick Experiments
Baseline experiments using PyClick require the PyPy interpreter and are maintained in a separate repository: https://github.com/philipphager/clax-baselines

## Reference
If CLAX is useful to you, please consider citing our paper: 

```
@inproceedings{hager2026clax,
  title = {CLAX: Fast and Flexible Neural Click Models in JAX},
  author  = {Philipp Hager and Onno Zoeter and Maarten de Rijke},
  year  = {2026},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`26)}
}
```
