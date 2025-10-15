# Welcome to CLAX

CLAX is a framework for training and evaluating fast and flexible neural clic models in JAX.

For example, training a [User Browsing Model](https://dl.acm.org/doi/abs/10.1145/1390334.1390392) in CLAX is as simple as:
```Python
from clax import Trainer, UserBrowsingModel
from flax import nnx
from optax import adamw

model = UserBrowsingModel(
    query_doc_pairs=100_000_000,
    positions=10,
    rngs=nnx.Rngs(42),
)
trainer = Trainer(
    optimizer=adamw(0.003),
    epochs=50,
)
train_df = trainer.train(model, train_loader, val_loader)
test_df = trainer.test(model, test_loader)
```

where `train_loader` and `val_loader` are, e.g., PyTorch data loaders.

## Installation
CLAX requires JAX. For installing JAX with CUDA support, please refer to the [JAX documentation](https://github.com/jax-ml/jax?tab=readme-ov-file#installation).
CLAX itself is available via pypi:
```bash
pip install clax-core
```


## Documentation
- [Overview of Click Models implemented in CLAX](/2-models/)
- [Evaluation metrics](/3-metrics/)
- Modularity in CLAX
- Implementing new click models in CLAX
- Datasets

## Reference
If you use CLAX, please consider citing our paper: 

```
@misc{hager2025clax,
  title = {CLAX: Fast and Flexible Neural Click Models in JAX},
  author  = {Philipp Hager and Onno Zoeter and Maarten de Rijke},
  year  = {2025},
  booktitle = {arxiv}
}
```
