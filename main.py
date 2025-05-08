import torch
import numpy as np
from functools import partial

import optax
from flax import nnx

from torch.utils.data import DataLoader
from tqdm import tqdm

from clix.datasets import YandexDataset
from clix.models.pbm import PositionBasedModel
from clix.trainer import Trainer


@partial(nnx.jit)
def _train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch,
):
    def loss_fn(model, batch):
        return model.log_loss(batch)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss


def main():
    rngs = nnx.Rngs(0)
    dataset = YandexDataset("data/yandex.csv")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        collate_fn=dataset.collate_fn,
    )
    model = PositionBasedModel(positions=10, query_doc_pairs=140_000, rngs=rngs)
    trainer = Trainer(optax.adam(1e-3), epochs=10, patience=0)
    trainer.train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
