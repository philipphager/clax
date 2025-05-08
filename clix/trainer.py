from copy import deepcopy
from functools import partial

from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        optimizer: GradientTransformation,
        epochs: int = 50,
        patience: int = 0,
    ):
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience
        self.metrics = {"loss": nnx.metrics.Average("loss")}

    def train(
        self,
        model: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        optimizer = nnx.Optimizer(model, self.optimizer)
        metrics = nnx.MultiMetric(**deepcopy(self.metrics))
        early_stopping = EarlyStopping(patience=self.patience)
        best_state = nnx.state(model)

        for epoch in range(self.epochs):
            model.train()

            for batch in tqdm(train_loader, desc=f"Train - Epoch: {epoch}"):
                self._train_step(model, optimizer, metrics, batch)

            train_metrics = metrics.compute()
            metrics.reset()

            model.eval()

            for batch in tqdm(val_loader, desc=f"Val - Epoch: {epoch}"):
                self._test_click_step(model, metrics, batch)

            val_metrics = metrics.compute()
            early_stopping = early_stopping.update(val_metrics["loss"])
            metrics.reset()

            print(
                f"Epoch {epoch} - "
                f"Train loss: {train_metrics['loss']:.8f}, "
                f"Val loss: {val_metrics['loss']:.8f}, "
                f"has improved: {early_stopping.has_improved}\n"
            )

            if early_stopping.has_improved:
                best_state = nnx.state(model)

            if early_stopping.should_stop:
                print("Stopping early, loading best model state")
                nnx.update(model, best_state)
                break

    @partial(nnx.jit, static_argnums=(0))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):
        def loss_fn(model, batch):
            return model.log_loss(batch)

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        metrics.update(loss=loss)
        optimizer.update(grads)

    @partial(nnx.jit, static_argnums=(0))
    def _test_click_step(
        self,
        model: nnx.Module,
        metrics: nnx.MultiMetric,
        batch,
    ):
        loss = model.log_loss(batch)
        metrics.update(loss=loss)
