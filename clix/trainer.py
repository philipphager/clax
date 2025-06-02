from functools import partial

import pandas as pd
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from torch.utils.data import DataLoader
from tqdm import tqdm

from clix.metrics import (
    LogLikelihood,
    Perplexity,
    ConditionalPerplexity,
    Average,
    MultiMetric,
)


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
        self.train_metrics = {
            "loss": Average("loss"),
        }
        self.test_metrics = {
            "loss": Average("loss"),
            "ll": LogLikelihood(),
            "ppl": Perplexity(),
            "cond_ppl": ConditionalPerplexity(),
        }

    def train(
        self,
        model: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> pd.DataFrame:

        optimizer = nnx.Optimizer(model, self.optimizer)
        train_metrics = MultiMetric(**self.train_metrics)
        val_metrics = MultiMetric(**self.test_metrics)

        early_stopping = EarlyStopping(patience=self.patience)
        best_state = nnx.state(model)

        for epoch in range(self.epochs):
            model.train()

            for batch in tqdm(train_loader, desc="Train"):
                self._train_step(model, optimizer, train_metrics, batch)

            train_results = train_metrics.compute(prefix="train_")
            train_metrics.reset()

            model.eval()

            for batch in tqdm(val_loader, desc="Val"):
                self._test_step(model, val_metrics, batch)

            val_results = val_metrics.compute(prefix="val_")
            val_metrics.reset()

            early_stopping = early_stopping.update(val_results["val_loss"])

            if early_stopping.has_improved:
                best_state = nnx.state(model)

            if early_stopping.should_stop:
                nnx.update(model, best_state)
                break

    def test(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
    ) -> pd.DataFrame:
        metrics = MultiMetric(**self.test_metrics)
        model.eval()

        for batch in tqdm(test_loader, desc="Test"):
            self._test_step(model, metrics, batch)

        results = metrics.compute(prefix="test_")
        metrics.reset()

    @partial(nnx.jit, static_argnums=(0))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):
        def loss_fn(model, batch):
            return model.compute_loss(batch).mean()

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        metrics.update(loss=loss)
        optimizer.update(grads)

    @partial(nnx.jit, static_argnums=(0))
    def _test_step(
        self,
        model: nnx.Module,
        metrics: nnx.MultiMetric,
        batch,
    ):
        loss = model.compute_loss(batch).mean()
        log_probs = model.predict_clicks(batch)
        conditional_log_probs = model.predict_conditional_clicks(batch)
        metrics.update(
            loss=loss,
            log_probs=log_probs,
            conditional_log_probs=conditional_log_probs,
            clicks=batch["clicks"],
            where=batch["mask"],
        )
