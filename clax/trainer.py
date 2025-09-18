from copy import deepcopy
from functools import partial
from typing import Dict, Optional

import pandas as pd
import rax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from progress_table import ProgressTable
from torch.utils.data import DataLoader

from clax.metrics import (
    LogLikelihood,
    Perplexity,
    ConditionalPerplexity,
    MultiMetric,
    RaxMetric,
    Metric,
    Average,
)
from clax.models.base import ClickModel


class Trainer:
    def __init__(
        self,
        optimizer: GradientTransformation,
        epochs: int = 50,
        early_stopping: EarlyStopping = EarlyStopping(patience=0, min_delta=1e-5),
    ):
        self.optimizer = optimizer
        self.epochs = epochs
        self.early_stopping = early_stopping

    def train(
        self,
        model: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        click_metrics: Optional[Dict[str, Metric]] = None,
    ) -> pd.DataFrame:
        train_metrics = MultiMetric(loss=Average("loss"))

        # Ensure the loss is always present during validation:
        click_metrics = click_metrics or self._default_click_metrics()
        click_metrics = {**click_metrics, "loss": Average("loss")}
        val_metrics = MultiMetric(**click_metrics)

        early_stopping = deepcopy(self.early_stopping)
        optimizer = nnx.Optimizer(model, self.optimizer)
        best_state = nnx.state(model)

        logger = ProgressTable(
            columns=[
                "epoch",
                "model",
                *train_metrics.compute(prefix="train_").keys(),
                *val_metrics.compute(prefix="val_").keys(),
                "has_improved",
                "should_stop",
            ],
            num_decimal_places=6,
            pbar_embedded=False,
            pbar_show_percents=True,
            pbar_style="angled alt red blue",
        )

        for epoch in logger(range(self.epochs), description="Epochs"):
            logger.update_from_dict({"epoch": epoch, "model": model.name})
            model.train()

            for batch in logger(train_loader, description="Train"):
                self._train_step(model, optimizer, train_metrics, batch)

            train_results = train_metrics.compute(prefix="train_")
            train_metrics.reset()
            logger.update_from_dict(train_results)

            model.eval()

            for batch in logger(val_loader, description="Val"):
                self._test_step(model, val_metrics, batch)

            val_results = val_metrics.compute(prefix="val_")
            val_metrics.reset()

            early_stopping = early_stopping.update(val_results["val_loss"])
            logger.update_from_dict(val_results)
            logger.update_from_dict(
                {
                    "has_improved": early_stopping.has_improved,
                    "should_stop": early_stopping.should_stop,
                }
            )

            if early_stopping.has_improved:
                best_state = nnx.state(model)

            if early_stopping.should_stop:
                nnx.update(model, best_state)
                break

            logger.next_row()

        logger.close()
        return logger.to_df()

    def test_clicks(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
        click_metrics: Optional[Dict[str, Metric]] = None,
    ) -> pd.DataFrame:
        test_metrics = click_metrics or self._default_click_metrics()
        test_metrics = {**test_metrics, "loss": Average("loss")}
        metrics = MultiMetric(**test_metrics)

        model.eval()
        logger = ProgressTable(
            columns=[
                "model",
                *metrics.compute(prefix="test_").keys(),
            ],
            pbar_embedded=False,
            pbar_show_percents=True,
            pbar_style="angled alt red blue",
        )
        logger.update("model", model.name)

        for batch in logger(test_loader, description="Test"):
            self._test_step(model, metrics, batch)

        results = metrics.compute(prefix="test_")
        metrics.reset()

        logger.update_from_dict(results)
        logger.close()
        return logger.to_df()

    def test_ranking(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
        ranking_metrics: Optional[Dict[str, Metric]] = None,
    ) -> pd.DataFrame:
        metrics = ranking_metrics or self._default_ranking_metrics()
        metrics = MultiMetric(**metrics)

        model.eval()
        logger = ProgressTable(
            columns=[
                "model",
                *metrics.compute(prefix="test_").keys(),
            ],
            pbar_embedded=False,
            pbar_show_percents=True,
            pbar_style="angled alt red blue",
        )
        logger.update("model", model.name)

        for batch in logger(test_loader, description="Test"):
            self._test_relevance_step(model, metrics, batch)

        results = metrics.compute(prefix="test_")
        metrics.reset()

        logger.update_from_dict(results)
        logger.close()
        return logger.to_df()

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
        cond_log_probs = model.predict_conditional_clicks(batch)
        metrics.update(
            loss=loss,
            log_probs=log_probs,
            cond_log_probs=cond_log_probs,
            clicks=batch["clicks"],
            where=batch["mask"],
        )

    @partial(nnx.jit, static_argnums=(0))
    def _test_relevance_step(
        self,
        model: ClickModel,
        metrics: nnx.MultiMetric,
        batch,
    ):
        scores = model.predict_relevance(batch)
        metrics.update(
            scores=scores,
            labels=batch["labels"],
            where=batch["mask"],
        )

    @staticmethod
    def _default_click_metrics() -> Dict[str, Metric]:
        return {
            "ll": LogLikelihood(),
            "ppl": Perplexity(),
            "cond_ppl": ConditionalPerplexity(),
        }

    @staticmethod
    def _default_ranking_metrics() -> Dict[str, Metric]:
        return {
            "dcg@10": RaxMetric(rax.dcg_metric, top_n=10),
            "dcg@5": RaxMetric(rax.dcg_metric, top_n=5),
            "dcg@3": RaxMetric(rax.dcg_metric, top_n=3),
            "dcg@1": RaxMetric(rax.dcg_metric, top_n=1),
            "mrr@10": RaxMetric(rax.mrr_metric, top_n=10),
        }
