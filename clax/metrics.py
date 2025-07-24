from copy import deepcopy
from typing import Optional, Any

import jax.numpy as jnp
from flax import nnx
from flax.nnx.object import Object
from flax.nnx.variablelib import Variable
from jax import Array

from clax.utils.math import log1mexp


class MetricState(Variable):
    pass


class Metric(Object):
    def reset(self) -> None:
        raise NotImplementedError("Must override `reset()` method.")

    def update(self, **kwargs) -> None:
        raise NotImplementedError("Must override `update()` method.")

    def compute(self):
        raise NotImplementedError("Must override `compute()` method.")


class MultiMetric(Metric):
    def __init__(self, **metrics):
        self.metric_names = []
        metrics = deepcopy(metrics)

        for metric_name, metric in metrics.items():
            self.metric_names.append(metric_name)
            vars(self)[metric_name] = metric

    def reset(self) -> None:
        for metric_name in self.metric_names:
            getattr(self, metric_name).reset()

    def update(self, **updates) -> None:
        for metric_name in self.metric_names:
            getattr(self, metric_name).update(**updates)

    def compute(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}{metric_name}": getattr(self, metric_name).compute()
            for metric_name in self.metric_names
        }

    def compute_per_rank(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}{metric_name}": getattr(self, metric_name).compute_per_rank()
            for metric_name in self.metric_names
            if isinstance(getattr(self, metric_name), RankBasedAverage)
        }


class Average(Metric):
    def __init__(self, argname: str):
        self.argname = argname
        self.total = MetricState(jnp.array(0, dtype=jnp.float32))
        self.count = MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total.value = jnp.array(0, dtype=jnp.float32)
        self.count.value = jnp.array(0, dtype=jnp.int32)

    def update(self, **kwargs) -> None:
        if self.argname not in kwargs:
            raise TypeError(f"Expected keyword argument '{self.argname}'")

        values = kwargs[self.argname]
        self.total.value += values if isinstance(values, (int, float)) else values.sum()
        self.count.value += 1 if isinstance(values, (int, float)) else values.size

    def compute(self) -> Array:
        return self.total.value / self.count.value


class RankBasedAverage(Metric):
    def __init__(self, positions: int = 10):
        self.positions = positions
        self.values_per_rank = nnx.metrics.MetricState(
            jnp.zeros(self.positions, dtype=jnp.float32)
        )
        self.counts_per_rank = nnx.metrics.MetricState(
            jnp.zeros(self.positions, dtype=jnp.int32)
        )

    def update_values(
        self,
        values: Array,
        *,
        where: Optional[Array] = None,
    ):
        if where is None:
            where = jnp.ones_like(values)

        self.values_per_rank.value += values.sum(axis=0, where=where)
        self.counts_per_rank.value += where.sum(axis=0)

    def reset(self):
        self.values_per_rank.value = jnp.zeros(self.positions, dtype=jnp.float32)
        self.counts_per_rank.value = jnp.zeros(self.positions, dtype=jnp.float32)

    def compute(self):
        value = self.values_per_rank.value.sum()
        count = self.counts_per_rank.value.sum()
        return value / count.clip(min=1)

    def compute_per_rank(self):
        return self.values_per_rank.value / self.counts_per_rank.value.clip(min=1)


class LogLikelihood(RankBasedAverage):
    def update(
        self,
        *,
        conditional_log_probs: Array,
        clicks: Array,
        where: Optional[Array] = None,
        **kwargs,
    ):
        p_click = conditional_log_probs
        p_no_click = log1mexp(conditional_log_probs)
        log_likelihood = clicks * p_click + (1 - clicks) * p_no_click

        super().update_values(log_likelihood, where=where)


class ConditionalPerplexity(RankBasedAverage):
    def update(
        self,
        *,
        conditional_log_probs: Array,
        clicks: Array,
        where: Optional[Array] = None,
        **kwargs,
    ):
        # Convert log probabilities ln(p) to log_2(p)
        p_click = conditional_log_probs / jnp.log(2)
        p_no_click = log1mexp(conditional_log_probs) / jnp.log(2)
        log_likelihood = clicks * p_click + (1 - clicks) * p_no_click

        super().update_values(log_likelihood, where=where)

    def compute(self):
        # Avg. cond. perplexity is calculated over all ranks
        return self.compute_per_rank().mean()

    def compute_per_rank(self):
        return 2 ** -super().compute_per_rank()


class Perplexity(RankBasedAverage):
    def update(
        self,
        *,
        log_probs: Array,
        clicks: Array,
        where: Optional[Array] = None,
        **kwargs,
    ):
        # Convert log probabilities ln(p) to log_2(p)
        p_click = log_probs / jnp.log(2)
        p_no_click = log1mexp(log_probs) / jnp.log(2)
        log_likelihood = clicks * p_click + (1 - clicks) * p_no_click

        super().update_values(log_likelihood, where=where)

    def compute(self):
        # Avg. perplexity is calculated over all ranks
        return self.compute_per_rank().mean()

    def compute_per_rank(self):
        return 2 ** -super().compute_per_rank()
