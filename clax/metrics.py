from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Any, Callable

import jax.numpy as jnp
from flax import nnx
from jax import Array

from clax.utils.math import log1mexp


class MetricState(nnx.variablelib.Variable):
    pass


class Metric(nnx.object.Object, ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


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
    def __init__(self, argname: str = "values"):
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
        return self.total.value / self.count.value.clip(1)


class RankBasedAverage(Metric, ABC):
    def __init__(self, max_positions: int = 100):
        self.max_positions = max_positions
        self.values_per_rank = nnx.metrics.MetricState(
            jnp.zeros(self.max_positions, dtype=jnp.float32)
        )
        self.counts_per_rank = nnx.metrics.MetricState(
            jnp.zeros(self.max_positions, dtype=jnp.int32)
        )

    @abstractmethod
    def update(self, **kwargs):
        pass

    def update_values(
        self,
        values: Array,
        *,
        where: Optional[Array] = None,
    ):
        n_batch, n_positions = values.shape

        if where is None:
            where = jnp.ones((n_batch, n_positions))

        self.values_per_rank.value = self.values_per_rank.value.at[:n_positions].add(
            values.sum(axis=0, where=where)
        )
        self.counts_per_rank.value = self.counts_per_rank.value.at[:n_positions].add(
            where.sum(axis=0)
        )

    def reset(self):
        self.values_per_rank.value = jnp.zeros(self.max_positions, dtype=jnp.float32)
        self.counts_per_rank.value = jnp.zeros(self.max_positions, dtype=jnp.int32)

    def compute(self):
        # Ignore positions with no observations:
        where = self.counts_per_rank.value > 0

        value = self.values_per_rank.value[where].sum()
        count = self.counts_per_rank.value[where].sum()
        return value / count.clip(min=1)

    def compute_per_rank(self):
        # Do not return positions with no observations:
        where = self.counts_per_rank.value > 0

        values = self.values_per_rank.value[where]
        counts = self.counts_per_rank.value[where]
        return values / counts.clip(min=1)


class LogLikelihood(RankBasedAverage):
    """
    Examples:
        Compute the mean log-likelihood over a single query with three documents
        from conditional log probabilities. Use `where = False` to mask out padding documents:

            ll = LogLikelihood()
            ll.update(
                cond_log_probs=jnp.array([[-0.01, -10.0, -0.7]]),
                clicks=jnp.array([[1, 0, 1]]),
                where=jnp.array([[True, True, True]]),
            )
            ll.compute()

        Compute the mean log-likelihood for each rank:

            ll.compute_per_rank()
    """

    def update(
        self,
        *,
        cond_log_probs: Array,
        clicks: Array,
        where: Optional[Array] = None,
        **kwargs,
    ):
        p_click = cond_log_probs
        p_no_click = log1mexp(cond_log_probs)
        log_likelihood = clicks * p_click + (1 - clicks) * p_no_click

        super().update_values(log_likelihood, where=where)


class ConditionalPerplexity(RankBasedAverage):
    """
    Examples:
        Compute the mean conditional perplexity over a single query with three documents
        from conditional log probabilities. Use `where = False` to mask out padding documents:

            cond_ppl = ConditionalPerplexity()
            cond_ppl.update(
                cond_log_probs=jnp.array([[-0.01, -10.0, -0.7]]),
                clicks=jnp.array([[1, 0, 1]]),
                where=jnp.array([[True, True, True]]),
            )
            cond_ppl.compute()

        Compute the mean conditional perplexity for each rank:

            cond_ppl.compute_per_rank()
    """

    def update(
        self,
        *,
        cond_log_probs: Array,
        clicks: Array,
        where: Optional[Array] = None,
        **kwargs,
    ):
        # Convert log probabilities ln(p) to log_2(p)
        p_click = cond_log_probs / jnp.log(2)
        p_no_click = log1mexp(cond_log_probs) / jnp.log(2)
        log_likelihood = clicks * p_click + (1 - clicks) * p_no_click

        super().update_values(log_likelihood, where=where)

    def compute(self):
        # Avg. cond. perplexity is the mean over ranks:
        return self.compute_per_rank().mean()

    def compute_per_rank(self):
        return 2 ** -super().compute_per_rank()


class Perplexity(RankBasedAverage):
    """
    Examples:
        Compute the mean (unconditional) perplexity over a single query with three documents
        from unconditional log probabilities. Use `where = False` to mask out padding documents:

            ppl = Perplexity()
            ppl.update(
                log_probs=jnp.array([[-0.01, -10.0, -0.7]]),
                clicks=jnp.array([[1, 0, 1]]),
                where=jnp.array([[True, True, True]]),
            )
            ppl.compute()

        Compute the mean perplexity for each rank:

            ppl.compute_per_rank()
    """

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
        # Avg. perplexity is the mean over ranks:
        return self.compute_per_rank().mean()

    def compute_per_rank(self):
        return 2 ** -super().compute_per_rank()


class RaxMetric(Average):
    def __init__(self, metric_fn: Callable, top_n: Optional[int] = None):
        super().__init__()
        self.metric_fn = metric_fn
        self.top_n = top_n

    def update(
        self,
        *,
        scores: Array,
        labels: Array,
        where: Optional[Array] = None,
        **kwargs,
    ):
        values = self.metric_fn(
            scores=scores,
            labels=labels,
            where=where,
            topn=self.top_n,
            reduce_fn=reduce_per_query,
        )
        super().update(values=values)


def reduce_per_query(loss: Array, where: Array) -> Array:
    loss = loss.reshape(len(loss), -1)
    where = where.reshape(len(where), -1)

    # Adopt Rax safe_reduce as jnp.mean can return NaN if all inputs are 0,
    # which happens easily for pairwise loss functions without any valid pair.
    # Replace NaNs with 0 after reduce, but propagate if the loss already contains NaNs:
    is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(loss)))
    output = jnp.mean(loss, where=where, axis=1)
    output = jnp.where(jnp.isnan(output) & is_input_valid, 0.0, output)

    return output
