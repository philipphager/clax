from typing import Dict

import jax
import jax.numpy as jnp

from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


class CascadeModel(nnx.Module):
    def __init__(
        self,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.relevance = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )
        self.min_probability = 0.00001

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = self.predict_clicks(batch)
        before_first_click = batch["clicks"].cumsum(axis=-1) <= 1
        clicks = jnp.where(before_first_click, clicks, self.min_probability)

        return batch["mask"] * clicks

    def predict_clicks(self, batch: Dict) -> Array:
        relevance = self.relevance(batch)

        examination = jnp.roll((1 - relevance), shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        examination = jnp.cumprod(examination, axis=-1)

        return batch["mask"] * examination * relevance

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        relevance = self.relevance(batch)
        clicks = jax.random.bernoulli(rngs(), relevance)
        before_first_click = clicks.cumsum(axis=-1) <= 1
        clicks = jnp.where(before_first_click, clicks, 0.0)

        return clicks
