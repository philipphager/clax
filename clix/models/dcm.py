from typing import Dict

import jax.numpy as jnp
import jax.random
from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


class DependentClickModel(nnx.Module):
    def __init__(
        self,
        positions: int,
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
        self.continuation = BernoulliEmbedding(
            use_feature="positions",
            parameters=positions + 1,
            rngs=rngs,
        )

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        relevance = self.relevance(batch)
        continuation = self.continuation(batch)
        n_batch, n_positions = relevance.shape

        examination = jnp.ones((n_batch, n_positions))

        for idx in range(n_positions - 1):
            click_prob = examination[:, idx] * relevance[:, idx]
            no_click_prob = 1 - click_prob
            examined_and_not_relevant = examination[:, idx] * (1 - relevance[:, idx])

            examination_after_click = continuation[:, idx]
            examination_after_no_click = examined_and_not_relevant / no_click_prob
            examination = examination.at[:, idx + 1].set(
                jnp.where(
                    clicks[:, idx],
                    examination_after_click,
                    examination_after_no_click,
                )
            )

        return batch["mask"] * examination * relevance

    def predict_clicks(self, batch: Dict) -> Array:
        relevance = self.relevance(batch)
        continuation = self.continuation(batch)

        examination = continuation * relevance + (1 - relevance)
        examination = jnp.roll(examination, shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        examination = jnp.cumprod(examination, axis=-1)

        return batch["mask"] * examination * relevance

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        relevance = self.relevance(batch)
        continuation = self.continuation(batch)
        n_batch, n_positions = relevance.shape

        clicks = jnp.zeros((n_batch, n_positions))
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = examination.at[:, 0].set(True)

        for idx in range(n_positions):
            click_prob = jnp.where(examination[:, idx], relevance[:, idx], 0.0)
            clicks = clicks.at[:, idx].set(jax.random.bernoulli(rngs(), click_prob))

            if idx < n_positions - 1:
                # Determine continuation probability:
                # - If clicked: use continuation probability
                # - If examined but not clicked: always continue (prob=1)
                # - If not examined: never continue (prob=0)
                continuation_prob = jnp.where(
                    examination[:, idx],
                    jnp.where(clicks[:, idx], continuation[:, idx], 1.0),
                    0.0,
                )
                examination = examination.at[:, idx + 1].set(
                    jax.random.bernoulli(rngs(), continuation_prob)
                )

        return batch["mask"] * clicks
