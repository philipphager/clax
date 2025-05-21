from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.math import logits_to_log_probs, logits_to_complement_log_probs
from clix.models.parameters import BernoulliEmbedding

MIN_LOG_PROB = jnp.log(1e-8)


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

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)

        return binary_cross_entropy(
            y_predict,
            y_true,
            where=batch["mask"],
            log_probs=True,
        )

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        click_log_probs = self.predict_clicks(batch)

        # Discard clicks after the first click by setting them to a minimum log prob:
        before_first_click = batch["clicks"].cumsum(axis=-1) <= 1
        click_log_probs = jnp.where(
            before_first_click,
            click_log_probs,
            MIN_LOG_PROB,
        )

        return click_log_probs

    def predict_clicks(self, batch: Dict) -> Array:
        rel_logits = self.relevance.logit(batch)

        # Compute log probabilities for relevance and non-relevance:
        rel_log_probs = logits_to_log_probs(rel_logits)
        non_rel_log_probs = logits_to_complement_log_probs(rel_logits)

        # Compute log examination, the first item is always examined:
        exam_log_probs = jnp.roll(non_rel_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + rel_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        relevance = self.relevance.prob(batch)
        clicks = jax.random.bernoulli(rngs(), relevance)

        # Only keep clicks if no other item was clicked before,
        # this is equivalent to sequential sampling:
        before_first_click = clicks.cumsum(axis=-1) <= 1
        clicks = jnp.where(before_first_click, clicks, 0.0)

        return clicks
