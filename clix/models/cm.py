from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.math import logits_to_log_probs, logits_to_complement_log_probs
from clix.models.parameters import BernoulliEmbedding

MIN_LOG_PROB = jnp.log(1e-8)


@struct.dataclass
class CascadeModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


class CascadeModel(nnx.Module):
    """
    Cascade Model (CM)

    The cascade model assumes users examine documents from top to bottom and stop
    after clicking the first relevant document. The model can only explain sessions
    with a single click.

    Model Assumptions:
    - Users examine documents sequentially from top to bottom
    - The first document is always examined
    - A click occurs if and only if a document is examined and attractive
    - Users stop examining after the first click

    References:
    - Craswell et al. (2008). "An experimental comparison of click position-bias models"
    """

    def __init__(
        self,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attraction = BernoulliEmbedding(
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
        no_clicks_before = self._no_clicks_before(batch["clicks"])
        click_log_probs = jnp.where(no_clicks_before, click_log_probs, MIN_LOG_PROB)

        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        attr_logits = self.attraction.logit(batch)

        # Compute log probabilities for relevance and non-relevance:
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)

        # Compute log examination, the first item is always examined:
        exam_log_probs = jnp.roll(non_attr_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        attr_probs = self.attraction.prob(batch)
        attraction = batch["mask"] & jax.random.bernoulli(rngs(), attr_probs)

        examination = self._no_clicks_before(attraction)
        clicks = examination & attraction

        return CascadeModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
        )

    @staticmethod
    def _no_clicks_before(clicks):
        """
        Check if there are no clicks before each position.
        """
        clicks_before = jnp.cumsum(clicks, axis=-1) - clicks
        return clicks_before == 0
