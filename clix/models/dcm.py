from typing import Dict

import jax.numpy as jnp
import jax.random
from flax import nnx
from flax import struct
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.math import (
    logits_to_log_probs,
    logits_to_complement_log_probs,
    log1mexp,
)
from clix.models.parameters import BernoulliEmbedding


@struct.dataclass
class DependentClickModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


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
        return binary_cross_entropy(
            y_predict,
            y_true,
            where=batch["mask"],
            log_probs=True,
        )

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        attr_logits = self.relevance.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)

        cont_log_probs = self.continuation.log_prob(batch)
        n_batch, n_positions = clicks.shape
        exam_log_probs = jnp.zeros((n_batch, n_positions))

        for idx in range(n_positions - 1):
            exam_after_click = cont_log_probs[:, idx]
            exam_and_non_attr_log_probs = (
                exam_log_probs[:, idx] + non_attr_log_probs[:, idx]
            )
            no_click_log_probs = log1mexp(
                exam_log_probs[:, idx] + attr_log_probs[:, idx]
            )
            exam_after_no_click = exam_and_non_attr_log_probs - no_click_log_probs
            exam_log_probs = exam_log_probs.at[:, idx + 1].set(
                jnp.where(
                    clicks[:, idx],
                    exam_after_click,
                    exam_after_no_click,
                )
            )

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        """
        Compute click log probabilities: log P(C=1|d,k) = log e_k + log a_d.
        Where:
        - log e_1 = 0
        - log e_{k + 1} = log e_k + log(exp(log a_d + log λ_k) + exp(log(1 - a_d)))
        """
        attr_logits = self.relevance.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)
        cont_log_probs = self.continuation.log_prob(batch)

        exam_log_probs = jnp.logaddexp(
            cont_log_probs + attr_log_probs,
            non_attr_log_probs,
        )
        exam_log_probs = jnp.roll(exam_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        mask = batch["mask"]
        attr_probs = self.relevance.prob(batch)
        continuation = self.continuation.prob(batch)
        n_batch, n_positions = mask.shape

        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        attraction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = examination.at[:, 0].set(mask[:, 0])

        for idx in range(n_positions):
            attraction_at_idx = jax.random.bernoulli(rngs(), attr_probs[:, idx])
            attraction = attraction.at[:, idx].set(mask[:, idx] & attraction_at_idx)
            clicks = clicks.at[:, idx].set(examination[:, idx] & attraction[:, idx])

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
                    mask[:, idx + 1] & jax.random.bernoulli(rngs(), continuation_prob)
                )

        return DependentClickModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
        )
