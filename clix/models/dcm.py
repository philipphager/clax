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
    """
    Dependent Click Model (DCM):
    1. Users examine documents sequentially from top to bottom
    2. They click if a document is examined AND attractive
    3. After clicking, they may continue with some probability (continuation) or stop
    4. After examining but not clicking, they always continue
    """
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

        # Get log probabilities:
        attr_logits = self.relevance.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)
        cont_log_probs = self.continuation.log_prob(batch)

        n_batch, n_positions = clicks.shape
        exam_log_probs = jnp.zeros((n_batch, n_positions))

        # Compute examination probabilities based on click history:
        for idx in range(n_positions - 1):
            exam_after_click = cont_log_probs[:, idx]
            exam_after_no_click = self._log_examination_after_no_click(
                current_exam_log_prob=exam_log_probs[:, idx],
                attraction_log_prob=attr_log_probs[:, idx],
                non_attraction_log_prob=non_attr_log_probs[:, idx]
            )
            next_exam_log_prob = jnp.where(
                clicks[:, idx],
                exam_after_click,
                exam_after_no_click,
            )
            exam_log_probs = exam_log_probs.at[:, idx + 1].set(next_exam_log_prob)

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        """
        Compute click log probabilities: log P(C=1|d,k) = log e_k + log a_d.
        Where:
        - log e_1 = 0
        - log e_{k + 1} = log e_k + log(exp(log a_d + log λ_k) + exp(log(1 - a_d)))
        """
        # Get log probabilities:
        attr_logits = self.relevance.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)
        cont_log_probs = self.continuation.log_prob(batch)

        # Compute examination log probability increments for each position:
        exam_increments = self._log_examination_step(
            attr_log_prob=attr_log_probs,
            non_attr_log_prob=non_attr_log_probs,
            cont_log_prob=cont_log_probs,
        )
        exam_increments = jnp.roll(exam_increments, shift=1, axis=-1)
        exam_increments = exam_increments.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_increments, axis=-1)

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        mask = batch["mask"]
        attr_probs = self.relevance.prob(batch)
        continuation = self.continuation.prob(batch)

        # Initialize outputs:
        n_batch, n_positions = mask.shape
        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        attraction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)

        # Always examine first position (if valid)
        examination = examination.at[:, 0].set(mask[:, 0])

        for idx in range(n_positions):
            attraction_at_idx = jax.random.bernoulli(rngs(), attr_probs[:, idx])
            attraction = attraction.at[:, idx].set(mask[:, idx] & attraction_at_idx)
            clicks = clicks.at[:, idx].set(examination[:, idx] & attraction[:, idx])

            # Update examination for next position:
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

    def _log_examination_after_no_click(
        self,
        current_exam_log_prob: Array,
        attraction_log_prob: Array,
        non_attraction_log_prob: Array
    ) -> Array:
        """
        Compute examination probability after not clicking:
        Formula: P(E_{k+1} = 1 | C_{d,k} = 0) = e_k(1-α_d) / (1-e_k α_d)
        In log space: log(e_k) + log(1-a_d) - log(1-e_k a_d)
        """
        numerator_log = current_exam_log_prob + non_attraction_log_prob
        denominator_log = log1mexp(current_exam_log_prob + attraction_log_prob)
        return numerator_log - denominator_log

    def _log_examination_step(
            self,
            attr_log_prob: Array,
            non_attr_log_prob: Array,
            cont_log_prob: Array,
    ) -> Array:
        """
        Compute one step of unconditional examination log probability:
        Formula: (a_d λ_k + (1-a_d))
        In log space: log(a_d λ_k + (1-a_d))
        """
        return jnp.logaddexp(cont_log_prob + attr_log_prob, non_attr_log_prob)
