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
    Dependent Click Model (DCM)

    DCM extends the cascade model to allow multiple clicks by introducing
    rank-dependent continuation probabilities. Users may continue examining
    after a click based on their current position in the ranking.

    Assumptions:
    - Users examine documents sequentially from top to bottom
    - A click occurs if and only if a document is examined and attractive
    - After clicking: continue with rank-dependent probability λ_r
    - After examining but not clicking: always continue (probability 1)
    - Implicit satisfaction: P(satisfied | click) = 1 - λ_r

    References:
    - Guo et al. (2009). "Efficient multiple-click models in web search"
    """
    name = "DCM"

    def __init__(
        self,
        positions: int,
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
        log_probs = self._get_log_probabilities(batch)

        # Initialize: first document always examined (log(1) = 0):
        n_batch, n_positions = clicks.shape
        exam_log_probs = jnp.zeros((n_batch, n_positions))

        # Compute examination probabilities based on click history:
        for idx in range(n_positions - 1):
            exam_after_click = log_probs["cont"][:, idx]
            exam_after_no_click = self._log_examination_after_no_click(
                current_exam_log_prob=exam_log_probs[:, idx],
                attraction_log_prob=log_probs["attr"][:, idx],
                non_attraction_log_prob=log_probs["non_attr"][:, idx],
            )
            exam_log_probs = exam_log_probs.at[:, idx + 1].set(
                jnp.where(
                    clicks[:, idx],
                    exam_after_click,
                    exam_after_no_click,
                )
            )

        click_log_probs = exam_log_probs + log_probs["attr"]
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        log_probs = self._get_log_probabilities(batch)

        # Compute examination log probability increments for each position:
        exam_log_probs = self._log_examination_step(
            attr_log_prob=log_probs["attr"],
            non_attr_log_prob=log_probs["non_attr"],
            cont_log_prob=log_probs["cont"],
        )
        exam_log_probs = jnp.roll(exam_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + log_probs["attr"]
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        mask = batch["mask"]
        attr_probs = self.attraction.prob(batch)
        continuation = self.continuation.prob(batch)

        n_batch, n_positions = mask.shape
        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        attraction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)

        # Always examine first position (if valid):
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
                should_continue = jax.random.bernoulli(rngs(), p=continuation_prob)
                examination = examination.at[:, idx + 1].set(
                    should_continue & batch["mask"][:, idx + 1]
                )

        return DependentClickModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
        )

    def _get_log_probabilities(self, batch: Dict) -> Dict[str, Array]:
        attr_logits = self.attraction.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)
        cont_log_probs = self.continuation.log_prob(batch)

        return {
            "attr": attr_log_probs,
            "non_attr": non_attr_log_probs,
            "cont": cont_log_probs,
        }

    @staticmethod
    def _log_examination_after_no_click(
        current_exam_log_prob: Array,
        attraction_log_prob: Array,
        non_attraction_log_prob: Array,
    ) -> Array:
        """
        Compute examination probability after not clicking.
        Formula: P(E_{r+1} = 1 | E_r = 1, C_r = 0) = [(1-α_r) × ε_r] / [1 - α_r × ε_r]
        In log space: log(1-α_r) + log(ε_r) - log(1 - α_r × ε_r)
        """
        numerator_log = current_exam_log_prob + non_attraction_log_prob
        denominator_log = log1mexp(current_exam_log_prob + attraction_log_prob)
        return numerator_log - denominator_log

    @staticmethod
    def _log_examination_step(
        attr_log_prob: Array,
        non_attr_log_prob: Array,
        cont_log_prob: Array,
    ) -> Array:
        """
        Compute one step of unconditional examination log probability.
        Formula: P(E_{r+1} = 1) = α_r × λ_r + (1-α_r) × 1
        In log space: log[α_r × λ_r + (1-α_r)]
        """
        return jnp.logaddexp(cont_log_prob + attr_log_prob, non_attr_log_prob)
