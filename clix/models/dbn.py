from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jaxlib.xla_extension import Array


from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding, BernoulliParameter
from clix.models.math import (
    logits_to_log_probs,
    logits_to_complement_log_probs,
    log1mexp,
)

@struct.dataclass
class DynamicBayesianNetworkOutput:
    clicks: Array
    examination: Array
    attraction: Array
    satisfaction: Array

class DynamicBayesianNetwork(nnx.Module):
    def __init__(
        self,
        query_doc_pairs: int,
        fix_continuation: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.fix_continuation = fix_continuation
        self.attraction = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )
        self.satisfaction = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )
        self.continuation = BernoulliParameter(rngs=rngs)

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"], log_probs=True)

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]

        # Get log probabilities:
        attr_logits = self.attraction.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)

        sat_logits = self.satisfaction.logit(batch)
        non_sat_log_probs = logits_to_complement_log_probs(sat_logits)

        cont_log_prob = jnp.array([0.0]) if self.fix_continuation else self.continuation.log_prob()

        # Initialize: first document always examined (log(1) = 0):
        n_batch, n_positions = clicks.shape
        exam_log_probs = jnp.zeros((n_batch, n_positions))

        # Compute examination probabilities based on click history:
        for idx in range(n_positions - 1):
            exam_after_click = self._log_examination_after_click(
                non_sat_log_probs=non_sat_log_probs[:, idx],
                cont_log_prob=cont_log_prob,
            )
            exam_after_no_click = self._log_examination_after_no_click(
                current_exam_log_prob=exam_log_probs[:, idx],
                attraction_log_prob=attr_log_probs[:, idx],
                non_attraction_log_prob=non_attr_log_probs[:, idx],
                cont_log_prob=cont_log_prob,
            )

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
        attr_logits = self.attraction.logit(batch)
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)

        sat_logits = self.satisfaction.logit(batch)
        non_sat_log_probs = logits_to_complement_log_probs(sat_logits)

        cont_log_prob = jnp.array([0.0]) if self.fix_continuation else self.continuation.log_prob()

        exam_log_probs = self._log_examination_step(
            attr_log_probs=attr_log_probs,
            non_attr_log_probs=non_attr_log_probs,
            non_sat_log_probs=non_sat_log_probs,
            cont_log_prob=cont_log_prob,
        )
        exam_log_probs = jnp.roll(exam_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    @staticmethod
    def _log_examination_step(
        attr_log_probs: Array,
        non_attr_log_probs: Array,
        non_sat_log_probs: Array,
        cont_log_prob: Array,
    ) -> Array:
        """
        Compute one step of unconditional examination log probability:
        Formula:
        In log space:
        """
        return cont_log_prob + jnp.logaddexp(
            attr_log_probs + non_sat_log_probs, non_attr_log_probs
        )


    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        mask = batch["mask"]
        n_batch, n_positions = mask.shape

        attr_probs = self.attraction.prob(batch)
        sat_probs = self.satisfaction.prob(batch)
        continuation = jnp.array([1.0]) if self.fix_continuation else self.continuation.prob()

        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        attraction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        satisfaction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        should_continue = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)

        # Always examine the first item (if valid)
        examination = examination.at[:, 0].set(batch["mask"][:, 0])

        for idx in range(n_positions):
            attraction_at_idx = jax.random.bernoulli(rngs(), attr_probs[:, idx])
            attraction = attraction.at[:, idx].set(mask[:, idx] & attraction_at_idx)
            clicks = clicks.at[:, idx].set(examination[:, idx] & attraction[:, idx])

            if idx < n_positions - 1:
                # Sample user satisfaction only for clicked items:
                satisfaction_probs = jnp.where(clicks[:, idx], sat_probs[:, idx], 0.0)
                satisfaction = satisfaction.at[:, idx].set(
                    jax.random.bernoulli(rngs(), p=satisfaction_probs)
                )

                # Sample user continuation:
                # - Users continue when not satisfied after click
                # - Users continue when examined but the item is not attractive/clicked
                continue_after_click = clicks[:, idx] & ~satisfaction[:, idx]
                continue_without_click = examination[:, idx] & ~clicks[:, idx]
                continuation_probs = continuation * (
                    continue_after_click | continue_without_click
                )
                should_continue = should_continue.at[:, idx].set(
                    jax.random.bernoulli(rngs(), p=continuation_probs)
                )
                examination = examination.at[:, idx + 1].set(
                    should_continue[:, idx] & batch["mask"][:, idx + 1]
                )

        return DynamicBayesianNetworkOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
            satisfaction=satisfaction,
        )

    @staticmethod
    def _log_examination_after_click(
            non_sat_log_probs: Array,
            cont_log_prob: Array,
    ) -> Array:
        """
        Compute log examination probability after clicking.
        Formula: e_{r+1} = (1 - σ_r) × γ
        In log space: log ε_{r+1} = log(1 - σ_r) + log γ
        """
        return cont_log_prob + non_sat_log_probs

    @staticmethod
    def _log_examination_after_no_click(
            current_exam_log_prob: Array,
            attraction_log_prob: Array,
            non_attraction_log_prob: Array,
            cont_log_prob: Array,
    ) -> Array:
        """
        Compute log examination probability after not clicking.
        Formula: ε_{r+1} = [(1 - α_r) × ε_r × γ] / [1 - α_r × ε_r]
        In log space: log ε_{r+1} = log(1 - α_r) + log ε_r + log γ - log(1 - α_r × ε_r)
        """
        numerator_log = current_exam_log_prob + non_attraction_log_prob + cont_log_prob
        denominator_log = log1mexp(current_exam_log_prob + attraction_log_prob)
        return numerator_log - denominator_log

    @staticmethod
    def _log_examination_step(
        attr_log_probs: Array,
        non_attr_log_probs: Array,
        non_sat_log_probs: Array,
        cont_log_prob: Array,
    ) -> Array:
        """
        Compute one step of unconditional examination log probability:
        Formula: γ × [α(1-σ) + (1-α)]
        In log space: log(γ) + log[α(1-σ) + (1-α)]
        """
        return cont_log_prob + jnp.logaddexp(
            attr_log_probs + non_sat_log_probs, non_attr_log_probs
        )
