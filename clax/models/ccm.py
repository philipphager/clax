from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import (
    ParameterConfig,
    GlobalParameter,
    init_parameter,
    Parameter,
)
from clax.parameters.defaults import default_attraction_config
from clax.utils.math import (
    logits_to_log_probs,
    logits_to_complement_log_probs,
    log1mexp,
)


@struct.dataclass
class ClickChainModelOutput:
    clicks: Array
    examination: Array
    attraction: Array
    satisfaction: Array


class ClickChainModel(ClickModel):
    """
    Click Chain Model (CCM)

    The CCM extends the DCM to allow users abandoning a session without any clicks
    by introducing probabilities for users to continue examination after not clicking
    a document, clicking but not being satisfied, and clicking and being satisfied.
    The CCM assumes that document attraction and satisfaction probabilities are identical.

    Args:
        query_doc_pairs (Optional[int], optional): Number of query-document
            pairs to allocate in an embedding table. This parameter is not
            used if a custom attraction module is provided.
        attraction (Optional[Parameter | ParameterConfig], optional): Custom
            attraction / satisfaction parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        rngs (nnx.Rngs): A NNX random number generator used for model
            initialization and stochastic operations.

    Examples:

        model = ClickChainModel(
            query_doc_pairs=1_000_000,
            rngs=nnx.Rngs(42)
        )

    References:
        Fan Guo, Chao Liu, Anitha Kannan, Tom Minka, Michael Taylor, Yi-Min Wang, and Christos Faloutsos.
        "Click Chain Model in Web Search."
        In WWW 2009.
    """

    name = "CCM"

    def __init__(
        self,
        query_doc_pairs: Optional[int] = None,
        attraction: Optional[Parameter | ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.attraction = init_parameter(
            "attraction",
            attraction,
            default_config_fn=default_attraction_config,
            default_config_args={"query_doc_pairs": query_doc_pairs},
            rngs=rngs,
        )
        # Continuation are global variables that don't depend on features.
        # These might be configurable in future versions if useful:
        self.continuation_exam_no_click = GlobalParameter(rngs=rngs)
        self.continuation_click_satisfied = GlobalParameter(rngs=rngs)
        self.continuation_click_not_satisfied = GlobalParameter(rngs=rngs)

    def compute_loss(self, batch: Dict, aggregate: bool = True):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(
            y_predict,
            y_true,
            where=batch["mask"],
            log_probs=True,
            aggregate=aggregate,
        )

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        n_batch, n_positions = clicks.shape
        log_probs = self._get_log_probabilities(batch)

        # First position is always examined (log(1) = 0):
        exam_log_probs = jnp.zeros((n_batch, n_positions))

        for idx in range(n_positions - 1):
            exam_after_click = self._log_examination_after_click(
                rel_log_prob=log_probs["rel"][:, idx],
                non_rel_log_prob=log_probs["non_rel"][:, idx],
                tau2_log_prob=log_probs["tau2"],
                tau3_log_prob=log_probs["tau3"],
            )
            exam_after_no_click = self._log_examination_after_no_click(
                current_exam_log_prob=exam_log_probs[:, idx],
                rel_log_prob=log_probs["rel"][:, idx],
                non_rel_log_prob=log_probs["non_rel"][:, idx],
                tau1_log_prob=log_probs["tau1"],
            )

            exam_log_probs = exam_log_probs.at[:, idx + 1].set(
                jnp.where(
                    clicks[:, idx],
                    exam_after_click,
                    exam_after_no_click,
                )
            )

        click_log_probs = exam_log_probs + log_probs["rel"]
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        log_probs = self._get_log_probabilities(batch)

        exam_log_probs = self._log_examination_step(
            rel_log_prob=log_probs["rel"],
            non_rel_log_prob=log_probs["non_rel"],
            tau1_log_prob=log_probs["tau1"],
            tau2_log_prob=log_probs["tau2"],
            tau3_log_prob=log_probs["tau3"],
        )
        exam_log_probs = jnp.roll(exam_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + log_probs["rel"]
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_relevance(self, batch: Dict) -> Array:
        return self.attraction.log_prob(batch)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> ClickChainModelOutput:
        rel_probs = self.attraction.prob(batch)
        tau1 = self.continuation_exam_no_click.prob()
        tau2 = self.continuation_click_not_satisfied.prob()
        tau3 = self.continuation_click_satisfied.prob()
        mask = batch["mask"]

        n_batch, n_positions = rel_probs.shape
        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        attraction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        satisfaction = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)

        # If valid, always examine the first item:
        examination = examination.at[:, 0].set(batch["mask"][:, 0])

        for idx in range(n_positions):
            attraction_at_idx = jax.random.bernoulli(rngs(), rel_probs[:, idx])
            attraction = attraction.at[:, idx].set(mask[:, idx] & attraction_at_idx)
            clicks = clicks.at[:, idx].set(examination[:, idx] & attraction[:, idx])

            if idx < n_positions - 1:
                sat_probs = jnp.where(clicks[:, idx], rel_probs[:, idx], 0.0)
                satisfaction = satisfaction.at[:, idx].set(
                    jax.random.bernoulli(rngs(), p=sat_probs)
                )

                continue_after_click_satisfied = clicks[:, idx] & satisfaction[:, idx]
                continue_after_click_not_satisfied = (
                    clicks[:, idx] & ~satisfaction[:, idx]
                )
                continue_after_no_click = examination[:, idx] & ~clicks[:, idx]

                continuation_probs = jnp.where(
                    continue_after_click_satisfied,
                    tau3,
                    jnp.where(
                        continue_after_click_not_satisfied,
                        tau2,
                        jnp.where(continue_after_no_click, tau1, 0.0),
                    ),
                )

                should_continue = jax.random.bernoulli(rngs(), p=continuation_probs)
                examination = examination.at[:, idx + 1].set(
                    should_continue & mask[:, idx + 1]
                )

        return ClickChainModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
            satisfaction=satisfaction,
        )

    def _get_log_probabilities(self, batch: Dict) -> Dict[str, Array]:
        rel_logits = self.attraction.logit(batch)
        rel_log_probs = logits_to_log_probs(rel_logits)
        non_rel_log_probs = logits_to_complement_log_probs(rel_logits)

        tau1_log_prob = self.continuation_exam_no_click.log_prob()
        tau2_log_prob = self.continuation_click_not_satisfied.log_prob()
        tau3_log_prob = self.continuation_click_satisfied.log_prob()

        return {
            "rel": rel_log_probs,
            "non_rel": non_rel_log_probs,
            "tau1": tau1_log_prob,
            "tau2": tau2_log_prob,
            "tau3": tau3_log_prob,
        }

    @staticmethod
    def _log_examination_after_click(
        rel_log_prob: Array,
        non_rel_log_prob: Array,
        tau2_log_prob: Array,
        tau3_log_prob: Array,
    ) -> Array:
        """
        Compute log examination probability after clicking.
        Formula: P(E_{r+1} = 1 | E_r = 1, C_r = 1) = α_r × τ3 + (1 - α_r) × τ2
        In log space: log[α_r × τ3 + (1 - α_r) × τ2]
        """
        satisfied_log = rel_log_prob + tau3_log_prob
        not_satisfied_log = non_rel_log_prob + tau2_log_prob
        return jnp.logaddexp(satisfied_log, not_satisfied_log)

    @staticmethod
    def _log_examination_after_no_click(
        current_exam_log_prob: Array,
        rel_log_prob: Array,
        non_rel_log_prob: Array,
        tau1_log_prob: Array,
    ) -> Array:
        """
        Compute log examination probability after not clicking.
        Formula: P(E_{r+1} = 1 | E_r = 1, C_r = 0) = [(1 - α_r) × ε_r × τ1] / [1 - α_r × ε_r]
        In log space: log(1 - α_r) + log ε_r + log τ1 - log(1 - α_r × ε_r)
        """
        numerator_log = non_rel_log_prob + current_exam_log_prob + tau1_log_prob
        denominator_log = log1mexp(rel_log_prob + current_exam_log_prob)
        return numerator_log - denominator_log

    @staticmethod
    def _log_examination_step(
        rel_log_prob: Array,
        non_rel_log_prob: Array,
        tau1_log_prob: Array,
        tau2_log_prob: Array,
        tau3_log_prob: Array,
    ) -> Array:
        """
        Compute one step of unconditional examination log probability.
        Formula: P(E_{r+1} = 1) = α × ((1-α) × τ2 + α × τ3) + (1-α) × τ1
        In log space: log[α × ((1-α) × τ2 + α × τ3) + (1-α) × τ1]
        """
        attraction_term = rel_log_prob + jnp.logaddexp(
            non_rel_log_prob + tau2_log_prob,
            rel_log_prob + tau3_log_prob,
        )
        non_attraction_term = non_rel_log_prob + tau1_log_prob

        return jnp.logaddexp(attraction_term, non_attraction_term)
