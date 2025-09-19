from typing import Dict, Optional

import jax.numpy as jnp
import jax.random
from flax import nnx
from flax import struct
from jax import Array

from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import ParameterConfig, init_parameter, Parameter
from clax.parameters.defaults import (
    default_continuation_config,
    default_attraction_config,
)
from clax.utils.math import (
    logits_to_log_probs,
    logits_to_complement_log_probs,
    log1mexp,
)


@struct.dataclass
class DependentClickModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


@struct.dataclass
class DependentClickModelConfig:
    attraction: ParameterConfig
    continuation: ParameterConfig


class DependentClickModel(ClickModel):
    """

    The Dependent Click Model (DCM) extends the cascade model to allow multiple clicks
    per session by introducing rank-dependent continuation probabilities.

    Args:
        positions (Optional[int], optional): Number of positions used to allocate
            continuation probabilities. This parameter is not used if a custom
            continuation module is provided.
        query_doc_pairs (Optional[int], optional): Number of query-document
            pairs to allocate in an embedding table. This parameter is not
            used if a custom attraction module is provided.
        attraction (Optional[Parameter | ParameterConfig], optional): Custom
            attraction parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        continuation (Optional[Parameter | ParameterConfig], optional): Custom
            continuation parameter deciding how likely a user is to continue their
            browsing session after clicking a document at the current position.
            Can be a parameter config or any subclass of the Parameter base class.
        rngs (nnx.Rngs): A NNX random number generator used for model
            initialization and stochastic operations.

    Examples:

        model = DependentClickModel(
            positions=10,
            query_doc_pairs=1_000_000,
            rngs=nnx.Rngs(42)
        )

    References:
        Fan Guo, Chao Liu, and Yi Min Wang.
        "Efficient multiple-click models in web search."
        In WSDM 2009.
    """

    name = "DCM"

    def __init__(
        self,
        positions: Optional[int] = None,
        query_doc_pairs: Optional[int] = None,
        attraction: Optional[Parameter | ParameterConfig] = None,
        continuation: Optional[Parameter | ParameterConfig] = None,
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
        self.continuation = init_parameter(
            "continuation",
            continuation,
            default_config_fn=default_continuation_config,
            default_config_args={"positions": positions},
            rngs=rngs,
        )

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

    def predict_relevance(self, batch: Dict) -> Array:
        return self.attraction.log_prob(batch)

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
