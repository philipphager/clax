from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array
from jax import lax

from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import ParameterConfig, init_parameter, Parameter
from clax.parameters.defaults import (
    default_attraction_config,
    default_ubm_examination_config,
)
from clax.utils.math import log1mexp


@struct.dataclass
class UserBrowsingModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


class UserBrowsingModel(ClickModel):
    """
    The UBM extends the PBM by making the examination probability depend on both the
    current position and the position of the last clicked document.

    Args:
        positions (Optional[int], optional): Number positions used to allocate a 2D
            embedding table of shape (positions, positions).
            This parameter is not used if a custom examination module is provided.
        query_doc_pairs (Optional[int], optional): Number of query-document
            pairs to allocate in an embedding table. This parameter is not
            used if a custom attraction module is provided.
        examination (Optional[Parameter | ParameterConfig], optional): Custom
            examination/bias parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        attraction (Optional[Parameter | ParameterConfig], optional): Custom
            attraction/relevance parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        rngs (nnx.Rngs): A NNX random number generator used for model
            initialization and stochastic operations.

    Examples:

        model = UserBrowsingModel(
            positions=10,
            query_doc_pairs=1_000_000,
            rngs=nnx.Rngs(42)
        )

    References:
        Georges E. Dupret and Benjamin Piwowarski.
        "A user browsing model to predict search engine click data from past observations."
        In SIGIR 2008.
    """

    name = "UBM"

    def __init__(
        self,
        positions: int,
        query_doc_pairs: Optional[int] = None,
        examination: Optional[Parameter | ParameterConfig] = None,
        attraction: Optional[Parameter | ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.positions = positions
        self.examination = init_parameter(
            "examination",
            examination,
            default_config_fn=default_ubm_examination_config,
            default_config_args={"positions": positions},
            rngs=rngs,
        )
        self.attraction = init_parameter(
            "attraction",
            attraction,
            default_config_fn=default_attraction_config,
            default_config_args={"query_doc_pairs": query_doc_pairs},
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
        positions = batch["positions"]

        last_clicked_positions = self._last_clicked_positions(positions, clicks)
        exam_log_probs = self.examination.log_prob(
            self._examination_parameters(
                positions,
                last_clicked_positions,
            )
        )
        attr_log_probs = self.attraction.log_prob(batch)
        click_log_probs = exam_log_probs + attr_log_probs

        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict):
        mask = batch["mask"]
        positions = batch["positions"]
        n_batch, n_positions = positions.shape

        click_log_probs = jnp.zeros((n_batch, n_positions))
        attr_log_probs = self.attraction.log_prob(batch)

        for current_idx in range(n_positions):
            scenario_log_probs = []

            for last_clicked_idx in range(-1, current_idx):
                # Each scenario represents one possible browsing history:
                # Predict clicks at the current_idx given the last clicked doc is at last_clicked_idx.
                last_click_log_prob = self._get_last_click_log_prob(
                    click_log_probs=click_log_probs,
                    last_clicked_idx=last_clicked_idx,
                )
                no_clicks_log_prob = self._compute_no_clicks_between_log_prob(
                    positions=positions,
                    attr_log_probs=attr_log_probs,
                    last_clicked_idx=last_clicked_idx,
                    current_idx=current_idx,
                )
                current_click_log_prob = self._compute_current_click_log_prob(
                    positions=positions,
                    attr_log_probs=attr_log_probs,
                    current_idx=current_idx,
                    last_clicked_idx=last_clicked_idx,
                )
                # The click probability of one scenario consists of:
                # The prob. of the last item to be clicked, no clicks between the
                # last item and the current item, and the current click probability.
                scenario_log_prob = (
                    last_click_log_prob + no_clicks_log_prob + current_click_log_prob
                )
                scenario_log_prob = jnp.where(
                    mask[:, current_idx], scenario_log_prob, -jnp.inf
                )
                scenario_log_probs.append(scenario_log_prob)

            # Marginalize over all scenarios:
            scenario_log_probs = jnp.stack(scenario_log_probs, axis=-1)
            scenario_log_probs = jax.scipy.special.logsumexp(
                scenario_log_probs,
                axis=-1,
            )
            click_log_probs = click_log_probs.at[:, current_idx].set(scenario_log_probs)

        return click_log_probs

    def predict_relevance(self, batch: Dict) -> Array:
        return self.attraction.log_prob(batch)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> UserBrowsingModelOutput:
        mask = batch["mask"]
        positions = batch["positions"]
        n_batch, n_positions = positions.shape

        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        examination = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        last_clicked_positions = jnp.zeros(n_batch, dtype=positions.dtype)

        attr_probs = self.attraction.prob(batch)
        attraction = mask & jax.random.bernoulli(rngs(), attr_probs)

        for idx in range(n_positions):
            exam_probs = self.examination.prob(
                self._examination_parameters(
                    positions[:, idx],
                    last_clicked_positions,
                )
            )
            examination_at_idx = jax.random.bernoulli(rngs(), p=exam_probs)
            examination = examination.at[:, idx].set(mask[:, idx] & examination_at_idx)
            clicks = clicks.at[:, idx].set(examination[:, idx] & attraction[:, idx])

            last_clicked_positions = jnp.where(
                clicks[:, idx],
                positions[:, idx],
                last_clicked_positions,
            )

        return UserBrowsingModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
        )

    def _examination_parameters(self, positions, last_clicked_positions):
        examination_idx = positions * self.positions + last_clicked_positions
        return {"examination_idx": examination_idx}

    def _get_last_click_log_prob(
        self,
        click_log_probs: Array,
        last_clicked_idx: int,
    ) -> Array:
        """
        Get log probability of the last click (or zero if no previous click).
        """
        if last_clicked_idx == -1:
            return jnp.zeros(click_log_probs.shape[0])
        else:
            return click_log_probs[:, last_clicked_idx]

    def _compute_no_clicks_between_log_prob(
        self,
        positions: Array,
        attr_log_probs: Array,
        last_clicked_idx: int,
        current_idx: int,
    ) -> Array:
        """
        Compute log probability of no clicks between last_clicked_idx and current_idx.
        """
        log_prob = jnp.zeros(positions.shape[0])

        for intermediate_idx in range(last_clicked_idx + 1, current_idx):
            intermediate_positions = positions[:, intermediate_idx]
            last_clicked_positions = self._get_last_clicked_positions(
                positions, last_clicked_idx
            )
            exam_log_prob = self.examination.log_prob(
                self._examination_parameters(
                    intermediate_positions, last_clicked_positions
                )
            )
            click_log_prob = exam_log_prob + attr_log_probs[:, intermediate_idx]
            no_click_log_prob = log1mexp(click_log_prob)
            log_prob += no_click_log_prob

        return log_prob

    def _compute_current_click_log_prob(
        self,
        positions: Array,
        attr_log_probs: Array,
        current_idx: int,
        last_clicked_idx: int,
    ) -> Array:
        """
        Compute log probability of click at current position given last clicked position.
        """
        # Get actual positions (not indices) for parameter lookup:
        current_positions = positions[:, current_idx]
        last_clicked_positions = self._get_last_clicked_positions(
            positions, last_clicked_idx
        )
        exam_log_prob = self.examination.log_prob(
            self._examination_parameters(current_positions, last_clicked_positions)
        )

        return exam_log_prob + attr_log_probs[:, current_idx]

    def _get_last_clicked_positions(
        self,
        positions: Array,
        last_clicked_idx: int,
    ) -> Array:
        """
        Get the actual position values for the last clicked index.
        """
        if last_clicked_idx == -1:
            return jnp.zeros(positions.shape[0], dtype=positions.dtype)
        else:
            return positions[:, last_clicked_idx]

    @staticmethod
    def _last_clicked_positions(positions: Array, clicks: Array) -> Array:
        """
        Find the position of the last clicked document for each position.
        Formula: r' = max{k ∈ {0,...,r-1} : c_k = 1}
        """
        # Filter clicked positions, e.g.: [1, 2, 3, 4], [1, 0, 0, 1] -> [1, 0, 0, 4]
        clicked_positions = jnp.where(clicks == 1, positions, 0)
        # Find the last clicked position for each item: [1, 0, 0, 4] -> [1, 1, 1, 4]
        # Assumes positions are sorted in ascending order!
        clicked_positions = lax.cummax(clicked_positions, axis=1)
        # Shift the clicked positions to the right to align with the next item:
        clicked_positions = jnp.roll(clicked_positions, shift=1, axis=1)
        # Set the first position to 0, as there is no previously clicked position:
        return clicked_positions.at[:, 0].set(0)
