from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array
from jax import lax

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


@struct.dataclass
class UserBrowsingModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


class UserBrowsingModel(nnx.Module):
    def __init__(
        self,
        positions: int,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.positions = positions
        self.examination = BernoulliEmbedding(
            use_feature="examination_idx",
            parameters=(positions + 1) ** 2,
            rngs=rngs,
        )
        self.attraction = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )
        self.rngs = rngs

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
        positions = batch["positions"]

        last_clicked_positions = self._last_clicked_positions(positions, clicks)
        exam_log_probs = self.examination.log_prob(self._examination_parameters(
            positions,
            last_clicked_positions,
        ))
        attr_log_probs = self.attraction.log_prob(batch)
        click_log_probs = exam_log_probs + attr_log_probs

        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict):
        pass

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
            exam_probs = self.examination.prob(self._examination_parameters(
                positions[:, idx],
                last_clicked_positions,
            ))
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

    @staticmethod
    def _last_clicked_positions(positions: Array, clicks: Array) -> Array:
        # Filter clicked positions, e.g.: [1, 2, 3, 4], [1, 0, 0, 1] -> [1, 0, 0, 4]
        clicked_positions = jnp.where(clicks == 1, positions, 0)
        # Find the last clicked position for each item: [1, 0, 0, 4] -> [1, 1, 1, 4]
        # Assumes positions are sorted in ascending order!
        clicked_positions = lax.cummax(clicked_positions, axis=1)
        # Shift the clicked positions to the right to align with the next item:
        clicked_positions = jnp.roll(clicked_positions, shift=1, axis=1)
        # Set the first position to 0, as there is no previously clicked position:
        return clicked_positions.at[:, 0].set(0)
