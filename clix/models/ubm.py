import numpy as np
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from distrax import Bernoulli, Distribution
from flax import nnx
from jax import Array

from clix.models import utils
from clix.models.parameters import BernoulliEmbedding
from clix.models.utils import last_clicked_positions as _last_clicked_positions


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
        self.relevance = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )
        self.rngs = rngs

    def compute_loss(self, batch: Dict):
        clicks = batch["clicks"]
        predicted_clicks = self.predict_conditional_clicks(batch)
        return -predicted_clicks.log_prob(clicks).mean(where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        positions = batch["positions"]

        last_clicked_positions = _last_clicked_positions(positions, clicks)
        examination_idx = positions * self.positions + last_clicked_positions
        examination = self.examination({"examination_idx": examination_idx})
        relevance = self.relevance(batch)

        return examination * relevance

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        positions = batch["positions"]
        mask = batch["mask"]
        n_batch, n_positions = positions.shape

        clicks = jnp.zeros((n_batch, n_positions))
        last_clicked_positions = jnp.zeros(n_batch, dtype=positions.dtype)

        relevance = self.relevance(batch)

        for i in range(n_positions):
            examination_idx = positions[:, i] * self.positions + last_clicked_positions
            examination = self.examination({"examination_idx": examination_idx})

            click_probs = mask[:, i] * examination * relevance[:, i]
            clicks_at_position = jax.random.bernoulli(rngs(), click_probs)
            clicks = clicks.at[:, i].set(clicks_at_position)

            last_clicked_positions = jnp.where(
                clicks_at_position > 0,
                positions[:, i],
                last_clicked_positions,
            )

        return clicks

