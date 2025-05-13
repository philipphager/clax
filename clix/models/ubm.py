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
        return -Bernoulli(predicted_clicks).log_prob(clicks).mean(where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        positions = batch["positions"]

        last_clicked_positions = _last_clicked_positions(positions, clicks)
        examination = self.predict_examination(positions, last_clicked_positions)
        relevance = self.relevance(batch)

        return examination * relevance

    def predict_clicks(self, batch: Dict):
        mask = batch["mask"]
        positions = batch["positions"]
        n_batch, n_positions = positions.shape

        click_probs = jnp.zeros((n_batch, n_positions))
        relevance = self.relevance(batch)

        def no_clicks_between(last_clicked_idx, current_idx, last_clicked_positions):
            prob = jnp.ones(n_batch)

            for idx in range(last_clicked_idx + 1, current_idx):
                examination = self.predict_examination(
                    positions[:, idx],
                    last_clicked_positions,
                )
                prob *= 1 - (examination * relevance[:, idx])

            return prob

        for idx in range(n_positions):
            rank_probs = jnp.zeros(n_batch)

            for last_clicked_idx in range(-1, idx):
                # We iterate over all possible last clicked positions
                # First, retrieve the click probability of the last clicked item

                if last_clicked_idx == -1:
                    # No previous click
                    last_clicked_positions = jnp.zeros(n_batch, dtype=positions.dtype)
                    last_click_prob = jnp.ones(n_batch)
                else:
                    last_clicked_positions = positions[:, last_clicked_idx]
                    last_click_prob = click_probs[:, last_clicked_idx]

                # Then, calculate the probability of no clicks between the last
                # clicked item and the current item:
                if idx > 0:
                    no_clicks_between_prob = no_clicks_between(
                        last_clicked_idx,
                        idx,
                        last_clicked_positions,
                    )
                else:
                    no_clicks_between_prob = jnp.ones(n_batch)

                # Finally, calculate the click probability at the current position,
                # conditioned on the last clicked item:
                examination = self.predict_examination(
                    positions[:, idx],
                    last_clicked_positions,
                )
                conditional_click_prob = mask[:, idx] * examination * relevance[:, idx]

                # Add contribution for each possible last clicked items:
                rank_probs += (
                    last_click_prob * no_clicks_between_prob * conditional_click_prob
                )

            click_probs = click_probs.at[:, idx].set(rank_probs)

        return click_probs

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        positions = batch["positions"]
        mask = batch["mask"]
        n_batch, n_positions = positions.shape

        clicks = jnp.zeros((n_batch, n_positions))
        last_clicked_positions = jnp.zeros(n_batch, dtype=positions.dtype)

        relevance = self.relevance(batch)

        for idx in range(n_positions):
            examination_idx = (
                positions[:, idx] * self.positions + last_clicked_positions
            )
            examination = self.examination({"examination_idx": examination_idx})

            click_probs = mask[:, idx] * examination * relevance[:, idx]
            clicks_at_position = jax.random.bernoulli(rngs(), click_probs)
            clicks = clicks.at[:, idx].set(clicks_at_position)

            last_clicked_positions = jnp.where(
                clicks_at_position > 0,
                positions[:, idx],
                last_clicked_positions,
            )

        return clicks

    def predict_examination(self, positions, last_clicked_positions):
        examination_idx = positions * self.positions + last_clicked_positions
        return self.examination({"examination_idx": examination_idx})
