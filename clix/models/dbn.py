from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


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
        self.continuation = BernoulliEmbedding(
            use_feature="continuation_idx",
            parameters=1,
            rngs=rngs,
        )

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        n_batch, n_positions = clicks.shape

        examination = jnp.ones((n_batch, n_positions))
        attraction = self.attraction(batch)
        satisfaction = self.satisfaction(batch)
        continuation = self._predict_continuation(batch)

        # Calculate examination probabilities based on observed clicks:
        for idx in range(n_positions - 1):
            click_prob = examination[:, idx] * attraction[:, idx]
            no_click_prob = 1 - click_prob

            examined_and_not_relevant = (
                examination[:, idx] * (1 - attraction[:, idx]) * continuation[:, idx]
            )
            # Not satisfied users will continue examination with continuation probability:
            examination_after_click = (1 - satisfaction[:, idx]) * continuation[:, idx]
            # Users examined but were not attracted (also not satisfied) with the current doc:
            examination_after_no_click = examined_and_not_relevant / no_click_prob
            examination = examination.at[:, idx + 1].set(
                jnp.where(
                    clicks[:, idx],
                    examination_after_click,
                    examination_after_no_click,
                )
            )

        return batch["mask"] * examination * attraction

    def predict_clicks(self, batch: Dict) -> Array:
        attraction = self.attraction(batch)
        satisfaction = self.satisfaction(batch)
        continuation = self._predict_continuation(batch)

        examination = continuation * (
            attraction * (1 - satisfaction) + (1 - attraction)
        )
        examination = jnp.roll(examination, shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        examination = jnp.cumprod(examination, axis=-1)

        return batch["mask"] * examination * attraction

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        attraction = self.attraction(batch)
        satisfaction = self.satisfaction(batch)
        continuation = self._predict_continuation(batch)
        n_batch, n_positions = attraction.shape

        clicks = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        is_examined = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        is_satisfied = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        should_continue = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)

        # Always examine the first item (if valid)
        is_examined = is_examined.at[:, 0].set(batch["mask"][:, 0])

        for idx in range(n_positions):
            click_probs = attraction[:, idx] * is_examined[:, idx]
            clicks = (
                clicks.at[:, idx]
                .set(jax.random.bernoulli(rngs(), p=click_probs))
                .astype(jnp.bool_)
            )

            if idx < n_positions - 1:
                # Sample user satisfaction, non-clicked items are never satisfactory:
                satisfaction_probs = clicks[:, idx] * satisfaction[:, idx]
                is_satisfied = is_satisfied.at[:, idx].set(
                    jax.random.bernoulli(rngs(), p=satisfaction_probs)
                )

                # Sample user continuation:
                # - Users continue when not satisfied after click
                # - Users continue when examined but the item is not attractive/clicked
                continue_after_click = clicks[:, idx] & ~is_satisfied[:, idx]
                continue_without_click = is_examined[:, idx] & ~clicks[:, idx]
                continuation_probs = continuation[:, idx] * (
                    continue_after_click | continue_without_click
                )
                should_continue = should_continue.at[:, idx].set(
                    jax.random.bernoulli(rngs(), p=continuation_probs)
                )
                is_examined = is_examined.at[:, idx + 1].set(
                    should_continue[:, idx] & batch["mask"][:, idx + 1]
                )

        clicks = clicks.astype(jnp.float32)
        return batch["mask"] * clicks

    def _predict_continuation(self, batch):
        if self.fix_continuation:
            # Users always continues when not satisfied:
            return jnp.ones_like(batch["positions"], dtype=jnp.float32)
        else:
            continuation_idx = jnp.zeros_like(batch["positions"])
            return self.continuation({"continuation_idx": continuation_idx})
