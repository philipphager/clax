from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


class RandomClickModel(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ctr = BernoulliEmbedding(
            use_feature="ctr_idx",
            parameters=1,
            rngs=rngs,
        )

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        return self.predict_clicks(batch)

    def predict_clicks(self, batch: Dict) -> Array:
        ctr_idx = jnp.zeros_like(batch["query_doc_ids"])
        click_probs = self.ctr({"ctr_idx": ctr_idx})
        return batch["mask"] * click_probs

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.predict_clicks(batch)
        clicks = jax.random.bernoulli(rngs(), click_probs)
        clicks = clicks.astype(jnp.float32)
        return batch["mask"] * clicks


class RankBasedCTRModel(nnx.Module):
    def __init__(
        self,
        positions: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ctr = BernoulliEmbedding(
            use_feature="positions",
            parameters=positions + 1,
            rngs=rngs,
        )

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        return self.predict_clicks(batch)

    def predict_clicks(self, batch: Dict) -> Array:
        click_probs = self.ctr(batch)
        return batch["mask"] * click_probs

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.predict_clicks(batch)
        clicks = jax.random.bernoulli(rngs(), click_probs)
        clicks = clicks.astype(jnp.float32)
        return batch["mask"] * clicks


class DocumentBasedCTRModel(nnx.Module):
    def __init__(
        self,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ctr = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        return self.predict_clicks(batch)

    def predict_clicks(self, batch: Dict) -> Array:
        click_probs = self.ctr(batch)
        return batch["mask"] * click_probs

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.predict_clicks(batch)
        clicks = jax.random.bernoulli(rngs(), click_probs)
        clicks = clicks.astype(jnp.float32)
        return batch["mask"] * clicks
