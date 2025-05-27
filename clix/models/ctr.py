from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


@struct.dataclass
class CTRModelOutput:
    clicks: Array


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
        return binary_cross_entropy(
            y_predict,
            y_true,
            where=batch["mask"],
            log_probs=True,
        )

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        ctr_idx = jnp.zeros_like(batch["query_doc_ids"])
        click_log_probs = self.ctr.log_prob({"ctr_idx": ctr_idx})
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> CTRModelOutput:
        ctr_idx = jnp.zeros_like(batch["query_doc_ids"])
        click_probs = self.ctr.prob({"ctr_idx": ctr_idx})
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


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
        return binary_cross_entropy(
            y_predict,
            y_true,
            where=batch["mask"],
            log_probs=True,
        )

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        click_log_probs = self.ctr.log_prob(batch)
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.ctr.prob(batch)
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


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
        return binary_cross_entropy(
            y_predict,
            y_true,
            where=batch["mask"],
            log_probs=True,
        )

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        click_log_probs = self.ctr.log_prob(batch)
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.ctr.prob(batch)
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)

class DocumentRankBasedCTRModel(nnx.Module):
    def __init__(
        self,
        query_doc_pairs: int,
        positions: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.positions = positions
        self.ctr = BernoulliEmbedding(
            use_feature="ctr_idx",
            parameters=(query_doc_pairs * positions),
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
        ctr_idx = batch["query_doc_ids"] * self.positions + batch["positions"]
        click_log_probs = self.ctr.log_prob({"ctr_idx": ctr_idx})
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        ctr_idx = batch["query_doc_ids"] * self.positions + batch["positions"]
        click_probs = self.ctr.prob({"ctr_idx": ctr_idx})
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)
