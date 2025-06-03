from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliEmbedding


@struct.dataclass
class CTRModelOutput:
    clicks: Array


class GlobalClickModel(nnx.Module):
    """
    Global/Random Click Model (GCTR)

    Assumptions:
    - All documents have the same probability of being clicked

    References:
    - Chuklin et al. (2015). "Click models for web search"
    """
    name = "GCTR"

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

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> CTRModelOutput:
        ctr_idx = jnp.zeros_like(batch["query_doc_ids"])
        click_probs = self.ctr.prob({"ctr_idx": ctr_idx})
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


class RankBasedCTRModel(nnx.Module):
    """
    Rank-based Click-Through Rate Model (RCTR).

    Models click probability as dependent only on document position/rank.
    Captures position bias where higher-ranked documents get more clicks
    regardless of their relevance.

    Assumptions:
    - Click probability depends only on document rank
    - Clicks are independent across positions
    - All documents at same rank have identical click probability

    References:
    - Chuklin et al. (2015). "Click models for web search"
    """
    name = "RCTR"

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

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.ctr.prob(batch)
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


class DocumentBasedCTRModel(nnx.Module):
    """
    Document-based Click-Through Rate Model (DCTR).

    Clicks depend only on the relevance of each query-document pair,
    ignoring position effects.

    Assumptions:
    - Click probability depends only on query-document pair
    - Clicks are independent across positions
    - No examination or position bias modeling

    References:
    - Chuklin et al. (2015). "Click models for web search"
    """
    name = "DCTR"

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
    """
    Document-Rank based Click-Through Rate Model (RDCTR).

    Models click probability based on both query-document pair and position.
    Prone to overfitting due to large number of parameters.

    Assumptions:
    - Click probability depends on both query-document pair and rank
    - Clicks are independent across positions

    References:
    - Deffayet et al. (2023). "Evaluating the robustness of click models to policy distributional shift"
    """
    name = "RDCTR"

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

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        ctr_idx = batch["query_doc_ids"] * self.positions + batch["positions"]
        click_probs = self.ctr.prob({"ctr_idx": ctr_idx})
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)
