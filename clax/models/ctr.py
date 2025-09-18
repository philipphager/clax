from typing import Dict, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import (
    GlobalParameter,
    build_parameter,
    EmbeddingParameterConfig,
    ParameterConfig,
    FullEmbedding,
    init_parameter,
    Parameter,
)
from clax.parameters.defaults import (
    default_examination_config,
    default_attraction_config,
)


@struct.dataclass
class CTRModelOutput:
    clicks: Array


class GlobalCTRModel(ClickModel):
    """
    Global/Random Click Model (GCTR)

    Assumptions:
    - All documents have the same probability of being clicked

    References:
    - Chuklin et al. (2015). "Click models for web search"
    """

    name = "GCTR"

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.ctr = GlobalParameter(rngs=rngs)

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
        ctr_idx = jnp.zeros_like(batch["clicks"])
        click_log_probs = self.ctr.log_prob({"ctr_idx": ctr_idx})
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def predict_relevance(self, batch: Dict) -> Array:
        ctr_idx = jnp.zeros_like(batch["labels"])
        click_log_probs = self.ctr.log_prob({"ctr_idx": ctr_idx})
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> CTRModelOutput:
        ctr_idx = jnp.zeros_like(batch["positions"])
        click_probs = self.ctr.prob({"ctr_idx": ctr_idx})
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


class RankCTRModel(ClickModel):
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
        positions: Optional[int] = None,
        ctr: Optional[Parameter | ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ctr = init_parameter(
            "ctr",
            ctr,
            default_config_fn=default_examination_config,
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
        click_log_probs = self.ctr.log_prob(batch)
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def predict_relevance(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.ctr.prob(batch)
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


class DocumentCTRModel(ClickModel):
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
        query_doc_pairs: Optional[int] = None,
        ctr: Optional[Parameter | ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ctr = init_parameter(
            "ctr",
            ctr,
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
        click_log_probs = self.ctr.log_prob(batch)
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def predict_relevance(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        click_probs = self.ctr.prob(batch)
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)


class DocumentRankCTRModel(ClickModel):
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

    name = "DRCTR"

    def __init__(
        self,
        positions: int,
        query_doc_pairs: int,
        embedding_fn: Callable = FullEmbedding,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.positions = positions
        # Unify API in future version:
        self.ctr = build_parameter(
            EmbeddingParameterConfig(
                use_feature="ctr_idx",
                parameters=(query_doc_pairs * positions),
                embedding_fn=embedding_fn,
            ),
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
        ctr_idx = batch["query_doc_ids"] * self.positions + batch["positions"]
        click_log_probs = self.ctr.log_prob({"ctr_idx": ctr_idx})
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def predict_relevance(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        ctr_idx = batch["query_doc_ids"] * self.positions + batch["positions"]
        click_probs = self.ctr.prob({"ctr_idx": ctr_idx})
        clicks = batch["mask"] & jax.random.bernoulli(rngs(), click_probs)
        return CTRModelOutput(clicks=clicks)
