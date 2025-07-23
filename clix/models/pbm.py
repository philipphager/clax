from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clix.models.base import ClickModel
from clix.models.loss import binary_cross_entropy
from clix.parameters import ParameterConfig, build_parameter


@struct.dataclass
class PositionBasedModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


@struct.dataclass
class PositionBasedModelConfig:
    examination: ParameterConfig
    attraction: ParameterConfig


class PositionBasedModel(ClickModel):
    """
    Position-Based Model (PBM)

    The PBM assumes uses click when they observed the position of an item and
    the displayed document is attractive/relevant.

    Assumptions:
    - A click occurs if and only if a document is examined and attractive
    - Examination probability depends only on document position
    - Attraction probability depends only on query-document pair
    - Examination and attraction are independent events
    - No sequential behavior (unlike cascade-based models)

    References:
    - Richardson et al. (2007). "Predicting clicks: estimating the click-through rate for new ads"
    - Craswell et al. (2008). "An experimental comparison of click position-bias models"
    """
    name = "PBM"

    def __init__(
        self,
        config: PositionBasedModelConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.examination = build_parameter(config.examination, rngs)
        self.attraction = build_parameter(config.attraction, rngs)

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
        exam_log_probs = self.examination.log_prob(batch)
        attr_log_probs = self.attraction.log_prob(batch)
        click_log_probs = exam_log_probs + attr_log_probs

        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> PositionBasedModelOutput:
        exam_probs = self.examination.prob(batch)
        attr_probs = self.attraction.prob(batch)

        examination = batch["mask"] & jax.random.bernoulli(rngs(), p=exam_probs)
        attraction = batch["mask"] & jax.random.bernoulli(rngs(), p=attr_probs)
        clicks = examination & attraction

        return PositionBasedModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
        )
