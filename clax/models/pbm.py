from typing import Dict, Optional

import jax
import jax.numpy as jnp
from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import ParameterConfig, build_parameter
from clax.parameters.defaults import (
    default_examination_config,
    default_attraction_config,
)
from flax import nnx
from flax import struct
from jax import Array


@struct.dataclass
class PositionBasedModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


class PositionBasedModel(ClickModel):
    name = "PBM"

    def __init__(
        self,
        positions: Optional[int] = None,
        query_doc_pairs: Optional[int] = None,
        examination_config: Optional[ParameterConfig] = None,
        attraction_config: Optional[ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        exam_config = examination_config or default_examination_config(positions)
        attr_config = attraction_config or default_attraction_config(query_doc_pairs)
        self.examination = build_parameter(exam_config, rngs)
        self.attraction = build_parameter(attr_config, rngs)

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
