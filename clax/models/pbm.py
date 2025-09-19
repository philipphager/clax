from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import ParameterConfig, Parameter, init_parameter
from clax.parameters.defaults import (
    default_examination_config,
    default_attraction_config,
)


@struct.dataclass
class PositionBasedModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


class PositionBasedModel(ClickModel):
    """
    The Position-based model (PBM) assumes that users examine ranks independently
    and only click on examined and attractive results.

    Args:
        positions (Optional[int], optional): Number position embeddings to allocate.
            This parameter is not used if a custom examination module is provided.
        query_doc_pairs (Optional[int], optional): Number of query-document
            pairs to allocate in an embedding table. This parameter is not
            used if a custom attraction module is provided.
        examination (Optional[Parameter | ParameterConfig], optional): Custom
            examination/bias parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        attraction (Optional[Parameter | ParameterConfig], optional): Custom
            attraction/relevance parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        rngs (nnx.Rngs): A NNX random number generator used for model
            initialization and stochastic operations.

    Examples:
        Creating a basic Position-based Model:

            model = PositionBasedModel(
                positions=10,
                query_doc_pairs=1_000_000,
                rngs=nnx.Rngs(42)
            )

        Configure a two-tower model with a linear combination of bias features and a
        deep-cross network for document attraction:

            model = PositionBasedModel(
                examination=LinearParameterConfig(
                    use_feature="bias_features",
                    features=8,
                ),
                attraction=DeepCrossParameterConfig(
                    use_feature="query_doc_features",
                    features=136,
                    cross_layers=2,
                    deep_layers=2,
                    combination=Combination.STACKED,
                ),
                rngs=nnx.Rngs(42),
            )

    References:
        Matthew Richardson, Ewa Dominowska, and Robert Ragno.
        "Predicting Clicks: Estimating the Click-through Rate for New Ads."
        In WWW 2007.

        Nick Craswell, Onno Zoeter, Michael Taylor, and Bill Ramsey.
        "An experimental comparison of click position-bias models."
        In WSDM 2008.
    """

    name = "PBM"

    def __init__(
        self,
        positions: Optional[int] = None,
        query_doc_pairs: Optional[int] = None,
        examination: Optional[Parameter | ParameterConfig] = None,
        attraction: Optional[Parameter | ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.examination = init_parameter(
            "examination",
            examination,
            default_config_fn=default_examination_config,
            default_config_args={"positions": positions},
            rngs=rngs,
        )
        self.attraction = init_parameter(
            "attraction",
            attraction,
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
        exam_log_probs = self.examination.log_prob(batch)
        attr_log_probs = self.attraction.log_prob(batch)
        click_log_probs = exam_log_probs + attr_log_probs

        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def predict_relevance(self, batch: Dict) -> Array:
        return self.attraction.log_prob(batch)

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
