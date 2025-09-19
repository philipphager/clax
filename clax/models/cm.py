from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clax.loss import binary_cross_entropy
from clax.models.base import ClickModel
from clax.parameters import ParameterConfig, init_parameter, Parameter
from clax.parameters.defaults import default_attraction_config
from clax.utils.math import logits_to_log_probs, logits_to_complement_log_probs


@struct.dataclass
class CascadeModelOutput:
    clicks: Array
    examination: Array
    attraction: Array


class CascadeModel(ClickModel):
    """
    The Cascade Model (CM) assumes that users scan results from top to bottom,
    click on the first attractive document they find, and then stop their search.

    Args:
        query_doc_pairs (Optional[int], optional): Number of query-document
            pairs to allocate in an embedding table. This parameter is not
            used if a custom attraction module is provided.
        attraction (Optional[Parameter | ParameterConfig], optional): Custom
            attraction parameter, which can be a parameter config or any
            subclass of the Parameter base class.
        rngs (nnx.Rngs): A NNX random number generator used for model
            initialization and stochastic operations.

    Examples:
        Creating a basic Cascade Model:

            model = CascadeModel(
                query_doc_pairs=1_000_000,
                rngs=nnx.Rngs(42),
            )

        Configure a deep network to user custom query-doc-features:

            attraction = DeepParameterConfig(
                use_feature="query_doc_features",
                features=16,
                layers=2,
                dropout=0.25,
            )
            model = CascadeModel(
                attraction=attraction,
                rngs=nnx.Rngs(42),
            )

    References:
        Nick Craswell, Onno Zoeter, Michael Taylor, and Bill Ramsey.
        "An experimental comparison of click position-bias models."
        In WSDM 2008.
    """

    name = "CM"

    def __init__(
        self,
        query_doc_pairs: Optional[int] = None,
        attraction: Optional[Parameter | ParameterConfig] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

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
        click_log_probs = self.predict_clicks(batch)

        # Discard clicks after the first click by setting them to a minimum log prob:
        no_clicks_before = self._no_clicks_before(batch["clicks"])
        click_log_probs = jnp.where(no_clicks_before, click_log_probs, jnp.log(1e-8))

        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def predict_clicks(self, batch: Dict) -> Array:
        attr_logits = self.attraction.logit(batch)

        # Compute log probabilities for relevance and non-relevance:
        attr_log_probs = logits_to_log_probs(attr_logits)
        non_attr_log_probs = logits_to_complement_log_probs(attr_logits)

        # Compute log examination, the first item is always examined:
        exam_log_probs = jnp.roll(non_attr_log_probs, shift=1, axis=-1)
        exam_log_probs = exam_log_probs.at[:, 0].set(0)
        exam_log_probs = jnp.cumsum(exam_log_probs, axis=-1)

        click_log_probs = exam_log_probs + attr_log_probs
        return jnp.where(batch["mask"], click_log_probs, -jnp.inf)

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        attr_probs = self.attraction.prob(batch)
        attraction = batch["mask"] & jax.random.bernoulli(rngs(), attr_probs)

        examination = self._no_clicks_before(attraction)
        clicks = examination & attraction

        return CascadeModelOutput(
            clicks=clicks,
            examination=examination,
            attraction=attraction,
        )

    def predict_relevance(self, batch: Dict) -> Array:
        return self.attraction.log_prob(batch)

    @staticmethod
    def _no_clicks_before(clicks):
        """
        Check if there are no clicks before each position.
        """
        clicks_before = jnp.cumsum(clicks, axis=-1) - clicks
        return clicks_before == 0
