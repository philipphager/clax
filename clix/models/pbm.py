from typing import Dict

import jax
from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.math import exp_logp
from clix.models.parameters import BernoulliEmbedding


class PositionBasedModel(nnx.Module):
    def __init__(
        self,
        positions: int,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.log_examination = BernoulliEmbedding(
            use_feature="positions",
            parameters=positions + 1,
            log_prob=True,
            rngs=rngs,
        )
        self.log_relevance = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            log_prob=True,
            rngs=rngs,
        )

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_click_log_probs(batch)
        return binary_cross_entropy(
            y_predict, y_true, where=batch["mask"], log_probs=True
        )

    def predict_click_log_probs(self, batch: Dict) -> Array:
        log_examination = self.log_examination(batch)
        log_relevance = self.log_relevance(batch)
        return log_examination + log_relevance

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        log_probs = self.predict_click_log_probs(batch)
        return exp_logp(log_probs, where=batch["mask"])

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        probs = self.predict_clicks(batch)
        return jax.random.bernoulli(rngs(), p=probs)
