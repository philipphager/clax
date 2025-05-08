from typing import Dict

from distrax import Bernoulli, Distribution
from flax import nnx

from clix.models.parameters import BernoulliEmbedding, BetaEmbedding


class PositionBasedModel(nnx.Module):
    def __init__(
        self,
        positions: int,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.examination = BernoulliEmbedding(
            use_feature="positions",
            parameters=positions,
            rngs=rngs,
        )
        self.relevance = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )

    def __call__(self, batch: Dict) -> Distribution:
        examination = self.examination(batch)
        relevance = self.relevance(batch)
        return Bernoulli(probs = examination * relevance)

    def log_loss(self, batch: Dict):
        clicks = batch["clicks"]
        predicted_clicks = self(batch)
        return -predicted_clicks.log_prob(clicks).mean(where=batch["mask"])
