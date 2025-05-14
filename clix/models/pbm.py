from typing import Dict

from flax import nnx
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
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

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        examination = self.examination(batch)
        relevance = self.relevance(batch)
        return batch["mask"] * examination * relevance

    def predict_clicks(self, batch: Dict) -> Array:
        return self.predict_conditional_clicks(batch)
