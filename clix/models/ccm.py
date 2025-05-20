import jax.numpy as jnp
import jax

from typing import Dict

from flax import nnx
from flax import struct
from jaxlib.xla_extension import Array

from clix.models.loss import binary_cross_entropy
from clix.models.parameters import BernoulliParameter, BernoulliEmbedding


@struct.dataclass
class ClickChainModelOutput:
    clicks: Array
    examination: Array
    attraction: Array
    satisfaction: Array


class ClickChainModel(nnx.Module):
    def __init__(
        self,
        query_doc_pairs: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        # In the CCM, both attraction and satisfaction are modeled as the same
        # variable, so we call it relevance here:
        self.relevance = BernoulliEmbedding(
            use_feature="query_doc_ids",
            parameters=query_doc_pairs,
            rngs=rngs,
        )
        self.continuation_exam_no_click = BernoulliParameter(rngs=rngs)
        self.continuation_click_satisfied = BernoulliParameter(rngs=rngs)
        self.continuation_click_not_satisfied = BernoulliParameter(rngs=rngs)

    def compute_loss(self, batch: Dict):
        y_true = batch["clicks"]
        y_predict = self.predict_conditional_clicks(batch)
        return binary_cross_entropy(y_predict, y_true, where=batch["mask"])

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        clicks = batch["clicks"]
        n_batch, n_positions = clicks.shape

        relevance = self.relevance(batch)
        tau1 = self.continuation_exam_no_click()
        tau2 = self.continuation_click_not_satisfied()
        tau3 = self.continuation_click_satisfied()

        # First position is always examined
        examination = jnp.ones((n_batch, n_positions))
        EPS = 1e-10

        for idx in range(n_positions - 1):
            continue_satisfied = relevance[:, idx] * tau3
            continue_not_satisfied = (1 - relevance[:, idx]) * tau2
            continue_after_click = continue_satisfied + continue_not_satisfied

            numerator = (1 - relevance[:, idx]) * examination[:, idx] * tau1
            denominator = 1 - relevance[:, idx] * examination[:, idx] + EPS
            continue_after_no_click = numerator / denominator

            examination = examination.at[:, idx + 1].set(
                jnp.where(
                    clicks[:, idx],
                    continue_after_click,
                    continue_after_no_click,
                )
            )

        return batch["mask"] * examination * relevance

    def predict_clicks(self, batch: Dict) -> Array:
        relevance = self.relevance(batch)
        tau1 = self.continuation_exam_no_click()
        tau2 = self.continuation_click_not_satisfied()
        tau3 = self.continuation_click_satisfied()

        examination_attractive = relevance * ((1 - relevance) * tau2 + relevance * tau3)
        examination_not_attractive = (1 - relevance) * tau1
        examination = examination_attractive + examination_not_attractive
        examination = jnp.roll(examination, shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        examination = jnp.cumprod(examination, axis=-1)

        return batch["mask"] * examination * relevance

    def sample_clicks(self, batch: Dict, rngs: nnx.Rngs) -> Array:
        relevance = self.relevance(batch)
        tau1 = self.continuation_exam_no_click()
        tau2 = self.continuation_click_not_satisfied()
        tau3 = self.continuation_click_satisfied()
        mask = batch["mask"]

        n_batch, n_positions = relevance.shape
        is_clicked = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        is_examined = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        is_attractive = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)
        is_satisfied = jnp.zeros((n_batch, n_positions), dtype=jnp.bool_)

        # If valid, always examine the first item:
        is_examined = is_examined.at[:, 0].set(batch["mask"][:, 0])

        for pos in range(n_positions):
            is_attractive = is_attractive.at[:, pos].set(
                jax.random.bernoulli(rngs(), p=relevance[:, pos])
            )
            is_clicked = is_clicked.at[:, pos].set(
                mask[:, pos] & is_examined[:, pos] & is_attractive[:, pos]
            )

            if pos < n_positions - 1:
                is_satisfied = is_satisfied.at[:, pos].set(
                    is_clicked[:, pos]
                    & jax.random.bernoulli(rngs(), p=relevance[:, pos])
                )

                continuation_prob = jnp.where(
                    is_examined[:, pos],
                    jnp.where(
                        is_clicked[:, pos],
                        jnp.where(
                            is_satisfied[:, pos],
                            tau3,  # Continue if clicked and satisfied
                            tau2,  # Continue if clicked and not satisfied
                        ),
                        tau1,  # Continue if no click
                    ),
                    0.0,  # Don't continue if previous position was not examined
                )

                is_examined = is_examined.at[:, pos + 1].set(
                    mask[:, pos + 1] & jax.random.bernoulli(rngs(), continuation_prob)
                )

        return ClickChainModelOutput(
            clicks=is_clicked,
            examination=is_examined,
            attraction=is_attractive,
            satisfaction=is_satisfied,
        )
