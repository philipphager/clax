import numpy as np
from typing import Dict, List

import jax
import jax.numpy as jnp
from flax import nnx
from flax import struct
from jax import Array

from clax.models.base import ClickModel


@struct.dataclass
class MixtureModelOutput:
    clicks: Array
    model_per_session: Array


class MixtureModel(ClickModel):
    name = "Mixture"

    def __init__(
        self,
        models: List[ClickModel],
        *,
        inverse_temperature: float = 1,
    ):
        super().__init__()
        self.models = models
        self.name = self._get_name()
        self.num_models = len(models)
        self.inverse_temperature = inverse_temperature
        self.model_prior_logits = nnx.Param(jnp.ones(self.num_models))

    def compute_loss(self, batch: Dict, aggregate: bool = True) -> Array:
        """Computes the negative log-likelihood loss for the mixture."""
        logits = self._get_posterior_log_probs(batch)
        loss = -nnx.logsumexp(logits, axis=1)
        return loss

    def _get_posterior_log_probs(self, batch: Dict) -> Array:
        """Compute the unnormalized log posterior"""
        prior_log_probs = self._get_prior_log_probs()
        session_loss_per_model = self._get_session_nll_per_model(batch)
        return -self.inverse_temperature * session_loss_per_model + prior_log_probs

    def _get_prior_log_probs(self) -> Array:
        """Compute a learnable log prior probability"""
        return jax.nn.log_softmax(self.model_prior_logits.value)

    def _get_session_nll_per_model(self, batch: Dict) -> Array:
        n_batch = batch["clicks"].shape[0]
        session_loss_per_model = jnp.zeros((n_batch, self.num_models))

        for i, model in enumerate(self.models):
            nll = model.compute_loss(batch, aggregate=False)
            session_loss = jnp.sum(nll, where=batch["mask"], axis=-1)
            session_loss_per_model = session_loss_per_model.at[:, i].set(session_loss)

        return session_loss_per_model

    def sample(self, batch: Dict, rngs: nnx.Rngs) -> MixtureModelOutput:
        n_batch = batch["positions"].shape[0]

        # Sample clicks from all models:
        model_clicks = jnp.stack(
            [model.sample(batch, rngs).clicks for model in self.models],
            axis=-1,
        )

        # Sample a given model per session:
        prior_logits = self.model_prior_logits.value
        model_idx = jax.random.categorical(
            key=rngs(),
            logits=prior_logits,
            shape=(n_batch,),
        )

        # Select clicks based on the sampled model:
        clicks = jnp.take_along_axis(model_clicks, model_idx[:, None, None], axis=-1)
        clicks = clicks.squeeze(axis=-1)

        return MixtureModelOutput(
            clicks=clicks,
            model_per_session=model_idx,
        )

    def predict_clicks(self, batch: Dict) -> Array:
        prior_log_probs = self._get_prior_log_probs()

        # Shape: (batch, positions, models)
        click_log_probs_per_model = jnp.stack(
            [model.predict_clicks(batch) for model in self.models], axis=-1
        )

        click_log_probs = click_log_probs_per_model + prior_log_probs
        click_log_probs = nnx.logsumexp(click_log_probs, axis=-1)

        return click_log_probs

    def predict_conditional_clicks(self, batch: Dict) -> Array:
        session_ll_per_model = -self._get_session_nll_per_model(batch)
        prior_log_probs = self._get_prior_log_probs()

        # Compute log-posterior for each model given observed sessions:
        posterior_log_probs = jax.nn.log_softmax(
            prior_log_probs + session_ll_per_model, axis=-1
        )

        # Shape: (batch, models) -> (batch, 1, models)
        posterior_log_probs = jnp.expand_dims(posterior_log_probs, axis=1)

        # Shape: (batch, positions, models)
        click_log_probs_per_model = jnp.stack(
            [model.predict_conditional_clicks(batch) for model in self.models],
            axis=-1,
        )

        click_log_probs = click_log_probs_per_model + posterior_log_probs
        click_log_probs = nnx.logsumexp(click_log_probs, axis=-1)

        return click_log_probs

    def predict_relevance(self, batch: Dict) -> Array:
        prior_log_probs = self._get_prior_log_probs()

        # Shape: (batch, positions, models)
        relevance_log_probs_per_model = jnp.stack(
            [model.predict_relevance(batch) for model in self.models], axis=-1
        )

        relevance_log_probs = relevance_log_probs_per_model + prior_log_probs
        return nnx.logsumexp(relevance_log_probs, axis=-1)

    def _get_name(self):
        model_names = ", ".join(
            [getattr(model, "name") for model in self.models if hasattr(model, "name")]
        )
        return f"Mixture ({model_names})"
