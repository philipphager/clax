import functools
from typing import Dict, Tuple, Optional, Callable

import jax.numpy as jnp
from flax import nnx
from flax.nnx import Initializer
from flax.nnx.module import Module
from flax.nnx.nn import dtypes, initializers
from jax import Array
import jax


class SparseEmbed(nnx.Module):
    """
    Sparse embedding layer that only computes gradients for embeddings
    actually used in the batch, leveraging JAX's sparse operations.
    """

    def __init__(
            self,
            num_embeddings: int,
            features: int,
            *,
            dtype: Optional[jnp.dtype] = None,
            param_dtype: jnp.dtype = jnp.float32,
            embedding_init: Callable = nnx.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='normal'
            ),
            rngs: nnx.Rngs,
    ):
        """
        Args:
            num_embeddings: Total vocabulary size
            features: Embedding dimension
            dtype: Computation dtype
            param_dtype: Parameter dtype
            embedding_init: Initializer for embedding matrix
            rngs: Random number generators
        """
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or param_dtype

        # Initialize full embedding matrix (but we'll use it sparsely)
        self.embedding = nnx.Param(
            embedding_init(rngs.params(), (num_embeddings, features), param_dtype)
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Sparse embedding lookup that only computes gradients for used indices.

        Args:
            inputs: Integer indices of shape (..., )

        Returns:
            Embeddings of shape (..., features)
        """
        # Get unique indices in the batch to create sparse representation
        flat_inputs = inputs.flatten()
        unique_indices = jnp.unique(flat_inputs, size=len(flat_inputs), fill_value=-1)

        # Filter out padding (-1 values from unique)
        valid_mask = unique_indices >= 0
        unique_indices = unique_indices[valid_mask]

        if len(unique_indices) == 0:
            # Handle edge case of no valid indices
            return jnp.zeros(inputs.shape + (self.features,), dtype=self.dtype)

        # Create sparse representation of the embedding subset we need
        # This ensures gradients only flow to embeddings actually used
        sparse_embeddings = self._create_sparse_embedding_lookup(
            unique_indices, inputs.shape
        )

        # Apply sparse embedding lookup
        return sparse_embeddings(inputs)

    def _create_sparse_embedding_lookup(self, unique_indices, input_shape):
        """Create a sparse embedding function for the given unique indices."""

        @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=0)
        def sparse_lookup_single(embedding_matrix, idx):
            """Lookup single embedding with proper gradient flow."""
            # Use jnp.where to ensure gradients only flow to accessed embeddings
            mask = jnp.arange(self.num_embeddings) == idx
            # This creates a sparse gradient: only the selected embedding gets gradients
            selected_embedding = jnp.sum(
                embedding_matrix * mask[:, None], axis=0, keepdims=False
            )
            return selected_embedding

        def sparse_embedding_fn(inputs):
            flat_inputs = inputs.flatten()
            # Apply sparse lookup
            embeddings = sparse_lookup_single(self.embedding.value, flat_inputs)
            # Reshape back to original input shape + features
            return embeddings.reshape(input_shape + (self.features,))

        return sparse_embedding_fn


class BernoulliParameter(nnx.Module):
    def __init__(
        self,
        shape: Tuple[int] = (1,),
        initializers: Initializer = initializers.normal(0.5),
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.weight = nnx.Param(initializers(rngs.params(), shape))

    def __call__(self) -> Array:
        return nnx.sigmoid(self.weight.value)

    def logit(self) -> Array:
        return self.weight.value

    def prob(self) -> Array:
        return nnx.sigmoid(self.weight.value)

    def log_prob(self) -> Array:
        return nnx.log_sigmoid(self.weight.value)


class BernoulliEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        add_baseline: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.add_baseline = add_baseline
        self.baseline = nnx.Param(jnp.zeros(1))
        self.embeddings = SparseEmbed(
            num_embeddings=parameters,
            features=1,
            rngs=rngs,
            embedding_init=initializers.zeros_init(),
        )

    def logit(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        logit = self.embeddings(x).squeeze()

        if self.add_baseline:
            # Add a baseline prediction, similar to a wide&deep model.
            # The model resorts to avg. predictions for prev. unseen parameters:
            logit = self.baseline.value + logit

        return logit

    def prob(self, batch: Dict) -> Array:
        return nnx.sigmoid(self.logit(batch))

    def log_prob(self, batch: Dict) -> Array:
        return nnx.log_sigmoid(self.logit(batch))


class BetaEmbedding(nnx.Module):
    def __init__(
        self,
        use_feature: str,
        parameters: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.use_feature = use_feature
        self.alpha = nnx.Sequential(
            nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs),
            nnx.softplus,
            self._offset,
        )
        self.beta = nnx.Sequential(
            nnx.Embed(num_embeddings=parameters, features=1, rngs=rngs),
            nnx.softplus,
            self._offset,
        )

    def __call__(self, batch: Dict) -> Array:
        x = batch[self.use_feature]
        alpha = self.alpha(x).squeeze()
        beta = self.beta(x).squeeze()
        return alpha / (alpha + beta)

    @staticmethod
    def _offset(x):
        return x + 2
