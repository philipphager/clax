# Evaluation Metrics

CLAX supports evaluation metrics for click and relevance prediction. For click prediction, CLAX implements log-likelihood, and conditional and unconditional perplexity. To evaluate ranking performance, CLAX supports ranking metrics such as nDCG or MRR from the [RAX](https://rax.readthedocs.io/en/stable/index.html) library.   

CLAX follows the design of [FLAX NNX metrics](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/metrics.html), which updates multiple metrics at once. Each metric picks the respective input parameters it requires and updates its internal state:
```Python
from clax.metrics import (
    MultiMetric,
    LogLikelihood,
    Perplexity,
    ConditionalPerplexity
)

metrics = MultiMetric(
    **{
        "ll": LogLikelihood(),
        "ppl": Perplexity(),
        "cond_ppl": ConditionalPerplexity(),
    }
)

metrics.update(
    log_probs=log_probs,
    conditional_log_probs=cond_log_probs,
    clicks=clicks,
    where=mask,
)

results = metrics.compute()
rank_results = metrics.compute_per_rank()
```
Finally, you can compute the metric value by calling `metric.compute()` or `metric.compute_per_rank()` if you want to compute the mean metric value per positions.


## Click Metrics
### Log-likelihood
The most common metric for click prediction is the log-likelihood, measuring how well a model fits observed clicks:

$$
\operatorname{LL}(\mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(d, k, c) \in \mathcal{D}} \Big[ c \log \hat{c} + (1 - c) \log \left(1 - \hat{c} \right) \Big],
$$

where $\hat{c} = P(C = 1 \mid d, k, C_{<k})$ are a model's click predictions for a document $d$ at rank $k$, conditioned on clicks observed before the current rank $C_{<k}$. Log-likelihood values are negative, with higher values (closer to zero) indicating better model fit.


::: clax.metrics.LogLikelihood
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

### Conditional Perplexity
Perplexity can offer a more intuitive interpretation than log-likelihood. It measures how __surprised__ a model is by the observed data, with a lower value indicating a better model fit. Intuitively, it represents the weighted average number of choices a model is considering. Perfect predictions yield a perplexity of $1$, while random guessing for binary outcomes gives a perplexity of $2$, as the model is as uncertain as a coin flip. Conditional perplexity is defined as:

$$
\operatorname{PPL}(\mathcal{D}) = 2^{- \frac{1}{|\mathcal{D}|} \sum_{(d, k, c) \in \mathcal{D}} \Big[ c \log_2 \hat{c} + (1 - c) \log_2 \left(1 - \hat{c} \right) \Big]},
$$

where $\hat{c} = P(C=1 \mid d, k, C_{<k})$ are a model's click predictions for a document $d$ at rank $k$, conditioned on clicks observed before the current rank $C_{<k}$. Note that models which adopt their behavior based on clicks in the current search session might score better in conditional predictions.

::: clax.metrics.ConditionalPerplexity
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

### Perplexity

Similar to conditional perplexity, unconditional perplexity measures how __surprised__ a model is by the observed data, with a lower value indicating a better model fit. However, in contrast to conditional perplexity and log-likelihood, unconditional perplexity is calculated from click predictions that do not take clicks from the current user session into account. Unconditional perplexity is defined as: 

$$
\operatorname{PPL}(\mathcal{D}) = 2^{- \frac{1}{|\mathcal{D}|} \sum_{(d, k, c) \in \mathcal{D}} \Big[ c \log_2 \hat{c} + (1 - c) \log_2 \left(1 - \hat{c} \right) \Big]},
$$

where $\hat{c} = P(C=1 \mid d, k)$ are a model's unconditional click predictions for a document $d$ at rank $k$. Meaning, a model has to predict all clicks for the current session without access to any clicks on earlier ranks in the session. 

::: clax.metrics.Perplexity
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

## Ranking Metrics
Instead of re-implementing ranking metrics in CLAX, we opted to integrate with [RAX](https://rax.readthedocs.io/en/stable/index.html) the most popular ranking library in JAX. You can use [any ranking metric from RAX](https://rax.readthedocs.io/en/stable/api.html#module-rax._src.metrics) by wrapping it in a `RaxMetric` object. Note that below, `score` is the relevance prediction of a click model and `label` is typically an expert-annotated relevance label:

```Python
import rax
from clax.metrics import RaxMetric, MultiMetric

metrics = MultiMetric(
    **{
        "dcg@10": RaxMetric(rax.dcg_metric, top_n=10),
        "mrr@10": RaxMetric(rax.mrr_metric, top_n=10),
    }
)

metrics.update(scores=scores, labels=labels, where=mask)
results = metrics.compute()
```
