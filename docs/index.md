# Welcome to CLAX

## Cascade Model (CM)

::: clax.CascadeModel
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

#### Unconditional click probability

The probability of a click at rank $k$ depends on the displayed document $d$ being attractive $\gamma_d$ and all preceding documents being unattractive:

$$\log P(C=1 \mid d, k) = \log \gamma_d + \sum_{i=1}^{k-1} \log(1 - \gamma_{d_i})$$

#### Conditional click probability

The Cascade Model can only explain a single click per list. All other documents after the first click, by definition, have a click probability of $0$. To avoid a log-likelihood of $-\infty$ in the conditional click predictions, we assign a very small default click probability to all documents following a click:

$$
\log P(C=1 \mid d, k, C_{<k}) =
\begin{cases}
    \log \gamma_d & \text{if } \sum_{i=1}^{k-1} c_i = 0 \\
    \text{min_log_prob} & \text{otherwise}
\end{cases}
$$
