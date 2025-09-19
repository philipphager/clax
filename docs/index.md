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

$$\log P(C=1 \mid d, k) = \log \gamma_d + \sum_{i=1}^{k-1} \log(1 - \gamma_{d_i}).$$

#### Conditional click probability

The Cascade Model can only explain a single click per list. All other documents after the first click, by definition, have a click probability of $0$. To avoid a log-likelihood of $-\infty$ in the conditional click predictions, we assign a very small default click probability to all documents following a click:

$$
\log P(C=1 \mid d, k, C_{<k}) =
\begin{cases}
    \log \gamma_d & \text{if } \sum_{i=1}^{k-1} c_i = 0 \\
    \text{min_log_prob} & \text{otherwise}.
\end{cases}
$$

------------------------------

## Position-based Model (PBM)

::: clax.PositionBasedModel
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

#### (Un)conditional click probability

The PBM assumes that clicks occurs only if a user first examines the result at rank k with probability $\theta_k$ and if the displayed document $d$ is attractive $\gamma_d$:

$$\log P(C = 1 \mid d, k) = \log \theta_k + \log \gamma_{d}.$$

------------------------------

## User Browsing Model (UBM)

::: clax.UserBrowsingModel
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

#### Unconditional click probability

As examination in the UBM depends on the last clicked position, predicting unconditional clicks on a new list of documents requires marginalizing over all possible last click positions $i < k$ before our current position:

$$\log P(C = 1 \mid d, k) = \log \left( \sum_{i=0}^{k - 1} P(C=1 \mid d_i, i) \cdot \left(\prod_{j=i+1}^{k - 1} (1 - \theta_{j,i}\gamma_{d_j})\right)  \theta_{k,i}\gamma_{d} \right).$$

Each term in the sum represents a path to the current document: the probability of clicking at a previous rank $i$, then not clicking on anything until rank $k$, and finally examining and clicking the document at rank $k$ given $i$ was the last clicked position.

#### Conditional click probability

The UBM assumes that examination at position $k$ depends also on the position of the last clicked document $k'$. Similar to the PBM, users only click on examined and attractive documents:

$$\log P(C=1 \mid d, k, C_{<k}) = \log \theta_{k, k'} + \log \gamma_{d}.$$
