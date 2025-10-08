# Overview of Click Models

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

------------------------------

## Dependent Click Model (DCM)

::: clax.DependentClickModel
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3


#### Unconditional click probability

The DCM assumes that users examine a list from top to bottom, click on attractive items $\gamma_d$, and after clicking have a rank-dependent probability $\lambda_k$ to continue browsing:

$$
\begin{split}
    \log P(C=1 \mid d, k) &= \log(\epsilon_{k}) + \log(\gamma_{d})\\
    \log(\epsilon_{k+1}) &= \log(\epsilon_k) + \log(\gamma_{d_k} \lambda_k + (1 - \gamma_{d_k})).\\
\end{split}
$$

#### Conditional click probability

Examination in the DCM changes based on the observed clicks. If a user clicks on a document, they continue to the next rank with probability $\lambda_k$ and if they do not click, we calculate the posterior probability of examining the next rank given that we observed no click using Bayes' rule:

$$
\begin{split}
    \log P(C=1 \mid d, k, C_{<k}) &= \log(\epsilon_{k}) + \log(\gamma_{d})\\
    \log(\epsilon_{k+1}) &= \log\left(c_k \lambda_k + (1 - c_k) \frac{(1 - \gamma_{d_k}) \epsilon_k}{1 - \gamma_{d_k} \epsilon_k}\right).\\
\end{split}
$$

------------------------------

## Click Chain Model (CCM)

::: clax.ClickChainModel
    options:
      docstring_style: google
      show_source: true
      show_base_class: false
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

#### Unconditional click probability

The click chain model (CCM) extends the DCM assuming a total of three continuation scenarios that do not only explain continuation after clicking a document but also allow users to abandon a session without any clicks. First, $\tau_1$ is the probability of a user continuing to the next document after not clicking on the current document. Second, if the user clicks on the current document but is not satisfied, $\tau_2$ is the probability of the user continuing to the next position. And lastly, $\tau_3$ is the probability that a user clicks on the current item, finds it satisfying, but still wants to continue to the next document:

$$
\begin{split}
    \log P(C=1 \mid d, k) &= \log(\gamma_d) + \log(\epsilon_k) \\
    \log(\epsilon_{k+1}) &= \log(\epsilon_k) \\
    &\quad + \log\left( \gamma_{d_k}((1-\gamma_{d_k})\tau_2 + \gamma_{d_k}\tau_3) \right. \\
    &\quad \left. + (1-\gamma_{d_k})\tau_1 \right).
\end{split}
$$

#### Conditional click probability

When conditioning on the observed clicks, the update rule for the examination probability changes based on the user's action at the current rank. If a click occurred, we compute continuation based on satisfaction (equal to attractiveness $\gamma_d$) and the continuation probabilities $\tau_2$ and $\tau_3$. If no click was observed, we compute the posterior log probability of continuing to the next rank:

$$
\begin{split}
    \log P(C=1 \mid d, k, C_{<k}) &= \log(\gamma_d) + \log(\epsilon_k) \\
    \log(\epsilon_{k+1}) &= c_k \left[ \log\left(\gamma_{d_k}\tau_3 + (1-\gamma_{d_k})\tau_2 \right) \right] \\
    &\quad + (1-c_k) \left[ \log(1-\gamma_{d_k}) + \log(\epsilon_k) + \log(\tau_1) - \log(1 - \gamma_{d_k}\epsilon_k) \right].
\end{split}
$$

------------------------------

## Dynamic Bayesian Network (DBN)


