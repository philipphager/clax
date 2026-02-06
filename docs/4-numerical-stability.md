# Numerical Stability

Optimizing complex likelihood expressions using gradient descent requires attention to numerical stability. The marginal likelihoods of many common click models contain products of small probabilities, which can lead to numerical underflow in finite-precision computer arithmetic. Below, we cover the techniques CLAX uses to stabilize complex likelihood expressions by performing all probability computations in log-space.

## Multiplication
By moving to log-probabilities, products of probabilities simplify to sums (and division to subtraction):

$$
    \log \left( \prod_{i = 1}^{n} p_i \right) = \sum_{i = 1}^{n} \log p_i,
$$

which essentially eliminates the concern of numerical underflow when multiplying small probabilities.

## Addition
While multiplication becomes stable (and faster) in log-space, the addition of probabilities becomes more complicated as it requires first exponentiating log probabilities. This reintroduces the problems we seek to avoid, as exponentiating large positive inputs lead to overflow and exponentiating large negative inputs lead to underflow. The standard solution is to avoid large inputs to the $\exp(\cdot)$ operation via the log-sum-exp trick:[^1]

$$
\texttt{log_sum_exp}(a) = a_{\text{max}} + \log \left( \sum_{i=1}^{n} \exp(a_i - a_{\text{max}}) \right),
$$

where $a = (a_1, \dots, a_n)$ is a vector of log values and $a_{\text{max}} = \max_i(a_i)$ is the maximum input value. The trick is prevalent in probabilistic modeling, and we also use it to transform the output logits of neural networks $x \in \mathbb{R}$ to log-probabilities by implementing numerically stable versions of the log-sigmoid functions:

$$
    \begin{split}
        \log(\sigma(x)) &= - \texttt{log_sum_exp}([0, -x]) \text{, or }\\
        \log(1 - \sigma(x)) &= - \texttt{log_sum_exp}([0, x]).\\
    \end{split}
$$

## Complements and cancellation
Sometimes we need to compute the log of a complement $\log(1 - p)$, e.g., in the binary-cross entropy loss or when computing [log-posteriors in the DBN](2-models.md#dynamic-bayesian-network-dbn). Performing this step directly from log-probability $\log p$ requires computing: $\log(1 - \exp(\log p))$.

This expression is numerically unstable in two ways: (i) underflow: when $p$ is very small, $\log p$ is very negative, causing $\exp(\log p)$ to underflow to zero; and (ii) catastrophic cancellation: when $p \approx 1$, we have $\exp(\log p) \approx 1$, making $1 - \exp(\log p) \approx 0$, since subtracting nearly equal floating point numbers leads to a loss of precision.[^2]

Therefore, we compute $\texttt{log1mexp}(x)$ as proposed by Mächler[^3] and adopted by major frameworks such as [TensorFlow](https://www.tensorflow.org/probability/api_docs/python/tfp/math/log1mexp) and [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log1mexp.html). Mächler proposes a piecewise approximation that switches between two stable expressions that are precise in different input ranges.[^4] For a log-probability $a \in \mathbb{R}, a \leq 0$:

$$
    \texttt{log1mexp}(a) =
    \begin{cases}
        \log(-\text{expm1}(a)) & \text{if } a > -\log(2) \\
        \text{log1p}(- \exp(a)) & \text{if } a \le -\log(2).
    \end{cases}
$$

The implementation relies on the standard functions $\text{log1p}(x)$, which accurately computes $\log(1 + x)$, and $\text{expm1}(x)$, which accurately computes $\exp(x) - 1$, to avoid catastrophic cancellation.

To summarize, CLAX performs all probability computations in log space for increased numerical stability, avoiding underflow and overflow as well as catastrophic cancellation. We list all models and their corresponding log-likelihood [here](2-models.md).

[^1]: Pierre Blanchard, Desmond J. Higham, and Nicholas J. Higham. "Accurate Computation of the Log-Sum-Exp and Softmax Functions". arXiv preprint arXiv:1909.03469, 2019.
[^2]: David Goldberg. "What Every Computer Scientist Should Know about Floating-Point Arithmetic". In ACM Computing Surveys, 1999.
[^3]: Martin Mächler. "Accurately Computing $\log(1 - \exp(-|a|))$ Assessed by the Rmpfr package". In The Comprehensive R Archive Network, 2012.
[^4]: Interested readers can find the motivation behind the switching point $\log(2)$ under [Section 2](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf).