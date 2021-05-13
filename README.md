# Robust Learning of Fixed-Structure Bayesian Networks in Nearly-Linear Time.
A MATLAB implementation of [Robust Learning of Fixed-Structure Bayesian Networks in Nearly-Linear Time](https://arxiv.org/abs/2105.05555).

Explanation of Files
===

Our algorithms (`robust` directory)
---
* `main.m`: Evaluate the performance of different algorithms (MLE, [Filtering](https://arxiv.org/abs/1606.07384), and ours) when the graph structure of the ground-truth Bayes net is an empty graph, a random tree, or a random graph.
* Our algorithm uses robust mean estimations in a black-box manner. We provide two examples of such algorithms.
  * `robust_mean_pgd.m`: run gradient descent on a natural non-convex formulation (see [High-Dimensional Robust Mean Estimation via Gradient Descent](https://arxiv.org/abs/2005.01378)).
  * `robust_mean_filter.m`: run one-iteration of the Filtering algorithm (see [Robust Estimators in High Dimensions without the Computational Intractability](https://arxiv.org/abs/1604.06443)).
* We use the following basic functions in `main.m`.
  * `dtv_bn.m`: Estimate the total variation distance (via sampling) between two Bayes nets that have the same structure.
  * `empirical_cond_mean.m`: Compute the empirical conditional probabilities of a known-structure Bayes net.
  * `empirical_parental_prob.m`: Compute the empirical parental configuration probabilities of a known-structure Bayes net.

Reference
===
This repository is an implementation of the paper [Robust Learning of Fixed-Structure Bayesian Networks in Nearly-Linear Time](https://arxiv.org/abs/2105.05555) which appeared in ICLR 2021, authored by [Yu Cheng](https://homepages.math.uic.edu/~yucheng) and Honghao Lin.

```
@inproceedings{ChengL21,
  author    = {Yu Cheng and
               Honghao Lin},
  title     = {Robust Learning of Fixed-Structure {B}ayesian Networks in Nearly-Linear Time},
  booktitle = {Proceedings of the 9th International Conference on Learning Representations (ICLR)},
  publisher = {OpenReview.net},
  year      = {2021}
}
```
