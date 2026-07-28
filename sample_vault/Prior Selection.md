---
title: Prior Selection
tags:
  - stats/bayesian
created: 2026-07-02
---

# Prior Selection

Choosing a prior is the most consequential and most debated step in Bayesian inference.

## Types of priors

- **Informative** — encodes genuine domain knowledge.
- **Weakly informative** — regularizes without dominating the data.
- **Non-informative / flat** — tries to "let the data speak"; often improper.

> [!warning] Flat priors are not always neutral
> A flat prior on one parameterization can be highly informative on another
> (Jeffreys' prior is one fix).

Related: [[Bayesian Reasoning]], [[Posterior Updating]].