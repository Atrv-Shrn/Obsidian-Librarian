---
title: Bayesian Reasoning
aliases:
  - Bayes
tags:
  - stats/bayesian
  - probability
created: 2026-07-01
---

# Bayesian Reasoning

> [!note] Core idea
> Bayesian reasoning updates a **prior** belief with new **evidence** to get a
> **posterior**, via [[Bayes' Theorem]].

## Bayes' Theorem

$$
P(H \mid E) = \frac{P(E \mid H)\, P(H)}{P(E)}
$$

- $P(H)$ — prior, $P(E \mid H)$ — likelihood, $P(H \mid E)$ — posterior, $P(E)$ — evidence.

See also [[Prior Selection]] and [[Posterior Updating]].

## Why it matters

Bayesian methods give a principled way to reason under uncertainty and to update
beliefs as data arrives. Frequentist approaches give point estimates; Bayes gives
a distribution over beliefs.

## Related

- [[MOC - Statistics]]
- [[Prior Selection]]
- [[Posterior Updating]]