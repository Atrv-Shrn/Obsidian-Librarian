---
title: Posterior Updating
tags:
  - stats/bayesian
created: 2026-07-03
---

# Posterior Updating

The posterior of one update becomes the prior of the next — beliefs accumulate
evidence sequentially.

## Conjugate priors

When prior and likelihood are conjugate, the posterior has a closed form. Example:
Beta prior + Binomial likelihood → Beta posterior. This makes sequential updating cheap.

## MCMC

For non-conjugate models we sample the posterior with MCMC (Metropolis-Hastings,
Hamiltonian Monte Carlo, NUTS). See [[Bayesian Reasoning]] for the full setup.

## Related

- [[Bayesian Reasoning]]
- [[Prior Selection]]
- [[MOC - Statistics]]