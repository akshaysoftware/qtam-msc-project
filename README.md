# QAOA Benchmark on the C8 MaxCut Instance

## Goals

This repository studies whether two quantum toolchains produce comparable behaviour when they implement the same variational algorithm on the same combinatorial optimisation problem.

The specific goals are:
1. Implement a single QAOA depth-1 (`p=1`) benchmark instance for MaxCut on the cycle graph with 8 vertices (`C8`).
2. Keep the **quantum circuit definition identical** across the Bloch and Python/Qiskit implementations.
3. Compare both toolchains under the **same classical optimisation routine** in a shared experiment runner.
4. Report reproducible metrics (optimised parameters, expected cut value, and runtimes).

## Methodology

### Problem formulation

Given a graph `G=(V,E)`, MaxCut asks for a bit assignment `z in {0,1}^|V|` that maximises:

`C(z) = sum over edges (i,j) in E of XOR(z_i, z_j)`

where `XOR` is exclusive-or.

Here, `G` is `C8` with:
- `V = {1,2,3,4,5,6,7,8}`
- `E = {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,1)}`

Implementation mapping:
- vertex `v` is mapped to qubit index `v-1` (so vertex 1 maps to `q[0]`, vertex 8 maps to `q[7]`).

### QAOA ansatz used in this project

We use QAOA with one layer (`p=1`) following Farhi et al. [1]:

1. Prepare `|+>^8` using Hadamard gates.
2. Apply the cost layer for each graph edge using:
   `CX(i,j) -> RZ(-gamma) on j -> CX(i,j)`.
3. Apply the mixer layer on each qubit using:
   `RX(2*beta)`.
4. Measure in the computational basis to obtain samples.

This is exactly what both implementations do:
- Bloch: `c8_maxcut_qaoa_p1.bloch`
- Python/Qiskit single-run: `c8_maxcut_qaoa_p1.py`

### Quantum and classical components

The workflow has two coupled parts:

1. Quantum evaluation:
   For fixed parameters `(gamma, beta)`, each toolchain generates a sample histogram from circuit measurements, then computes the expected cut:
   `E[C] = sum over bitstrings z of p(z) * C(z)`.
2. Classical optimisation:
   The runner uses parameter-shift gradients [2,3] with gradient ascent to update `(gamma, beta)`:
   - `dF/dgamma ~= 0.5 * (F(gamma + pi/2, beta) - F(gamma - pi/2, beta))`
   - `dF/dbeta  ~= 0.5 * (F(gamma, beta + pi/2) - F(gamma, beta - pi/2))`

Update rule:
- `gamma <- gamma + eta * dF/dgamma`
- `beta  <- beta  + eta * dF/dbeta`

Default learning rate: `eta = 0.12`.

### Optimisation schedule and sampling policy

The experiment runner is [`run_experiment.py`](./run_experiment.py). 
It performs:
1. **25 optimisation iterations** per toolchain (not continuous optimisation during every script output stage).
2. One final evaluation at the optimised parameters.

Each optimisation iteration uses parameter-shift, which evaluates the objective 5 times (`F0`, `gamma+`, `gamma-`, `beta+`, `beta-`).  
Therefore, optimisation uses `25 * 5 = 125` objective evaluations per toolchain, followed by one final post-optimisation evaluation for reporting.

By default, both optimisation evaluations and final evaluation use `8192` shots (`EXPERIMENT_SHOTS=8192`) for both toolchains, so the shot budget is matched.

## How to Replicate

### Prerequisites

1. Python 3.12
2. Poetry
3. Bloch CLI on `PATH` (binary name `bloch`)

This experiment was conducted using the original Bloch compiler.
The implementation is available in
[`bloch-labs/bloch-legacy`](https://github.com/bloch-labs/bloch-legacy).

> Tested against tag `v1.0.3`

Active development of the language continues at
[Bloch Basis](https://github.com/bloch-labs/basis).

### Installation

```bash
poetry install
```

### A. Run standalone Bloch implementation (single histogram, no optimisation)

```bash
bloch c8_maxcut_qaoa_p1.bloch --shots=8192
```

### B. Run standalone Python implementation (single histogram, no optimisation)

```bash
poetry run python c8_maxcut_qaoa_p1.py
```

Optional overrides:
- `QAOA_GAMMA`, `QAOA_BETA`, `QAOA_SHOTS`, `QAOA_SEED`

### C. Run full cross-toolchain experiment (shared optimisation + metrics)

```bash
poetry run python run_experiment.py
```

This runs Bloch and Qiskit with the same optimisation algorithm and prints:
- optimised parameters `(gamma, beta)`
- optimisation runtime
- final evaluation runtime
- final expected cut value
- top outcomes and histograms

Useful environment overrides:
- `EXPERIMENT_SHOTS` (default `8192`)
- `OPT_ITERATIONS` (default `25`)
- `OPT_ETA`, `OPT_GAMMA0`, `OPT_BETA0`
- `BLOCH_CLI`, `BLOCH_PROGRAM`

## References

[1] E. Farhi, J. Goldstone, and S. Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028, 2014.  
[2] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, *Quantum circuit learning*, Phys. Rev. A 98, 032309, 2018.  
[3] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, *Evaluating analytic gradients on quantum hardware*, Phys. Rev. A 99, 032331, 2019.
