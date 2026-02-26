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

Given a graph \(G=(V,E)\), MaxCut asks for a bit assignment \(z \in \{0,1\}^{|V|}\) that maximises:
\[
C(z) = \sum_{(i,j)\in E} z_i \oplus z_j
\]
where \(\oplus\) is XOR.

Here, \(G\) is `C8` with edges:
`(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)`.

### QAOA ansatz used in this project

We use QAOA with one layer (`p=1`) following Farhi et al. [1]:

1. Prepare \(|+\rangle^{\otimes 8}\) using Hadamard gates.
2. Apply the cost layer for each graph edge using:
   `CX(i,j) -> RZ(-gamma) on j -> CX(i,j)`.
3. Apply the mixer layer on each qubit using:
   `RX(2*beta)`.
4. Measure in the computational basis to obtain samples.

This is exactly what both implementations do:
- Bloch: [`c8_maxcut_qaoa_p1.bloch`]
- Python/Qiskit single-run: [`c8_maxcut_qaoa_p1.py`]

### Quantum and classical components

The workflow has two coupled parts:

1. Quantum evaluation:
   For fixed parameters \((\gamma,\beta)\), each toolchain generates a sample histogram from circuit measurements, then computes the expected cut:
   \[
   \mathbb{E}[C] = \sum_{z} p(z)\,C(z)
   \]
2. Classical optimisation:
   The runner uses parameter-shift gradients [2,3] with gradient ascent to update \((\gamma,\beta)\):
   - \(\partial_\gamma F \approx \frac{1}{2}\left[F(\gamma+\pi/2,\beta)-F(\gamma-\pi/2,\beta)\right]\)
   - \(\partial_\beta F \approx F(\gamma,\beta+\pi/4)-F(\gamma,\beta-\pi/4)\)

The update rule is:
\[
\gamma \leftarrow \gamma + \eta\,\partial_\gamma F,\quad
\beta \leftarrow \beta + \eta\,\partial_\beta F
\]
with learning rate \(\eta=0.12\) by default.

### Optimisation schedule and sampling policy

The experiment runner is [`run_experiment.py`](/home/akshay/documents/msc/project/run_experiment.py).  
It performs:
1. **25 optimisation iterations** per toolchain (not continuous optimisation during every script output stage).
2. One final evaluation at the optimised parameters.

Each optimisation iteration uses parameter-shift, which evaluates the objective 5 times (`F0`, `gamma+`, `gamma-`, `beta+`, `beta-`).  
Therefore, optimisation uses \(25 \times 5 = 125\) objective evaluations per toolchain, followed by one final post-optimisation evaluation for reporting.

By default, both optimisation evaluations and final evaluation use `8192` shots (`EXPERIMENT_SHOTS=8192`) for both toolchains, so the shot budget is matched.

## How to Replicate

### Prerequisites

1. Python 3.12
2. Poetry
3. Bloch CLI on `PATH` (binary name `bloch`)

Bloch installation reference:
- <https://docs.bloch-labs.com/bloch/overview/>

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
- optimised parameters \((\gamma,\beta)\)
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
