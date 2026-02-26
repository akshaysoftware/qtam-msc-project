import os
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

N = 8
# Graph is expressed in 1-based labels:
# V = {1,2,...,8}, E = {(1,2),(2,3),...,(7,8),(8,1)}
VERTICES: List[int] = list(range(1, N + 1))
EDGES_1BASED: List[Tuple[int, int]] = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 1),
]

GAMMA = float(os.environ.get("QAOA_GAMMA", "0.7"))
BETA = float(os.environ.get("QAOA_BETA", "0.3"))
SHOTS = int(os.environ.get("QAOA_SHOTS", "8192"))
SEED = int(os.environ.get("QAOA_SEED", "1234"))


def qiskit_counts_for_params(gamma: float, beta: float, shots: int, seed: int) -> Dict[str, int]:
    qc = QuantumCircuit(N)

    for vertex in VERTICES:
        qc.h(vertex - 1)

    for (u, v) in EDGES_1BASED:
        qu_u = u - 1
        qu_v = v - 1
        qc.cx(qu_u, qu_v)
        qc.rz(-gamma, qu_v)
        qc.cx(qu_u, qu_v)

    for vertex in VERTICES:
        qc.rx(2.0 * beta, vertex - 1)

    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2

    rng = np.random.default_rng(seed)
    outcomes = rng.choice(len(probs), size=shots, p=probs)

    counts: Dict[str, int] = {}
    for z in outcomes:
        bitstring = format(z, f"0{N}b")[::-1]
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


def print_histogram_table(title: str, counts: Dict[str, int], max_rows: int = 20) -> None:
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    items = sorted(counts.items(), key=lambda x: -x[1])[:max_rows]

    print(f"\n{title}")
    print("outcome  | count |  prob")
    print("---------+-------+------")
    for b, c in items:
        print(f"{b} | {c:5d} | {probs[b]:.6f}")


def main() -> None:
    counts = qiskit_counts_for_params(GAMMA, BETA, SHOTS, SEED)
    print(
        "C8 QAOA p=1 (Python/Qiskit) "
        f"gamma={GAMMA:.8f} beta={BETA:.8f} shots={SHOTS} seed={SEED}"
    )
    print_histogram_table("QISKIT histogram", counts, max_rows=20)


if __name__ == "__main__":
    main()
