import os
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

N = 8
EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 0),
]

GAMMA = float(os.environ.get("QAOA_GAMMA", "0.7"))
BETA = float(os.environ.get("QAOA_BETA", "0.3"))
SHOTS = int(os.environ.get("QAOA_SHOTS", "8192"))
SEED = int(os.environ.get("QAOA_SEED", "1234"))


def qiskit_counts_for_params(gamma: float, beta: float, shots: int, seed: int) -> Dict[str, int]:
    qc = QuantumCircuit(N)

    for i in range(N):
        qc.h(i)

    for (i, j) in EDGES:
        qc.cx(i, j)
        qc.rz(-gamma, j)
        qc.cx(i, j)

    for i in range(N):
        qc.rx(2.0 * beta, i)

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
