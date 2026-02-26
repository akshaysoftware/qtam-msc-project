import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Shared C8 MaxCut problem definition
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


@dataclass
class OptConfig:
    iterations: int = int(os.environ.get("OPT_ITERATIONS", "25"))
    shots: int = int(os.environ.get("EXPERIMENT_SHOTS", "8192"))
    eta: float = float(os.environ.get("OPT_ETA", "0.12"))
    gamma0: float = float(os.environ.get("OPT_GAMMA0", "0.7"))
    beta0: float = float(os.environ.get("OPT_BETA0", "0.3"))
    shift_gamma: float = math.pi / 2
    shift_beta: float = math.pi / 4  # RX(2 * beta)


@dataclass
class EvalResult:
    counts: Dict[str, int]
    expected_cut: float
    elapsed_seconds: float


def cut_value_c8(bitstring: str) -> int:
    bits = [int(b) for b in bitstring]
    return sum(bits[i] ^ bits[j] for (i, j) in EDGES)


def expected_cut_from_counts(counts: Dict[str, int], total: int) -> float:
    return sum(cut_value_c8(b) * c for b, c in counts.items()) / float(total)


def normalise_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def print_histogram_table(title: str, counts: Dict[str, int], max_rows: int = 20) -> None:
    probs = normalise_counts(counts)
    items = sorted(counts.items(), key=lambda x: -x[1])[:max_rows]
    print(f"\n{title}")
    print("outcome  | count |  prob")
    print("---------+-------+------")
    for b, c in items:
        print(f"{b} | {c:5d} | {probs[b]:.6f}")


def print_top_outcomes(title: str, counts: Dict[str, int], max_rows: int = 10) -> None:
    total = sum(counts.values())
    print(f"\n{title}")
    for bs, c in sorted(counts.items(), key=lambda x: -x[1])[:max_rows]:
        print(" ", bs, c, f"p={c/total:.4f}", "cut=", cut_value_c8(bs))


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


def evaluate_qiskit(gamma: float, beta: float, shots: int, seed: int) -> EvalResult:
    t0 = time.perf_counter()
    counts = qiskit_counts_for_params(gamma, beta, shots, seed=seed)
    elapsed = time.perf_counter() - t0
    return EvalResult(
        counts=counts,
        expected_cut=expected_cut_from_counts(counts, shots),
        elapsed_seconds=elapsed,
    )


GAMMA_RE = re.compile(r"(final\s+float\s+gamma\s*=\s*)(-?\d+(?:\.\d+)?)(f\s*;)")
BETA_RE = re.compile(r"(final\s+float\s+beta\s*=\s*)(-?\d+(?:\.\d+)?)(f\s*;)")
ROW_RE = re.compile(r"^\s*([01]{8})\s*\|\s*(\d+)\s*\|\s*([0-9]*\.?[0-9]+)\s*$")


def set_bloch_params_in_file(path: Path, gamma: float, beta: float) -> None:
    text = path.read_text(encoding="utf-8")

    def repl_gamma(match: re.Match) -> str:
        return f"{match.group(1)}{gamma:.8f}{match.group(3)}"

    def repl_beta(match: re.Match) -> str:
        return f"{match.group(1)}{beta:.8f}{match.group(3)}"

    text, n_gamma = GAMMA_RE.subn(repl_gamma, text, count=1)
    if n_gamma == 0:
        raise RuntimeError(
            "Could not replace gamma in Bloch file.\n"
            "Expected a line like: final float gamma = 0.7000f;"
        )

    text, n_beta = BETA_RE.subn(repl_beta, text, count=1)
    if n_beta == 0:
        raise RuntimeError(
            "Could not replace beta in Bloch file.\n"
            "Expected a line like: final float beta  = 0.3000f;"
        )

    path.write_text(text, encoding="utf-8")


def run_bloch_and_parse_counts(bloch_cli: str, bloch_program: Path, shots: int) -> Dict[str, int]:
    try:
        proc = subprocess.run(
            [bloch_cli, str(bloch_program), f"--shots={shots}"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Bloch CLI '{bloch_cli}' was not found. Install Bloch and ensure it is on PATH."
        ) from exc

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Bloch failed:\n{output}")

    counts: Dict[str, int] = {}
    for line in output.splitlines():
        match = ROW_RE.match(line)
        if match:
            bitstring = match.group(1)
            counts[bitstring] = counts.get(bitstring, 0) + int(match.group(2))

    if not counts:
        raise RuntimeError(
            "Could not parse histogram rows from Bloch output.\n"
            "Expected lines like: 01010101 | 123 | 0.125\n\n"
            f"Raw output (first 2000 chars):\n{output[:2000]}"
        )
    return counts


def evaluate_bloch(
    bloch_cli: str,
    bloch_program: Path,
    gamma: float,
    beta: float,
    shots: int,
) -> EvalResult:
    t0 = time.perf_counter()
    set_bloch_params_in_file(bloch_program, gamma, beta)
    counts = run_bloch_and_parse_counts(bloch_cli, bloch_program, shots)
    elapsed = time.perf_counter() - t0
    total = sum(counts.values())
    return EvalResult(
        counts=counts,
        expected_cut=expected_cut_from_counts(counts, total),
        elapsed_seconds=elapsed,
    )


Evaluator = Callable[[float, float, int], float]


def parameter_shift_step(
    eval_f: Evaluator,
    gamma: float,
    beta: float,
    cfg: OptConfig,
    base_seed: int,
    step_id: int,
) -> Tuple[float, float, float]:
    # 0=F0, 1=gp, 2=gm, 3=bp, 4=bm
    def seed(k: int) -> int:
        return base_seed + 1000 * step_id + k

    f0 = eval_f(gamma, beta, seed(0))
    f_gp = eval_f(gamma + cfg.shift_gamma, beta, seed(1))
    f_gm = eval_f(gamma - cfg.shift_gamma, beta, seed(2))
    dgamma = 0.5 * (f_gp - f_gm)

    f_bp = eval_f(gamma, beta + cfg.shift_beta, seed(3))
    f_bm = eval_f(gamma, beta - cfg.shift_beta, seed(4))
    dbeta = f_bp - f_bm
    return f0, dgamma, dbeta


def optimise(name: str, eval_f: Evaluator, cfg: OptConfig, base_seed: int = 1234) -> Tuple[float, float]:
    gamma, beta = cfg.gamma0, cfg.beta0
    print(f"\n=== Optimising {name} ===")
    for t in range(cfg.iterations):
        f0, dgamma, dbeta = parameter_shift_step(eval_f, gamma, beta, cfg, base_seed, t)
        gamma += cfg.eta * dgamma
        beta += cfg.eta * dbeta
        print(
            f"[{name}] iter={t:02d} "
            f"F={f0:.6f} gamma={gamma:.6f} beta={beta:.6f} "
            f"(dg={dgamma:.6f}, db={dbeta:.6f})"
        )
    return gamma, beta


def main() -> None:
    bloch_cli = os.environ.get("BLOCH_CLI", "bloch")
    bloch_program = Path(os.environ.get("BLOCH_PROGRAM", "c8_maxcut_qaoa_p1.bloch"))
    cfg = OptConfig()
    original_bloch_source = bloch_program.read_text(encoding="utf-8")

    print(
        "Experiment configuration: "
        f"iterations={cfg.iterations}, shots_per_eval={cfg.shots}, eta={cfg.eta}, "
        f"gamma0={cfg.gamma0}, beta0={cfg.beta0}"
    )

    def eval_f_bloch(gamma: float, beta: float, seed: int) -> float:
        _ = seed  # Bloch CLI currently does not expose a seed parameter.
        return evaluate_bloch(
            bloch_cli=bloch_cli,
            bloch_program=bloch_program,
            gamma=gamma,
            beta=beta,
            shots=cfg.shots,
        ).expected_cut

    def eval_f_qiskit(gamma: float, beta: float, seed: int) -> float:
        return evaluate_qiskit(gamma=gamma, beta=beta, shots=cfg.shots, seed=seed).expected_cut

    try:
        t0 = time.perf_counter()
        g_bloch, b_bloch = optimise("BLOCH", eval_f_bloch, cfg)
        bloch_optimisation_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        g_qiskit, b_qiskit = optimise("QISKIT", eval_f_qiskit, cfg)
        qiskit_optimisation_s = time.perf_counter() - t0

        print("\n=== Final evaluation (BLOCH) ===")
        bloch_final = evaluate_bloch(
            bloch_cli=bloch_cli,
            bloch_program=bloch_program,
            gamma=g_bloch,
            beta=b_bloch,
            shots=cfg.shots,
        )
        print_top_outcomes("BLOCH top outcomes", bloch_final.counts, max_rows=10)
        print(f"Expected cut: {bloch_final.expected_cut:.6f}")
        print(f"Elapsed (final evaluation): {bloch_final.elapsed_seconds:.3f}s")

        print("\n=== Final evaluation (QISKIT) ===")
        qiskit_final = evaluate_qiskit(
            gamma=g_qiskit,
            beta=b_qiskit,
            shots=cfg.shots,
            seed=1234,
        )
        print_top_outcomes("QISKIT top outcomes", qiskit_final.counts, max_rows=10)
        print(f"Expected cut: {qiskit_final.expected_cut:.6f}")
        print(f"Elapsed (final evaluation): {qiskit_final.elapsed_seconds:.3f}s")

        print_histogram_table("BLOCH histogram", bloch_final.counts, max_rows=20)
        print_histogram_table("QISKIT histogram", qiskit_final.counts, max_rows=20)

        print("\n=== Metrics summary ===")
        print("toolchain | gamma | beta | optimisation_s | final_eval_s | expected_cut")
        print("----------+-------+------+----------------+--------------+------------")
        print(
            f"BLOCH     | {g_bloch:.6f} | {b_bloch:.6f} | "
            f"{bloch_optimisation_s:14.3f} | {bloch_final.elapsed_seconds:12.3f} | "
            f"{bloch_final.expected_cut:.6f}"
        )
        print(
            f"QISKIT    | {g_qiskit:.6f} | {b_qiskit:.6f} | "
            f"{qiskit_optimisation_s:14.3f} | {qiskit_final.elapsed_seconds:12.3f} | "
            f"{qiskit_final.expected_cut:.6f}"
        )
    finally:
        bloch_program.write_text(original_bloch_source, encoding="utf-8")


if __name__ == "__main__":
    main()
