"""First test: 20 sparse 10x10 matrices over 20 timesteps, measure PARDISO solve time."""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

N = 10
N_STEPS = 20
REPEATS = 51
OUT = Path("simple_results")


def make_matrix(step):
    progress = (step - 1) / (N_STEPS - 1)
    diagonal = 4.0 - 2.5 * progress
    off_diagonal = -1.0 - 0.3 * progress

    A = sp.diags(
        [off_diagonal * np.ones(N - 1),
         diagonal * np.ones(N),
         off_diagonal * np.ones(N - 1)],
        [-1, 0, 1],
        format="lil",
    )

    if step >= 8:
        for i in range(N - 3):
            A[i, i + 3] = -0.35
            A[i + 3, i] = -0.35

    if step >= 15:
        for i in range(N - 5):
            A[i, i + 5] = -0.25
            A[i + 5, i] = -0.25

    return A.tocsr()


def get_features(A):
    n = A.shape[0]
    rows, cols = A.nonzero()
    values = A.data

    diagonal = np.abs(A.diagonal())
    row_sums = np.abs(A).sum(axis=1).A1
    off_diagonal_sums = row_sums - diagonal

    return {
        "size": n,
        "nonzeros": A.nnz,
        "density": A.nnz / (n * n),
        "bandwidth": int(np.abs(rows - cols).max()),
        "avg_nonzeros_per_row": A.nnz / n,
        "diagonal_dominance": float(np.min((diagonal - off_diagonal_sums) / diagonal)),
        "largest_value": float(np.abs(values).max()),
        "smallest_value": float(np.abs(values).min()),
    }


def solve_and_time(A, b, repeats=REPEATS):
    import pypardiso
    for _ in range(3):
        pypardiso.spsolve(A, b)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        x = pypardiso.spsolve(A, b)
        times.append(time.perf_counter() - start)
    error = np.linalg.norm(A @ x - b) / np.linalg.norm(b)

    return float(np.median(times)), float(error)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"Building {N_STEPS} matrices of size {N}x{N} and timing each solve.\n")
    print(f"{'step':>5} {'nonzeros':>9} {'bandwidth':>10} {'diag dom':>9} "
          f"{'time (us)':>11} {'error':>10}")
    print("-" * 60)

    rows = []
    for step in range(1, N_STEPS + 1):
        A = make_matrix(step)
        b = rng.standard_normal(N)

        features = get_features(A)
        solve_time, error = solve_and_time(A, b)

        if error > 1e-8:
            raise RuntimeError(f"Step {step}: solve failed, error = {error:.1e}")

        rows.append({"step": step, **features,
                     "solve_time_s": solve_time, "error": error})

        print(f"{step:>5} {features['nonzeros']:>9} {features['bandwidth']:>10} "
              f"{features['diagonal_dominance']:>9.3f} "
              f"{solve_time * 1e6:>11.1f} {error:>10.1e}")

    data = pd.DataFrame(rows)
    data.to_csv(OUT / "dataset.csv", index=False)

    times_us = data["solve_time_s"] * 1e6
    feature_names = [c for c in data.columns
                     if c not in ("step", "solve_time_s", "error")
                     and data[c].std() > 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(data["step"], data["nonzeros"], marker="o", label="nonzeros")
    axes[0].plot(data["step"], data["bandwidth"], marker="s", label="bandwidth")
    axes[0].set_xlabel("timestep")
    axes[0].set_title("Nonzeros and bandwidth")
    axes[0].legend()

    axes[1].plot(data["step"], data["diagonal_dominance"], marker="o")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("diagonal dominance")
    axes[1].set_title("Diagonal dominance")

    axes[2].plot(data["step"], times_us, marker="o")
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("solve time (us)")
    axes[2].set_title("Solve time")

    fig.tight_layout()
    fig.savefig(OUT / "plots.png", dpi=150)

    print(f"\nSaved:")
    print(f"   {OUT / 'dataset.csv'}   ({len(data)} rows)")
    print(f"   {OUT / 'plots.png'}")
    print(f"\nFor ML training later:")
    print(f"   X = {feature_names}")
    print(f"   y = solve_time_s")


if __name__ == "__main__":
    main()
