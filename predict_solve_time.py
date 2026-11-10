"""Predict PARDISO solve time with linear regression: run 10 steps, predict the next 10, until step 200."""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pypardiso import PyPardisoSolver
from sklearn.linear_model import LinearRegression

N = 2000
N_STEPS = 200
BLOCK = 10
REPEATS = 21
OUT = Path("predict_results")

BAND_STEPS = [35, 75, 115, 155, 195]
BAND_OFFSETS = [3, 6, 10, 15, 21]
BAND_VALUES = [-0.35, -0.30, -0.25, -0.20, -0.15]

FEATURES = ["nonzeros", "bandwidth"]


def make_matrix(step):
    progress = (step - 1) / (N_STEPS - 1)
    values = [-1.0 - 0.3 * progress, 8.0 - 2.0 * progress, -1.0 - 0.3 * progress]
    offsets = [-1, 0, 1]
    for band_step, offset, value in zip(BAND_STEPS, BAND_OFFSETS, BAND_VALUES):
        if step >= band_step:
            values += [value, value]
            offsets += [-offset, offset]
    return sp.diags(values, offsets, shape=(N, N), format="csr")


def get_features(A):
    rows, cols = A.nonzero()
    return {"nonzeros": A.nnz, "bandwidth": int(np.abs(rows - cols).max())}


def measure_time(A, b):
    solver = PyPardisoSolver()
    for _ in range(3):
        solver.solve(A, b)

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        x = solver.solve(A, b)  # PARDISO call
        times.append(time.perf_counter() - start)
    solver.free_memory(everything=True)

    error = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    if error > 1e-8:
        raise RuntimeError(f"solve failed, error = {error:.1e}")
    return float(np.median(times)) * 1000


def percent_error(predicted, actual):
    return float(np.mean(np.abs(predicted - actual) / actual) * 100)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []

    print(f"Matrix size {N}x{N}, {N_STEPS} timesteps, run {BLOCK} / predict {BLOCK}.\n")
    print(f"{'predicted steps':>16} {'trained on':>11} {'error':>8}")
    print("-" * 40)

    for start in range(1, N_STEPS + 1, 2 * BLOCK):
        run_steps = range(start, start + BLOCK)
        predict_steps = range(start + BLOCK, start + 2 * BLOCK)

        for step in run_steps:
            A = make_matrix(step)
            b = rng.standard_normal(N)
            rows.append({"step": step, "phase": "run", **get_features(A),
                         "actual_ms": measure_time(A, b)})

        train = pd.DataFrame([r for r in rows if r["phase"] == "run"])
        model = LinearRegression().fit(train[FEATURES], train["actual_ms"])

        block = []
        for step in predict_steps:
            A = make_matrix(step)
            b = rng.standard_normal(N)
            features = get_features(A)
            predicted = model.predict(pd.DataFrame([features])[FEATURES])[0]
            block.append({"step": step, "phase": "predict", **features,
                          "actual_ms": measure_time(A, b),
                          "predicted_ms": predicted})
        rows += block

        block = pd.DataFrame(block)
        error = percent_error(block["predicted_ms"], block["actual_ms"])
        print(f"{predict_steps[0]:>7}-{predict_steps[-1]:<8} {len(train):>7} steps "
              f"{error:>7.1f}%")

    data = pd.DataFrame(rows)
    data.to_csv(OUT / "results.csv", index=False)

    pred = data[data["phase"] == "predict"]
    print(f"\nAverage error over all predicted steps: "
          f"{percent_error(pred['predicted_ms'], pred['actual_ms']):.1f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(data["step"], data["actual_ms"], color="black", label="Measured time")
    plt.scatter(pred["step"], pred["predicted_ms"], color="red", marker="x",
                label="Predicted time")
    plt.xlabel("Time step")
    plt.ylabel("Time (ms)")
    plt.title("Measured and predicted solve time")
    plt.legend()
    plt.grid()
    plt.savefig(OUT / "graph1.png", dpi=150)
    plt.close()

    blocks, errors = [], []
    for s in range(BLOCK + 1, N_STEPS + 1, 2 * BLOCK):
        part = pred[(pred["step"] >= s) & (pred["step"] <= s + BLOCK - 1)]
        blocks.append(f"{s}-{s + BLOCK - 1}")
        errors.append(percent_error(part["predicted_ms"], part["actual_ms"]))

    plt.figure(figsize=(10, 5))
    plt.bar(blocks, errors, color="red")
    plt.xlabel("Predicted time steps")
    plt.ylabel("Error (%)")
    plt.title("Prediction error for each block")
    plt.grid(axis="y")
    plt.savefig(OUT / "graph2.png", dpi=150)
    plt.close()

    print(f"\nSaved:\n   {OUT / 'results.csv'}\n   {OUT / 'graph1.png'}\n   {OUT / 'graph2.png'}")


if __name__ == "__main__":
    main()
