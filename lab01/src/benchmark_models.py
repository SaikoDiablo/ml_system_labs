import os
import time
from pathlib import Path
from statistics import median

import joblib
import matplotlib

# Use a non-interactive backend so the plot saves on any computer.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import psutil

from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from print_versions import print_versions


# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

SEED = 42
TEST_SIZE = 0.30
TRAINING_RUNS = 5
INFERENCE_RUNS = 100

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "lab01" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def train_model(model, X, y):
    """Train one model and return the trained model."""
    model.fit(X, y)
    return model


def measure_training_time(model, X, y):
    """
    Perform one warm-up training run.
    Then train five fresh copies and return the median time.
    """

    # Warm-up: not included in the final measurements.
    train_model(clone(model), X, y)

    times = []

    for run_number in range(TRAINING_RUNS):
        fresh_model = clone(model)

        start = time.perf_counter()
        train_model(fresh_model, X, y)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        print(
            f"    Training run {run_number + 1}: "
            f"{elapsed:.6f} seconds"
        )

    return median(times)


def measure_inference_time(model, sample):
    """
    Measure single-sample prediction latency.
    The first prediction is a warm-up and is excluded.
    """

    model.predict(sample)

    times = []

    for _ in range(INFERENCE_RUNS):
        start = time.perf_counter()
        model.predict(sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Convert seconds to milliseconds.
    return median(times) * 1000


def measure_process_memory(model, X, y):
    """
    Measure process memory before and after training.
    This is a simple RSS measurement, not true peak memory.
    """

    process = psutil.Process(os.getpid())

    memory_before = process.memory_info().rss
    trained_model = train_model(clone(model), X, y)
    memory_after = process.memory_info().rss

    memory_before_mb = memory_before / (1024 * 1024)
    memory_after_mb = memory_after / (1024 * 1024)

    memory_change_mb = memory_after_mb - memory_before_mb

    return memory_after_mb, memory_change_mb, trained_model


# --------------------------------------------------
# 3. Print environment versions
# --------------------------------------------------

print_versions()

print("\n=== EXPERIMENT SETTINGS ===")
print("Random seed:", SEED)
print("Test size:", TEST_SIZE)
print("Training runs:", TRAINING_RUNS)
print("Inference runs:", INFERENCE_RUNS)
print("Results directory:", RESULTS_DIR)


# --------------------------------------------------
# 4. Load and split the dataset
# --------------------------------------------------

print("\n=== DATASET ===")

# This dataset is included in scikit-learn.
# No external download is necessary.
dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

print("Dataset: Breast Cancer Wisconsin")
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# --------------------------------------------------
# 5. Define the models
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=1,
    ),
}


# --------------------------------------------------
# 6. Train and evaluate both models
# --------------------------------------------------

accuracy_results = []
training_results = []
inference_results = []
resource_results = []

process = psutil.Process(os.getpid())

for model_name, model in models.items():

    print(f"\n=== {model_name} ===")

    # Measure training time using the required warm-up and repetitions.
    print("Measuring training time...")
    training_seconds = measure_training_time(
        model, X_train, y_train
    )

    # Measure peak process memory during a separate training run.
    print("Measuring process memory...")
    process_memory_mb, memory_change_mb, trained_model = (
    measure_process_memory(model, X_train, y_train)
    )

    # Evaluate on the held-out test set.
    predictions = trained_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Measure prediction latency for one sample.
    sample = X_test.iloc[[0]]
    inference_ms = measure_inference_time(trained_model, sample)

    # Save the model.
    safe_name = model_name.lower().replace(" ", "_")
    model_path = RESULTS_DIR / f"{safe_name}.joblib"

    joblib.dump(trained_model, model_path)
    model_size_bytes = model_path.stat().st_size
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_bytes / (1024 * 1024)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Median training time: {training_seconds:.6f} seconds")
    print(f"Median single-sample inference: {inference_ms:.6f} ms")
    print(f"Process memory after training: {process_memory_mb:.2f} MiB")
    print(f"Memory change during training: {memory_change_mb:.2f} MiB")
    print(f"Saved model size: {model_size_kb:.2f} KiB")

    accuracy_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
    })

    training_results.append({
        "model": model_name,
        "median_training_seconds": training_seconds,
        "training_runs": TRAINING_RUNS,
        "warmup_runs": 1,
    })

    inference_results.append({
        "model": model_name,
        "median_single_sample_latency_ms": inference_ms,
        "inference_runs": INFERENCE_RUNS,
        "warmup_runs": 1,
    })

    resource_results.append({
        "model": model_name,
        "model_size_bytes": model_size_bytes,
        "model_size_kb": model_size_kb,
        "model_size_mb": model_size_mb,
        "process_memory_after_training_mib": process_memory_mb,
        "memory_change_during_training_mib": memory_change_mb,
    })


# --------------------------------------------------
# 7. Save results as CSV files
# --------------------------------------------------

pd.DataFrame(accuracy_results).to_csv(
    RESULTS_DIR / "accuracy.csv", index=False
)

pd.DataFrame(training_results).to_csv(
    RESULTS_DIR / "training_times.csv", index=False
)

pd.DataFrame(inference_results).to_csv(
    RESULTS_DIR / "inference_times.csv", index=False
)

pd.DataFrame(resource_results).to_csv(
    RESULTS_DIR / "resource_measurements.csv", index=False
)


# --------------------------------------------------
# 8. Compare model file sizes with resource budgets
# --------------------------------------------------

# These are illustrative size limits for comparison.
# Replace them if your lab manual specifies different limits.
budgets = [
    {"environment": "Cloud", "model_size_limit_mb": 1024.0},
    {"environment": "Edge", "model_size_limit_mb": 256.0},
    {"environment": "Mobile", "model_size_limit_mb": 64.0},
    {"environment": "TinyML", "model_size_limit_mb": 0.25},
]

budget_rows = []

for result in resource_results:
    for budget in budgets:
        budget_rows.append({
            "model": result["model"],
            "environment": budget["environment"],
            "model_size_mb": result["model_size_mb"],
            "model_size_limit_mb": budget["model_size_limit_mb"],
            "fits_model_file_size_limit": (
                result["model_size_mb"]
                <= budget["model_size_limit_mb"]
            ),
            "fits_100_kb_model_limit": result["model_size_kb"] <= 100,
        })

pd.DataFrame(budget_rows).to_csv(
    RESULTS_DIR / "resource_budget.csv", index=False
)


# --------------------------------------------------
# 9. Create a comparison plot
# --------------------------------------------------

accuracy_df = pd.DataFrame(accuracy_results)
inference_df = pd.DataFrame(inference_results)

comparison_df = accuracy_df.merge(
    inference_df,
    on="model",
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].bar(
    comparison_df["model"],
    comparison_df["accuracy"],
)
axes[0].set_title("Test Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.05)
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(
    comparison_df["model"],
    comparison_df["median_single_sample_latency_ms"],
)
axes[1].set_title("Single-Sample Inference Latency")
axes[1].set_ylabel("Median latency (ms)")
axes[1].tick_params(axis="x", rotation=15)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# 10. Print the final summary
# --------------------------------------------------

print("\n=== FINAL RESULTS ===")
print(pd.DataFrame(accuracy_results).to_string(index=False))

print("\n=== SAVED FILES ===")
for path in sorted(RESULTS_DIR.iterdir()):
    if path.is_file():
        print(path.name)

print("\nExperiment completed successfully.")