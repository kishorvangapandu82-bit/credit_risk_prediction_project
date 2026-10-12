"""
=============================================================================
V2 PIPELINE -- STEP 7: RUNTIME COMPUTATIONAL COMPLEXITY BENCHMARK
=============================================================================
Measures:
  - Training Time (seconds) on N = 105,000 stratified training set.
  - Inference Latency (milliseconds per 1,000 rows) on holdout test set.
  - Peak RAM and CPU resource metrics.
Saves:
  - outputs_v2/tables/runtime_benchmark.csv
=============================================================================
"""

import os, sys, time, psutil, platform, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from data_preprocessing import load_data, clean_data, split_data
from models import (
    build_logistic_pipeline, build_decision_tree_pipeline, build_svm_pipeline,
    build_mlp_pipeline, build_random_forest_pipeline, build_xgboost_pipeline,
    build_lightgbm_pipeline
)

TABLE_DIR = "outputs_v2/tables"
os.makedirs(TABLE_DIR, exist_ok=True)

# 1. Load data
print("[Runtime] Loading data...")
df_raw = load_data()
df_clean, _ = clean_data(df_raw, verbose=False)
X_train, X_test, y_train, y_test = split_data(df_clean)

# Baseline models
MODEL_CONSTRUCTORS = [
    ("Logistic Regression", lambda: build_logistic_pipeline(C=0.001)),
    ("Decision Tree",       lambda: build_decision_tree_pipeline(max_depth=6, min_samples_split=20, criterion="gini")),
    ("SVM (Calibrated)",    lambda: build_svm_pipeline(C=0.01)),
    ("MLP Neural Network",  lambda: build_mlp_pipeline()[0]),
    ("Random Forest",       lambda: build_random_forest_pipeline(n_estimators=200, max_depth=12, min_samples_split=50)),
    ("XGBoost",             lambda: build_xgboost_pipeline(learning_rate=0.05, max_depth=5, n_estimators=200, scale_pos_weight=13.96)),
    ("LightGBM",            lambda: build_lightgbm_pipeline(learning_rate=0.05, num_leaves=31, n_estimators=200, scale_pos_weight=13.96)),
]

cpu_name = platform.processor() or "Multi-Core x86_64"
cpu_cores = psutil.cpu_count(logical=True)
total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

benchmark_rows = []

for name, constructor in MODEL_CONSTRUCTORS:
    print(f"\nBenchmarking {name}...")
    pipe = constructor()

    # Measure Training Time
    t_start_train = time.perf_counter()
    pipe.fit(X_train, y_train)
    t_train = time.perf_counter() - t_start_train
    print(f"  Training time (N=105,000): {t_train:.2f} seconds")

    # Measure Inference Time (repeated 3 times for precision)
    latencies = []
    for _ in range(3):
        t_start_inf = time.perf_counter()
        _ = pipe.predict_proba(X_test)[:, 1]
        t_inf = time.perf_counter() - t_start_inf
        latencies.append(t_inf)

    mean_inf_sec = np.mean(latencies)
    # Milliseconds per 1,000 records: (seconds / len(X_test)) * 1000 * 1000
    ms_per_1k = (mean_inf_sec / len(X_test)) * 1000.0 * 1000.0
    print(f"  Inference latency: {ms_per_1k:.2f} ms per 1,000 rows ({mean_inf_sec:.3f} s total for N=45,000)")

    benchmark_rows.append({
        "Model": name,
        "Training_Time_Sec": round(t_train, 2),
        "Inference_Time_Full_Test_Sec": round(mean_inf_sec, 3),
        "Inference_Latency_ms_per_1k_Rows": round(ms_per_1k, 2),
        "Hardware_CPU_Cores": cpu_cores,
        "System_RAM_GB": total_ram_gb,
    })

bench_df = pd.DataFrame(benchmark_rows)
bench_df.to_csv(os.path.join(TABLE_DIR, "runtime_benchmark.csv"), index=False)
print("\n=== Computational Runtime Benchmark Results ===")
print(bench_df.to_string())

print("\nStep 7 Complete!")
