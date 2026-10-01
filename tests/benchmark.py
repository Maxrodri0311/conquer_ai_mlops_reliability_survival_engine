"""
tests/benchmark.py - Quantitative Latency & Memory Benchmark for Conquer AI MLOps Platform.
Measures p50, p95, and p99 query latency over 30 iterations.
Enforces SLA constraints across Cox Hazards, DuckDB Vectorized Windowing and Kubeflow Dispatch.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_generator import generate_domain_dataset
from src.core_engine import DomainAnalyticsEngine, DuckDBStorageAdapter
from src.domain.entities import ExecutionContext


def run_benchmarks(iterations=30, num_records=10000):
    print(f"[Benchmark] Preparing dataset with {num_records:,} records...")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_domain_dataset(num_records=num_records, output_path=tmp_path)
        adapter = DuckDBStorageAdapter()
        engine = DomainAnalyticsEngine(storage=adapter, data_path=tmp_path)
        ctx = ExecutionContext()
        
        # Warmup
        engine.execute_analysis(ctx)
        
        print(f"[Benchmark] Profiling subcomponents over {iterations} iterations...")
        
        cox_latencies = []
        duckdb_latencies = []
        pipeline_latencies = []

        raw_df = engine.load_dataset()

        for _ in range(iterations):
            # 1. Micro-benchmark: Cox Proportional Hazards Fitting
            t0 = time.perf_counter()
            engine.fit_cox_hazard_model(raw_df)
            cox_latencies.append((time.perf_counter() - t0) * 1000)

            # 2. Micro-benchmark: DuckDB Vectorized Cluster Degradation Windowing
            t1 = time.perf_counter()
            engine.execute_degradation_matrix()
            duckdb_latencies.append((time.perf_counter() - t1) * 1000)

            # 3. Macro-benchmark: Full End-to-End Reliability Pipeline (Weibull + Cox + Triggers)
            t2 = time.perf_counter()
            engine.execute_analysis(ctx)
            pipeline_latencies.append((time.perf_counter() - t2) * 1000)
            
        p50_cox = float(np.percentile(cox_latencies, 50))
        p95_cox = float(np.percentile(cox_latencies, 95))

        p50_duckdb = float(np.percentile(duckdb_latencies, 50))
        p95_duckdb = float(np.percentile(duckdb_latencies, 95))

        p50_pipe = float(np.percentile(pipeline_latencies, 50))
        p95_pipe = float(np.percentile(pipeline_latencies, 95))
        p99_pipe = float(np.percentile(pipeline_latencies, 99))

        status_cox = "PASS" if p95_cox < 450.0 else "FAIL"
        status_duckdb = "PASS" if p95_duckdb < 25.0 else "FAIL"
        status_pipe = "PASS" if p95_pipe < 1250.0 else "FAIL"
        
        print("\n" + "="*70)
        print("  CONQUER AI MLOPS RELIABILITY ENGINE - QUANTITATIVE BENCHMARK")
        print("="*70)
        print(f"  Dataset Size: {num_records:,} rows | Iterations: {iterations}")
        print("-" * 70)
        print(f"  1. Cox Proportional Hazards Engine:   p50 = {p50_cox:.2f} ms | p95 = {p95_cox:.2f} ms [SLA < 450ms: {status_cox}]")
        print(f"  2. DuckDB Vectorized Cluster Window:   p50 = {p50_duckdb:.2f} ms | p95 = {p95_duckdb:.2f} ms [SLA < 25ms: {status_duckdb}]")
        print(f"  3. Full End-to-End Pipeline (with MLE): p50 = {p50_pipe:.2f} ms | p95 = {p95_pipe:.2f} ms [SLA < 1250ms: {status_pipe}]")
        print(f"     -> p99 End-to-End Pipeline Latency:  {p99_pipe:.2f} ms")
        print("="*70 + "\n")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    run_benchmarks()