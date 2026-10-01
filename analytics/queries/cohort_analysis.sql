-- ==============================================================================
-- Conquer AI MLOps Platform - Model Family Degradation & Reliability Window
-- Dialect: ANSI SQL / DuckDB / PostgreSQL Compatible
-- ==============================================================================

WITH model_base_metrics AS (
    SELECT
        model_id,
        model_family,
        environment,
        inference_throughput_qps,
        feature_psi_drift,
        p99_latency_ms,
        gpu_memory_utilization_pct,
        days_active,
        drift_detected,
        -- Early reliability milestones
        CASE WHEN days_active >= 7 THEN 1.0 ELSE 0.0 END AS reliable_d7,
        CASE WHEN days_active >= 14 THEN 1.0 ELSE 0.0 END AS reliable_d14,
        CASE WHEN days_active >= 30 THEN 1.0 ELSE 0.0 END AS reliable_d30,
        CASE WHEN drift_detected THEN 1.0 ELSE 0.0 END AS drifted,
        CASE WHEN feature_psi_drift >= 0.25 THEN 1 ELSE 0 END AS critical_drift
    FROM telemetry_events
),
cohort_summary AS (
    SELECT
        model_family,
        environment,
        COUNT(DISTINCT model_id) AS total_deployments,
        ROUND(AVG(inference_throughput_qps), 1) AS avg_qps,
        ROUND(AVG(feature_psi_drift), 4) AS avg_psi_drift,
        ROUND(AVG(p99_latency_ms), 1) AS avg_p99_ms,
        ROUND(AVG(gpu_memory_utilization_pct), 1) AS avg_gpu_pct,
        ROUND(AVG(reliable_d7) * 100.0, 1) AS reliability_d7_pct,
        ROUND(AVG(reliable_d14) * 100.0, 1) AS reliability_d14_pct,
        ROUND(AVG(reliable_d30) * 100.0, 1) AS reliability_d30_pct,
        ROUND(AVG(drifted) * 100.0, 1) AS drift_rate_pct,
        SUM(critical_drift) AS critical_models_count,
        DENSE_RANK() OVER (
            PARTITION BY environment 
            ORDER BY COUNT(DISTINCT model_id) DESC
        ) AS cluster_scale_rank
    FROM model_base_metrics
    GROUP BY model_family, environment
)
SELECT
    model_family,
    environment,
    total_deployments,
    avg_qps,
    avg_psi_drift,
    avg_p99_ms,
    avg_gpu_pct,
    reliability_d14_pct,
    drift_rate_pct,
    critical_models_count,
    cluster_scale_rank
FROM cohort_summary
ORDER BY critical_models_count DESC, total_deployments DESC;