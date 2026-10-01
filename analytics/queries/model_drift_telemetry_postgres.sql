-- ==============================================================================
-- Conquer AI MLOps Platform - PostgreSQL Advanced Telemetry & Drift Analysis
-- Target Role: Senior Data Scientist | Stack: PostgreSQL 16 + Kubeflow Pipelines
-- Dialect: PostgreSQL (Window Functions, CTEs, Filter Aggregations, Math Scoring)
-- ==============================================================================

WITH rolling_inference_telemetry AS (
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
        event_timestamp,
        -- Window function: rank models by drift severity within each cluster
        DENSE_RANK() OVER (
            PARTITION BY environment 
            ORDER BY feature_psi_drift DESC
        ) AS cluster_drift_rank,
        -- Running average PSI drift across the model family
        AVG(feature_psi_drift) OVER (
            PARTITION BY model_family
        ) AS family_baseline_psi,
        -- P95 Latency threshold per model family
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p99_latency_ms) OVER (
            PARTITION BY model_family
        ) AS family_p95_latency_ceiling
    FROM model_inference_telemetry
),
cox_hazard_scoring AS (
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
        cluster_drift_rank,
        family_baseline_psi,
        family_p95_latency_ceiling,
        -- Cox Proportional Hazard Multiplier derived in pure PostgreSQL SQL
        -- Formula: exp( beta_psi * (PSI - 0.10) + beta_qps * (QPS/100) + beta_gpu * (GPU/100) )
        ROUND(
            EXP(
                LEAST(
                    4.0,
                    GREATEST(
                        -2.5,
                        (3.20 * (feature_psi_drift - 0.10)) +
                        (0.45 * ((inference_throughput_qps - 80.0) / 100.0)) +
                        (0.35 * ((gpu_memory_utilization_pct - 60.0) / 100.0))
                    )
                )
            )::numeric,
            3
        ) AS relative_hazard_multiplier,
        -- Estimated survival probability using parametric Weibull baseline (k=1.45, lambda=42.0)
        ROUND(
            (EXP(-POWER(days_active / 42.0, 1.45)) * 100.0)::numeric,
            1
        ) AS estimated_survival_prob_pct
    FROM rolling_inference_telemetry
),
retraining_dispatch_triage AS (
    SELECT
        model_id,
        model_family,
        environment,
        inference_throughput_qps,
        feature_psi_drift,
        p99_latency_ms,
        days_active,
        relative_hazard_multiplier,
        estimated_survival_prob_pct,
        CASE
            WHEN feature_psi_drift >= 0.25 THEN 'CRITICAL_DRIFT_IMMEDIATE_RETRAIN'
            WHEN estimated_survival_prob_pct < 65.0 THEN 'PREDICTIVE_DEGRADATION_SCHEDULED'
            WHEN p99_latency_ms > family_p95_latency_ceiling THEN 'LATENCY_SPIKE_MONITOR'
            ELSE 'STABLE_HEALTHY'
        END AS mlops_action_tier,
        CASE
            WHEN feature_psi_drift >= 0.25 OR estimated_survival_prob_pct < 65.0
            THEN '/apis/v1/kubeflow/pipelines/retrain-' || model_family
            ELSE NULL
        END AS kubeflow_webhook_endpoint
    FROM cox_hazard_scoring
)
SELECT
    model_id,
    model_family,
    environment,
    inference_throughput_qps,
    feature_psi_drift,
    p99_latency_ms,
    days_active,
    relative_hazard_multiplier,
    estimated_survival_prob_pct,
    mlops_action_tier,
    kubeflow_webhook_endpoint
FROM retraining_dispatch_triage
WHERE mlops_action_tier IN ('CRITICAL_DRIFT_IMMEDIATE_RETRAIN', 'PREDICTIVE_DEGRADATION_SCHEDULED')
ORDER BY feature_psi_drift DESC, relative_hazard_multiplier DESC
LIMIT 50;
