-- ==============================================================================
-- Conquer AI MLOps Platform - Continuous Drift & Survival Acceleration Rollup
-- Target: PostgreSQL 16 Materialized Views & Windowed Analytical Marts
-- Role: Senior Data Scientist / Data Engineering Specialist
-- Focus: Drift Acceleration, Moving Z-Scores, Decile Segmentation & Concurrent Refresh
-- ==============================================================================

SET search_path TO mlops_telemetry, public;

-- Drop existing materialized view if recreating
DROP MATERIALIZED VIEW IF EXISTS mv_model_drift_continuous_rollup CASCADE;

-- ==============================================================================
-- Materialized Mart: Daily Continuous Rollup with Statistical Windowing
-- ==============================================================================
CREATE MATERIALIZED VIEW mv_model_drift_continuous_rollup AS
WITH daily_model_slices AS (
    SELECT
        model_id,
        model_family,
        environment,
        DATE_TRUNC('day', event_timestamp) AS observation_day,
        COUNT(*) AS total_inferences_logged,
        AVG(inference_throughput_qps) AS avg_daily_qps,
        AVG(feature_psi_drift) AS avg_daily_psi,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p99_latency_ms) AS p95_latency_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY p99_latency_ms) AS p99_latency_ceiling_ms,
        AVG(gpu_memory_utilization_pct) AS avg_gpu_mem_pct,
        MAX(days_active) AS current_days_active,
        BOOL_OR(drift_detected) AS any_drift_flagged
    FROM model_inference_telemetry
    GROUP BY model_id, model_family, environment, DATE_TRUNC('day', event_timestamp)
),
temporal_acceleration_window AS (
    SELECT
        model_id,
        model_family,
        environment,
        observation_day,
        total_inferences_logged,
        avg_daily_qps,
        avg_daily_psi,
        p95_latency_ms,
        p99_latency_ceiling_ms,
        avg_gpu_mem_pct,
        current_days_active,
        any_drift_flagged,

        -- 1. Lagged PSI and Velocity (First Derivative of Drift)
        LAG(avg_daily_psi, 1) OVER (
            PARTITION BY model_id ORDER BY observation_day
        ) AS previous_day_psi,
        
        avg_daily_psi - COALESCE(LAG(avg_daily_psi, 1) OVER (
            PARTITION BY model_id ORDER BY observation_day
        ), avg_daily_psi) AS psi_velocity,

        -- 2. Rolling 7-Day Moving Baseline and Variance
        AVG(avg_daily_psi) OVER (
            PARTITION BY model_id ORDER BY observation_day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_mean_psi,

        STDDEV_SAMP(avg_daily_psi) OVER (
            PARTITION BY model_id ORDER BY observation_day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_stddev_psi,

        -- 3. Decile Risk Segmentation across all active models in the cluster
        NTILE(10) OVER (
            PARTITION BY observation_day, environment
            ORDER BY avg_daily_psi DESC
        ) AS cluster_drift_risk_decile,

        -- 4. Percent Rank of Latency Degradation
        PERCENT_RANK() OVER (
            PARTITION BY observation_day, model_family
            ORDER BY p99_latency_ceiling_ms ASC
        ) AS latency_percent_rank
    FROM daily_model_slices
)
SELECT
    model_id,
    model_family,
    environment,
    observation_day,
    total_inferences_logged,
    ROUND(avg_daily_qps, 2) AS avg_daily_qps,
    ROUND(avg_daily_psi, 4) AS avg_daily_psi,
    ROUND(previous_day_psi, 4) AS previous_day_psi,
    ROUND(psi_velocity, 4) AS psi_velocity,
    ROUND(rolling_7d_mean_psi, 4) AS rolling_7d_mean_psi,
    ROUND(COALESCE(rolling_7d_stddev_psi, 0.0), 4) AS rolling_7d_stddev_psi,
    
    -- Statistical Z-Score of current drift relative to running operational variance
    CASE 
        WHEN COALESCE(rolling_7d_stddev_psi, 0.0) > 0.0001 THEN
            ROUND((avg_daily_psi - rolling_7d_mean_psi) / rolling_7d_stddev_psi, 2)
        ELSE 0.0
    END AS drift_statistical_z_score,

    ROUND(p95_latency_ms, 2) AS p95_latency_ms,
    ROUND(p99_latency_ceiling_ms, 2) AS p99_latency_ceiling_ms,
    ROUND(avg_gpu_mem_pct, 2) AS avg_gpu_mem_pct,
    current_days_active,
    any_drift_flagged,
    cluster_drift_risk_decile,
    ROUND(latency_percent_rank::NUMERIC, 4) AS latency_percent_rank,

    -- Dynamic Categorization Engine
    CASE 
        WHEN avg_daily_psi >= 0.25 OR cluster_drift_risk_decile = 1 THEN 'CRITICAL_DISPATCH_REQUIRED'
        WHEN avg_daily_psi >= 0.10 OR psi_velocity > 0.05 THEN 'WARNING_DRIFT_ACCELERATING'
        ELSE 'STABLE_OPERATIONAL'
    END AS operational_reliability_status,

    CURRENT_TIMESTAMP AS refreshed_at
FROM temporal_acceleration_window;

-- ==============================================================================
-- Materialized View Indexing for Fast Concurrent Refresh & Dashboard Querying
-- ==============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS uq_idx_continuous_rollup_day 
    ON mv_model_drift_continuous_rollup (model_id, observation_day);

CREATE INDEX IF NOT EXISTS idx_continuous_rollup_status 
    ON mv_model_drift_continuous_rollup (operational_reliability_status, cluster_drift_risk_decile);

-- Automated Concurrent Refresh Command
COMMENT ON MATERIALIZED VIEW mv_model_drift_continuous_rollup IS 
    'Precomputed daily statistical telemetry rollup. Refresh via: REFRESH MATERIALIZED VIEW CONCURRENTLY mlops_telemetry.mv_model_drift_continuous_rollup;';
