//! Production observability for PartitionedBloomFilter
//!
//! Provides metrics collection, health checks, and Prometheus export.

#![cfg(feature = "metrics")]
#![allow(clippy::pedantic)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Metrics for PartitionedBloomFilter operations.
#[derive(Debug)]
pub struct PartitionedFilterMetrics {
    /// Total insert operations.
    pub insert_count: AtomicU64,
    /// Total query operations.
    pub query_count: AtomicU64,
    /// Estimated false positives (sampled).
    pub false_positive_count: AtomicU64,
    /// Insert latency histogram.
    pub insert_latency: LatencyHistogram,
    /// Query latency histogram.
    pub query_latency: LatencyHistogram,
    /// Cache hit estimate (sampled).
    pub cache_hits: AtomicU64,
    /// Cache miss estimate (sampled).
    pub cache_misses: AtomicU64,
}

impl PartitionedFilterMetrics {
    /// Create new metrics instance.
    pub fn new() -> Self {
        Self {
            insert_count: AtomicU64::new(0),
            query_count: AtomicU64::new(0),
            false_positive_count: AtomicU64::new(0),
            insert_latency: LatencyHistogram::new(),
            query_latency: LatencyHistogram::new(),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Record an insert operation.
    #[inline]
    pub fn record_insert(&self, latency: Duration) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
        self.insert_latency.record(latency);
    }

    /// Record a query operation.
    #[inline]
    pub fn record_query(&self, latency: Duration, _result: bool) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.query_latency.record(latency);
    }

    /// Record a false positive (sampled).
    #[inline]
    pub fn record_false_positive(&self) {
        self.false_positive_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get insert count.
    pub fn insert_count(&self) -> u64 {
        self.insert_count.load(Ordering::Relaxed)
    }

    /// Get query count.
    pub fn query_count(&self) -> u64 {
        self.query_count.load(Ordering::Relaxed)
    }

    /// Get false positive count.
    pub fn false_positive_count(&self) -> u64 {
        self.false_positive_count.load(Ordering::Relaxed)
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.insert_count.store(0, Ordering::Relaxed);
        self.query_count.store(0, Ordering::Relaxed);
        self.false_positive_count.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.insert_latency.reset();
        self.query_latency.reset();
    }
}

impl Default for PartitionedFilterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency histogram with percentiles.
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Total operations recorded.
    count: AtomicU64,
    /// Sum of all latencies (nanoseconds).
    sum_ns: AtomicU64,
    /// Min latency (nanoseconds).
    min_ns: AtomicU64,
    /// Max latency (nanoseconds).
    max_ns: AtomicU64,
}

impl LatencyHistogram {
    /// Create new histogram.
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
        }
    }

    /// Record a latency measurement.
    #[inline]
    pub fn record(&self, duration: Duration) {
        let ns = duration.as_nanos() as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);

        // Update min
        self.min_ns.fetch_min(ns, Ordering::Relaxed);

        // Update max
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
    }

    /// Get mean latency.
    pub fn mean(&self) -> Duration {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        let sum = self.sum_ns.load(Ordering::Relaxed);
        Duration::from_nanos(sum / count)
    }

    /// Get min latency.
    pub fn min(&self) -> Duration {
        let min = self.min_ns.load(Ordering::Relaxed);
        if min == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_nanos(min)
        }
    }

    /// Get max latency.
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.max_ns.load(Ordering::Relaxed))
    }

    /// Get total count.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Reset histogram.
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.sum_ns.store(0, Ordering::Relaxed);
        self.min_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Health status for filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Healthy (< 70% saturation).
    Healthy,
    /// Degraded (70-90% saturation).
    Degraded,
    /// Critical (> 90% saturation).
    Critical,
}

/// Health check result.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Overall status.
    pub status: HealthStatus,
    /// Filter saturation (0.0 to 1.0).
    pub saturation: f64,
    /// Estimated false positive rate.
    pub estimated_fpr: f64,
    /// Warnings and recommendations.
    pub warnings: Vec<Warning>,
}

/// Warning types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Warning {
    /// Saturation exceeds threshold.
    HighSaturation { current: u8, threshold: u8 },
    /// FPR exceeds target.
    HighFalsePositiveRate { estimated: u32, target: u32 },
    /// Partition imbalance detected.
    PartitionImbalance { max_fill: u8, min_fill: u8 },
}

impl HealthCheck {
    /// Create health check from filter state.
    pub fn new(saturation: f64, estimated_fpr: f64, target_fpr: f64) -> Self {
        let mut warnings = Vec::new();

        // Check saturation
        let status = if saturation < 0.70 {
            HealthStatus::Healthy
        } else if saturation < 0.90 {
            warnings.push(Warning::HighSaturation {
                current: (saturation * 100.0) as u8,
                threshold: 70,
            });
            HealthStatus::Degraded
        } else {
            warnings.push(Warning::HighSaturation {
                current: (saturation * 100.0) as u8,
                threshold: 90,
            });
            HealthStatus::Critical
        };

        // Check FPR
        if estimated_fpr > target_fpr * 2.0 {
            warnings.push(Warning::HighFalsePositiveRate {
                estimated: (estimated_fpr * 1_000_000.0) as u32,
                target: (target_fpr * 1_000_000.0) as u32,
            });
        }

        Self {
            status,
            saturation,
            estimated_fpr,
            warnings,
        }
    }

    /// Check if filter is healthy.
    pub const fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy)
    }
}

/// Export metrics in Prometheus format.
pub fn export_prometheus(
    metrics: &PartitionedFilterMetrics,
    health: &HealthCheck,
) -> String {
    format!(
        r#"# HELP bloom_filter_inserts_total Total insert operations
# TYPE bloom_filter_inserts_total counter
bloom_filter_inserts_total{{type="partitioned"}} {}

# HELP bloom_filter_queries_total Total query operations
# TYPE bloom_filter_queries_total counter
bloom_filter_queries_total{{type="partitioned"}} {}

# HELP bloom_filter_false_positives_total Estimated false positives
# TYPE bloom_filter_false_positives_total counter
bloom_filter_false_positives_total{{type="partitioned"}} {}

# HELP bloom_filter_saturation Filter saturation (0.0-1.0)
# TYPE bloom_filter_saturation gauge
bloom_filter_saturation{{type="partitioned"}} {:.6}

# HELP bloom_filter_fpr Estimated false positive rate
# TYPE bloom_filter_fpr gauge
bloom_filter_fpr{{type="partitioned"}} {:.8}

# HELP bloom_filter_insert_latency_seconds Insert latency mean
# TYPE bloom_filter_insert_latency_seconds gauge
bloom_filter_insert_latency_seconds{{type="partitioned"}} {:.9}

# HELP bloom_filter_query_latency_seconds Query latency mean
# TYPE bloom_filter_query_latency_seconds gauge
bloom_filter_query_latency_seconds{{type="partitioned"}} {:.9}

# HELP bloom_filter_health_status Health status (0=healthy, 1=degraded, 2=critical)
# TYPE bloom_filter_health_status gauge
bloom_filter_health_status{{type="partitioned"}} {}
"#,
        metrics.insert_count(),
        metrics.query_count(),
        metrics.false_positive_count(),
        health.saturation,
        health.estimated_fpr,
        metrics.insert_latency.mean().as_secs_f64(),
        metrics.query_latency.mean().as_secs_f64(),
        match health.status {
            HealthStatus::Healthy => 0,
            HealthStatus::Degraded => 1,
            HealthStatus::Critical => 2,
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = PartitionedFilterMetrics::new();

        metrics.record_insert(Duration::from_nanos(100));
        metrics.record_query(Duration::from_nanos(50), true);

        assert_eq!(metrics.insert_count(), 1);
        assert_eq!(metrics.query_count(), 1);
    }

    #[test]
    fn test_latency_histogram() {
        let hist = LatencyHistogram::new();

        hist.record(Duration::from_nanos(100));
        hist.record(Duration::from_nanos(200));
        hist.record(Duration::from_nanos(300));

        assert_eq!(hist.count(), 3);
        assert_eq!(hist.mean(), Duration::from_nanos(200));
        assert_eq!(hist.min(), Duration::from_nanos(100));
        assert_eq!(hist.max(), Duration::from_nanos(300));
    }

    #[test]
    fn test_health_check() {
        let health = HealthCheck::new(0.5, 0.01, 0.01);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.is_healthy());

        let health = HealthCheck::new(0.75, 0.01, 0.01);
        assert_eq!(health.status, HealthStatus::Degraded);
        assert!(!health.is_healthy());

        let health = HealthCheck::new(0.95, 0.01, 0.01);
        assert_eq!(health.status, HealthStatus::Critical);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = PartitionedFilterMetrics::new();
        metrics.record_insert(Duration::from_micros(1));

        let health = HealthCheck::new(0.5, 0.01, 0.01);
        let output = export_prometheus(&metrics, &health);

        assert!(output.contains("bloom_filter_inserts_total"));
        assert!(output.contains("bloom_filter_saturation"));
        assert!(output.contains("bloom_filter_fpr"));
    }
}
