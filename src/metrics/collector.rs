//! Unified metrics collection interface.
//!
//! Provides a central collector that aggregates all metrics types
//! (false positive tracking, latency histograms, operation counts) with
//! export capabilities for monitoring systems.
//!
//! # Design
//!
//! The collector acts as a facade over individual metric types, providing:
//! - Unified recording interface
//! - Aggregated snapshots
//! - Export to multiple formats (JSON, Prometheus)
//! - Labeled metrics for multi-dimensional analysis
//!
//! # Examples
//!
//! ## Basic Collection
//!
//! ```
//! use bloomcraft::metrics::MetricsCollector;
//!
//! let metrics = MetricsCollector::new();
//!
//! metrics.record_insert();
//! metrics.record_query(true);
//! metrics.record_query(false);
//!
//! let snapshot = metrics.snapshot();
//! println!("Queries: {}, Inserts: {}", 
//!     snapshot.total_queries, snapshot.total_inserts);
//! ```

use super::{FalsePositiveTracker, LatencyHistogram, LatencyStats};
use super::tracker::FpTrackerSnapshot;
use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Unified metrics collector for Bloom filters.
///
/// Aggregates all metric types with a simple recording interface.
/// Thread-safe and suitable for concurrent use.
pub struct MetricsCollector {
    /// False positive rate tracking
    fp_tracker: Arc<FalsePositiveTracker>,
    /// Query latency histogram
    query_latency: Option<Arc<LatencyHistogram>>,
    /// Insert latency histogram
    insert_latency: Option<Arc<LatencyHistogram>>,
    /// Operation counters
    counters: Arc<OperationCounters>,
    /// Start time for uptime tracking
    start_time: Instant,
}

/// Operation counters for basic metrics.
struct OperationCounters {
    inserts: AtomicU64,
    queries: AtomicU64,
    removes: AtomicU64,
    clears: AtomicU64,
}

impl OperationCounters {
    fn new() -> Self {
        Self {
            inserts: AtomicU64::new(0),
            queries: AtomicU64::new(0),
            removes: AtomicU64::new(0),
            clears: AtomicU64::new(0),
        }
    }
}

impl MetricsCollector {
    /// Create a new metrics collector with basic tracking.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::metrics::MetricsCollector;
    ///
    /// let metrics = MetricsCollector::new();
    /// ```
    pub fn new() -> Self {
        Self {
            fp_tracker: Arc::new(FalsePositiveTracker::new(10_000)),
            query_latency: None,
            insert_latency: None,
            counters: Arc::new(OperationCounters::new()),
            start_time: Instant::now(),
        }
    }

    /// Create a collector with latency histogram tracking enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::metrics::MetricsCollector;
    ///
    /// let metrics = MetricsCollector::with_histogram();
    /// ```
    pub fn with_histogram() -> Self {
        Self {
            fp_tracker: Arc::new(FalsePositiveTracker::new(10_000)),
            query_latency: Some(Arc::new(LatencyHistogram::new())),
            insert_latency: Some(Arc::new(LatencyHistogram::new())),
            counters: Arc::new(OperationCounters::new()),
            start_time: Instant::now(),
        }
    }

    /// Create a collector with custom configuration.
    pub fn with_config(expected_items: usize, track_latency: bool) -> Self {
        Self {
            fp_tracker: Arc::new(FalsePositiveTracker::new(expected_items)),
            query_latency: if track_latency {
                Some(Arc::new(LatencyHistogram::new()))
            } else {
                None
            },
            insert_latency: if track_latency {
                Some(Arc::new(LatencyHistogram::new()))
            } else {
                None
            },
            counters: Arc::new(OperationCounters::new()),
            start_time: Instant::now(),
        }
    }

    /// Record an insert operation.
    ///
    /// This is lock-free and extremely fast (~5ns).
    pub fn record_insert(&self) {
        self.counters.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an insert operation with latency.
    pub fn record_insert_latency(&self, latency: Duration) {
        self.record_insert();
        if let Some(ref histogram) = self.insert_latency {
            histogram.record(latency);
        }
    }

    /// Record a query operation.
    ///
    /// # Arguments
    ///
    /// * `result` - Whether the filter returned positive (true) or negative (false)
    pub fn record_query(&self, result: bool) {
        self.counters.queries.fetch_add(1, Ordering::Relaxed);
        if result {
            self.fp_tracker.record_positive();
        } else {
            self.fp_tracker.record_negative();
        }
    }

    /// Record a query operation with latency.
    pub fn record_query_latency(&self, result: bool, latency: Duration) {
        self.record_query(result);
        if let Some(ref histogram) = self.query_latency {
            histogram.record(latency);
        }
    }

    /// Record a confirmed query result.
    ///
    /// Use this when you can verify if the item is actually in the set.
    ///
    /// # Arguments
    ///
    /// * `filter_result` - What the filter returned
    /// * `actually_present` - Whether the item is actually in the set
    pub fn record_confirmed_query(&self, filter_result: bool, actually_present: bool) {
        self.counters.queries.fetch_add(1, Ordering::Relaxed);
        self.fp_tracker.record_confirmed(filter_result, actually_present);
    }

    /// Record a remove operation.
    pub fn record_remove(&self) {
        self.counters.removes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a clear operation.
    pub fn record_clear(&self) {
        self.counters.clears.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total insert count.
    pub fn total_inserts(&self) -> u64 {
        self.counters.inserts.load(Ordering::Relaxed)
    }

    /// Get total query count.
    pub fn total_queries(&self) -> u64 {
        self.counters.queries.load(Ordering::Relaxed)
    }

    /// Get total remove count.
    pub fn total_removes(&self) -> u64 {
        self.counters.removes.load(Ordering::Relaxed)
    }

    /// Get uptime duration.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get query latency histogram.
    pub fn query_latency_histogram(&self) -> Option<&LatencyHistogram> {
        self.query_latency.as_deref()
    }

    /// Get insert latency histogram.
    pub fn insert_latency_histogram(&self) -> Option<&LatencyHistogram> {
        self.insert_latency.as_deref()
    }

    /// Get false positive tracker.
    pub fn fp_tracker(&self) -> &FalsePositiveTracker {
        &self.fp_tracker
    }

    /// Get complete metrics snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_inserts: self.total_inserts(),
            total_queries: self.total_queries(),
            total_removes: self.total_removes(),
            uptime: self.uptime(),
            fp_tracker: self.fp_tracker.snapshot(),
            query_latency: self.query_latency.as_ref().map(|h| h.snapshot()),
            insert_latency: self.insert_latency.as_ref().map(|h| h.snapshot()),
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.counters.inserts.store(0, Ordering::Relaxed);
        self.counters.queries.store(0, Ordering::Relaxed);
        self.counters.removes.store(0, Ordering::Relaxed);
        self.counters.clears.store(0, Ordering::Relaxed);

        self.fp_tracker.reset();

        if let Some(ref histogram) = self.query_latency {
            histogram.reset();
        }
        if let Some(ref histogram) = self.insert_latency {
            histogram.reset();
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            fp_tracker: Arc::clone(&self.fp_tracker),
            query_latency: self.query_latency.as_ref().map(Arc::clone),
            insert_latency: self.insert_latency.as_ref().map(Arc::clone),
            counters: Arc::clone(&self.counters),
            start_time: self.start_time,
        }
    }
}

/// Complete snapshot of all metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total number of insert operations performed.
    pub total_inserts: u64,
    /// Total number of query operations performed.
    pub total_queries: u64,
    /// Total number of remove operations performed.
    pub total_removes: u64,
    /// Time elapsed since the collector was created.
    pub uptime: Duration,
    /// Snapshot of false positive tracking statistics.
    pub fp_tracker: FpTrackerSnapshot,
    /// Query latency statistics (if histogram tracking enabled).
    pub query_latency: Option<LatencyStats>,
    /// Insert latency statistics (if histogram tracking enabled).
    pub insert_latency: Option<LatencyStats>,
}

impl MetricsSnapshot {
    /// Calculate queries per second.
    pub fn queries_per_second(&self) -> f64 {
        let secs = self.uptime.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_queries as f64 / secs
    }

    /// Calculate inserts per second.
    pub fn inserts_per_second(&self) -> f64 {
        let secs = self.uptime.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_inserts as f64 / secs
    }

    /// Export to JSON string.
    ///
    /// This method is only available when the `serde` feature is enabled.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        #[derive(serde::Serialize)]
        struct JsonSnapshot {
            total_inserts: u64,
            total_queries: u64,
            total_removes: u64,
            uptime_secs: f64,
            queries_per_second: f64,
            inserts_per_second: f64,
            false_positive_rate: f64,
            query_latency: Option<JsonLatencyStats>,
            insert_latency: Option<JsonLatencyStats>,
        }

        #[derive(serde::Serialize)]
        struct JsonLatencyStats {
            mean_us: u128,
            min_us: u128,
            max_us: u128,
            p50_us: u128,
            p90_us: u128,
            p95_us: u128,
            p99_us: u128,
        }

        let json_snapshot = JsonSnapshot {
            total_inserts: self.total_inserts,
            total_queries: self.total_queries,
            total_removes: self.total_removes,
            uptime_secs: self.uptime.as_secs_f64(),
            queries_per_second: self.queries_per_second(),
            inserts_per_second: self.inserts_per_second(),
            false_positive_rate: self.fp_tracker.current_fp_rate,
            query_latency: self.query_latency.as_ref().map(|stats| JsonLatencyStats {
                mean_us: stats.mean.as_micros(),
                min_us: stats.min.as_micros(),
                max_us: stats.max.as_micros(),
                p50_us: stats.p50.as_micros(),
                p90_us: stats.p90.as_micros(),
                p95_us: stats.p95.as_micros(),
                p99_us: stats.p99.as_micros(),
            }),
            insert_latency: self.insert_latency.as_ref().map(|stats| JsonLatencyStats {
                mean_us: stats.mean.as_micros(),
                min_us: stats.min.as_micros(),
                max_us: stats.max.as_micros(),
                p50_us: stats.p50.as_micros(),
                p90_us: stats.p90.as_micros(),
                p95_us: stats.p95.as_micros(),
                p99_us: stats.p99.as_micros(),
            }),
        };

        serde_json::to_string_pretty(&json_snapshot)
            .map_err(|e| crate::error::BloomCraftError::serialization_error(e.to_string()))
    }

    /// Export to Prometheus format.
    pub fn to_prometheus_format(&self, prefix: &str) -> String {
        let mut lines = Vec::new();

        // Operation counters
        lines.push(format!("# HELP {}_inserts_total Total number of insert operations", prefix));
        lines.push(format!("# TYPE {}_inserts_total counter", prefix));
        lines.push(format!("{}_inserts_total {}", prefix, self.total_inserts));

        lines.push(format!("# HELP {}_queries_total Total number of query operations", prefix));
        lines.push(format!("# TYPE {}_queries_total counter", prefix));
        lines.push(format!("{}_queries_total {}", prefix, self.total_queries));

        // Throughput
        lines.push(format!("# HELP {}_queries_per_second Current query rate", prefix));
        lines.push(format!("# TYPE {}_queries_per_second gauge", prefix));
        lines.push(format!("{}_queries_per_second {:.2}", prefix, self.queries_per_second()));

        // False positive rate
        lines.push(format!("# HELP {}_false_positive_rate Current false positive rate", prefix));
        lines.push(format!("# TYPE {}_false_positive_rate gauge", prefix));
        lines.push(format!("{}_false_positive_rate {:.6}", prefix, self.fp_tracker.current_fp_rate));

        // Query latency
        if let Some(ref stats) = self.query_latency {
            lines.push(format!("# HELP {}_query_latency_seconds Query latency", prefix));
            lines.push(format!("# TYPE {}_query_latency_seconds summary", prefix));
            lines.push(format!("{}_query_latency_seconds{{quantile=\"0.5\"}} {:.9}", 
                prefix, stats.p50.as_secs_f64()));
            lines.push(format!("{}_query_latency_seconds{{quantile=\"0.9\"}} {:.9}", 
                prefix, stats.p90.as_secs_f64()));
            lines.push(format!("{}_query_latency_seconds{{quantile=\"0.99\"}} {:.9}", 
                prefix, stats.p99.as_secs_f64()));
        }

        lines.join("
")
    }
}

/// Trait for types that can expose filter metrics.
pub trait FilterMetrics {
    /// Get metrics collector.
    fn metrics(&self) -> Option<&MetricsCollector>;

    /// Record an insert operation.
    fn record_insert_metric(&self) {
        if let Some(metrics) = self.metrics() {
            metrics.record_insert();
        }
    }

    /// Record a query operation.
    fn record_query_metric(&self, result: bool) {
        if let Some(metrics) = self.metrics() {
            metrics.record_query(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_collector() {
        let collector = MetricsCollector::new();
        assert_eq!(collector.total_inserts(), 0);
        assert_eq!(collector.total_queries(), 0);
    }

    #[test]
    fn test_record_operations() {
        let collector = MetricsCollector::new();

        collector.record_insert();
        collector.record_query(true);
        collector.record_query(false);

        assert_eq!(collector.total_inserts(), 1);
        assert_eq!(collector.total_queries(), 2);
    }

    #[test]
    fn test_with_histogram() {
        let collector = MetricsCollector::with_histogram();

        collector.record_query_latency(true, Duration::from_micros(100));
        collector.record_insert_latency(Duration::from_micros(50));

        assert!(collector.query_latency_histogram().is_some());
        assert!(collector.insert_latency_histogram().is_some());
    }

    #[test]
    fn test_snapshot() {
        let collector = MetricsCollector::new();

        collector.record_insert();
        collector.record_query(true);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.total_inserts, 1);
        assert_eq!(snapshot.total_queries, 1);
    }

    #[test]
    fn test_throughput_calculation() {
        let collector = MetricsCollector::new();

        for _ in 0..1000 {
            collector.record_query(true);
        }

        std::thread::sleep(Duration::from_millis(100));

        let snapshot = collector.snapshot();
        let qps = snapshot.queries_per_second();
        assert!(qps > 0.0);
    }

    #[test]
    fn test_reset() {
        let collector = MetricsCollector::new();

        collector.record_insert();
        collector.record_query(true);

        assert_eq!(collector.total_inserts(), 1);

        collector.reset();

        assert_eq!(collector.total_inserts(), 0);
        assert_eq!(collector.total_queries(), 0);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_json_export() {
        let collector = MetricsCollector::with_histogram();

        collector.record_insert();
        collector.record_query_latency(true, Duration::from_micros(100));

        let snapshot = collector.snapshot();
        let json = snapshot.to_json().unwrap();

        assert!(json.contains("total_inserts"));
        assert!(json.contains("total_queries"));
    }

    #[test]
    fn test_prometheus_export() {
        let collector = MetricsCollector::new();

        collector.record_insert();
        collector.record_query(true);

        let snapshot = collector.snapshot();
        let prometheus = snapshot.to_prometheus_format("bloomcraft");

        assert!(prometheus.contains("bloomcraft_inserts_total"));
        assert!(prometheus.contains("bloomcraft_queries_total"));
    }

    #[test]
    fn test_clone() {
        let collector = MetricsCollector::new();
        collector.record_insert();

        let cloned = collector.clone();
        assert_eq!(cloned.total_inserts(), 1);

        // Both share the same underlying data
        cloned.record_insert();
        assert_eq!(collector.total_inserts(), 2);
    }

    #[test]
    fn test_concurrent_collection() {
        use std::sync::Arc;
        use std::thread;

        let collector = Arc::new(MetricsCollector::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let collector_clone = Arc::clone(&collector);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    collector_clone.record_insert();
                    collector_clone.record_query(true);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(collector.total_inserts(), 10_000);
        assert_eq!(collector.total_queries(), 10_000);
    }

    #[test]
    fn test_confirmed_queries() {
        let collector = MetricsCollector::new();

        // True positive
        collector.record_confirmed_query(true, true);
        // False positive
        collector.record_confirmed_query(true, false);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.fp_tracker.total_positives, 2);
        assert_eq!(snapshot.fp_tracker.true_positives, 1);
    }
}
