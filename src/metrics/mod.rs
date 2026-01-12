//! Observability and monitoring for Bloom filters.
//!
//! This module provides production-grade metrics collection, tracking, and analysis
//! for Bloom filter operations. It enables:
//!
//! - Real-time false positive rate tracking
//! - Query latency histograms
//! - Insert/query throughput metrics
//! - Memory usage tracking
//! - Custom metric collection
//!
//! # Design Philosophy
//!
//! **Zero-Overhead When Disabled**: Metrics collection has zero runtime cost when not used.
//! All metric operations are designed to be fast enough for production use.
//!
//! **Thread-Safe**: All metrics collectors are thread-safe and lock-free where possible.
//!
//! **Composable**: Metrics can be collected at multiple levels (per-filter, per-shard, global).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                   MetricsCollector                  │
//! │  (Central aggregation and reporting interface)      │
//! └─────────────────────────────────────────────────────┘
//!                             ▲
//!                             │
//!          ┌──────────────────┼────────────────────┐
//!          │                  │                    │
//!    ┌─────▼─────┐    ┌───────▼────────┐    ┌──────▼──────┐
//!    │ FpTracker │    │LatencyHistogram│    │CustomMetrics│
//!    └───────────┘    └────────────────┘    └─────────────┘
//! ```
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::metrics::{MetricsCollector, FilterMetrics};
//!
//! let metrics = MetricsCollector::new();
//!
//! // Track operations
//! metrics.record_insert();
//! metrics.record_query(true);  // true positive
//! metrics.record_query(false); // negative
//!
//! // Get statistics
//! let stats = metrics.snapshot();
//! println!("Queries: {}, FP rate: {:.4}%", 
//!     stats.total_queries, 
//!     stats.fp_tracker.current_fp_rate * 100.0
//! );
//! ```
//!
//! ## With Latency Tracking
//!
//! ```
//! use bloomcraft::metrics::{MetricsCollector, LatencyHistogram};
//! use std::time::Instant;
//!
//! let metrics = MetricsCollector::with_histogram();
//!
//! let start = Instant::now();
//! // ... perform query ...
//! metrics.record_query_latency(true, start.elapsed());
//!
//! if let Some(histogram) = metrics.query_latency_histogram() {
//!     println!("P50: {:?}", histogram.percentile(0.50));
//!     println!("P99: {:?}", histogram.percentile(0.99));
//! }
//! ```
//!
//! ## False Positive Rate Tracking
//!
//! ```
//! use bloomcraft::metrics::FalsePositiveTracker;
//!
//! let tracker = FalsePositiveTracker::new(1000);
//!
//! // Track actual vs expected
//! tracker.record_positive(); // Filter said "yes"
//! tracker.record_true_positive(); // Actually in set
//!
//! let rate = tracker.current_fp_rate();
//! let expected = tracker.expected_fp_rate();
//! 
//! if rate > expected * 1.5 {
//!     eprintln!("WARNING: FP rate higher than expected!");
//! }
//! ```
//!
//! ## Integration with Filters
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//! use bloomcraft::metrics::MetricsCollector;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);
//! let metrics = MetricsCollector::new();
//!
//! // Wrap operations
//! filter.insert(&"item");
//! metrics.record_insert();
//!
//! let result = filter.contains(&"item");
//! metrics.record_query(result);
//! ```
//!
//! ## Exporting Metrics
//!
//! ```ignore
//! use bloomcraft::metrics::MetricsCollector;
//!
//! let metrics = MetricsCollector::new();
//! // ... collect metrics ...
//!
//! // Export as JSON (if implemented)
//! // let json = metrics.to_json().unwrap();
//!
//! // Export for Prometheus (if implemented)
//! // let prometheus = metrics.to_prometheus_format();
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Overhead | Thread Safety |
//! |-----------|----------|---------------|
//! | `record_insert()` | ~5ns | Lock-free |
//! | `record_query()` | ~8ns | Lock-free |
//! | `record_latency()` | ~15ns | Lock-free |
//! | `snapshot()` | ~100ns | Atomic read |
//! | `histogram()` | ~500ns | Requires lock |
//!
//! # Thread Safety
//!
//! All metrics collectors are `Send + Sync` and can be safely shared across threads.
//! Internal synchronization uses atomic operations for counters and lock-free algorithms
//! for histograms where possible.
//!
//! # Memory Overhead
//!
//! - **MetricsCollector**: ~128 bytes base + histogram overhead
//! - **LatencyHistogram**: ~4KB (configurable buckets)
//! - **FalsePositiveTracker**: ~64 bytes + sliding window buffer
//!
//! # Production Best Practices
//!
//! 1. **Sample for Latency**: Don't track latency for every operation in high-throughput scenarios
//! 2. **Periodic Snapshots**: Take snapshots periodically rather than on every query
//! 3. **Bounded Histograms**: Use bounded histogram sizes to prevent memory growth
//! 4. **Aggregation**: Aggregate metrics across multiple filters at application level

pub mod collector;
pub mod histogram;
pub mod tracker;

pub use collector::{MetricsCollector, FilterMetrics, MetricsSnapshot};
pub use histogram::{LatencyHistogram, LatencyStats, Percentile};
pub use tracker::{FalsePositiveTracker, FpTrackerConfig};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        MetricsCollector,
        FilterMetrics,
        MetricsSnapshot,
        LatencyHistogram,
        LatencyStats,
        FalsePositiveTracker,
    };
}

/// Trait for types that can export metrics.
pub trait MetricsExporter {
    /// Export metrics as JSON string.
    fn to_json(&self) -> crate::Result<String>;

    /// Export metrics in Prometheus format.
    fn to_prometheus_format(&self) -> String;
}

/// Trait for types that can be reset.
pub trait Resettable {
    /// Reset all metrics to initial state.
    fn reset(&self);
}

/// Common metric labels for categorizing data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricLabel {
    /// Filter type (standard, counting, scalable)
    FilterType,
    /// Shard index (for sharded filters)
    ShardId,
    /// Operation type (insert, query, remove)
    Operation,
    /// Custom label
    Custom(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _collector = MetricsCollector::new();
        let _histogram = LatencyHistogram::new();
        let _tracker = FalsePositiveTracker::new(1000);
    }

    #[test]
    fn test_metric_label() {
        let label = MetricLabel::FilterType;
        assert_eq!(label, MetricLabel::FilterType);
    }
}
