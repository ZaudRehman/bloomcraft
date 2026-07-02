//! Metrics collection for Bloom filters.
//!
//! Tracks false positive rates, operation counts, and latency distributions.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────┐
//! │                MetricsCollector               │
//! │  (facade aggregating all metric types)        │
//! └───────────────────────────────────────────────┘
//!                        │
//!         ┌──────────────┼──────────────┐
//!         │              │              │
//!   ┌─────▼─────┐  ┌─────▼──────┐  ┌───▼──────┐
//!   │FpTracker  │  │LatencyHist │  │ Counters │
//!   └───────────┘  └────────────┘  └──────────┘
//! ```
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`collector`] | `MetricsCollector` — unified facade |
//! | [`histogram`] | `LatencyHistogram` — percentile tracking |
//! | [`tracker`] | `FalsePositiveTracker` — FP rate tracking |
//! | [`partitioned_metrics`] | Metrics + health checks for `PartitionedBloomFilter` (feature `metrics`) |
//!
//! # Examples
//!
//! ```
//! use bloomcraft::metrics::MetricsCollector;
//!
//! let m = MetricsCollector::new();
//! m.record_insert();
//! m.record_query(true);
//! let s = m.snapshot();
//! println!("queries={} inserts={}", s.total_queries, s.total_inserts);
//! ```
//!
//! # Thread Safety
//!
//! All types are `Send + Sync`. Recording operations use relaxed atomic
//! loads/stores. Snapshot methods that lock (e.g., sliding window reads)
//! use `std::sync::Mutex` internally.

pub mod collector;
pub mod histogram;
pub mod tracker;

#[cfg(feature = "metrics")]
pub mod partitioned_metrics;

pub use collector::{MetricsCollector, FilterMetrics, MetricsSnapshot};
pub use histogram::{LatencyHistogram, LatencyStats};
pub use tracker::{FalsePositiveTracker, FpTrackerConfig};

#[cfg(feature = "metrics")]
pub use partitioned_metrics::{
    PartitionedFilterMetrics, 
    HealthCheck, 
    HealthStatus,
    export_prometheus
};

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

}
