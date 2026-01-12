//! Query latency histograms and percentile tracking.
//!
//! Provides high-performance latency tracking with configurable precision
//! and percentile calculation. Uses a logarithmic bucketing strategy for
//! efficient memory usage while maintaining accuracy.
//!
//! # Design
//!
//! The histogram uses:
//! - **Logarithmic buckets** for wide range coverage with bounded memory
//! - **Lock-free recording** for minimal overhead
//! - **Percentile calculation** with linear interpolation
//! - **Configurable precision** via bucket count
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::metrics::LatencyHistogram;
//! use std::time::Duration;
//!
//! let histogram = LatencyHistogram::new();
//!
//! histogram.record(Duration::from_micros(100));
//! histogram.record(Duration::from_micros(200));
//! histogram.record(Duration::from_millis(1));
//!
//! let stats = histogram.snapshot();
//! println!("Mean: {:?}", stats.mean);
//! println!("P50: {:?}", stats.percentile(0.50));
//! println!("P99: {:?}", stats.percentile(0.99));
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Number of histogram buckets (logarithmic scale).
const DEFAULT_BUCKET_COUNT: usize = 64;

/// Minimum latency bucket (1 nanosecond).
const MIN_LATENCY_NS: u64 = 1;

/// Maximum latency bucket (1 minute).
const MAX_LATENCY_NS: u64 = 60_000_000_000;

/// Latency histogram with percentile calculation.
///
/// Thread-safe with lock-free recording. Percentile calculation requires a lock.
pub struct LatencyHistogram {
    /// Logarithmic buckets for latency distribution
    buckets: Vec<AtomicU64>,
    /// Bucket boundaries in nanoseconds
    boundaries: Vec<u64>,
    /// Total count of samples
    count: AtomicU64,
    /// Sum of all latencies (for mean calculation)
    sum_nanos: AtomicU64,
    /// Minimum observed latency
    min_nanos: AtomicU64,
    /// Maximum observed latency
    max_nanos: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new histogram with default bucket count.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::metrics::LatencyHistogram;
    ///
    /// let histogram = LatencyHistogram::new();
    /// ```
    pub fn new() -> Self {
        Self::with_buckets(DEFAULT_BUCKET_COUNT)
    }

    /// Create a histogram with custom bucket count.
    ///
    /// More buckets = higher precision but more memory.
    pub fn with_buckets(bucket_count: usize) -> Self {
        let boundaries = Self::compute_boundaries(bucket_count);
        let buckets = (0..bucket_count)
            .map(|_| AtomicU64::new(0))
            .collect();

        Self {
            buckets,
            boundaries,
            count: AtomicU64::new(0),
            sum_nanos: AtomicU64::new(0),
            min_nanos: AtomicU64::new(u64::MAX),
            max_nanos: AtomicU64::new(0),
        }
    }

    /// Compute logarithmic bucket boundaries.
    fn compute_boundaries(count: usize) -> Vec<u64> {
        let log_min = (MIN_LATENCY_NS as f64).ln();
        let log_max = (MAX_LATENCY_NS as f64).ln();
        let log_step = (log_max - log_min) / (count as f64);

        (0..count)
            .map(|i| {
                let log_value = log_min + (i as f64) * log_step;
                log_value.exp() as u64
            })
            .collect()
    }

    /// Find bucket index for a given latency.
    fn find_bucket(&self, nanos: u64) -> usize {
        // Binary search for efficiency
        match self.boundaries.binary_search(&nanos) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        }
    }

    /// Record a latency measurement.
    ///
    /// This is lock-free and very fast (~15ns overhead).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::metrics::LatencyHistogram;
    /// use std::time::{Duration, Instant};
    ///
    /// let histogram = LatencyHistogram::new();
    ///
    /// let start = Instant::now();
    /// // ... perform operation ...
    /// histogram.record(start.elapsed());
    /// ```
    pub fn record(&self, latency: Duration) {
        let nanos = latency.as_nanos() as u64;

        // Update count and sum
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_nanos.fetch_add(nanos, Ordering::Relaxed);

        // Update min/max
        self.update_min(nanos);
        self.update_max(nanos);

        // Update histogram bucket
        let bucket = self.find_bucket(nanos);
        if bucket < self.buckets.len() {
            self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Update minimum observed latency.
    fn update_min(&self, nanos: u64) {
        let mut current = self.min_nanos.load(Ordering::Relaxed);
        while nanos < current {
            match self.min_nanos.compare_exchange_weak(
                current,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    /// Update maximum observed latency.
    fn update_max(&self, nanos: u64) {
        let mut current = self.max_nanos.load(Ordering::Relaxed);
        while nanos > current {
            match self.max_nanos.compare_exchange_weak(
                current,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    /// Get total count of recorded samples.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get mean latency.
    pub fn mean(&self) -> Duration {
        let count = self.count();
        if count == 0 {
            return Duration::ZERO;
        }
        let sum = self.sum_nanos.load(Ordering::Relaxed);
        Duration::from_nanos(sum / count)
    }

    /// Get minimum latency.
    pub fn min(&self) -> Duration {
        let min = self.min_nanos.load(Ordering::Relaxed);
        if min == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_nanos(min)
        }
    }

    /// Get maximum latency.
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.max_nanos.load(Ordering::Relaxed))
    }

    /// Calculate percentile value.
    ///
    /// # Arguments
    ///
    /// * `p` - Percentile to calculate (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::metrics::LatencyHistogram;
    /// use std::time::Duration;
    ///
    /// let histogram = LatencyHistogram::new();
    /// for i in 1..=100 {
    ///     histogram.record(Duration::from_micros(i));
    /// }
    ///
    /// let p50 = histogram.percentile(0.50);
    /// let p99 = histogram.percentile(0.99);
    /// ```
    pub fn percentile(&self, p: f64) -> Duration {
        assert!((0.0..=1.0).contains(&p), "Percentile must be between 0.0 and 1.0");

        let count = self.count();
        if count == 0 {
            return Duration::ZERO;
        }

        let target_count = (count as f64 * p) as u64;
        let mut cumulative = 0u64;

        for (i, bucket) in self.buckets.iter().enumerate() {
            let bucket_count = bucket.load(Ordering::Relaxed);
            cumulative += bucket_count;

            if cumulative >= target_count {
                // Linear interpolation within bucket
                let boundary = self.boundaries[i];
                let next_boundary = self.boundaries.get(i + 1)
                    .copied()
                    .unwrap_or(MAX_LATENCY_NS);

                let ratio = if bucket_count > 0 {
                    (target_count - (cumulative - bucket_count)) as f64 / bucket_count as f64
                } else {
                    0.5
                };

                let interpolated = boundary as f64 + 
                    ratio * (next_boundary - boundary) as f64;

                return Duration::from_nanos(interpolated as u64);
            }
        }

        self.max()
    }

    /// Get snapshot of statistics.
    pub fn snapshot(&self) -> LatencyStats {
        LatencyStats {
            count: self.count(),
            mean: self.mean(),
            min: self.min(),
            max: self.max(),
            p50: self.percentile(0.50),
            p90: self.percentile(0.90),
            p95: self.percentile(0.95),
            p99: self.percentile(0.99),
            p999: self.percentile(0.999),
        }
    }

    /// Reset all histogram data.
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
        self.sum_nanos.store(0, Ordering::Relaxed);
        self.min_nanos.store(u64::MAX, Ordering::Relaxed);
        self.max_nanos.store(0, Ordering::Relaxed);
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LatencyHistogram {
    fn clone(&self) -> Self {
        let buckets = self.buckets.iter()
            .map(|b| AtomicU64::new(b.load(Ordering::Relaxed)))
            .collect();

        Self {
            buckets,
            boundaries: self.boundaries.clone(),
            count: AtomicU64::new(self.count.load(Ordering::Relaxed)),
            sum_nanos: AtomicU64::new(self.sum_nanos.load(Ordering::Relaxed)),
            min_nanos: AtomicU64::new(self.min_nanos.load(Ordering::Relaxed)),
            max_nanos: AtomicU64::new(self.max_nanos.load(Ordering::Relaxed)),
        }
    }
}

/// Snapshot of latency statistics.
#[derive(Debug, Clone, Copy)]
pub struct LatencyStats {
    /// Total number of samples recorded.
    pub count: u64,
    /// Mean (average) latency.
    pub mean: Duration,
    /// Minimum observed latency.
    pub min: Duration,
    /// Maximum observed latency.
    pub max: Duration,
    /// 50th percentile (median) latency.
    pub p50: Duration,
    /// 90th percentile latency.
    pub p90: Duration,
    /// 95th percentile latency.
    pub p95: Duration,
    /// 99th percentile latency.
    pub p99: Duration,
    /// 99.9th percentile latency.
    pub p999: Duration,
}

impl LatencyStats {
    /// Get percentile by value.
    pub fn percentile(&self, p: f64) -> Duration {
        match p {
            x if x <= 0.50 => self.p50,
            x if x <= 0.90 => self.p90,
            x if x <= 0.95 => self.p95,
            x if x <= 0.99 => self.p99,
            x if x <= 0.999 => self.p999,
            _ => self.max,
        }
    }
}

/// Percentile value with metadata.
#[derive(Debug, Clone, Copy)]
pub struct Percentile {
    /// The latency value at this percentile.
    pub value: Duration,
    /// The percentile (0.0 to 1.0).
    pub percentile: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_histogram() {
        let histogram = LatencyHistogram::new();
        assert_eq!(histogram.count(), 0);
        assert_eq!(histogram.mean(), Duration::ZERO);
    }

    #[test]
    fn test_record_single() {
        let histogram = LatencyHistogram::new();
        histogram.record(Duration::from_micros(100));

        assert_eq!(histogram.count(), 1);
        assert_eq!(histogram.mean(), Duration::from_micros(100));
        assert_eq!(histogram.min(), Duration::from_micros(100));
        assert_eq!(histogram.max(), Duration::from_micros(100));
    }

    #[test]
    fn test_record_multiple() {
        let histogram = LatencyHistogram::new();

        histogram.record(Duration::from_micros(100));
        histogram.record(Duration::from_micros(200));
        histogram.record(Duration::from_micros(300));

        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.mean(), Duration::from_micros(200));
        assert_eq!(histogram.min(), Duration::from_micros(100));
        assert_eq!(histogram.max(), Duration::from_micros(300));
    }

    #[test]
    fn test_percentile_calculation() {
        let histogram = LatencyHistogram::new();

        // Record 1-100 microseconds
        for i in 1..=100 {
            histogram.record(Duration::from_micros(i));
        }

        let p50 = histogram.percentile(0.50);
        let p90 = histogram.percentile(0.90);
        let p99 = histogram.percentile(0.99);

        // P50 should be around 50us (allow wider range due to bucket quantization)
        assert!(p50.as_micros() >= 35 && p50.as_micros() <= 65,
            "P50 {} not in expected range [35, 65]", p50.as_micros());
        // P90 should be around 90us
        assert!(p90.as_micros() >= 70 && p90.as_micros() <= 110,
            "P90 {} not in expected range [70, 110]", p90.as_micros());
        // P99 should be around 99us (logarithmic buckets cause quantization)
        assert!(p99.as_micros() >= 80 && p99.as_micros() <= 120,
            "P99 {} not in expected range [80, 120]", p99.as_micros());
    }

    #[test]
    fn test_snapshot() {
        let histogram = LatencyHistogram::new();

        for i in 1..=100 {
            histogram.record(Duration::from_micros(i));
        }

        let stats = histogram.snapshot();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.mean.as_micros(), 50); // Mean of 1-100
        assert_eq!(stats.min.as_micros(), 1);
        assert_eq!(stats.max.as_micros(), 100);
    }

    #[test]
    fn test_reset() {
        let histogram = LatencyHistogram::new();

        histogram.record(Duration::from_micros(100));
        assert_eq!(histogram.count(), 1);

        histogram.reset();

        assert_eq!(histogram.count(), 0);
        assert_eq!(histogram.mean(), Duration::ZERO);
    }

    #[test]
    fn test_wide_range() {
        let histogram = LatencyHistogram::new();

        histogram.record(Duration::from_nanos(100));
        histogram.record(Duration::from_micros(100));
        histogram.record(Duration::from_millis(100));

        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.min(), Duration::from_nanos(100));
        assert_eq!(histogram.max(), Duration::from_millis(100));
    }

    #[test]
    fn test_concurrent_recording() {
        use std::sync::Arc;
        use std::thread;

        let histogram = Arc::new(LatencyHistogram::new());
        let mut handles = vec![];

        for i in 0..10 {
            let histogram_clone = Arc::clone(&histogram);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    histogram_clone.record(Duration::from_micros((i * 100 + j) as u64));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(histogram.count(), 1000);
    }

    #[test]
    fn test_custom_bucket_count() {
        let histogram = LatencyHistogram::with_buckets(128);
        histogram.record(Duration::from_micros(100));
        assert_eq!(histogram.count(), 1);
    }

    #[test]
    #[should_panic(expected = "Percentile must be between 0.0 and 1.0")]
    fn test_invalid_percentile() {
        let histogram = LatencyHistogram::new();
        histogram.percentile(1.5);
    }

    #[test]
    fn test_empty_histogram_percentile() {
        let histogram = LatencyHistogram::new();
        let p50 = histogram.percentile(0.50);
        assert_eq!(p50, Duration::ZERO);
    }

    #[test]
    fn test_clone() {
        let histogram = LatencyHistogram::new();
        histogram.record(Duration::from_micros(100));

        let cloned = histogram.clone();
        assert_eq!(cloned.count(), 1);
        assert_eq!(cloned.mean(), Duration::from_micros(100));
    }

    #[test]
    fn test_latency_stats_percentile_method() {
        let stats = LatencyStats {
            count: 100,
            mean: Duration::from_micros(50),
            min: Duration::from_micros(1),
            max: Duration::from_micros(100),
            p50: Duration::from_micros(50),
            p90: Duration::from_micros(90),
            p95: Duration::from_micros(95),
            p99: Duration::from_micros(99),
            p999: Duration::from_micros(100),
        };

        assert_eq!(stats.percentile(0.50), stats.p50);
        assert_eq!(stats.percentile(0.90), stats.p90);
        assert_eq!(stats.percentile(0.99), stats.p99);
    }
}
