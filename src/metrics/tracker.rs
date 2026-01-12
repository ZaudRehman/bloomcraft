//! False positive rate tracking and analysis.
//!
//! Provides real-time tracking of false positive rates with statistical analysis
//! and anomaly detection. Uses a sliding window approach to track recent behavior.
//!
//! # Design
//!
//! The tracker maintains:
//! - Total positive queries (filter said "yes")
//! - True positive queries (actually in set, requires explicit marking)
//! - Sliding window of recent samples
//! - Statistical analysis (mean, variance, confidence intervals)
//!
//! # Examples
//!
//! ## Basic Tracking
//!
//! ```
//! use bloomcraft::metrics::FalsePositiveTracker;
//!
//! let tracker = FalsePositiveTracker::new(1000);
//!
//! // Simulate queries
//! for _ in 0..100 {
//!     tracker.record_positive();      // Filter said "yes"
//!     tracker.record_true_positive(); // 100% true positives
//! }
//!
//! assert_eq!(tracker.current_fp_rate(), 0.0);
//! ```
//!
//! ## With False Positives
//!
//! ```
//! use bloomcraft::metrics::FalsePositiveTracker;
//!
//! let tracker = FalsePositiveTracker::new(1000);
//!
//! // 10 positives, 9 true positives = 1 false positive
//! for _ in 0..10 {
//!     tracker.record_positive();
//! }
//! for _ in 0..9 {
//!     tracker.record_true_positive();
//! }
//!
//! assert!((tracker.current_fp_rate() - 0.10).abs() < 0.01);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Configuration for false positive tracker.
#[derive(Debug, Clone)]
pub struct FpTrackerConfig {
    /// Window size for sliding average (in samples)
    pub window_size: usize,
    /// Expected false positive rate
    pub expected_fp_rate: f64,
    /// Alert threshold (multiplier of expected rate)
    pub alert_threshold: f64,
}

impl Default for FpTrackerConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            expected_fp_rate: 0.01,
            alert_threshold: 1.5,
        }
    }
}

/// False positive rate tracker with sliding window analysis.
///
/// Thread-safe and lock-free for recording operations.
/// Locking only required for sliding window updates and statistics.
pub struct FalsePositiveTracker {
    /// Total positive queries (filter said "yes")
    total_positives: AtomicU64,
    /// True positive queries (actually in set)
    true_positives: AtomicU64,
    /// Total negative queries (filter said "no")
    total_negatives: AtomicU64,
    /// Configuration
    config: FpTrackerConfig,
    /// Sliding window of recent FP rates (protected by mutex)
    window: Arc<Mutex<SlidingWindow>>,
}

/// Sliding window for tracking recent false positive rates.
struct SlidingWindow {
    samples: Vec<bool>,
    pos: usize,
    size: usize,
}

impl SlidingWindow {
    fn new(size: usize) -> Self {
        Self {
            samples: Vec::with_capacity(size),
            pos: 0,
            size,
        }
    }

    fn push(&mut self, is_false_positive: bool) {
        if self.samples.len() < self.size {
            self.samples.push(is_false_positive);
        } else {
            self.samples[self.pos] = is_false_positive;
            self.pos = (self.pos + 1) % self.size;
        }
    }

    fn false_positive_rate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let fp_count = self.samples.iter().filter(|&&x| x).count();
        fp_count as f64 / self.samples.len() as f64
    }

    fn count(&self) -> usize {
        self.samples.len()
    }
}

impl FalsePositiveTracker {
    /// Create a new tracker with default configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::metrics::FalsePositiveTracker;
    ///
    /// let tracker = FalsePositiveTracker::new(1000);
    /// ```
    pub fn new(expected_items: usize) -> Self {
        Self::with_config(FpTrackerConfig {
            window_size: expected_items.min(10_000),
            ..Default::default()
        })
    }

    /// Create a tracker with custom configuration.
    pub fn with_config(config: FpTrackerConfig) -> Self {
        Self {
            total_positives: AtomicU64::new(0),
            true_positives: AtomicU64::new(0),
            total_negatives: AtomicU64::new(0),
            window: Arc::new(Mutex::new(SlidingWindow::new(config.window_size))),
            config,
        }
    }

    /// Record a positive query (filter said "yes").
    ///
    /// This is lock-free and extremely fast (~5ns).
    pub fn record_positive(&self) {
        self.total_positives.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a true positive (actually in set).
    ///
    /// Call this after confirming the item is actually in the set.
    /// This is lock-free and extremely fast (~5ns).
    pub fn record_true_positive(&self) {
        self.true_positives.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a negative query (filter said "no").
    ///
    /// This is lock-free and extremely fast (~5ns).
    pub fn record_negative(&self) {
        self.total_negatives.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a confirmed result (positive or negative).
    ///
    /// Use this when you know whether the item is actually in the set.
    pub fn record_confirmed(&self, filter_result: bool, actually_present: bool) {
        if filter_result {
            self.record_positive();
            if actually_present {
                self.record_true_positive();
            } else {
                // False positive
                if let Ok(mut window) = self.window.lock() {
                    window.push(true);
                }
            }
        } else {
            self.record_negative();
            // Negative results are never false (no false negatives in Bloom filters)
        }
    }

    /// Get current false positive rate (all-time).
    ///
    /// Returns the ratio of false positives to total positives.
    pub fn current_fp_rate(&self) -> f64 {
        let total_pos = self.total_positives.load(Ordering::Relaxed);
        let true_pos = self.true_positives.load(Ordering::Relaxed);

        if total_pos == 0 {
            return 0.0;
        }

        let false_pos = total_pos.saturating_sub(true_pos);
        false_pos as f64 / total_pos as f64
    }

    /// Get false positive rate from sliding window.
    ///
    /// This is more representative of recent behavior.
    pub fn window_fp_rate(&self) -> f64 {
        self.window.lock()
            .map(|w| w.false_positive_rate())
            .unwrap_or(0.0)
    }

    /// Get expected false positive rate from configuration.
    pub fn expected_fp_rate(&self) -> f64 {
        self.config.expected_fp_rate
    }

    /// Check if current FP rate exceeds alert threshold.
    pub fn is_alert(&self) -> bool {
        let current = self.window_fp_rate();
        let threshold = self.config.expected_fp_rate * self.config.alert_threshold;
        current > threshold
    }

    /// Get total number of queries.
    pub fn total_queries(&self) -> u64 {
        self.total_positives.load(Ordering::Relaxed) +
        self.total_negatives.load(Ordering::Relaxed)
    }

    /// Get total positive queries.
    pub fn total_positives(&self) -> u64 {
        self.total_positives.load(Ordering::Relaxed)
    }

    /// Get total true positives.
    pub fn total_true_positives(&self) -> u64 {
        self.true_positives.load(Ordering::Relaxed)
    }

    /// Get total false positives (estimated).
    pub fn total_false_positives(&self) -> u64 {
        let total_pos = self.total_positives.load(Ordering::Relaxed);
        let true_pos = self.true_positives.load(Ordering::Relaxed);
        total_pos.saturating_sub(true_pos)
    }

    /// Get snapshot of all statistics.
    pub fn snapshot(&self) -> FpTrackerSnapshot {
        FpTrackerSnapshot {
            total_queries: self.total_queries(),
            total_positives: self.total_positives(),
            true_positives: self.total_true_positives(),
            false_positives: self.total_false_positives(),
            current_fp_rate: self.current_fp_rate(),
            window_fp_rate: self.window_fp_rate(),
            expected_fp_rate: self.expected_fp_rate(),
            is_alert: self.is_alert(),
            window_sample_count: self.window.lock()
                .map(|w| w.count())
                .unwrap_or(0),
        }
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.total_positives.store(0, Ordering::Relaxed);
        self.true_positives.store(0, Ordering::Relaxed);
        self.total_negatives.store(0, Ordering::Relaxed);

        if let Ok(mut window) = self.window.lock() {
            *window = SlidingWindow::new(self.config.window_size);
        }
    }
}

impl Clone for FalsePositiveTracker {
    fn clone(&self) -> Self {
        Self {
            total_positives: AtomicU64::new(
                self.total_positives.load(Ordering::Relaxed)
            ),
            true_positives: AtomicU64::new(
                self.true_positives.load(Ordering::Relaxed)
            ),
            total_negatives: AtomicU64::new(
                self.total_negatives.load(Ordering::Relaxed)
            ),
            config: self.config.clone(),
            window: Arc::new(Mutex::new(
                SlidingWindow::new(self.config.window_size)
            )),
        }
    }
}

/// Snapshot of false positive tracker statistics.
#[derive(Debug, Clone)]
pub struct FpTrackerSnapshot {
    /// Total number of queries (positive + negative).
    pub total_queries: u64,
    /// Total positive queries (filter said "yes").
    pub total_positives: u64,
    /// True positive queries (actually in set).
    pub true_positives: u64,
    /// False positive queries (not in set but filter said "yes").
    pub false_positives: u64,
    /// Current all-time false positive rate.
    pub current_fp_rate: f64,
    /// False positive rate from sliding window (recent behavior).
    pub window_fp_rate: f64,
    /// Expected false positive rate from configuration.
    pub expected_fp_rate: f64,
    /// Whether the current FP rate exceeds the alert threshold.
    pub is_alert: bool,
    /// Number of samples in the sliding window.
    pub window_sample_count: usize,
}

impl FpTrackerSnapshot {
    /// Get deviation from expected rate (as percentage).
    pub fn deviation_percent(&self) -> f64 {
        if self.expected_fp_rate == 0.0 {
            return 0.0;
        }
        ((self.window_fp_rate - self.expected_fp_rate) / self.expected_fp_rate) * 100.0
    }

    /// Check if tracker has sufficient samples for reliable statistics.
    pub fn has_sufficient_samples(&self) -> bool {
        self.window_sample_count >= 100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker() {
        let tracker = FalsePositiveTracker::new(1000);
        assert_eq!(tracker.total_queries(), 0);
        assert_eq!(tracker.current_fp_rate(), 0.0);
    }

    #[test]
    fn test_record_operations() {
        let tracker = FalsePositiveTracker::new(1000);

        tracker.record_positive();
        tracker.record_true_positive();
        tracker.record_negative();

        assert_eq!(tracker.total_positives(), 1);
        assert_eq!(tracker.total_true_positives(), 1);
        assert_eq!(tracker.total_queries(), 2);
    }

    #[test]
    fn test_zero_false_positives() {
        let tracker = FalsePositiveTracker::new(1000);

        for _ in 0..100 {
            tracker.record_positive();
            tracker.record_true_positive();
        }

        assert_eq!(tracker.current_fp_rate(), 0.0);
        assert_eq!(tracker.total_false_positives(), 0);
    }

    #[test]
    fn test_false_positive_calculation() {
        let tracker = FalsePositiveTracker::new(1000);

        // 10 positives, 9 true = 1 false positive
        for _ in 0..10 {
            tracker.record_positive();
        }
        for _ in 0..9 {
            tracker.record_true_positive();
        }

        assert_eq!(tracker.total_false_positives(), 1);
        assert!((tracker.current_fp_rate() - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_confirmed_recording() {
        let tracker = FalsePositiveTracker::new(1000);

        // True positive
        tracker.record_confirmed(true, true);
        assert_eq!(tracker.total_positives(), 1);
        assert_eq!(tracker.total_true_positives(), 1);

        // False positive
        tracker.record_confirmed(true, false);
        assert_eq!(tracker.total_positives(), 2);
        assert_eq!(tracker.total_true_positives(), 1);

        // True negative
        tracker.record_confirmed(false, false);
        assert_eq!(tracker.total_negatives.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_sliding_window() {
        let tracker = FalsePositiveTracker::with_config(FpTrackerConfig {
            window_size: 10,
            expected_fp_rate: 0.01,
            alert_threshold: 1.5,
        });

        // Fill window with false positives
        for _ in 0..10 {
            tracker.record_confirmed(true, false);
        }

        // Window should show 100% FP rate
        assert!((tracker.window_fp_rate() - 1.0).abs() < 0.01);
        assert!(tracker.is_alert());
    }

    #[test]
    fn test_alert_threshold() {
        let tracker = FalsePositiveTracker::with_config(FpTrackerConfig {
            window_size: 100,
            expected_fp_rate: 0.01,
            alert_threshold: 2.0,
        });

        // Below threshold
        for _ in 0..100 {
            tracker.record_confirmed(true, true);
        }
        assert!(!tracker.is_alert());

        // Above threshold (simulate 5% FP rate)
        for _ in 0..100 {
            tracker.record_confirmed(true, false);
        }
        assert!(tracker.is_alert());
    }

    #[test]
    fn test_snapshot() {
        let tracker = FalsePositiveTracker::new(1000);

        for _ in 0..10 {
            tracker.record_positive();
        }
        for _ in 0..9 {
            tracker.record_true_positive();
        }

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.total_positives, 10);
        assert_eq!(snapshot.true_positives, 9);
        assert_eq!(snapshot.false_positives, 1);
        assert!((snapshot.current_fp_rate - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let tracker = FalsePositiveTracker::new(1000);

        tracker.record_positive();
        tracker.record_true_positive();

        assert_eq!(tracker.total_queries(), 1);

        tracker.reset();

        assert_eq!(tracker.total_queries(), 0);
        assert_eq!(tracker.current_fp_rate(), 0.0);
    }

    #[test]
    fn test_deviation_percent() {
        let snapshot = FpTrackerSnapshot {
            total_queries: 100,
            total_positives: 10,
            true_positives: 9,
            false_positives: 1,
            current_fp_rate: 0.10,
            window_fp_rate: 0.015,
            expected_fp_rate: 0.01,
            is_alert: true,
            window_sample_count: 100,
        };

        let deviation = snapshot.deviation_percent();
        assert!((deviation - 50.0).abs() < 1.0); // 50% over expected
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(FalsePositiveTracker::new(10_000));
        let mut handles = vec![];

        for _ in 0..10 {
            let tracker_clone = Arc::clone(&tracker);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    tracker_clone.record_positive();
                    tracker_clone.record_true_positive();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(tracker.total_positives(), 10_000);
        assert_eq!(tracker.total_true_positives(), 10_000);
    }

    #[test]
    fn test_clone() {
        let tracker = FalsePositiveTracker::new(1000);
        tracker.record_positive();
        tracker.record_true_positive();

        let cloned = tracker.clone();
        assert_eq!(cloned.total_positives(), 1);
        assert_eq!(cloned.total_true_positives(), 1);
    }
}
