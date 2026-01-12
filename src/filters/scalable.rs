//! Scalable Bloom filter with dynamic growth.
//!
//! A scalable Bloom filter automatically grows to accommodate more elements while
//! maintaining a target false positive rate. This was proposed by Almeida et al. in 2007.
//!
//! # Key Innovation
//!
//! Instead of a single fixed-size filter, use a sequence of standard Bloom filters:
//! - When a filter reaches capacity, add a new filter
//! - Each successive filter has a tighter error rate
//! - Query checks all filters (union)
//!
//! # Algorithm
//!
//! ```text
//! Scalable Bloom Filter = [Filter₀, Filter₁, Filter₂, ...]
//!
//! Where:
//! - Size(Filterᵢ) = s₀ × rⁱ
//! - FPR(Filterᵢ) = p₀ × rⁱ
//! - s₀ = initial capacity
//! - p₀ = initial error rate
//! - r = error tightening ratio (typically 0.5)
//! ```
//!
//! # False Positive Rate
//!
//! The overall false positive rate is bounded by:
//!
//! ```text
//! P(false positive) ≤ Σᵢ P(Filterᵢ) = p₀ × Σᵢ rⁱ = p₀ / (1 - r)
//! ```
//!
//! With r = 0.5, the total FPR ≤ 2 × p₀
//!
//! # Growth Strategy
//!
//! Two common strategies:
//!
//! 1. Constant Growth: Each filter same size (simple)
//! 2. Geometric Growth: Each filter 2x larger (space efficient)
//!
//! # Trade-offs
//!
//! | Aspect | Standard Bloom | Scalable Bloom |
//! |--------|----------------|----------------|
//! | Fixed size | Yes | No (grows dynamically) |
//! | FPR | Constant | Increases slowly |
//! | Memory | Pre-allocated | Grows on demand |
//! | Query time | O(k) | O(k × n_filters) |
//! | Best for | Known size | Unknown/unbounded size |
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! // Start with capacity for 1000 items, 1% FPR
//! let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01);
//!
//! // Insert any number of items - filter grows automatically
//! for i in 0..10_000 {
//!     filter.insert(&i);
//! }
//!
//! // Query works across all sub-filters
//! assert!(filter.contains(&42));
//! assert!(filter.contains(&9999));
//! ```
//!
//! ## Growth Monitoring
//!
//! ```
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
//!
//! for i in 0..500 {
//!     filter.insert(&i);
//! }
//!
//! println!("Number of sub-filters: {}", filter.filter_count());
//! println!("Total capacity: {}", filter.total_capacity());
//! println!("Fill rate: {:.2}%", filter.fill_rate() * 100.0);
//! ```
//!
//! ## Custom Growth Parameters
//!
//! ```
//! use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};
//!
//! let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::with_strategy(
//!     1000,           // Initial capacity
//!     0.01,           // Target FPR
//!     0.5,            // Error tightening ratio
//!     GrowthStrategy::Geometric(2.0) // 2x growth
//! );
//! ```
//!
//! # References
//!
//! - Almeida, P. S., Baquero, C., Preguiça, N., & Hutchison, D. (2007).
//!   "Scalable Bloom Filters". Information Processing Letters, 101(6), 255-261.

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::filter::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Maximum number of sub-filters to prevent unbounded growth and overflow.
///
/// This limit ensures:
/// - Capacity calculations don't overflow (even with geometric growth)
/// - Query time remains reasonable (O(k × MAX_FILTERS))
/// - Memory usage stays within practical bounds
const MAX_FILTERS: usize = 64;

/// Minimum FPR to prevent precision loss in deep hierarchies.
///
/// For deep filter hierarchies (n > 30), f64 precision can degrade.
/// This clamp prevents FPR from becoming unrealistically small.
const MIN_FPR: f64 = 1e-15;

/// Growth strategy for scalable Bloom filters.
///
/// Determines how the capacity increases with each new sub-filter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GrowthStrategy {
    /// Constant size: all sub-filters have the same capacity
    ///
    /// Memory: O(n) filters of size s
    /// Good for: Predictable memory usage
    Constant,

    /// Geometric growth: each filter is scale × larger
    ///
    /// Memory: More space efficient (fewer large filters)
    /// Good for: Unknown size, space efficiency
    ///
    /// Typically scale = 2.0 (double each time)
    Geometric(f64),
}

impl Default for GrowthStrategy {
    fn default() -> Self {
        Self::Geometric(2.0)
    }
}

/// Scalable Bloom filter that grows dynamically.
///
/// Maintains a sequence of standard Bloom filters, adding new ones as needed.
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Memory Layout
///
/// ```text
/// ScalableBloomFilter {
///     filters: Vec<StandardBloomFilter>,  // Sequence of sub-filters
///     initial_capacity: usize,             // Capacity of first filter
///     target_fpr: f64,                     // Target error rate
///     error_ratio: f64,                    // Error tightening ratio
///     growth: GrowthStrategy,              // How to grow
///     fill_threshold: f64,                 // When to add new filter
/// }
/// ```
///
/// # Thread Safety
///
/// - **Insert**: Requires `&mut self` (exclusive access)
/// - **Query**: Thread-safe (immutable access)
/// - Each sub-filter uses lock-free operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "H: BloomHasher + Clone + Default",
        deserialize = "H: BloomHasher + Clone + Default"
    ))
)]
pub struct ScalableBloomFilter<T, H = StdHasher>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Sequence of standard Bloom filters
    filters: Vec<StandardBloomFilter<T, H>>,

    /// Initial capacity per filter
    initial_capacity: usize,

    /// Target false positive rate for first filter
    target_fpr: f64,

    /// Error tightening ratio (r)
    error_ratio: f64,

    /// Growth strategy
    growth: GrowthStrategy,

    /// Fill threshold to trigger new filter (0.5 = 50%)
    fill_threshold: f64,

    /// Hash function
    #[cfg_attr(feature = "serde", serde(skip))]
    hasher: H,

    /// Total items inserted
    total_items: usize,

    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

impl<T> ScalableBloomFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new scalable Bloom filter with default parameters.
    ///
    /// Uses:
    /// - Error ratio: 0.5 (each filter has half the error rate)
    /// - Growth strategy: Geometric(2.0) (double capacity each time)
    /// - Fill threshold: 0.5 (add new filter at 50% fill)
    ///
    /// # Arguments
    ///
    /// * `initial_capacity` - Capacity of first filter
    /// * `target_fpr` - Target false positive rate
    ///
    /// # Panics
    ///
    /// Panics if `target_fpr` is not in (0, 1) or `initial_capacity` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let filter: ScalableBloomFilter<String> = ScalableBloomFilter::new(1000, 0.01);
    /// ```
    #[must_use]
    pub fn new(initial_capacity: usize, target_fpr: f64) -> Self {
        Self::with_hasher(initial_capacity, target_fpr, StdHasher::new())
    }

    /// Create a scalable Bloom filter with custom growth strategy.
    ///
    /// # Arguments
    ///
    /// * `initial_capacity` - Capacity of first filter
    /// * `target_fpr` - Target false positive rate
    /// * `error_ratio` - Error tightening ratio (typically 0.5)
    /// * `growth` - Growth strategy
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};
    ///
    /// let filter: ScalableBloomFilter<String> = ScalableBloomFilter::with_strategy(
    ///     1000,
    ///     0.01,
    ///     0.5,
    ///     GrowthStrategy::Geometric(2.0)
    /// );
    /// ```
    #[must_use]
    pub fn with_strategy(
        initial_capacity: usize,
        target_fpr: f64,
        error_ratio: f64,
        growth: GrowthStrategy,
    ) -> Self {
        Self::with_strategy_and_hasher(
            initial_capacity,
            target_fpr,
            error_ratio,
            growth,
            StdHasher::new(),
        )
    }
}

impl<T, H> ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Create a new scalable Bloom filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `initial_capacity` - Capacity of first filter
    /// * `target_fpr` - Target false positive rate
    /// * `hasher` - Custom hash function
    #[must_use]
    pub fn with_hasher(initial_capacity: usize, target_fpr: f64, hasher: H) -> Self {
        Self::with_strategy_and_hasher(
            initial_capacity,
            target_fpr,
            0.5,
            GrowthStrategy::default(),
            hasher,
        )
    }

    /// Create a scalable Bloom filter with full customization.
    ///
    /// # Arguments
    ///
    /// * `initial_capacity` - Capacity of first filter
    /// * `target_fpr` - Target false positive rate
    /// * `error_ratio` - Error tightening ratio
    /// * `growth` - Growth strategy
    /// * `hasher` - Hash function
    #[must_use]
    pub fn with_strategy_and_hasher(
        initial_capacity: usize,
        target_fpr: f64,
        error_ratio: f64,
        growth: GrowthStrategy,
        hasher: H,
    ) -> Self {
        assert!(initial_capacity > 0, "initial_capacity must be > 0");
        assert!(
            target_fpr > 0.0 && target_fpr < 1.0,
            "target_fpr must be in (0, 1)"
        );
        assert!(
            error_ratio > 0.0 && error_ratio < 1.0,
            "error_ratio must be in (0, 1)"
        );

        let mut filter = Self {
            filters: Vec::new(),
            initial_capacity,
            target_fpr,
            error_ratio,
            growth,
            fill_threshold: 0.5,
            hasher: hasher.clone(),
            total_items: 0,
            _phantom: PhantomData,
        };

        // Create initial filter (cannot fail for index 0)
        let _ = filter.try_add_filter();
        filter
    }

    /// Add a new sub-filter to the sequence.
    ///
    /// The new filter has:
    /// - Capacity based on growth strategy
    /// - Tighter error rate (target_fpr × error_ratio^n)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if filter was added successfully
    /// - `Err(BloomCraftError)` if maximum filters reached or capacity overflow
    fn try_add_filter(&mut self) -> Result<()> {
        let filter_index = self.filters.len();

        // Check if we've reached maximum number of filters
        if filter_index >= MAX_FILTERS {
            return Err(BloomCraftError::invalid_parameters(
                format!("Maximum filter count ({}) reached", MAX_FILTERS)
            ));
        }

        // Calculate capacity for new filter with overflow protection
        let capacity = match self.growth {
            GrowthStrategy::Constant => self.initial_capacity,
            GrowthStrategy::Geometric(scale) => {
                if filter_index == 0 {
                    self.initial_capacity
                } else {
                    // Use checked arithmetic to prevent overflow
                    let scale_int = scale as usize;
                    let exponent = filter_index as u32;

                    // Calculate scale^filter_index with overflow checking
                    let growth_factor = scale_int
                        .checked_pow(exponent)
                        .ok_or_else(|| BloomCraftError::invalid_parameters(
                            "Capacity growth overflow"
                        ))?;

                    self.initial_capacity
                        .checked_mul(growth_factor)
                        .ok_or_else(|| BloomCraftError::invalid_parameters(
                            "Capacity multiplication overflow"
                        ))?
                }
            }
        };

        // Calculate error rate for new filter with precision clamping
        let fpr = (self.target_fpr * self.error_ratio.powi(filter_index as i32)).max(MIN_FPR);

        // Create and add new filter
        let new_filter = StandardBloomFilter::with_hasher(capacity, fpr, self.hasher.clone());
        self.filters.push(new_filter);

        Ok(())
    }

    /// Check if current filter needs to grow.
    ///
    /// Returns true if the latest filter exceeds fill threshold.
    fn should_grow(&self) -> bool {
        if let Some(current) = self.filters.last() {
            current.fill_rate() >= self.fill_threshold
        } else {
            false
        }
    }

    /// Get the number of sub-filters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
    /// assert_eq!(filter.filter_count(), 1);
    ///
    /// for i in 0..200 {
    ///     filter.insert(&i);
    /// }
    ///
    /// assert!(filter.filter_count() > 1);
    /// ```
    #[must_use]
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Get the total capacity across all filters.
    #[must_use]
    pub fn total_capacity(&self) -> usize {
        self.filters.iter().map(|f| f.expected_items()).sum()
    }

    /// Get the total number of items inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_items
    }

    /// Check if the filter is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_items == 0
    }

    /// Calculate overall fill rate across all filters.
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        if self.filters.is_empty() {
            return 0.0;
        }

        let total_bits: usize = self.filters.iter().map(|f| f.size()).sum();
        let set_bits: usize = self.filters.iter().map(|f| f.count_set_bits()).sum();

        set_bits as f64 / total_bits as f64
    }

    /// Estimate the current false positive rate.
    ///
    /// Uses the union bound: P(FP) ≤ Σᵢ P(Filterᵢ)
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        self.filters.iter().map(|f| f.estimate_fpr()).sum()
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.filters.iter().map(|f| f.memory_usage()).sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    /// Insert an item into the filter.
    ///
    /// Automatically adds a new sub-filter if current one is full.
    /// If maximum filter count is reached, continues inserting into the last filter.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
    ///
    /// // Insert many items - filter grows automatically
    /// for i in 0..1000 {
    ///     filter.insert(&i);
    /// }
    /// ```
    pub fn insert(&mut self, item: &T) {
        // Check if we need to add a new filter
        if self.should_grow() {
            // Try to add new filter, but continue even if we hit max filters
            let _ = self.try_add_filter();
        }

        // Insert into the current (last) filter
        if let Some(current) = self.filters.last_mut() {
            current.insert(item);
            self.total_items += 1;
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// Queries all sub-filters (returns true if any filter contains the item).
    ///
    /// # Arguments
    ///
    /// * `item` - Item to check
    ///
    /// # Returns
    ///
    /// - `true`: Item might be in the set (or false positive)
    /// - `false`: Item is definitely not in the set
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);
    /// filter.insert(&"hello");
    ///
    /// assert!(filter.contains(&"hello"));
    /// assert!(!filter.contains(&"world"));
    /// ```
    #[must_use]
    pub fn contains(&self, item: &T) -> bool {
        self.filters.iter().any(|filter| filter.contains(item))
    }

    /// Clear all sub-filters.
    ///
    /// Removes all items and resets to a single empty filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);
    /// filter.insert(&"hello");
    /// filter.clear();
    ///
    /// assert!(filter.is_empty());
    /// assert_eq!(filter.filter_count(), 1);
    /// ```
    pub fn clear(&mut self) {
        self.filters.clear();
        self.total_items = 0;
        let _ = self.try_add_filter();
    }

    /// Insert multiple items in batch.
    pub fn insert_batch(&mut self, items: &[T]) {
        for item in items {
            self.insert(item);
        }
    }

    /// Check multiple items in batch.
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Get statistics for each sub-filter.
    ///
    /// Returns a vector of (capacity, fill_rate, estimated_fpr) tuples.
    #[must_use]
    pub fn filter_stats(&self) -> Vec<(usize, f64, f64)> {
        self.filters
            .iter()
            .map(|f| (f.expected_items(), f.fill_rate(), f.estimate_fpr()))
            .collect()
    }

    /// Get the growth strategy.
    #[must_use]
    pub fn growth_strategy(&self) -> GrowthStrategy {
        self.growth
    }

    /// Get the error tightening ratio.
    #[must_use]
    pub fn error_ratio(&self) -> f64 {
        self.error_ratio
    }

    /// Get the fill threshold.
    #[must_use]
    pub fn fill_threshold(&self) -> f64 {
        self.fill_threshold
    }

    /// Set the fill threshold (when to add new filter).
    ///
    /// # Arguments
    ///
    /// * `threshold` - New threshold in range (0, 1)
    ///
    /// # Panics
    ///
    /// Panics if threshold is not in (0, 1).
    pub fn set_fill_threshold(&mut self, threshold: f64) {
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "threshold must be in (0, 1)"
        );
        self.fill_threshold = threshold;
    }
}

impl<T, H> BloomFilter<T> for ScalableBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        ScalableBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        ScalableBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        ScalableBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        ScalableBloomFilter::len(self)
    }

    fn is_empty(&self) -> bool {
        ScalableBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.filters.iter().map(|f| f.expected_items()).sum()
    }

    fn bit_count(&self) -> usize {
        self.filters.iter().map(|f| f.bit_count()).sum()
    }

    fn hash_count(&self) -> usize {
        self.filters.first().map(|f| f.hash_count()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: ScalableBloomFilter<String> = ScalableBloomFilter::new(1000, 0.01);
        assert_eq!(filter.filter_count(), 1);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);

        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_automatic_growth() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01);
        assert_eq!(filter.filter_count(), 1);

        // Insert enough items to trigger growth
        for i in 0..100 {
            filter.insert(&i);
        }

        assert!(filter.filter_count() > 1, "Filter should have grown");
    }

    #[test]
    fn test_geometric_growth() {
        let mut filter =
            ScalableBloomFilter::with_strategy(10, 0.01, 0.5, GrowthStrategy::Geometric(2.0));

        for i in 0..200 {
            filter.insert(&i);
        }

        let stats = filter.filter_stats();
        if stats.len() >= 2 {
            // Second filter should be roughly 2x larger
            let ratio = stats[1].0 as f64 / stats[0].0 as f64;
            assert!(ratio > 1.5 && ratio < 2.5, "Growth ratio: {}", ratio);
        }
    }

    #[test]
    fn test_constant_growth() {
        let mut filter =
            ScalableBloomFilter::with_strategy(10, 0.01, 0.5, GrowthStrategy::Constant);

        for i in 0..100 {
            filter.insert(&i);
        }

        let stats = filter.filter_stats();
        if stats.len() >= 2 {
            // All filters should have same capacity
            assert_eq!(stats[0].0, stats[1].0);
        }
    }

    #[test]
    fn test_clear() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01);

        for i in 0..100 {
            filter.insert(&i);
        }

        filter.clear();

        assert!(filter.is_empty());
        assert_eq!(filter.filter_count(), 1);
        assert!(!filter.contains(&42));
    }

    #[test]
    fn test_len() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);
        assert_eq!(filter.len(), 0);

        filter.insert(&"a");
        filter.insert(&"b");
        filter.insert(&"c");

        assert_eq!(filter.len(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);
        assert!(filter.is_empty());

        filter.insert(&"test");
        assert!(!filter.is_empty());

        filter.clear();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_fill_rate() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
        assert_eq!(filter.fill_rate(), 0.0);

        for i in 0..50 {
            filter.insert(&i);
        }

        let fill = filter.fill_rate();
        assert!(fill > 0.0 && fill < 1.0);
    }

    #[test]
    fn test_estimate_fpr() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);

        for i in 0..50 {
            filter.insert(&i);
        }

        let fpr = filter.estimate_fpr();
        assert!(fpr > 0.0 && fpr < 1.0);
    }

    #[test]
    fn test_memory_usage() {
        let filter: ScalableBloomFilter<String> = ScalableBloomFilter::new(1000, 0.01);
        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_insert_batch() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);

        let items = vec!["a", "b", "c", "d"];
        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);

        filter.insert(&"a");
        filter.insert(&"b");

        let queries = vec!["a", "b", "c", "d"];
        let results = filter.contains_batch(&queries);

        assert_eq!(results, vec![true, true, false, false]);
    }

    #[test]
    fn test_filter_stats() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01);

        for i in 0..100 {
            filter.insert(&i);
        }

        let stats = filter.filter_stats();
        assert!(!stats.is_empty());

        for (capacity, fill_rate, fpr) in stats {
            assert!(capacity > 0);
            assert!(fill_rate >= 0.0 && fill_rate <= 1.0);
            assert!(fpr >= 0.0 && fpr <= 1.0);
        }
    }

    #[test]
    fn test_set_fill_threshold() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);

        filter.set_fill_threshold(0.8);
        assert_eq!(filter.fill_threshold(), 0.8);
    }

    #[test]
    #[should_panic(expected = "threshold must be in (0, 1)")]
    fn test_invalid_fill_threshold() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
        filter.set_fill_threshold(1.5);
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);

        BloomFilter::insert(&mut filter, &"test");
        assert!(BloomFilter::contains(&filter, &"test"));
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    #[test]
    fn test_no_false_negatives() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01);

        let items = vec!["apple", "banana", "cherry", "date", "elderberry"];
        for item in &items {
            filter.insert(item);
        }

        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_large_scale_insertion() {
        let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(100, 0.01);

        // Insert 10,000 items
        for i in 0..10_000 {
            filter.insert(&i);
        }

        assert_eq!(filter.len(), 10_000);
        assert!(filter.filter_count() > 1);

        // Verify all items present
        for i in 0..10_000 {
            assert!(filter.contains(&i), "False negative for {}", i);
        }
    }

    #[test]
    fn test_error_ratio() {
        let filter: ScalableBloomFilter<i32> =
            ScalableBloomFilter::with_strategy(100, 0.01, 0.3, GrowthStrategy::Constant);

        assert_eq!(filter.error_ratio(), 0.3);
    }

    #[test]
    fn test_growth_strategy() {
        let filter: ScalableBloomFilter<i32> =
            ScalableBloomFilter::with_strategy(100, 0.01, 0.5, GrowthStrategy::Geometric(3.0));

        assert_eq!(filter.growth_strategy(), GrowthStrategy::Geometric(3.0));
    }

    #[test]
    fn test_total_capacity() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);

        for i in 0..500 {
            filter.insert(&i);
        }

        let capacity = filter.total_capacity();
        assert!(capacity >= 100);
    }

    #[test]
    fn test_clone() {
        let mut filter1 = ScalableBloomFilter::new(100, 0.01);
        filter1.insert(&"test");

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"test"));
        assert_eq!(filter1.filter_count(), filter2.filter_count());
    }

    #[test]
    fn test_growth_strategy_default() {
        let strategy = GrowthStrategy::default();
        assert_eq!(strategy, GrowthStrategy::Geometric(2.0));
    }

    #[test]
    fn test_max_filters_limit() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1, 0.01);

        // Insert enough items to try to exceed MAX_FILTERS
        // With capacity 1 and fill threshold 0.5, this should trigger many growth attempts
        for i in 0..1000 {
            filter.insert(&i);
        }

        // Should not exceed MAX_FILTERS
        assert!(filter.filter_count() <= MAX_FILTERS);

        // All items should still be queryable
        for i in 0..1000 {
            assert!(filter.contains(&i), "False negative for {}", i);
        }
    }

    #[test]
    fn test_fpr_precision_clamp() {
        // Create filter with deep hierarchy to test MIN_FPR clamping
        let mut filter: ScalableBloomFilter<i32> =
            ScalableBloomFilter::with_strategy(10, 0.01, 0.1, GrowthStrategy::Geometric(2.0));

        // Trigger multiple growths
        for i in 0..500 {
            filter.insert(&i);
        }

        // FPR should never be less than MIN_FPR
        let fpr = filter.estimate_fpr();
        assert!(fpr >= MIN_FPR);
    }
}
