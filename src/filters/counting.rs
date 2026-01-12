 //! Counting Bloom filter with deletion support.
//!
//! A counting Bloom filter extends the standard Bloom filter by using counters instead
//! of bits, enabling element deletion. This was first proposed by Fan et al. in 2000.
//!
//! # Key Innovation
//!
//! Instead of a bit array, use an array of counters:
//! - Insert: Increment k counters
//! - Delete: Decrement k counters
//! - Query: Check if all k counters > 0
//!
//! # Trade-offs
//!
//! | Aspect | Standard Bloom | Counting Bloom |
//! |--------|----------------|----------------|
//! | Insert | O(k) | O(k) |
//! | Query | O(k) | O(k) |
//! | Delete | Not supported | O(k) |
//! | Space | 1 bit per position | 3-4 bits per position |
//! | False positives | Yes | Yes |
//! | False negatives | Never | Possible if deleted too many times |
//!
//! # Counter Size Selection
//!
//! For n items and m counters with k hash functions:
//! - 4-bit counters: Good for most cases (max count = 15)
//! - 8-bit counters: More resilient (max count = 255)
//! - 16-bit counters: High-load scenarios
//!
//! The probability of counter overflow is negligible for properly sized filters.
//!
//! # Mathematical Foundation
//!
//! Maximum expected counter value:
//! ```text
//! E[max counter] ≈ (kn/m) + sqrt((kn/m) × ln(m))
//! ```
//!
//! For standard parameters (k≈7, load factor < 0.5), 4-bit counters suffice.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let mut filter = CountingBloomFilter::new(10_000, 0.01);
//!
//! // Insert items
//! filter.insert(&"hello");
//! filter.insert(&"world");
//! assert!(filter.contains(&"hello"));
//!
//! // Delete items
//! filter.delete(&"hello");
//! assert!(!filter.contains(&"hello"));
//! assert!(filter.contains(&"world")); // Still there
//! ```
//!
//! ## Counter Overflow Protection
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let mut filter = CountingBloomFilter::with_counter_size(1000, 0.01, 4);
//!
//! // Insert same item multiple times
//! for _ in 0..10 {
//!     filter.insert(&"item");
//! }
//!
//! // Check if any counter overflowed
//! if filter.has_overflowed() {
//!     println!("Warning: Counter overflow detected!");
//! }
//! ```
//!
//! # References
//!
//! - Fan, L., Cao, P., Almeida, J., & Broder, A. Z. (2000). "Summary cache: a scalable
//!   wide-area web cache sharing protocol". IEEE/ACM Transactions on Networking.
//! - Bonomi, F., Mitzenmacher, M., Panigrahy, R., Singh, S., & Varghese, G. (2006).
//!   "An improved construction for counting bloom filters".

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]

use crate::core::filter::BloomFilter;
use crate::core::params::{optimal_k, optimal_m};
use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy};
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Convert a hashable item to bytes using Rust's `Hash` trait.
///
/// This is the bridge between generic `T: Hash` and the `&[u8]` API
/// required by `BloomHasher`.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Counter size options for counting Bloom filters.
///
/// Determines the maximum value each counter can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CounterSize {
    /// 4-bit counters (max value: 15)
    ///
    /// Memory: 0.5 bytes per counter
    /// Good for: Standard use cases with proper sizing
    FourBit,

    /// 8-bit counters (max value: 255)
    ///
    /// Memory: 1 byte per counter
    /// Good for: Most production use cases
    EightBit,

    /// 16-bit counters (max value: 65535)
    ///
    /// Memory: 2 bytes per counter
    /// Good for: High-load or skewed distributions
    SixteenBit,
}

impl CounterSize {
    /// Get the maximum value for this counter size.
    #[must_use]
    pub const fn max_value(self) -> u16 {
        match self {
            Self::FourBit => 15,
            Self::EightBit => 255,
            Self::SixteenBit => 65535,
        }
    }

    /// Get the number of bits per counter.
    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            Self::FourBit => 4,
            Self::EightBit => 8,
            Self::SixteenBit => 16,
        }
    }
}

/// Counting Bloom filter supporting insertions, deletions, and queries.
///
/// Uses 8-bit atomic counters for thread-safe operations with overflow detection.
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Memory Layout
///
/// ```text
/// CountingBloomFilter {
///     counters: Vec<AtomicU8>,    // m counters
///     k: usize,                    // number of hash functions
///     max_count: u8,               // maximum counter value
///     overflow_count: AtomicUsize, // number of overflows
/// }
/// ```
///
/// # Thread Safety
///
/// All operations are thread-safe using atomic counters:
/// - Insert: Atomic increment
/// - Delete: Atomic decrement
/// - Query: Atomic load
#[derive(Debug)]
pub struct CountingBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Array of atomic counters
    counters: Vec<AtomicU8>,

    /// Number of hash functions (k)
    k: usize,

    /// Maximum counter value before saturation
    max_count: u8,

    /// Number of counter overflows detected
    overflow_count: std::sync::atomic::AtomicUsize,

    /// Hash function
    hasher: H,

    /// Hash strategy for generating indices
    strategy: EnhancedDoubleHashing,

    /// Expected number of items (for statistics)
    expected_items: usize,

    /// Target false positive rate (for statistics)
    target_fpr: f64,

    /// Phantom data for type parameter T
    _phantom: PhantomData<T>,
}

impl<T> CountingBloomFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new counting Bloom filter with default hasher.
    ///
    /// Uses 8-bit counters by default.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items (n)
    /// * `fpr` - Target false positive rate (0 < fpr < 1)
    ///
    /// # Panics
    ///
    /// Panics if `fpr` is not in (0, 1) or `expected_items` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let filter: CountingBloomFilter<String> = CountingBloomFilter::new(10_000, 0.01);
    /// ```
    #[must_use]
    pub fn new(expected_items: usize, fpr: f64) -> Self {
        Self::with_hasher(expected_items, fpr, StdHasher::new())
    }

    /// Create a counting Bloom filter with specific counter size.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items
    /// * `fpr` - Target false positive rate
    /// * `counter_bits` - Bits per counter (4 or 8)
    ///
    /// # Panics
    ///
    /// Panics if counter_bits is not 4 or 8.
    ///
    /// # Note
    ///
    /// This implementation uses 8-bit atomic counters internally. The `counter_bits`
    /// parameter controls the logical maximum value:
    /// - 4-bit: max value 15 (space-efficient, suitable for most use cases)
    /// - 8-bit: max value 255 (more resilient to high-frequency items)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// // Use 4-bit counters for space efficiency
    /// let filter: CountingBloomFilter<String> =
    ///     CountingBloomFilter::with_counter_size(10_000, 0.01, 4);
    /// ```
    #[must_use]
    pub fn with_counter_size(expected_items: usize, fpr: f64, counter_bits: usize) -> Self {
        let max_count = match counter_bits {
            4 => 15,
            8 => 255,
            _ => panic!(
                "counter_bits must be 4 or 8 (this implementation uses 8-bit atomic storage)"
            ),
        };

        let mut filter = Self::new(expected_items, fpr);
        filter.max_count = max_count;
        filter
    }
}

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Create a new counting Bloom filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items
    /// * `fpr` - Target false positive rate
    /// * `hasher` - Custom hash function
    ///
    /// # Panics
    ///
    /// Panics if `fpr` is not in (0, 1) or `expected_items` is 0.
    #[must_use]
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fpr > 0.0 && fpr < 1.0,
            "fpr must be in range (0, 1), got {}",
            fpr
        );

        let m = optimal_m(expected_items, fpr);
        let k = optimal_k(expected_items, m);

        Self {
            counters: (0..m).map(|_| AtomicU8::new(0)).collect(),
            k,
            max_count: 255, // 8-bit default
            overflow_count: std::sync::atomic::AtomicUsize::new(0),
            hasher,
            strategy: EnhancedDoubleHashing,
            expected_items,
            target_fpr: fpr,
            _phantom: PhantomData,
        }
    }

    /// Create a counting Bloom filter with explicit parameters.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of counters
    /// * `k` - Number of hash functions
    /// * `hasher` - Hash function
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    #[must_use]
    pub fn with_params(m: usize, k: usize, hasher: H) -> Self {
        assert!(m > 0, "m must be > 0");
        assert!(k > 0, "k must be > 0");

        Self {
            counters: (0..m).map(|_| AtomicU8::new(0)).collect(),
            k,
            max_count: 255,
            overflow_count: std::sync::atomic::AtomicUsize::new(0),
            hasher,
            strategy: EnhancedDoubleHashing,
            expected_items: 0,
            target_fpr: 0.0,
            _phantom: PhantomData,
        }
    }

    /// Create a counting Bloom filter with full parameter control.
    ///
    /// This constructor is used by the builder pattern to create filters
    /// with all parameters explicitly specified.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of counters
    /// * `k` - Number of hash functions
    /// * `max_count` - Maximum counter value (typically 15 or 255)
    /// * `_strategy` - Hash strategy (currently ignored, uses EnhancedDoubleHashing)
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    #[must_use]
    pub fn with_full_params(m: usize, k: usize, max_count: u8, _strategy: crate::hash::HashStrategy) -> Self
    where
        H: Default,
    {
        assert!(m > 0, "m must be > 0");
        assert!(k > 0, "k must be > 0");

        // Note: Currently uses EnhancedDoubleHashing regardless of strategy parameter
        // This is because the struct's strategy field is typed as EnhancedDoubleHashing
        Self {
            counters: (0..m).map(|_| AtomicU8::new(0)).collect(),
            k,
            max_count,
            overflow_count: std::sync::atomic::AtomicUsize::new(0),
            hasher: H::default(),
            strategy: EnhancedDoubleHashing,
            expected_items: 0,
            target_fpr: 0.0,
            _phantom: PhantomData,
        }
    }

    /// Get the number of counters (m).
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.counters.len()
    }

    /// Get the number of hash functions (k).
    #[must_use]
    #[inline]
    pub fn hash_count(&self) -> usize {
        self.k
    }

    /// Get the maximum counter value.
    #[must_use]
    #[inline]
    pub fn max_count(&self) -> u8 {
        self.max_count
    }

    /// Get the number of counter overflows that occurred.
    ///
    /// # Returns
    ///
    /// Total number of overflow events
    #[must_use]
    pub fn overflow_count(&self) -> usize {
        self.overflow_count.load(Ordering::Relaxed)
    }

    /// Check if any counter has overflowed.
    #[must_use]
    pub fn has_overflowed(&self) -> bool {
        self.overflow_count() > 0
    }

    /// Increment a counter at the given index.
    ///
    /// Returns true if increment succeeded, false if counter is at max.
    #[inline]
    fn increment_counter(&self, index: usize) -> bool {
        let counter = &self.counters[index];
        let mut current = counter.load(Ordering::Relaxed);

        loop {
            if current >= self.max_count {
                // Counter at maximum, record overflow
                self.overflow_count.fetch_add(1, Ordering::Relaxed);
                return false;
            }

            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Decrement a counter at the given index.
    ///
    /// Returns true if decrement succeeded, false if counter was already 0.
    #[inline]
    fn decrement_counter(&self, index: usize) -> bool {
        let counter = &self.counters[index];
        let mut current = counter.load(Ordering::Relaxed);

        loop {
            if current == 0 {
                return false; // Already at zero
            }

            match counter.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Get the value of a counter at the given index.
    #[inline]
    fn get_counter(&self, index: usize) -> u8 {
        self.counters[index].load(Ordering::Relaxed)
    }

    /// Insert an item into the filter.
    ///
    /// Increments k counters. If any counter reaches maximum value,
    /// it saturates (doesn't wrap around) and the overflow is recorded.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// ```
    #[inline]
    pub fn insert(&mut self, item: &T) {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        for idx in indices {
            self.increment_counter(idx);
        }
    }

    /// Delete an item from the filter.
    ///
    /// First checks if the item appears to be in the filter (all counters > 0),
    /// then decrements k counters. This prevents false negatives that could occur
    /// from deleting items that were never inserted.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to delete
    ///
    /// # Returns
    ///
    /// `true` if the item was found and all counters were decremented successfully,
    /// `false` if the item was not in the filter (no counters were modified)
    ///
    /// # Safety
    ///
    /// This method is safe to call even if the item was never inserted - it will
    /// simply return `false` without modifying any counters, preventing false
    /// negatives for other items.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    ///
    /// assert!(filter.delete(&"hello"));  // Returns true - item was present
    /// assert!(!filter.contains(&"hello"));
    ///
    /// assert!(!filter.delete(&"never_inserted"));  // Returns false - item not present
    /// ```
    pub fn delete(&mut self, item: &T) -> bool {
        // First check if the item appears to be in the filter
        // This prevents false negatives from deleting non-existent items
        if !self.contains(item) {
            return false;
        }

        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        let mut all_decremented = true;

        for idx in indices {
            if !self.decrement_counter(idx) {
                all_decremented = false;
            }
        }

        all_decremented
    }

    /// Force delete an item from the filter without checking if it exists.
    ///
    /// **WARNING**: This method can cause false negatives if used to delete
    /// items that were never inserted. Use `delete()` instead for safe deletion.
    ///
    /// This method is provided for advanced use cases where you are certain
    /// the item exists and want to avoid the overhead of the contains check.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to delete
    ///
    /// # Returns
    ///
    /// `true` if all counters were decremented successfully (were > 0),
    /// `false` if any counter was already at 0
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    ///
    /// // Only use this if you're certain the item exists
    /// filter.delete_unchecked(&"hello");
    /// ```
    pub fn delete_unchecked(&mut self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        let mut all_decremented = true;

        for idx in indices {
            if !self.decrement_counter(idx) {
                all_decremented = false;
            }
        }

        all_decremented
    }

    /// Check if an item might be in the filter.
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
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    ///
    /// assert!(filter.contains(&"hello"));
    /// assert!(!filter.contains(&"world"));
    /// ```
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        indices.iter().all(|&idx| self.get_counter(idx) > 0)
    }

    /// Clear all counters in the filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// filter.clear();
    /// assert!(!filter.contains(&"hello"));
    /// ```
    pub fn clear(&mut self) {
        for counter in &self.counters {
            counter.store(0, Ordering::Relaxed);
        }
        self.overflow_count.store(0, Ordering::Relaxed);
    }

    /// Check if the filter is empty (all counters are 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counters
            .iter()
            .all(|c| c.load(Ordering::Relaxed) == 0)
    }

    /// Count the number of non-zero counters.
    ///
    /// This gives an approximate measure of how many positions are occupied.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        self.counters
            .iter()
            .filter(|c| c.load(Ordering::Relaxed) > 0)
            .count()
    }

    /// Calculate the fill rate (fraction of non-zero counters).
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        self.count_nonzero() as f64 / self.size() as f64
    }

    /// Get the target false positive rate the filter was configured with.
    ///
    /// This is the FPR the filter was designed to achieve when filled
    /// to its expected capacity.
    #[must_use]
    pub fn target_false_positive_rate(&self) -> f64 {
        self.target_fpr
    }

    /// Estimate the current false positive rate.
    ///
    /// Uses the standard Bloom filter formula with proper estimation of n
    /// from the fill rate.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let nonzero = self.count_nonzero();
        let m = self.size() as f64;
        let k = self.k as f64;

        if nonzero == 0 {
            return 0.0;
        }

        let fill_rate = nonzero as f64 / m;

        if fill_rate >= 1.0 {
            return 1.0;
        }

        // Estimate n (number of items) from fill rate
        // fill_rate ≈ 1 - e^(-kn/m)
        // => n ≈ -(m/k) × ln(1 - fill_rate)
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();

        // Calculate FPR using the standard formula: (1 - e^(-kn/m))^k
        let exponent = -k * estimated_n / m;
        (1.0 - exponent.exp()).powf(k)
    }

    /// Get the maximum counter value currently in the filter.
    ///
    /// Useful for monitoring counter saturation.
    #[must_use]
    pub fn max_counter_value(&self) -> u8 {
        self.counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0)
    }

    /// Get the average counter value (excluding zeros).
    #[must_use]
    pub fn avg_counter_value(&self) -> f64 {
        let sum: usize = self
            .counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed) as usize)
            .sum();

        let nonzero = self.count_nonzero();

        if nonzero == 0 {
            return 0.0;
        }

        sum as f64 / nonzero as f64
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.counters.len() * std::mem::size_of::<AtomicU8>() + std::mem::size_of::<Self>()
    }

    /// Get the raw counter data as bytes.
    ///
    /// This is useful for serialization.
    #[must_use]
    pub fn raw_counters(&self) -> Vec<u8> {
        self.counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }

    /// Get the number of bits per counter.
    #[must_use]
    pub fn counter_bits(&self) -> u8 {
        if self.max_count <= 15 {
            4
        } else {
            8
        }
    }

    /// Get the number of hash functions (alias for hash_count).
    #[must_use]
    #[inline]
    pub fn num_hashes(&self) -> usize {
        self.k
    }

    /// Get the hash strategy used by this filter.
    ///
    /// Returns the strategy enum variant for serialization.
    #[must_use]
    pub fn hash_strategy(&self) -> crate::hash::HashStrategy {
        // This filter uses EnhancedDoubleHashing internally
        crate::hash::HashStrategy::EnhancedDouble
    }

    /// Create a filter from raw parts (for deserialization).
    ///
    /// # Arguments
    ///
    /// * `size` - Number of counters
    /// * `k` - Number of hash functions
    /// * `max_count` - Maximum counter value
    /// * `_strategy` - Hash strategy (currently ignored, uses EnhancedDoubleHashing)
    /// * `counters` - Raw counter data
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid.
    pub fn from_raw(
        size: usize,
        k: usize,
        max_count: u8,
        _strategy: crate::hash::HashStrategy,
        counters: &[u8],
    ) -> crate::error::Result<Self>
    where
        H: Default,
    {
        if size == 0 {
            return Err(crate::error::BloomCraftError::invalid_filter_size(size));
        }
        if k == 0 || k > 32 {
            return Err(crate::error::BloomCraftError::invalid_hash_count(k, 1, 32));
        }
        if counters.len() < size {
            return Err(crate::error::BloomCraftError::invalid_parameters(
                format!("Counter data too small: expected at least {} bytes, got {}", size, counters.len())
            ));
        }

        let atomic_counters: Vec<AtomicU8> = counters.iter()
            .take(size)
            .map(|&v| AtomicU8::new(v))
            .collect();

        Ok(Self {
            counters: atomic_counters,
            k,
            max_count,
            overflow_count: std::sync::atomic::AtomicUsize::new(0),
            hasher: H::default(),
            strategy: EnhancedDoubleHashing,
            expected_items: 0,
            target_fpr: 0.0,
            _phantom: PhantomData,
        })
    }

    /// Insert multiple items in batch.
    pub fn insert_batch(&mut self, items: &[T]) {
        for item in items {
            self.insert(item);
        }
    }

    /// Delete multiple items in batch.
    ///
    /// # Returns
    ///
    /// Number of items successfully deleted (items that were actually in the filter)
    pub fn delete_batch(&mut self, items: &[T]) -> usize {
        items.iter().filter(|item| self.delete(item)).count()
    }

    /// Force delete multiple items in batch without checking existence.
    ///
    /// **WARNING**: This can cause false negatives. Use `delete_batch()` for safe deletion.
    ///
    /// # Returns
    ///
    /// Number of items where all counters were successfully decremented
    pub fn delete_batch_unchecked(&mut self, items: &[T]) -> usize {
        items
            .iter()
            .filter(|item| self.delete_unchecked(item))
            .count()
    }

    /// Check multiple items in batch.
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Get a histogram of counter values.
    ///
    /// Returns a vector where index i contains the count of counters with value i.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter: CountingBloomFilter<&str> = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    ///
    /// let histogram = filter.counter_histogram();
    /// println!("Counters with value 0: {}", histogram[0]);
    /// println!("Counters with value 1: {}", histogram[1]);
    /// ```
    #[must_use]
    pub fn counter_histogram(&self) -> Vec<usize> {
        let max_val = self.max_counter_value() as usize;
        let mut histogram = vec![0; max_val + 1];

        for counter in &self.counters {
            let val = counter.load(Ordering::Relaxed) as usize;
            histogram[val] += 1;
        }

        histogram
    }

    /// Get the number of counters that have reached the saturation limit.
    ///
    /// This is a diagnostic metric to monitor filter health.
    #[must_use]
    pub fn saturated_counter_count(&self) -> usize {
        self.counters
            .iter()
            .filter(|c| c.load(Ordering::Relaxed) >= self.max_count)
            .count()
    }
}

impl<T, H> BloomFilter<T> for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    fn insert(&mut self, item: &T) {
        CountingBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        CountingBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        CountingBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        self.count_nonzero()
    }

    fn is_empty(&self) -> bool {
        CountingBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.size()
    }

    fn hash_count(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: CountingBloomFilter<String> = CountingBloomFilter::new(1000, 0.01);
        assert!(filter.size() > 0);
        assert!(filter.hash_count() > 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_delete() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));

        filter.delete(&"hello");
        assert!(!filter.contains(&"hello"));
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        // Deleting item that was never inserted should return false
        // and NOT modify any counters (preventing false negatives)
        let result = filter.delete(&"ghost");
        assert!(!result); // Should indicate item was not found

        // Verify no counters were modified
        assert!(filter.is_empty());
    }

    #[test]
    fn test_delete_unchecked_nonexistent() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        // Insert an item first
        filter.insert(&"existing");

        // delete_unchecked on non-existent item will return false
        // because counters are at 0 (can't decrement)
        let result = filter.delete_unchecked(&"ghost");
        assert!(!result);

        // The existing item should still be there
        assert!(filter.contains(&"existing"));
    }

    #[test]
    fn test_multiple_inserts_and_deletes() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        // Insert same item multiple times
        filter.insert(&"item");
        filter.insert(&"item");
        filter.insert(&"item");

        assert!(filter.contains(&"item"));

        // Delete once - should still be present
        filter.delete(&"item");
        assert!(filter.contains(&"item"));

        // Delete again
        filter.delete(&"item");
        assert!(filter.contains(&"item"));

        // Delete third time - now should be gone
        filter.delete(&"item");
        assert!(!filter.contains(&"item"));
    }

    #[test]
    fn test_clear() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        filter.insert(&"a");
        filter.insert(&"b");
        filter.insert(&"c");

        filter.clear();

        assert!(filter.is_empty());
        assert!(!filter.contains(&"a"));
        assert!(!filter.contains(&"b"));
        assert!(!filter.contains(&"c"));
    }

    #[test]
    fn test_is_empty() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);
        assert!(filter.is_empty());

        filter.insert(&"test");
        assert!(!filter.is_empty());

        filter.delete(&"test");
        assert!(filter.is_empty());
    }

    #[test]
    fn test_count_nonzero() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);
        assert_eq!(filter.count_nonzero(), 0);

        filter.insert(&"test");
        assert!(filter.count_nonzero() > 0);
    }

    #[test]
    fn test_fill_rate() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);
        assert_eq!(filter.fill_rate(), 0.0);

        for i in 0..100 {
            filter.insert(&i);
        }

        let fill_rate = filter.fill_rate();
        assert!(fill_rate > 0.0 && fill_rate < 1.0);
    }

    #[test]
    fn test_estimate_fpr() {
        let mut filter = CountingBloomFilter::new(10_000, 0.01);

        for i in 0..5000 {
            filter.insert(&i);
        }

        let fpr = filter.estimate_fpr();
        assert!(fpr > 0.0 && fpr < 1.0);
    }

    #[test]
    fn test_max_counter_value() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);
        assert_eq!(filter.max_counter_value(), 0);

        filter.insert(&"test");
        assert!(filter.max_counter_value() > 0);
    }

    #[test]
    fn test_avg_counter_value() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);
        assert_eq!(filter.avg_counter_value(), 0.0);

        filter.insert(&"test");
        assert!(filter.avg_counter_value() > 0.0);
    }

    #[test]
    fn test_counter_overflow() {
        let mut filter = CountingBloomFilter::with_counter_size(10, 0.5, 4);
        // max_count = 15 for 4-bit counters

        // Insert same item many times to force overflow
        for _ in 0..20 {
            filter.insert(&"overflow_test");
        }

        assert!(filter.has_overflowed());
        assert!(filter.overflow_count() > 0);
    }

    #[test]
    fn test_memory_usage() {
        let filter: CountingBloomFilter<String> = CountingBloomFilter::new(10_000, 0.01);
        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_insert_batch() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        let items = vec!["a", "b", "c", "d"];
        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_delete_batch() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        let items = vec!["a", "b", "c", "d"];
        filter.insert_batch(&items);

        let deleted = filter.delete_batch(&items);
        assert_eq!(deleted, 4);

        for item in &items {
            assert!(!filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        filter.insert(&"a");
        filter.insert(&"b");

        let queries = vec!["a", "b", "c", "d"];
        let results = filter.contains_batch(&queries);

        assert_eq!(results, vec![true, true, false, false]);
    }

    #[test]
    fn test_counter_histogram() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        filter.insert(&"test");

        let histogram = filter.counter_histogram();
        assert!(histogram.len() > 0);
        assert!(histogram[0] > 0); // Should have many zero counters
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        BloomFilter::insert(&mut filter, &"test");
        assert!(BloomFilter::contains(&filter, &"test"));
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    #[test]
    fn test_no_false_negatives_without_deletion() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        let items = vec!["apple", "banana", "cherry", "date"];
        for item in &items {
            filter.insert(item);
        }

        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_independent_items() {
        let mut filter = CountingBloomFilter::new(1000, 0.01);

        filter.insert(&"a");
        filter.insert(&"b");

        filter.delete(&"a");

        assert!(!filter.contains(&"a"));
        assert!(filter.contains(&"b")); // Should still be there
    }

    #[test]
    fn test_with_params() {
        let filter: CountingBloomFilter<String, StdHasher> =
            CountingBloomFilter::with_params(1000, 7, StdHasher::new());

        assert_eq!(filter.size(), 1000);
        assert_eq!(filter.hash_count(), 7);
    }

    #[test]
    fn test_counter_size_enum() {
        assert_eq!(CounterSize::FourBit.max_value(), 15);
        assert_eq!(CounterSize::EightBit.max_value(), 255);
        assert_eq!(CounterSize::SixteenBit.max_value(), 65535);

        assert_eq!(CounterSize::FourBit.bits(), 4);
        assert_eq!(CounterSize::EightBit.bits(), 8);
        assert_eq!(CounterSize::SixteenBit.bits(), 16);
    }

    #[test]
    fn test_saturated_counter_count() {
        let mut filter = CountingBloomFilter::with_counter_size(10, 0.5, 4);

        // Saturate some counters
        for _ in 0..20 {
            filter.insert(&"test");
        }

        let saturated = filter.saturated_counter_count();
        assert!(saturated > 0);
    }
}
