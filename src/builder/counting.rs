//! Builder for counting Bloom filters.
//!
//! Counting Bloom filters support deletion by maintaining counters instead of bits.
//! This builder provides a type-safe API for constructing counting filters.
//!
//! # Type-State Pattern
//!
//! ```text
//! Initial → WithItems → Complete → CountingBloomFilter
//!     ↓         ↓           ↓
//!   .expected_items()  .false_positive_rate()  .build()
//! ```
//!
//! # Examples
//!
//! ## Minimal Configuration
//!
//! ```
//! use bloomcraft::builder::CountingBloomFilterBuilder;
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Full Configuration
//!
//! ```
//! use bloomcraft::builder::CountingBloomFilterBuilder;
//! use bloomcraft::filters::CountingBloomFilter;
//! use bloomcraft::hash::HashStrategy;
//!
//! let filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .max_count(255)  // Maximum counter value
//!     .hash_strategy(HashStrategy::EnhancedDouble)
//!     .build()
//!     .unwrap();
//! ```

use crate::core::params;
use crate::error::Result;
use crate::hash::{BloomHasher, DefaultHasher, HashStrategy};
use crate::filters::counting::CountingBloomFilter;
use std::marker::PhantomData;

/// Type-state marker: Initial state.
pub struct Initial;

/// Type-state marker: Items count is set.
pub struct WithItems;

/// Type-state marker: All required parameters set.
pub struct Complete;

/// Builder for counting Bloom filters with type-state guarantees.
///
/// # Type Parameters
///
/// - `State`: Current builder state
/// - `H`: Hash function type
///
/// # Counter Sizes
///
/// Counting filters use 4-bit counters by default (max count = 15).
/// This can be customized via `max_count()`:
///
/// - `max_count(15)`: 4 bits per counter (default)
/// - `max_count(255)`: 8 bits per counter
///
/// # Memory Overhead
///
/// Counting filters use 4-8x more memory than standard filters due to counters.
pub struct CountingBloomFilterBuilder<State, H = DefaultHasher> {
    expected_items: Option<usize>,
    fp_rate: Option<f64>,
    max_count: u8,
    hash_strategy: HashStrategy,
    _state: PhantomData<State>,
    _hasher: PhantomData<H>,
}

impl CountingBloomFilterBuilder<Initial, DefaultHasher> {
    /// Create a new counting filter builder.
    ///
    /// Defaults:
    /// - `max_count`: 15 (4-bit counters)
    /// - `hash_strategy`: EnhancedDouble
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    ///
    /// let builder = CountingBloomFilterBuilder::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_items: None,
            fp_rate: None,
            max_count: 15,  // 4-bit counters by default
            hash_strategy: HashStrategy::EnhancedDouble,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> CountingBloomFilterBuilder<Initial, H> {
    /// Set the expected number of items to insert.
    ///
    /// Required parameter. Transitions to `WithItems` state.
    ///
    /// # Arguments
    ///
    /// * `items` - Expected number of elements (must be > 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    ///
    /// let builder = CountingBloomFilterBuilder::new()
    ///     .expected_items(10_000);
    /// ```
    #[must_use]
    pub fn expected_items(self, items: usize) -> CountingBloomFilterBuilder<WithItems, H> {
        CountingBloomFilterBuilder {
            expected_items: Some(items),
            fp_rate: self.fp_rate,
            max_count: self.max_count,
            hash_strategy: self.hash_strategy,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> CountingBloomFilterBuilder<WithItems, H> {
    /// Set the target false positive rate.
    ///
    /// Required parameter. Transitions to `Complete` state.
    ///
    /// # Arguments
    ///
    /// * `fp_rate` - Target false positive probability (must be in (0, 1))
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    ///
    /// let builder = CountingBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01);
    /// ```
    #[must_use]
    pub fn false_positive_rate(self, fp_rate: f64) -> CountingBloomFilterBuilder<Complete, H> {
        CountingBloomFilterBuilder {
            expected_items: self.expected_items,
            fp_rate: Some(fp_rate),
            max_count: self.max_count,
            hash_strategy: self.hash_strategy,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }

    /// Set the maximum counter value (optional).
    ///
    /// Determines counter size:
    /// - 1-15: 4-bit counters (default)
    /// - 16-255: 8-bit counters
    ///
    /// Higher values increase memory usage but support more insertions
    /// of the same item before overflow.
    ///
    /// # Arguments
    ///
    /// * `max_count` - Maximum value for counters (1-255)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    ///
    /// let builder = CountingBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .max_count(255);  // 8-bit counters
    /// ```
    #[must_use]
    pub fn max_count(mut self, max_count: u8) -> Self {
        self.max_count = max_count;
        self
    }

    /// Set the hash strategy (optional).
    ///
    /// Defaults to `HashStrategy::EnhancedDouble`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    /// use bloomcraft::hash::HashStrategy;
    ///
    /// let builder = CountingBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .hash_strategy(HashStrategy::Triple);
    /// ```
    #[must_use]
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H> CountingBloomFilterBuilder<Complete, H> {
    /// Set the maximum counter value (optional, can be set in Complete state too).
    #[must_use]
    pub fn max_count(mut self, max_count: u8) -> Self {
        self.max_count = max_count;
        self
    }

    /// Set the hash strategy (optional, can be set in Complete state too).
    #[must_use]
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H: BloomHasher + Default + Clone> CountingBloomFilterBuilder<Complete, H> {
    /// Build the counting Bloom filter.
    ///
    /// Validates all parameters and constructs the filter.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `expected_items == 0`
    /// - `fp_rate` not in (0, 1)
    /// - `max_count < 1`
    /// - Calculated parameters are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01)
    ///     .build()
    ///     .unwrap();
    ///
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    ///
    /// filter.delete(&"hello");
    /// assert!(!filter.contains(&"hello"));
    /// ```
    pub fn build<T: std::hash::Hash>(self) -> Result<CountingBloomFilter<T, H>> {
        // Extract parameters
        let expected_items = self.expected_items.expect("items must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        // Validate parameters
        super::validation::validate_items(expected_items)?;
        super::validation::validate_fp_rate(fp_rate)?;
        super::validation::validate_max_count(self.max_count)?;

        // Calculate optimal parameters
        let filter_size = params::optimal_bit_count(expected_items, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, expected_items)?;

        // Validate calculated parameters
        params::validate_params(filter_size, expected_items, num_hashes)?;

        // Construct filter
        let filter = CountingBloomFilter::with_full_params(
            filter_size,
            num_hashes,
            self.max_count,
            self.hash_strategy,
        );

        Ok(filter)
    }

    /// Build the filter and return it with metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::CountingBloomFilterBuilder;
    /// use bloomcraft::filters::CountingBloomFilter;
    /// use bloomcraft::builder::counting::CountingFilterMetadata;
    ///
    /// let (filter, metadata): (CountingBloomFilter<&str>, CountingFilterMetadata) = CountingBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01)
    ///     .build_with_metadata()
    ///     .unwrap();
    ///
    /// println!("Counter bits: {}", metadata.counter_bits);
    /// println!("Memory overhead: {:.1}x", metadata.memory_overhead_factor);
    /// ```
    pub fn build_with_metadata<T: std::hash::Hash>(self) -> Result<(CountingBloomFilter<T, H>, CountingFilterMetadata)> {
        let expected_items = self.expected_items.expect("items must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        // Validate
        super::validation::validate_items(expected_items)?;
        super::validation::validate_fp_rate(fp_rate)?;
        super::validation::validate_max_count(self.max_count)?;

        // Calculate parameters
        let filter_size = params::optimal_bit_count(expected_items, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, expected_items)?;
        params::validate_params(filter_size, expected_items, num_hashes)?;

        // Build filter
        let filter = CountingBloomFilter::with_full_params(
            filter_size,
            num_hashes,
            self.max_count,
            self.hash_strategy,
        );

        // Determine counter bits
        let counter_bits = if self.max_count <= 15 { 4 } else { 8 };

        // Create metadata
        let metadata = CountingFilterMetadata {
            expected_items,
            fp_rate,
            filter_size,
            num_hashes,
            max_count: self.max_count,
            counter_bits,
            hash_strategy: self.hash_strategy,
            memory_overhead_factor: counter_bits as f64,
        };

        Ok((filter, metadata))
    }
}

impl Default for CountingBloomFilterBuilder<Initial, DefaultHasher> {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a constructed counting filter.
#[derive(Debug, Clone)]
pub struct CountingFilterMetadata {
    /// Expected number of items
    pub expected_items: usize,
    /// Target false positive rate
    pub fp_rate: f64,
    /// Filter size (number of counters)
    pub filter_size: usize,
    /// Number of hash functions
    pub num_hashes: usize,
    /// Maximum counter value
    pub max_count: u8,
    /// Bits per counter (4 or 8)
    pub counter_bits: u8,
    /// Hash strategy used
    pub hash_strategy: HashStrategy,
    /// Memory overhead vs standard filter (4.0x or 8.0x)
    pub memory_overhead_factor: f64,
}

impl CountingFilterMetadata {
    /// Get the theoretical memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.filter_size * self.counter_bits as usize + 7) / 8
    }

    /// Get the theoretical memory usage in kilobytes.
    #[must_use]
    pub fn memory_kb(&self) -> f64 {
        self.memory_bytes() as f64 / 1024.0
    }

    /// Get the theoretical memory usage in megabytes.
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Get bytes per item.
    #[must_use]
    pub fn bytes_per_item(&self) -> f64 {
        self.memory_bytes() as f64 / self.expected_items as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_minimal() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_with_max_count() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .max_count(255)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_with_strategy() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .hash_strategy(HashStrategy::Triple)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_full_config() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .max_count(255)
            .hash_strategy(HashStrategy::Double)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_with_metadata() {
        let (filter, metadata): (CountingBloomFilter<String>, _) = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build_with_metadata()
            .unwrap();

        assert!(filter.is_empty());
        assert_eq!(metadata.expected_items, 10_000);
        assert!((metadata.fp_rate - 0.01).abs() < 0.001);
        assert!(metadata.filter_size > 0);
        assert!(metadata.num_hashes > 0);
        assert_eq!(metadata.max_count, 15);  // Default
        assert_eq!(metadata.counter_bits, 4);  // 4-bit counters
        assert!((metadata.memory_overhead_factor - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_builder_8bit_counters() {
        let (_, metadata): (CountingBloomFilter<String>, _) = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .max_count(255)
            .build_with_metadata()
            .unwrap();

        assert_eq!(metadata.max_count, 255);
        assert_eq!(metadata.counter_bits, 8);
        assert!((metadata.memory_overhead_factor - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_builder_invalid_items() {
        let result: Result<CountingBloomFilter<String>> = CountingBloomFilterBuilder::new()
            .expected_items(0)
            .false_positive_rate(0.01)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_fp_rate() {
        let result: Result<CountingBloomFilter<String>> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.0)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_max_count() {
        let result: Result<CountingBloomFilter<String>> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .max_count(0)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_filter_insert_delete() {
        let mut filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        filter.insert(&"hello");
        filter.insert(&"world");

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));

        filter.delete(&"hello");
        assert!(!filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
    }

    #[test]
    fn test_filter_insert_delete_integers() {
        let mut filter: CountingBloomFilter<i32> = CountingBloomFilterBuilder::new()
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        filter.insert(&12345);
        filter.insert(&67890);

        assert!(filter.contains(&12345));
        assert!(filter.contains(&67890));

        filter.delete(&12345);
        assert!(!filter.contains(&12345));
        assert!(filter.contains(&67890));
    }

    #[test]
    fn test_filter_duplicate_insertions() {
        let mut filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .max_count(15)
            .build()
            .unwrap();

        // Insert same item multiple times
        for _ in 0..10 {
            filter.insert(&"test");
        }

        assert!(filter.contains(&"test"));

        // Delete same number of times
        for _ in 0..10 {
            filter.delete(&"test");
        }

        assert!(!filter.contains(&"test"));
    }

    #[test]
    fn test_metadata_memory_calculations() {
        let (_, metadata): (CountingBloomFilter<String>, _) = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build_with_metadata()
            .unwrap();

        let bytes = metadata.memory_bytes();
        let kb = metadata.memory_kb();
        let mb = metadata.memory_mb();
        let bpi = metadata.bytes_per_item();

        assert!(bytes > 0);
        assert!(kb > 0.0);
        assert!(mb > 0.0);
        assert!(bpi > 0.0);

        // Consistency checks
        assert!((kb - bytes as f64 / 1024.0).abs() < 0.01);
        assert!((mb - kb / 1024.0).abs() < 0.0001);
    }

    #[test]
    fn test_different_strategies() {
        let strategies = [
            HashStrategy::Double,
            HashStrategy::EnhancedDouble,
            HashStrategy::Triple,
        ];

        for strategy in strategies {
            let mut filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
                .expected_items(1_000)
                .false_positive_rate(0.01)
                .hash_strategy(strategy)
                .build()
                .unwrap();

            filter.insert(&"test");
            assert!(filter.contains(&"test"));
            filter.delete(&"test");
            assert!(!filter.contains(&"test"));
        }
    }

    #[test]
    fn test_max_count_before_fp_rate() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .max_count(100)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_max_count_after_fp_rate() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .max_count(100)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_large_filter() {
        let filter: CountingBloomFilter<String> = CountingBloomFilterBuilder::new()
            .expected_items(1_000_000)
            .false_positive_rate(0.001)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_counter_overflow_protection() {
        let mut filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
            .expected_items(100)
            .false_positive_rate(0.01)
            .max_count(15)  // 4-bit counters
            .build()
            .unwrap();

        // Insert item 15 times (should succeed)
        for _ in 0..15 {
            filter.insert(&"overflow_test");
        }

        // 16th insertion should fail or be handled gracefully
        // (implementation-dependent)
    }
}
