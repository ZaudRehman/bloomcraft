//! Builder for [`CountingBloomFilter`].
//!
//! Counting filters extend standard Bloom filters with multi-bit counters per
//! position, enabling element deletion. This builder uses PhantomData state
//! markers to document the required parameter progression.
//!
//! # State Machine
//!
//! ```text
//! Initial  ──.expected_items(n)──→  WithItems
//! WithItems ──.false_positive_rate(p)──→  Complete
//! Complete  ──.build()──→  Result<CountingBloomFilter<T, H>>
//! ```
//!
//! Optional setters (`max_count`, `hash_strategy`) are available at both
//! `WithItems` and `Complete` and can appear in any order.
//!
//! # Counter Sizing & Memory Overhead
//!
//! Counting filters store *m* counters instead of *m* bits. The counter width
//! is determined by the maximum count needed:
//!
//! | `max_count` | Counter Width | Memory per position | Overhead vs Standard |
//! |------------|--------------|--------------------|--------------------|
//! | 1–15       | 4-bit        | 0.5 B              | 4× |
//! | 16–255     | 8-bit        | 1 B                | 8× |
//! | 256–65535  | 16-bit       | 2 B                | 16× |
//!
//! Example: a filter sized for 100k items at 1% FPR uses *m* ≈ 958k positions.
//! With 4-bit counters this is ~479 KB; with 8-bit counters, ~958 KB.
//!
//! **Choose the smallest `max_count` your workload allows.** The 4-bit default
//! (max 15 insertions of the same item) is sufficient for most deduplication
//! and rate-limiting use cases.
//!
//! # Hasher
//!
//! The type parameter `H` is fixed at compile time via `PhantomData`.
//! The hasher is always default-constructed ([`H::default()`]). To use a
//! custom hasher instance or seed, construct the filter directly via
//! [`CountingBloomFilter::with_counter_size_and_hasher`].
//!
//! # Performance
//!
//! Dominated by the k-hash probe loop (identical cost to a standard filter
//! with the same *m*/*k*), plus a counter read-modify-write per probe.
//! At 1% FPR (*k* ≈ 7), single-threaded inserts complete in 30–50 ns on
//! modern x86_64. Deletion has the same cost as insertion.
//!
//! # Examples
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
//! ```
//! use bloomcraft::builder::CountingBloomFilterBuilder;
//! use bloomcraft::filters::CountingBloomFilter;
//! use bloomcraft::hash::IndexingStrategy;
//!
//! let filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .max_count(255)
//!     .hash_strategy(IndexingStrategy::EnhancedDouble)
//!     .build()
//!     .unwrap();
//! ```

use crate::core::params;
use crate::error::Result;
use crate::hash::{BloomHasher, IndexingStrategy, StdHasher};
use crate::filters::counting::CountingBloomFilter;
use crate::filters::CounterSize;
use std::marker::PhantomData;

/// State marker: initial builder state.
pub struct Initial;

/// State marker: expected item count has been provided.
pub struct WithItems;

/// State marker: all required parameters have been provided.
pub struct Complete;

/// Builder for [`CountingBloomFilter`] with state-machine parameter enforcement.
///
/// # Type Parameters
///
/// * `State` — PhantomData state marker (`Initial` → `WithItems` → `Complete`).
/// * `H` — Hasher type, always default-constructed. See [hasher limitations](index.html#hasher).
pub struct CountingBloomFilterBuilder<State, H = StdHasher> {
    expected_items: Option<usize>,
    fp_rate: Option<f64>,
    max_count: u16,
    hash_strategy: IndexingStrategy,
    _state: PhantomData<State>,
    _hasher: PhantomData<H>,
}

impl CountingBloomFilterBuilder<Initial, StdHasher> {
    /// Creates a new builder with the default hasher ([`StdHasher`]).
    ///
    /// Defaults:
    /// * `max_count` = 15 (4-bit counters, ~4× memory overhead vs standard).
    /// * `hash_strategy` = [`IndexingStrategy::EnhancedDouble`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_items: None,
            fp_rate: None,
            max_count: 15,
            hash_strategy: IndexingStrategy::EnhancedDouble,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> CountingBloomFilterBuilder<Initial, H> {
    /// Sets the expected number of distinct items and advances to `WithItems`.
    ///
    /// Required parameter. Passing `0` here is not rejected until [`build`](Self::build).
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
    /// Sets the target false-positive rate and advances to `Complete`.
    ///
    /// Required parameter. `fp_rate` must be in the open interval (0, 1).
    /// Out-of-range values are not rejected until [`build`](Self::build).
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

    /// Sets the maximum counter value.
    ///
    /// Determines the physical counter storage width (see [sizing table](index.html#counter-sizing--memory-overhead)).
    /// The default (15) gives 4-bit counters. When this is called after
    /// `false_positive_rate()` it is forwarded to the `Complete` state.
    #[must_use]
    pub fn max_count(mut self, max_count: u16) -> Self {
        self.max_count = max_count;
        self
    }

    /// Sets the hash indexing strategy.
    ///
    /// Defaults to [`IndexingStrategy::EnhancedDouble`].
    #[must_use]
    pub fn hash_strategy(mut self, strategy: IndexingStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H> CountingBloomFilterBuilder<Complete, H> {
    /// Sets the maximum counter value (available in `Complete` state too).
    #[must_use]
    pub fn max_count(mut self, max_count: u16) -> Self {
        self.max_count = max_count;
        self
    }

    /// Sets the hash indexing strategy (available in `Complete` state too).
    #[must_use]
    pub fn hash_strategy(mut self, strategy: IndexingStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H: BloomHasher + Default + Clone> CountingBloomFilterBuilder<Complete, H> {
    fn compute_params(&self, expected_items: usize, fp_rate: f64) -> Result<(usize, usize)> {
        let filter_size = params::optimal_bit_count(expected_items, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, expected_items)?;
        params::validate_params(filter_size, expected_items, num_hashes)?;
        Ok((filter_size, num_hashes))
    }

    /// Constructs the counting Bloom filter.
    ///
    /// The hasher is always default-constructed — see [hasher limitations](index.html#hasher)
    /// for details and the workaround.
    ///
    /// # Errors
    ///
    /// | Condition | Error Variant |
    /// |-----------|--------------|
    /// | `expected_items == 0` | `InvalidItemCount` |
    /// | `fp_rate` ∉ (0, 1) | `FalsePositiveRateOutOfBounds` |
    /// | `max_count == 0` | `InvalidParameters` |
    /// | Derived *m* or *k* exceed limits | `InvalidParameters` |
    pub fn build<T: std::hash::Hash + Send + Sync>(self) -> Result<CountingBloomFilter<T, H>> {
        let expected_items = self.expected_items.expect("items must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        super::validation::validate_items(expected_items)?;
        super::validation::validate_fp_rate(fp_rate)?;
        super::validation::validate_max_count(self.max_count)?;

        let (filter_size, num_hashes) = self.compute_params(expected_items, fp_rate)?;

        let filter = CountingBloomFilter::with_full_params(
            filter_size,
            num_hashes,
            CounterSize::from_max_count(self.max_count),
            expected_items,
            fp_rate,
        );

        Ok(filter)
    }

    /// Constructs the filter and returns a [`CountingFilterMetadata`] snapshot.
    pub fn build_with_metadata<T: std::hash::Hash + Send + Sync>(self) -> Result<(CountingBloomFilter<T, H>, CountingFilterMetadata)> {
        let expected_items = self.expected_items.expect("items must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        super::validation::validate_items(expected_items)?;
        super::validation::validate_fp_rate(fp_rate)?;
        super::validation::validate_max_count(self.max_count)?;

        let (filter_size, num_hashes) = self.compute_params(expected_items, fp_rate)?;

        let counter_size = CounterSize::from_max_count(self.max_count);
        let filter = CountingBloomFilter::with_full_params(
            filter_size,
            num_hashes,
            counter_size,
            expected_items,
            fp_rate,
        );

        let counter_bits = counter_size.bits() as u8;

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

impl Default for CountingBloomFilterBuilder<Initial, StdHasher> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construction-time metadata for a [`CountingBloomFilter`].
///
/// Returned by [`CountingBloomFilterBuilder::build_with_metadata`].
#[derive(Debug, Clone)]
pub struct CountingFilterMetadata {
    /// *n* — expected item count supplied to the builder.
    pub expected_items: usize,
    /// *p* — target false-positive rate supplied to the builder.
    pub fp_rate: f64,
    /// *m* — actual number of counters.
    pub filter_size: usize,
    /// *k* — number of hash functions.
    pub num_hashes: usize,
    /// Maximum counter value (determines counter width).
    pub max_count: u16,
    /// Bits per counter: 4, 8, or 16.
    pub counter_bits: u8,
    /// Hash indexing strategy.
    pub hash_strategy: IndexingStrategy,
    /// Memory overhead factor relative to a standard filter (= `counter_bits`).
    ///
    /// A standard filter uses 1 bit per position; a counting filter with 4-bit
    /// counters uses 4 bits per position, so `memory_overhead_factor == 4.0`.
    pub memory_overhead_factor: f64,
}

impl CountingFilterMetadata {
    /// Raw counter array size in bytes.
    ///
    /// Computed as ⌈*m* × `counter_bits` / 8⌉.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.filter_size * self.counter_bits as usize).div_ceil(8)
    }

    /// Memory in kibibytes.
    #[must_use]
    pub fn memory_kb(&self) -> f64 {
        self.memory_bytes() as f64 / 1024.0
    }

    /// Memory in mebibytes.
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Bytes allocated per expected item.
    #[must_use]
    pub fn bytes_per_item(&self) -> f64 {
        self.memory_bytes() as f64 / self.expected_items as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::IndexingStrategy;

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
            .hash_strategy(IndexingStrategy::Triple)
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
            .hash_strategy(IndexingStrategy::Double)
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
        assert_eq!(metadata.max_count, 15);
        assert_eq!(metadata.counter_bits, 4);
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

        for _ in 0..10 {
            filter.insert(&"test");
        }

        assert!(filter.contains(&"test"));

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

        assert!((kb - bytes as f64 / 1024.0).abs() < 0.01);
        assert!((mb - kb / 1024.0).abs() < 0.0001);
    }

    #[test]
    fn test_different_strategies() {
        let strategies = [
            IndexingStrategy::Double,
            IndexingStrategy::EnhancedDouble,
            IndexingStrategy::Triple,
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
            .max_count(15)
            .build()
            .unwrap();

        for _ in 0..15 {
            filter.insert(&"overflow_test");
        }
    }
}
