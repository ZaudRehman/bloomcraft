//! Builder for standard Bloom filters.
//!
//! # Type-State Pattern
//!
//! This builder uses the type-state pattern to ensure required parameters
//! are provided at compile time. The builder progresses through states:
//!
//! ```text
//! Initial → WithItems → Complete → StandardBloomFilter
//!     ↓         ↓           ↓
//!   .expected_items()  .false_positive_rate()  .build()
//! ```
//!
//! # Examples
//!
//! ## Minimal Configuration
//!
//! ```
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Full Configuration
//!
//! ```
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::hash::HashStrategy;
//!
//! let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .hash_strategy(HashStrategy::EnhancedDouble)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Error Handling
//!
//! ```
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let result: Result<StandardBloomFilter<&str>, _> = StandardBloomFilterBuilder::new()
//!     .expected_items(0)  // Invalid!
//!     .false_positive_rate(0.01)
//!     .build();
//!
//! assert!(result.is_err());
//! ```

use crate::core::params;
use crate::error::Result;
use crate::hash::{BloomHasher, DefaultHasher, HashStrategy};
use crate::filters::standard::StandardBloomFilter;
use std::marker::PhantomData;

/// Type-state marker: Initial state (no parameters set).
pub struct Initial;

/// Type-state marker: Items count is set.
pub struct WithItems;

/// Type-state marker: All required parameters set.
pub struct Complete;

/// Builder for standard Bloom filters with type-state guarantees.
///
/// # Type Parameters
///
/// - `State`: Current builder state (Initial, WithItems, Complete)
/// - `H`: Hash function type (defaults to DefaultHasher)
///
/// # Thread Safety
///
/// Builder is not thread-safe (not `Send + Sync`). Create filters, then share them.
pub struct StandardBloomFilterBuilder<State, H = DefaultHasher> {
    expected_items: Option<usize>,
    fp_rate: Option<f64>,
    hash_strategy: HashStrategy,
    _state: PhantomData<State>,
    _hasher: PhantomData<H>,
}

impl StandardBloomFilterBuilder<Initial, DefaultHasher> {
    /// Create a new standard filter builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    ///
    /// let builder = StandardBloomFilterBuilder::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_items: None,
            fp_rate: None,
            hash_strategy: HashStrategy::EnhancedDouble,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> StandardBloomFilterBuilder<Initial, H> {
    /// Set the expected number of items to insert.
    ///
    /// This is a required parameter. Transitions builder to `WithItems` state.
    ///
    /// # Arguments
    ///
    /// * `items` - Expected number of elements (must be > 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    ///
    /// let builder = StandardBloomFilterBuilder::new()
    ///     .expected_items(10_000);
    /// ```
    #[must_use]
    pub fn expected_items(self, items: usize) -> StandardBloomFilterBuilder<WithItems, H> {
        StandardBloomFilterBuilder {
            expected_items: Some(items),
            fp_rate: self.fp_rate,
            hash_strategy: self.hash_strategy,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> StandardBloomFilterBuilder<WithItems, H> {
    /// Set the target false positive rate.
    ///
    /// This is a required parameter. Transitions builder to `Complete` state.
    ///
    /// # Arguments
    ///
    /// * `fp_rate` - Target false positive probability (must be in (0, 1))
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    ///
    /// let builder = StandardBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01);  // 1% false positive rate
    /// ```
    #[must_use]
    pub fn false_positive_rate(self, fp_rate: f64) -> StandardBloomFilterBuilder<Complete, H> {
        StandardBloomFilterBuilder {
            expected_items: self.expected_items,
            fp_rate: Some(fp_rate),
            hash_strategy: self.hash_strategy,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }

    /// Set the hash strategy (optional).
    ///
    /// Defaults to `HashStrategy::EnhancedDouble` if not specified.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Hash strategy (Double, EnhancedDouble, or Triple)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    /// use bloomcraft::hash::HashStrategy;
    ///
    /// let builder = StandardBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .hash_strategy(HashStrategy::Triple);
    /// ```
    #[must_use]
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H> StandardBloomFilterBuilder<Complete, H> {
    /// Set the hash strategy (optional, can be set in Complete state too).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    /// use bloomcraft::hash::HashStrategy;
    ///
    /// let builder = StandardBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01)
    ///     .hash_strategy(HashStrategy::Double);
    /// ```
    #[must_use]
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H: BloomHasher + Default + Clone> StandardBloomFilterBuilder<Complete, H> {
    /// Build the standard Bloom filter.
    ///
    /// Validates all parameters and constructs the filter.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `expected_items == 0`
    /// - `fp_rate` not in (0, 1)
    /// - Calculated parameters are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert!(filter.is_empty());
    /// ```
    pub fn build<T: std::hash::Hash>(self) -> Result<StandardBloomFilter<T, H>> {
        // Extract parameters (safe because we're in Complete state)
        let expected_items = self.expected_items.expect("items must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        // Validate parameters
        super::validation::validate_items(expected_items)?;
        super::validation::validate_fp_rate(fp_rate)?;

        // Calculate optimal parameters
        let filter_size = params::optimal_bit_count(expected_items, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, expected_items)?;

        // Validate calculated parameters
        params::validate_params(filter_size, expected_items, num_hashes)?;

        // Construct filter
        Ok(StandardBloomFilter::with_params(filter_size, num_hashes, H::default())?)
    }

    /// Build the filter and return it with metadata.
    ///
    /// Returns a tuple of (filter, metadata) where metadata contains
    /// the calculated parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::builder::standard::FilterMetadata;
    ///
    /// let (filter, metadata): (StandardBloomFilter<&str>, FilterMetadata) = StandardBloomFilterBuilder::new()
    ///     .expected_items(10_000)
    ///     .false_positive_rate(0.01)
    ///     .build_with_metadata()
    ///     .unwrap();
    ///
    /// println!("Filter size: {} bits", metadata.filter_size);
    /// println!("Hash functions: {}", metadata.num_hashes);
    /// println!("Bytes per item: {:.2}", metadata.bytes_per_item);
    /// ```
    pub fn build_with_metadata<T: std::hash::Hash>(self) -> Result<(StandardBloomFilter<T, H>, FilterMetadata)> {
        let expected_items = self.expected_items.expect("items must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        // Validate
        super::validation::validate_items(expected_items)?;
        super::validation::validate_fp_rate(fp_rate)?;

        // Calculate parameters
        let filter_size = params::optimal_bit_count(expected_items, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, expected_items)?;
        params::validate_params(filter_size, expected_items, num_hashes)?;

        // Build filter
        let filter = StandardBloomFilter::with_params(filter_size, num_hashes, H::default());

        // Create metadata
        let metadata = FilterMetadata {
            expected_items,
            fp_rate,
            filter_size,
            num_hashes,
            hash_strategy: self.hash_strategy,
            bytes_per_item: filter_size as f64 / 8.0 / expected_items as f64,
        };

        Ok((filter?, metadata))
    }
}

impl Default for StandardBloomFilterBuilder<Initial, DefaultHasher> {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a constructed filter.
///
/// Contains the parameters used to create the filter, useful for
/// monitoring, debugging, and optimization.
#[derive(Debug, Clone)]
pub struct FilterMetadata {
    /// Expected number of items
    pub expected_items: usize,
    /// Target false positive rate
    pub fp_rate: f64,
    /// Actual filter size in bits
    pub filter_size: usize,
    /// Number of hash functions
    pub num_hashes: usize,
    /// Hash strategy used
    pub hash_strategy: HashStrategy,
    /// Memory efficiency (bytes per item)
    pub bytes_per_item: f64,
}

impl FilterMetadata {
    /// Get the theoretical memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.filter_size + 7) / 8  // Round up to nearest byte
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_minimal() {
        let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_with_strategy() {
        let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .hash_strategy(HashStrategy::Triple)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_with_metadata() {
        let (filter, metadata): (StandardBloomFilter<String>, _) = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build_with_metadata()
            .unwrap();

        assert!(filter.is_empty());
        assert_eq!(metadata.expected_items, 10_000);
        assert!((metadata.fp_rate - 0.01).abs() < 0.001);
        assert!(metadata.filter_size > 0);
        assert!(metadata.num_hashes > 0);
        assert!(metadata.bytes_per_item > 0.0);
    }

    #[test]
    fn test_builder_invalid_items() {
        let result: Result<StandardBloomFilter<String>> = StandardBloomFilterBuilder::new()
            .expected_items(0)
            .false_positive_rate(0.01)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_fp_rate_zero() {
        let result: Result<StandardBloomFilter<String>> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.0)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_fp_rate_one() {
        let result: Result<StandardBloomFilter<String>> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(1.0)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_fp_rate_negative() {
        let result: Result<StandardBloomFilter<String>> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(-0.1)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_fp_rate_too_large() {
        let result: Result<StandardBloomFilter<String>> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(1.5)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_memory_calculations() {
        let (_, metadata): (StandardBloomFilter<String>, _) = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build_with_metadata()
            .unwrap();

        let bytes = metadata.memory_bytes();
        let kb = metadata.memory_kb();
        let mb = metadata.memory_mb();

        assert!(bytes > 0);
        assert!(kb > 0.0);
        assert!(mb > 0.0);

        // Consistency checks
        assert!((kb - bytes as f64 / 1024.0).abs() < 0.01);
        assert!((mb - kb / 1024.0).abs() < 0.0001);
    }

    #[test]
    fn test_filter_functionality_strings() {
        let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        filter.insert(&"hello");
        filter.insert(&"world");

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(!filter.contains(&"missing"));
    }

    #[test]
    fn test_filter_functionality_integers() {
        let filter: StandardBloomFilter<i32> = StandardBloomFilterBuilder::new()
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        filter.insert(&12345);
        filter.insert(&67890);

        assert!(filter.contains(&12345));
        assert!(filter.contains(&67890));
        assert!(!filter.contains(&99999));
    }

    #[test]
    fn test_different_strategies() {
        let strategies = [
            HashStrategy::Double,
            HashStrategy::EnhancedDouble,
            HashStrategy::Triple,
        ];

        for strategy in strategies {
            let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
                .expected_items(1_000)
                .false_positive_rate(0.01)
                .hash_strategy(strategy)
                .build()
                .unwrap();

            filter.insert(&"test");
            assert!(filter.contains(&"test"));
        }
    }

    #[test]
    fn test_builder_reusability() {
        // Can't reuse builder after build() due to move semantics
        let builder = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01);

        let _filter: StandardBloomFilter<String> = builder.build().unwrap();
        // builder is moved, can't use again - this is intentional
    }

    #[test]
    fn test_large_filter() {
        let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
            .expected_items(1_000_000)
            .false_positive_rate(0.001)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_small_filter() {
        let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
            .expected_items(100)
            .false_positive_rate(0.1)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_strategy_can_be_set_before_fp_rate() {
        let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .hash_strategy(HashStrategy::Double)
            .false_positive_rate(0.01)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }

    #[test]
    fn test_strategy_can_be_set_after_fp_rate() {
        let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .hash_strategy(HashStrategy::Double)
            .build()
            .unwrap();

        assert!(filter.is_empty());
    }
}
