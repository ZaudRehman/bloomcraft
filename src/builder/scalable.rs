//! Builder for scalable Bloom filters.
//!
//! Scalable Bloom filters automatically grow as more items are inserted,
//! maintaining a target false positive rate across expansions.
//!
//! # Type-State Pattern
//!
//! ```text
//! Initial → WithCapacity → Complete → ScalableBloomFilter
//!     ↓          ↓             ↓
//!   .initial_capacity()  .false_positive_rate()  .build()
//! ```
//!
//! # Growth Strategy
//!
//! When a filter slice fills up:
//! 1. Create new slice with capacity = old_capacity × growth_factor
//! 2. Tighten FP rate: new_fp_rate = old_fp_rate × tightening_ratio
//! 3. Append to filter chain
//!
//! This ensures the overall FP rate stays bounded.
//!
//! # Examples
//!
//! ## Minimal Configuration
//!
//! ```
//! use bloomcraft::builder::ScalableBloomFilterBuilder;
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let filter: ScalableBloomFilter<&str> = ScalableBloomFilterBuilder::new()
//!     .initial_capacity(1_000)
//!     .false_positive_rate(0.01)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Full Configuration
//!
//! ```
//! use bloomcraft::builder::ScalableBloomFilterBuilder;
//! use bloomcraft::filters::ScalableBloomFilter;
//! use bloomcraft::hash::HashStrategy;
//!
//! let filter: ScalableBloomFilter<&str> = ScalableBloomFilterBuilder::new()
//!     .initial_capacity(1_000)
//!     .false_positive_rate(0.01)
//!     .growth_factor(2.0)       // Double capacity each growth
//!     .tightening_ratio(0.85)   // Tighten FP rate by 15%
//!     .hash_strategy(HashStrategy::EnhancedDouble)
//!     .build()
//!     .unwrap();
//! ```

use crate::core::params;
use crate::error::Result;
use crate::hash::{BloomHasher, DefaultHasher, HashStrategy};
use crate::filters::scalable::ScalableBloomFilter;
use std::marker::PhantomData;

/// Type-state marker: Initial state.
pub struct Initial;

/// Type-state marker: Initial capacity is set.
pub struct WithCapacity;

/// Type-state marker: All required parameters set.
pub struct Complete;

/// Default growth factor for scalable filters.
const DEFAULT_GROWTH_FACTOR: f64 = 2.0;

/// Default tightening ratio for false positive rates.
const DEFAULT_TIGHTENING_RATIO: f64 = 0.85;

/// Builder for scalable Bloom filters with type-state guarantees.
pub struct ScalableBloomFilterBuilder<State, H = DefaultHasher> {
    initial_capacity: Option<usize>,
    fp_rate: Option<f64>,
    growth_factor: f64,
    tightening_ratio: f64,
    hash_strategy: HashStrategy,
    _state: PhantomData<State>,
    _hasher: PhantomData<H>,
}

impl ScalableBloomFilterBuilder<Initial, DefaultHasher> {
    /// Create a new scalable Bloom filter builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::ScalableBloomFilterBuilder;
    ///
    /// let builder = ScalableBloomFilterBuilder::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            initial_capacity: None,
            fp_rate: None,
            growth_factor: DEFAULT_GROWTH_FACTOR,
            tightening_ratio: DEFAULT_TIGHTENING_RATIO,
            hash_strategy: HashStrategy::EnhancedDouble,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> ScalableBloomFilterBuilder<Initial, H> {
    /// Set the initial capacity for the first filter slice.
    ///
    /// This is the number of items the first slice can hold before
    /// the filter grows.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity (must be > 0)
    #[must_use]
    pub fn initial_capacity(self, capacity: usize) -> ScalableBloomFilterBuilder<WithCapacity, H> {
        ScalableBloomFilterBuilder {
            initial_capacity: Some(capacity),
            fp_rate: self.fp_rate,
            growth_factor: self.growth_factor,
            tightening_ratio: self.tightening_ratio,
            hash_strategy: self.hash_strategy,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }
}

impl<H> ScalableBloomFilterBuilder<WithCapacity, H> {
    /// Set the target false positive rate for the first slice.
    ///
    /// Subsequent slices will have tighter FP rates based on the
    /// tightening ratio.
    ///
    /// # Arguments
    ///
    /// * `fp_rate` - Target false positive rate (must be in (0, 1))
    #[must_use]
    pub fn false_positive_rate(self, fp_rate: f64) -> ScalableBloomFilterBuilder<Complete, H> {
        ScalableBloomFilterBuilder {
            initial_capacity: self.initial_capacity,
            fp_rate: Some(fp_rate),
            growth_factor: self.growth_factor,
            tightening_ratio: self.tightening_ratio,
            hash_strategy: self.hash_strategy,
            _state: PhantomData,
            _hasher: PhantomData,
        }
    }

    /// Set the growth factor for capacity expansion.
    ///
    /// When a slice fills up, the next slice will have capacity
    /// multiplied by this factor. Default is 2.0.
    ///
    /// # Arguments
    ///
    /// * `factor` - Growth factor (must be > 1.0 and <= 10.0)
    #[must_use]
    pub fn growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Set the tightening ratio for false positive rates.
    ///
    /// Each new slice will have its FP rate multiplied by this ratio.
    /// Default is 0.85 (15% tighter each slice).
    ///
    /// # Arguments
    ///
    /// * `ratio` - Tightening ratio (must be in (0, 1))
    #[must_use]
    pub fn tightening_ratio(mut self, ratio: f64) -> Self {
        self.tightening_ratio = ratio;
        self
    }

    /// Set the hash strategy for the filter.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Hash strategy to use
    #[must_use]
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H> ScalableBloomFilterBuilder<Complete, H> {
    /// Set the growth factor for capacity expansion.
    ///
    /// When a slice fills up, the next slice will have capacity
    /// multiplied by this factor. Default is 2.0.
    ///
    /// # Arguments
    ///
    /// * `factor` - Growth factor (must be > 1.0 and <= 10.0)
    #[must_use]
    pub fn growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Set the tightening ratio for false positive rates.
    ///
    /// Each new slice will have its FP rate multiplied by this ratio.
    /// Default is 0.85 (15% tighter each slice).
    ///
    /// # Arguments
    ///
    /// * `ratio` - Tightening ratio (must be in (0, 1))
    #[must_use]
    pub fn tightening_ratio(mut self, ratio: f64) -> Self {
        self.tightening_ratio = ratio;
        self
    }

    /// Set the hash strategy for the filter.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Hash strategy to use
    #[must_use]
    pub fn hash_strategy(mut self, strategy: HashStrategy) -> Self {
        self.hash_strategy = strategy;
        self
    }
}

impl<H: BloomHasher + Default + Clone> ScalableBloomFilterBuilder<Complete, H> {
    /// Build the scalable Bloom filter.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::builder::ScalableBloomFilterBuilder;
    ///
    /// let filter = ScalableBloomFilterBuilder::new()
    ///     .initial_capacity(1_000)
    ///     .false_positive_rate(0.01)
    ///     .build::<String>()
    ///     .unwrap();
    /// ```
    pub fn build<T: std::hash::Hash>(self) -> Result<ScalableBloomFilter<T, H>> {
        let initial_capacity = self.initial_capacity.expect("capacity must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        super::validation::validate_items(initial_capacity)?;
        super::validation::validate_fp_rate(fp_rate)?;
        super::validation::validate_growth_factor(self.growth_factor)?;
        super::validation::validate_tightening_ratio(self.tightening_ratio)?;

        let filter_size = params::optimal_bit_count(initial_capacity, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, initial_capacity)?;
        params::validate_params(filter_size, initial_capacity, num_hashes)?;

        let growth = match self.hash_strategy {
            HashStrategy::Double | HashStrategy::EnhancedDouble | HashStrategy::Triple => {
                crate::filters::scalable::GrowthStrategy::Geometric(self.growth_factor)
            }
        };

        let filter = ScalableBloomFilter::with_strategy_and_hasher(
            initial_capacity,
            fp_rate,
            self.tightening_ratio,
            growth,
            H::default(),
        );

        Ok(filter?)
    }

    /// Build the scalable Bloom filter with metadata.
    ///
    /// Returns both the filter and metadata about its configuration,
    /// useful for monitoring and capacity planning.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameters are invalid.
    pub fn build_with_metadata<T: std::hash::Hash>(self) -> Result<(ScalableBloomFilter<T, H>, ScalableFilterMetadata)> {
        let initial_capacity = self.initial_capacity.expect("capacity must be set");
        let fp_rate = self.fp_rate.expect("fp_rate must be set");

        super::validation::validate_items(initial_capacity)?;
        super::validation::validate_fp_rate(fp_rate)?;
        super::validation::validate_growth_factor(self.growth_factor)?;
        super::validation::validate_tightening_ratio(self.tightening_ratio)?;

        let filter_size = params::optimal_bit_count(initial_capacity, fp_rate)?;
        let num_hashes = params::optimal_hash_count(filter_size, initial_capacity)?;
        params::validate_params(filter_size, initial_capacity, num_hashes)?;

        let growth = crate::filters::scalable::GrowthStrategy::Geometric(self.growth_factor);

        let filter = ScalableBloomFilter::with_strategy_and_hasher(
            initial_capacity,
            fp_rate,
            self.tightening_ratio,
            growth,
            H::default(),
        );

        let max_fp_rate_bound = fp_rate / (1.0 - self.tightening_ratio);

        let metadata = ScalableFilterMetadata {
            initial_capacity,
            initial_fp_rate: fp_rate,
            growth_factor: self.growth_factor,
            tightening_ratio: self.tightening_ratio,
            hash_strategy: self.hash_strategy,
            initial_filter_size: filter_size,
            initial_num_hashes: num_hashes,
            max_fp_rate_bound,
        };

        Ok((filter?, metadata))
    }
}

impl Default for ScalableBloomFilterBuilder<Initial, DefaultHasher> {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a constructed scalable filter.
///
/// Contains configuration and computed parameters for capacity planning
/// and monitoring.
#[derive(Debug, Clone)]
pub struct ScalableFilterMetadata {
    /// Initial capacity of the first slice.
    pub initial_capacity: usize,
    /// Initial false positive rate of the first slice.
    pub initial_fp_rate: f64,
    /// Growth factor for capacity expansion.
    pub growth_factor: f64,
    /// Tightening ratio for FP rate reduction.
    pub tightening_ratio: f64,
    /// Hash strategy used by the filter.
    pub hash_strategy: HashStrategy,
    /// Bit count of the initial filter slice.
    pub initial_filter_size: usize,
    /// Number of hash functions in the initial slice.
    pub initial_num_hashes: usize,
    /// Upper bound on overall false positive rate.
    pub max_fp_rate_bound: f64,
}

impl ScalableFilterMetadata {
    /// Calculate the capacity of slice n (0-indexed).
    ///
    /// # Arguments
    ///
    /// * `n` - Slice index (0 = first slice)
    #[must_use]
    pub fn slice_capacity(&self, n: usize) -> usize {
        (self.initial_capacity as f64 * self.growth_factor.powi(n as i32)) as usize
    }

    /// Calculate the false positive rate of slice n (0-indexed).
    ///
    /// # Arguments
    ///
    /// * `n` - Slice index (0 = first slice)
    #[must_use]
    pub fn slice_fp_rate(&self, n: usize) -> f64 {
        self.initial_fp_rate * self.tightening_ratio.powi(n as i32)
    }

    /// Calculate total capacity across n slices.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of slices
    #[must_use]
    pub fn total_capacity(&self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let mut total = 0;
        for i in 0..n {
            total += self.slice_capacity(i);
        }
        total
    }

    /// Calculate how many slices are needed for a target capacity.
    ///
    /// # Arguments
    ///
    /// * `target_capacity` - Desired total capacity
    #[must_use]
    pub fn slices_for_capacity(&self, target_capacity: usize) -> usize {
        let mut slices = 0;
        let mut total = 0;
        while total < target_capacity {
            total += self.slice_capacity(slices);
            slices += 1;
            if slices > 100 {
                break;
            }
        }
        slices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_minimal() {
        let filter: ScalableBloomFilter<String> = ScalableBloomFilterBuilder::new()
            .initial_capacity(1_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_builder_invalid_capacity() {
        let result: Result<ScalableBloomFilter<String>> = ScalableBloomFilterBuilder::new()
            .initial_capacity(0)
            .false_positive_rate(0.01)
            .build();
        assert!(result.is_err());
    }
}
