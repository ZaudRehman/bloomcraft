//! Builder for [`ScalableBloomFilter`].
//!
//! Scalable Bloom filters ([Almeida et al., 2007])
//! add new internal filter slices on demand, keeping the overall false-positive
//! rate below a user-specified bound even when the final cardinality is unknown
//! at construction time.
//!
//! # State Machine
//!
//! ```text
//! Initial  ──.initial_capacity(n)──→  WithCapacity
//! WithCapacity ──.false_positive_rate(p)──→  Complete
//! Complete  ──.build()──→  Result<ScalableBloomFilter<T, H>>
//! ```
//!
//! Optional setters (`growth_factor`, `tightening_ratio`) are available at both
//! `WithCapacity` and `Complete` and can appear in any order.
//!
//! # Growth Model
//!
//! When the active slice reaches its fill threshold, the filter appends a new
//! slice with:
//!
//! * Capacity = previous slice capacity × `growth_factor`
//! * FP rate  = previous slice FP rate × `tightening_ratio`
//!
//! The FP rates form a geometric series *p*₀, *p*₀·*r*, *p*₀·*r*², …,
//! so the overall upper bound on the combined FP rate is the series sum:
//!
//! *p*∞ ≤ *p*₀ / (1 − *r*)
//!
//! where *r* is the tightening ratio. For the defaults (*p*₀ = 1%, *r* = 0.85),
//! *p*∞ ≈ 6.7 %. The bound is conservative because the union bound assumes each
//! slice's FP events are independent.
//!
//! | Tightening Ratio | FP Bound (× initial) | Slices to 100× initial capacity |
//! |-----------------|---------------------|--------------------------------|
//! | 0.50            | 2.0×                | 8 |
//! | 0.75            | 4.0×                | 17 |
//! | 0.85 (default)  | 6.7×                | 27 |
//! | 0.90            | 10.0×               | 42 |
//!
//! Total capacity after *k* slices follows: *C*₀ × (1 − *g*ⁱ) / (1 − *g*)
//! when *g* ≠ 1, where *g* is the growth factor and *C*₀ the initial capacity.
//!
//! # Limits
//!
//! * Maximum slices: [`MAX_FILTERS`] (crate-internal constant, typically 32).
//! * Growth factor: (1.0, 10.0].
//! * Tightening ratio: (0.0, 1.0).
//! * Once [`MAX_FILTERS`] is reached, the filter returns
//!   [`CapacityExceeded`](crate::error::BloomCraftError::CapacityExceeded) on further insert
//!   attempts (unless `CapacityExhaustedBehavior::Silent` is set).
//!
//! # Performance
//!
//! Single insert/contains latency grows linearly with the number of slices.
//! A query probes every slice until a miss or until all slices contain the item,
//! so worst-case latency = *s* × single-filter latency. With 4-bit or 8-bit
//! counters, 4–8 slices remain sub-microsecond on modern hardware.
//!
//! Batch operations (`insert_batch`, `contains_batch`) amortize the slice
//! iteration overhead, achieving near-single-filter throughput per slice.
//!
//! # Examples
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
//! ```
//! use bloomcraft::builder::ScalableBloomFilterBuilder;
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let filter: ScalableBloomFilter<&str> = ScalableBloomFilterBuilder::new()
//!     .initial_capacity(1_000)
//!     .false_positive_rate(0.01)
//!     .growth_factor(2.0)
//!     .tightening_ratio(0.85)
//!     .build()
//!     .unwrap();
//! ```
//!
//! [Almeida et al., 2007]: https://doi.org/10.1007/978-3-540-72986-0_17
//! [`MAX_FILTERS`]: crate::filters::scalable::MAX_FILTERS

use crate::core::params;
use crate::error::Result;
use crate::filters::scalable::ScalableBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::marker::PhantomData;

/// State marker: initial builder state.
pub struct Initial;

/// State marker: initial capacity has been provided.
pub struct WithCapacity;

/// State marker: all required parameters have been provided.
pub struct Complete;

const DEFAULT_GROWTH_FACTOR: f64 = 2.0;
const DEFAULT_TIGHTENING_RATIO: f64 = 0.85;

/// Builder for [`ScalableBloomFilter`] with state-machine parameter enforcement.
pub struct ScalableBloomFilterBuilder<State, H = StdHasher> {
    initial_capacity: Option<usize>,
    fp_rate: Option<f64>,
    growth_factor: f64,
    tightening_ratio: f64,
    _state: PhantomData<State>,
    hasher: H,
}

impl ScalableBloomFilterBuilder<Initial, StdHasher> {
    /// Creates a new builder with the default hasher ([`StdHasher`]).
    ///
    /// Defaults:
    /// * `growth_factor` = 2.0 (geometric, doubles each slice).
    /// * `tightening_ratio` = 0.85 (15 % tighter per slice).
    #[must_use]
    pub fn new() -> Self {
        Self {
            initial_capacity: None,
            fp_rate: None,
            growth_factor: DEFAULT_GROWTH_FACTOR,
            tightening_ratio: DEFAULT_TIGHTENING_RATIO,
            _state: PhantomData,
            hasher: StdHasher::new(),
        }
    }
}

impl<H> ScalableBloomFilterBuilder<Initial, H> {
    /// Sets the initial capacity (number of items the first slice can hold)
    /// and advances to `WithCapacity`.
    ///
    /// Required parameter. The first slice is sized for this many items at the
    /// target FP rate. Subsequent slices grow geometrically by `growth_factor`.
    #[must_use]
    pub fn initial_capacity(self, capacity: usize) -> ScalableBloomFilterBuilder<WithCapacity, H> {
        ScalableBloomFilterBuilder {
            initial_capacity: Some(capacity),
            fp_rate: self.fp_rate,
            growth_factor: self.growth_factor,
            tightening_ratio: self.tightening_ratio,
            _state: PhantomData,
            hasher: self.hasher,
        }
    }
}

impl<H> ScalableBloomFilterBuilder<WithCapacity, H> {
    /// Sets the target false-positive rate for the first slice and advances
    /// to `Complete`.
    ///
    /// The overall FP rate bound is `fp_rate / (1 - tightening_ratio)`.
    /// See [growth model](index.html#growth-model).
    #[must_use]
    pub fn false_positive_rate(self, fp_rate: f64) -> ScalableBloomFilterBuilder<Complete, H> {
        ScalableBloomFilterBuilder {
            initial_capacity: self.initial_capacity,
            fp_rate: Some(fp_rate),
            growth_factor: self.growth_factor,
            tightening_ratio: self.tightening_ratio,
            _state: PhantomData,
            hasher: self.hasher,
        }
    }

    /// Sets the geometric growth factor for slice capacity.
    ///
    /// Each new slice has capacity = previous slice capacity × `factor`.
    /// Default: 2.0. Must satisfy (1.0, 10.0].
    #[must_use]
    pub fn growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Sets the tightening ratio for per-slice FP rates.
    ///
    /// Each new slice has FP rate = previous slice FP rate × `ratio`.
    /// Default: 0.85. Must be in (0, 1). Lower values keep the overall FP rate
    /// tighter but increase per-slice memory.
    #[must_use]
    pub fn tightening_ratio(mut self, ratio: f64) -> Self {
        self.tightening_ratio = ratio;
        self
    }
}

impl<H> ScalableBloomFilterBuilder<Complete, H> {
    /// Sets the geometric growth factor (available in `Complete` state too).
    #[must_use]
    pub fn growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Sets the tightening ratio (available in `Complete` state too).
    #[must_use]
    pub fn tightening_ratio(mut self, ratio: f64) -> Self {
        self.tightening_ratio = ratio;
        self
    }
}

impl<H: BloomHasher + Default + Clone> ScalableBloomFilterBuilder<Complete, H> {
    /// Constructs the scalable Bloom filter.
    ///
    /// # Errors
    ///
    /// | Condition | Error Variant |
    /// |-----------|--------------|
    /// | `initial_capacity == 0` | `InvalidItemCount` |
    /// | `fp_rate` ∉ (0, 1) | `FalsePositiveRateOutOfBounds` |
    /// | `growth_factor ≤ 1.0` or `> 10.0` | `InvalidParameters` |
    /// | `tightening_ratio` ∉ (0, 1) | `InvalidParameters` |
    /// | Derived *m* or *k* exceed limits | `InvalidParameters` |
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

        let growth = crate::filters::scalable::GrowthStrategy::Geometric(self.growth_factor);

        ScalableBloomFilter::with_strategy_and_hasher(
            initial_capacity,
            fp_rate,
            self.tightening_ratio,
            growth,
            self.hasher,
        )
    }

    /// Constructs the filter and returns a [`ScalableFilterMetadata`] snapshot.
    ///
    /// The metadata is computed from the builder's parameters and is guaranteed
    /// consistent with the returned filter.
    ///
    /// # Errors
    ///
    /// Same conditions as [`build`](Self::build).
    pub fn build_with_metadata<T: std::hash::Hash>(
        self,
    ) -> Result<(ScalableBloomFilter<T, H>, ScalableFilterMetadata)> {
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
            self.hasher,
        )?;

        let max_fp_rate_bound = fp_rate / (1.0 - self.tightening_ratio);

        let metadata = ScalableFilterMetadata {
            initial_capacity,
            initial_fp_rate: fp_rate,
            growth_factor: self.growth_factor,
            tightening_ratio: self.tightening_ratio,
            initial_filter_size: filter_size,
            initial_num_hashes: num_hashes,
            max_fp_rate_bound,
        };

        Ok((filter, metadata))
    }
}

impl Default for ScalableBloomFilterBuilder<Initial, StdHasher> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construction-time metadata for a [`ScalableBloomFilter`].
///
/// Returned by [`ScalableBloomFilterBuilder::build_with_metadata`].
/// Provides capacity planning helpers ([`slice_capacity`](Self::slice_capacity),
/// [`total_capacity`](Self::total_capacity), [`slices_for_capacity`](Self::slices_for_capacity)).
#[derive(Debug, Clone)]
pub struct ScalableFilterMetadata {
    /// *C*₀ — capacity of the first filter slice.
    pub initial_capacity: usize,
    /// *p*₀ — false-positive rate of the first slice.
    pub initial_fp_rate: f64,
    /// *g* — geometric growth factor for slice capacity.
    pub growth_factor: f64,
    /// *r* — geometric tightening ratio for per-slice FP rate.
    pub tightening_ratio: f64,
    /// *m* — bit count of the initial filter slice.
    pub initial_filter_size: usize,
    /// *k* — hash count of the initial filter slice.
    pub initial_num_hashes: usize,
    /// Upper bound on the overall FP rate: *p*₀ / (1 − *r*).
    pub max_fp_rate_bound: f64,
}

impl ScalableFilterMetadata {
    /// Capacity of slice *i* (0-indexed).
    ///
    /// Returns `usize::MAX` when the computation would overflow or produce
    /// a non-finite value.
    #[must_use]
    pub fn slice_capacity(&self, i: usize) -> usize {
        const MAX_CAP: f64 = usize::MAX as f64;
        let computed = self.initial_capacity as f64 * self.growth_factor.powi(i as i32);
        if computed > MAX_CAP || !computed.is_finite() {
            return usize::MAX;
        }
        computed as usize
    }

    /// FP rate of slice *i* (0-indexed).
    #[must_use]
    pub fn slice_fp_rate(&self, i: usize) -> f64 {
        self.initial_fp_rate * self.tightening_ratio.powi(i as i32)
    }

    /// Total capacity across the first *s* slices.
    ///
    /// Returns the sum of [`slice_capacity`](Self::slice_capacity) for
    /// *i* = 0 .. *s*−1.
    #[must_use]
    pub fn total_capacity(&self, s: usize) -> usize {
        let mut total = 0usize;
        for i in 0..s {
            total = total.saturating_add(self.slice_capacity(i));
        }
        total
    }

    /// Minimum number of slices needed to reach `target_capacity`.
    ///
    /// Bounded by [`MAX_FILTERS`](crate::filters::scalable::MAX_FILTERS); returns `MAX_FILTERS + 1` if the target
    /// cannot be reached within the slice limit.
    #[must_use]
    pub fn slices_for_capacity(&self, target_capacity: usize) -> usize {
        let mut slices = 0;
        let mut total = 0usize;
        while total < target_capacity {
            total = total.saturating_add(self.slice_capacity(slices));
            slices += 1;
            if slices > crate::filters::scalable::MAX_FILTERS {
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
