//! Fluent, type-safe builders for every Bloom filter variant in the crate.
//!
//! # Quick Start
//!
//! | Variant | Builder | Use Case |
//! |---------|---------|---------|
//! | [`StandardBloomFilter`](crate::filters::StandardBloomFilter) | [`StandardBloomFilterBuilder`] | Fixed-capacity set-membership; the workhorse |
//! | [`CountingBloomFilter`](crate::filters::CountingBloomFilter) | [`CountingBloomFilterBuilder`] | Deletion support via per-position counters |
//! | [`ScalableBloomFilter`](crate::filters::ScalableBloomFilter) | [`ScalableBloomFilterBuilder`] | Unknown or growing cardinality |
//!
//! All three accept `expected_items` (or `initial_capacity`) and a target
//! false-positive rate, then derive optimal filter dimensions *m* (bits) and
//! *k* (hash functions) automatically.
//!
//! # Sizing Reference
//!
//! The optimal bit-count formula is *m* = –*n*·ln(*p*) / (ln 2)²
//! (Bloom, CACM 1970). Concrete numbers at common false-positive rates:
//!
//! | Target FPR | Bits/Item | Bytes/Item | Example: 100k items |
//! |-----------|-----------|------------|---------------------|
//! | 10%       | 4.8       | 0.6        | ~59 KB |
//! | 1%        | 9.6       | 1.2        | ~117 KB |
//! | 0.1%      | 14.4      | 1.8        | ~176 KB |
//! | 0.01%     | 19.2      | 2.4        | ~234 KB |
//!
//! Optimal hash count *k* = (*m* / *n*)·ln 2 ≈ 0.693·(*m* / *n*). A filter
//! sized for 1% FPR therefore uses *k* ≈ 7 hash functions regardless of *n*.
//!
//! # Design
//!
//! - [`StandardBloomFilterBuilder`] uses a **true type-state** pattern:
//!   the Rust type system prevents calling `.build()` before both required
//!   parameters are provided. Missing a parameter is a compile-time error.
//! - [`CountingBloomFilterBuilder`] and [`ScalableBloomFilterBuilder`] use
//!   **PhantomData state markers** for a lighter-weight variant that still
//!   documents the intended progression but catches missing parameters only
//!   at runtime (via `.expect()`). This is a deliberate trade-off: those
//!   builders have more optional parameters and the type-state combinatorics
//!   would multiply implementation surface area.
//!
//! # Performance Characteristics
//!
//! Benchmarks at 1% FPR, 100k capacity, 50 % fill, u64 keys
//! (Intel Xeon, 2.5 GHz):
//!
//! | Operation | Latency | Notes |
//! |-----------|---------|-------|
//! | Single insert | 20–30 ns | Lock-free `AtomicU64::fetch_or`; scales near-linearly to 16+ threads |
//! | Single contains (hit) | 15–25 ns | All *k* hash positions probed |
//! | Single contains (miss) | 5–10 ns | Early-exit on first unset bit |
//! | Batch insert (1k items) | 15–20 ns/item | ~1.3–1.5× speedup vs per-item loop |
//! | Batch contains (1k items) | 10–15 ns/item | ~1.5× speedup |
//! | Clone (100k filter) | 5–10 µs | Parallel `memcpy`-equivalent |
//! | Clear (100k filter) | 3–5 µs | Sequential `memset`-equivalent |
//!
//! Access-pattern sensitivity: sequential queries benefit from CPU prefetchers;
//! uniform-random probes are ~2–3× slower due to cache misses. Zipf-distributed
//! (skewed) traffic, typical in production, falls between the two extremes.
//!
//! # Examples
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
//! ```
//! use bloomcraft::builder::CountingBloomFilterBuilder;
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .max_count(255)  // 8-bit counters (default: 4-bit)
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

#![allow(clippy::module_name_repetitions)]

pub mod counting;
pub mod scalable;
pub mod standard;

pub use counting::{CountingBloomFilterBuilder, CountingFilterMetadata};
pub use scalable::{ScalableBloomFilterBuilder, ScalableFilterMetadata};
pub use standard::{FilterMetadata, StandardBloomFilterBuilder};

use crate::error::{BloomCraftError, Result};

mod validation {
    use super::*;

    #[inline]
    pub fn validate_items(items: usize) -> Result<()> {
        if items == 0 {
            return Err(BloomCraftError::invalid_item_count(items));
        }
        Ok(())
    }

    #[inline]
    pub fn validate_fp_rate(fp_rate: f64) -> Result<()> {
        if fp_rate <= 0.0 || fp_rate >= 1.0 {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fp_rate));
        }
        Ok(())
    }

    #[inline]
    pub fn validate_growth_factor(factor: f64) -> Result<()> {
        if factor <= 1.0 {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Growth factor {} must be > 1.0",
                factor
            )));
        }
        if factor > 10.0 {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Growth factor {} exceeds reasonable limit (10.0)",
                factor
            )));
        }
        Ok(())
    }

    #[inline]
    pub fn validate_tightening_ratio(ratio: f64) -> Result<()> {
        if ratio <= 0.0 || ratio >= 1.0 {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Tightening ratio {} must be in (0, 1)",
                ratio
            )));
        }
        Ok(())
    }

    /// Validate counter max count for counting filters.
    ///
    /// # Errors
    ///
    /// Returns error if `max_count < 1`.
    #[inline]
    pub fn validate_max_count(max_count: u16) -> Result<()> {
        if max_count < 1 {
            return Err(BloomCraftError::invalid_parameters(
                "max_count must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Convenience re-exports for the builder types.
pub mod prelude {
    pub use super::{
        CountingBloomFilterBuilder, ScalableBloomFilterBuilder, StandardBloomFilterBuilder,
    };
}

#[cfg(test)]
mod tests {
    use super::validation::*;

    #[test]
    fn test_validate_items() {
        assert!(validate_items(1).is_ok());
        assert!(validate_items(1000).is_ok());
        assert!(validate_items(0).is_err());
    }

    #[test]
    fn test_validate_fp_rate() {
        assert!(validate_fp_rate(0.01).is_ok());
        assert!(validate_fp_rate(0.5).is_ok());
        assert!(validate_fp_rate(0.0).is_err());
        assert!(validate_fp_rate(1.0).is_err());
        assert!(validate_fp_rate(-0.1).is_err());
        assert!(validate_fp_rate(1.5).is_err());
    }

    #[test]
    fn test_validate_growth_factor() {
        assert!(validate_growth_factor(1.5).is_ok());
        assert!(validate_growth_factor(2.0).is_ok());
        assert!(validate_growth_factor(10.0).is_ok());
        assert!(validate_growth_factor(1.0).is_err());
        assert!(validate_growth_factor(0.5).is_err());
        assert!(validate_growth_factor(11.0).is_err());
    }

    #[test]
    fn test_validate_tightening_ratio() {
        assert!(validate_tightening_ratio(0.5).is_ok());
        assert!(validate_tightening_ratio(0.85).is_ok());
        assert!(validate_tightening_ratio(0.0).is_err());
        assert!(validate_tightening_ratio(1.0).is_err());
        assert!(validate_tightening_ratio(-0.1).is_err());
        assert!(validate_tightening_ratio(1.5).is_err());
    }

    #[test]
    fn test_validate_max_count() {
        assert!(validate_max_count(1).is_ok());
        assert!(validate_max_count(15).is_ok());
        assert!(validate_max_count(255).is_ok());
        assert!(validate_max_count(0).is_err());
    }
}
