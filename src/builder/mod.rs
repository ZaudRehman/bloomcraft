//! Builder pattern for Bloom filter construction.
//!
//! This module provides fluent, type-safe builders for all Bloom filter variants
//! with compile-time guarantees that required parameters are provided.
//!
//! # Design Philosophy
//!
//! ## Type-State Pattern
//!
//! Builders use the type-state pattern to enforce parameter requirements at
//! compile time. Each builder progresses through states, with methods only
//! available in appropriate states.
//!
//! ## Error Handling
//!
//! - **Compile-time errors**: Missing required parameters
//! - **Runtime errors**: Invalid parameter values (out of range)
//!
//! # Examples
//!
//! ## Standard Filter Builder
//!
//! ```
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! // Type-safe: can't forget required parameters
//! let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Counting Filter Builder
//!
//! ```
//! use bloomcraft::builder::CountingBloomFilterBuilder;
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let filter: CountingBloomFilter<&str> = CountingBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .max_count(255)  // Optional, defaults to 15
//!     .build()
//!     .unwrap();
//! ```
//!
//! ## Scalable Filter Builder
//!
//! ```
//! use bloomcraft::builder::ScalableBloomFilterBuilder;
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let filter: ScalableBloomFilter<&str> = ScalableBloomFilterBuilder::new()
//!     .initial_capacity(1_000)
//!     .false_positive_rate(0.01)
//!     .growth_factor(2.0)      // Optional
//!     .tightening_ratio(0.85)  // Optional
//!     .build()
//!     .unwrap();
//! ```
//!
//! # Builder Comparison
//!
//! | Builder | Required Parameters | Key Optional Parameters |
//! |---------|---------------------|-------------------------|
//! | Standard | items, fp_rate | hash_strategy |
//! | Counting | items, fp_rate | max_count, hash_strategy |
//! | Scalable | initial_capacity, fp_rate | growth_factor, tightening_ratio |

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod counting;
pub mod scalable;
pub mod standard;

pub use counting::CountingBloomFilterBuilder;
pub use scalable::ScalableBloomFilterBuilder;
pub use standard::StandardBloomFilterBuilder;

use crate::error::{BloomCraftError, Result};

/// Common validation functions for all builders.
mod validation {
    use super::*;

    /// Validate expected items count.
    ///
    /// # Errors
    ///
    /// Returns error if `items == 0`.
    #[inline]
    pub fn validate_items(items: usize) -> Result<()> {
        if items == 0 {
            return Err(BloomCraftError::invalid_item_count(items));
        }
        Ok(())
    }

    /// Validate false positive rate.
    ///
    /// # Errors
    ///
    /// Returns error if `fp_rate` is not in (0, 1).
    #[inline]
    pub fn validate_fp_rate(fp_rate: f64) -> Result<()> {
        if fp_rate <= 0.0 || fp_rate >= 1.0 {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fp_rate));
        }
        Ok(())
    }

    /// Validate growth factor for scalable filters.
    ///
    /// # Errors
    ///
    /// Returns error if `factor <= 1.0` or `factor > 10.0`.
    #[inline]
    pub fn validate_growth_factor(factor: f64) -> Result<()> {
        if factor <= 1.0 {
            return Err(BloomCraftError::invalid_parameters(
                format!("Growth factor {} must be > 1.0", factor),
            ));
        }
        if factor > 10.0 {
            return Err(BloomCraftError::invalid_parameters(
                format!("Growth factor {} exceeds reasonable limit (10.0)", factor),
            ));
        }
        Ok(())
    }

    /// Validate tightening ratio for scalable filters.
    ///
    /// # Errors
    ///
    /// Returns error if `ratio` is not in (0, 1).
    #[inline]
    pub fn validate_tightening_ratio(ratio: f64) -> Result<()> {
        if ratio <= 0.0 || ratio >= 1.0 {
            return Err(BloomCraftError::invalid_parameters(
                format!("Tightening ratio {} must be in (0, 1)", ratio),
            ));
        }
        Ok(())
    }

    /// Validate counter max count for counting filters.
    ///
    /// # Errors
    ///
    /// Returns error if `max_count < 1` or `max_count > 255`.
    #[inline]
    pub fn validate_max_count(max_count: u8) -> Result<()> {
        if max_count < 1 {
            return Err(BloomCraftError::invalid_parameters(
                "max_count must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Prelude for convenient builder imports.
pub mod prelude {
    pub use super::{
        CountingBloomFilterBuilder,
        ScalableBloomFilterBuilder,
        StandardBloomFilterBuilder,
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
