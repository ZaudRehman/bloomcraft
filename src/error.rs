//! Error types for BloomCraft operations.
//!
//! This module provides comprehensive error handling for all Bloom filter operations.
//! All errors are structured for ergonomic handling and clear error messages.
//!
//! # Error Propagation
//!
//! ```
//! use bloomcraft::{Result, BloomCraftError};
//! use bloomcraft::core::params::{optimal_bit_count, optimal_hash_count};
//!
//! fn create_filter_params(n: usize, fp: f64) -> Result<(usize, usize)> {
//!     let m = optimal_bit_count(n, fp)?;
//!     let k = optimal_hash_count(m, n)?;
//!     Ok((m, k))
//! }
//! # let result = create_filter_params(1000, 0.01);
//! # assert!(result.is_ok());
//! ```

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::fmt;

/// Result type alias for BloomCraft operations.
///
/// This is the standard Result type used throughout the crate.
/// All fallible operations return [`Result<T>`] where the error type is [`BloomCraftError`].
///
/// # Examples
/// ```
/// use bloomcraft::Result;
///
/// fn validate_params(n: usize, fp: f64) -> Result<()> {
///     if n == 0 {
///         return Err(bloomcraft::BloomCraftError::invalid_item_count(n));
///     }
///     Ok(())
/// }
/// # let result = validate_params(1000, 0.01);
/// # assert!(result.is_ok());
/// ```
pub type Result<T> = std::result::Result<T, BloomCraftError>;

/// Errors that can occur during Bloom filter operations.
///
/// This enum covers all possible error conditions in BloomCraft.
/// Each variant contains relevant context to help diagnose issues.
///
/// # Design Notes
/// - `Clone` + `PartialEq` enable testing and error comparison
/// - `Debug` required by std::error::Error trait
/// - All variants include sufficient context for debugging
#[derive(Debug, Clone, PartialEq)]
pub enum BloomCraftError {
    /// Invalid filter parameters provided during construction.
    ///
    /// This occurs when parameters don't satisfy mathematical constraints
    /// or would result in a non-functional filter.
    InvalidParameters {
        /// Human-readable description of what's invalid.
        message: String,
    },

    /// False positive rate out of valid bounds (0, 1).
    ///
    /// Bloom filters require 0 < ε < 1. Values outside this range
    /// are mathematically meaningless.
    ///
    /// # Examples
    /// - ε = 0: Would require infinite memory
    /// - ε = 1: Filter accepts everything (useless)
    /// - ε < 0: Negative probability (nonsensical)
    /// - ε > 1: Probability > 100% (nonsensical)
    FalsePositiveRateOutOfBounds {
        /// The invalid false positive rate that was provided.
        fp_rate: f64,
    },

    /// Expected items count is invalid.
    ///
    /// Occurs when n ≤ 0, which would result in undefined behavior
    /// in parameter calculations (division by zero, log of zero, etc.)
    InvalidItemCount {
        /// The invalid count that was provided.
        count: usize,
    },

    /// Filter capacity would be exceeded by operation.
    ///
    /// Some filter variants have hard capacity limits. This error
    /// prevents inserting beyond those limits.
    CapacityExceeded {
        /// Maximum capacity of the filter.
        capacity: usize,
        /// Number of items attempted to insert.
        attempted: usize,
    },

    /// Operation requires features not supported by this variant.
    ///
    /// For example, trying to remove items from a standard Bloom filter
    /// (which doesn't support deletion).
    UnsupportedOperation {
        /// Name of the operation attempted.
        operation: String,
        /// Name of the filter variant.
        variant: String,
    },

    /// Parameters are incompatible between two filters.
    ///
    /// Occurs during merge/union operations when filters have
    /// different sizes, hash functions, or other critical parameters.
    IncompatibleFilters {
        /// Description of the incompatibility.
        reason: String,
    },

    /// Hash function configuration is invalid.
    ///
    /// This can occur if the number of hash functions is 0 or
    /// exceeds practical limits (typically 32).
    InvalidHashCount {
        /// The invalid hash count provided.
        count: usize,
        /// Minimum allowed value.
        min: usize,
        /// Maximum allowed value.
        max: usize,
    },

    /// Bit array size is invalid.
    ///
    /// Filter size must be positive and within system memory limits.
    InvalidFilterSize {
        /// The invalid size in bits.
        size: usize,
    },

    /// Serialization or deserialization failed.
    ///
    /// This wraps underlying I/O or format errors.
    #[cfg(feature = "serde")]
    SerializationError {
        /// Description of what failed.
        message: String,
    },

    /// Internal invariant violated.
    ///
    /// This should never occur in correct usage. If it does,
    /// it indicates a bug in BloomCraft itself.
    InternalError {
        /// Description of the invariant that was violated.
        message: String,
    },

    /// Counter overflow: attempted to increment beyond maximum value.
    CounterOverflow {
        /// Maximum value the counter can hold
        max_value: u64,
    },

    /// Counter underflow: attempted to decrement below minimum value.
    CounterUnderflow {
        /// Minimum value (always 0 for unsigned counters)
        min_value: u64,
    },

    /// Occurs when attempting to access or modify a bit at an index >= length.
    IndexOutOfBounds {
        /// The invalid index that was accessed.
        index: usize,
        /// The valid length of the bit vector.
        length: usize,
    },

    /// Occurs in range-based operations like `set_range` or `get_range`.
    InvalidRange {
        /// Start index of the range.
        start: usize,
        /// End index of the range (exclusive).
        end: usize,
        /// Length of the bit vector.
        length: usize,
        /// Description of why the range is invalid.
        reason: String,
    },
}

impl fmt::Display for BloomCraftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters { message } => {
                write!(f, "Invalid Bloom filter parameters: {}.", message)
            }
            Self::FalsePositiveRateOutOfBounds { fp_rate } => {
                write!(
                    f,
                    "False positive rate {} is out of bounds. Must be in range (0, 1).",
                    fp_rate
                )
            }
            Self::InvalidItemCount { count } => {
                write!(
                    f,
                    "Invalid item count: {}. Expected items must be greater than 0.",
                    count
                )
            }
            Self::CapacityExceeded { capacity, attempted } => {
                write!(
                    f,
                    "Filter capacity of {} items exceeded. Attempted to insert {} items.",
                    capacity, attempted
                )
            }
            Self::UnsupportedOperation { operation, variant } => {
                write!(
                    f,
                    "Operation '{}' is not supported by {} Bloom filter variant.",
                    operation, variant
                )
            }
            Self::IncompatibleFilters { reason } => {
                write!(
                    f,
                    "Cannot perform operation on incompatible filters: {}.",
                    reason
                )
            }
            Self::InvalidHashCount { count, min, max } => {
                write!(
                    f,
                    "Invalid hash function count: {}. Must be in range [{}, {}].",
                    count, min, max
                )
            }
            Self::InvalidFilterSize { size } => {
                write!(
                    f,
                    "Invalid filter size: {} bits. Must be positive and within memory limits.",
                    size
                )
            }
            #[cfg(feature = "serde")]
            Self::SerializationError { message } => {
                write!(f, "Serialization error: {}.", message)
            }
            Self::InternalError { message } => {
                write!(
                    f,
                    "Internal error (this is a bug in BloomCraft): {}.",
                    message
                )
            }
            Self::CounterOverflow { max_value } => {
                write!(
                    f,
                    "Counter overflow: attempted to increment beyond maximum value {}",
                    max_value
                )
            }
            Self::CounterUnderflow { min_value } => {
                write!(
                    f,
                    "Counter underflow: attempted to decrement below minimum value {}",
                    min_value
                )
            }
            Self::IndexOutOfBounds { index, length } => {
                write!(
                    f,
                    "Index {} out of bounds for bit vector of length {}",
                    index, length
                )
            }
            Self::InvalidRange {
                start,
                end,
                length,
                reason,
            } => {
                write!(
                    f,
                    "Invalid range [{}..{}) for bit vector of length {}: {}",
                    start, end, length, reason
                )
            }
        }
    }
}

impl std::error::Error for BloomCraftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // No nested errors in our current implementation
        None
    }
}

impl BloomCraftError {
    /// Create an `InvalidParameters` error with a formatted message.
    ///
    /// This is a convenience constructor for building detailed error messages.
    ///
    /// # Examples
    /// ```
    /// use bloomcraft::BloomCraftError;
    ///
    /// let err = BloomCraftError::invalid_parameters(
    ///     format!("m={} and k={} would result in degenerate filter", 100, 50)
    /// );
    /// ```
    #[must_use]
    pub fn invalid_parameters(message: impl Into<String>) -> Self {
        Self::InvalidParameters {
            message: message.into(),
        }
    }

    /// Create a `FalsePositiveRateOutOfBounds` error.
    #[must_use]
    pub fn fp_rate_out_of_bounds(fp_rate: f64) -> Self {
        Self::FalsePositiveRateOutOfBounds { fp_rate }
    }

    /// Create an `InvalidItemCount` error.
    #[must_use]
    pub fn invalid_item_count(count: usize) -> Self {
        Self::InvalidItemCount { count }
    }

    /// Create a `CapacityExceeded` error.
    #[must_use]
    pub fn capacity_exceeded(capacity: usize, attempted: usize) -> Self {
        Self::CapacityExceeded { capacity, attempted }
    }

    /// Create an `UnsupportedOperation` error.
    #[must_use]
    pub fn unsupported_operation(operation: impl Into<String>, variant: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            variant: variant.into(),
        }
    }

    /// Create an `IncompatibleFilters` error.
    #[must_use]
    pub fn incompatible_filters(reason: impl Into<String>) -> Self {
        Self::IncompatibleFilters {
            reason: reason.into(),
        }
    }

    /// Create an `InvalidHashCount` error.
    #[must_use]
    pub fn invalid_hash_count(count: usize, min: usize, max: usize) -> Self {
        Self::InvalidHashCount { count, min, max }
    }

    /// Create an `InvalidFilterSize` error.
    #[must_use]
    pub fn invalid_filter_size(size: usize) -> Self {
        Self::InvalidFilterSize { size }
    }

    /// Create a `SerializationError`.
    #[cfg(feature = "serde")]
    #[must_use]
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    /// Create an `InternalError`.
    ///
    /// This should only be used for conditions that indicate bugs in BloomCraft.
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Create a `CounterOverflow` error.
    #[must_use]
    pub fn counter_overflow(max_value: u64) -> Self {
        Self::CounterOverflow { max_value }
    }

    /// Create a `CounterUnderflow` error.
    #[must_use]
    pub fn counter_underflow(min_value: u64) -> Self {
        Self::CounterUnderflow { min_value }
    }

      /// Create an `IndexOutOfBounds` error.
    #[must_use]
    pub fn index_out_of_bounds(index: usize, length: usize) -> Self {
        Self::IndexOutOfBounds { index, length }
    }

    /// Create an `InvalidRange` error.
    #[must_use]
    pub fn invalid_range(start: usize, end: usize, length: usize, reason: impl Into<String>) -> Self {
        Self::InvalidRange {
            start,
            end,
            length,
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_invalid_parameters() {
        let err = BloomCraftError::invalid_parameters("test message");
        let display = format!("{err}");
        assert!(display.contains("Invalid Bloom filter parameters"));
        assert!(display.contains("test message"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_fp_rate_out_of_bounds() {
        let err = BloomCraftError::fp_rate_out_of_bounds(1.5);
        let display = format!("{err}");
        assert!(display.contains("1.5"));
        assert!(display.contains("out of bounds"));
        assert!(display.contains("(0, 1)"));
    }

    #[test]
    fn test_error_display_invalid_item_count() {
        let err = BloomCraftError::invalid_item_count(0);
        let display = format!("{err}");
        assert!(display.contains("0"));
        assert!(display.contains("greater than 0"));
    }

    #[test]
    fn test_error_display_capacity_exceeded() {
        let err = BloomCraftError::capacity_exceeded(1000, 1500);
        let display = format!("{err}");
        assert!(display.contains("1000"));
        assert!(display.contains("1500"));
        assert!(display.contains("exceeded"));
    }

    #[test]
    fn test_error_display_unsupported_operation() {
        let err = BloomCraftError::unsupported_operation("remove", "Standard");
        let display = format!("{err}");
        assert!(display.contains("remove"));
        assert!(display.contains("Standard"));
        assert!(display.contains("not supported"));
    }

    #[test]
    fn test_error_display_incompatible_filters() {
        let err = BloomCraftError::incompatible_filters("different sizes");
        let display = format!("{err}");
        assert!(display.contains("incompatible"));
        assert!(display.contains("different sizes"));
    }

    #[test]
    fn test_error_display_invalid_hash_count() {
        let err = BloomCraftError::invalid_hash_count(0, 1, 32);
        let display = format!("{err}");
        assert!(display.contains("0"));
        assert!(display.contains("[1, 32]"));
    }

    #[test]
    fn test_error_display_invalid_filter_size() {
        let err = BloomCraftError::invalid_filter_size(0);
        let display = format!("{err}");
        assert!(display.contains("0 bits"));
        assert!(display.contains("positive"));
    }

    #[test]
    fn test_error_display_index_out_of_bounds() {
        let err = BloomCraftError::index_out_of_bounds(150, 100);
        let display = format!("{}", err);
        assert!(display.contains("150"));
        assert!(display.contains("100"));
        assert!(display.contains("out of bounds"));
    }

    #[test]
    fn test_error_display_invalid_range() {
        let err = BloomCraftError::invalid_range(50, 150, 100, "end exceeds length");
        let display = format!("{}", err);
        assert!(display.contains("[50..150)"));
        assert!(display.contains("100"));
        assert!(display.contains("end exceeds length"));
    }

    #[test]
    fn test_error_display_counter_overflow() {
        let err = BloomCraftError::counter_overflow(u64::MAX);
        let display = format!("{}", err);
        assert!(display.contains("overflow"));
        assert!(display.contains(&u64::MAX.to_string()));
    }

    #[test]
    fn test_error_display_counter_underflow() {
        let err = BloomCraftError::counter_underflow(0);
        let display = format!("{}", err);
        assert!(display.contains("underflow"));
        assert!(display.contains("0"));
    }

    #[test]
    fn test_error_display_internal_error() {
        let err = BloomCraftError::internal_error("impossible state reached");
        let display = format!("{err}");
        assert!(display.contains("Internal error"));
        assert!(display.contains("bug"));
        assert!(display.contains("impossible state reached"));
    }

    #[test]
    fn test_error_implements_std_error() {
        let _err: Box<dyn std::error::Error> = Box::new(BloomCraftError::invalid_parameters("test"));
    }

    #[test]
    fn test_error_clone() {
        let err1 = BloomCraftError::invalid_parameters("test");
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }
        
        assert_eq!(returns_result().unwrap(), 42);
    }

    #[test]
    fn test_convenience_constructors() {
        let _ = BloomCraftError::invalid_parameters("test");
        let _ = BloomCraftError::fp_rate_out_of_bounds(1.5);
        let _ = BloomCraftError::invalid_item_count(0);
        let _ = BloomCraftError::capacity_exceeded(100, 200);
        let _ = BloomCraftError::unsupported_operation("op", "variant");
        let _ = BloomCraftError::incompatible_filters("reason");
        let _ = BloomCraftError::invalid_hash_count(0, 1, 10);
        let _ = BloomCraftError::invalid_filter_size(0);
        let _ = BloomCraftError::index_out_of_bounds(100, 50);
        let _ = BloomCraftError::invalid_range(50, 150, 100, "end exceeds length");
        let _ = BloomCraftError::counter_overflow(u64::MAX);
        let _ = BloomCraftError::counter_underflow(0);
        let _ = BloomCraftError::internal_error("test");
    }

    #[test]
    fn test_error_propagation_with_question_mark() {
        fn inner() -> Result<()> {
            Err(BloomCraftError::invalid_item_count(0))
        }

        fn outer() -> Result<()> {
            inner()?;
            Ok(())
        }

        assert!(outer().is_err());
    }
}
