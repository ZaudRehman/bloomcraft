//! Error types for BloomCraft operations.
//!
//! This module defines the single error type [`BloomCraftError`] and the
//! crate-wide [`Result`] alias.
//!
//! # Error taxonomy
//!
//! | Category      | Variants | Handling |
//! |---|---|---|
//! | **Construction** | [`InvalidParameters`], [`FalsePositiveRateOutOfBounds`], [`InvalidItemCount`], [`InvalidHashCount`], [`InvalidFilterSize`], [`InvalidRange`] | Handle during setup — reject invalid config at startup. |
//! | **Runtime** | [`CapacityExceeded`], [`MaxFiltersExceeded`], [`UnsupportedOperation`], [`IncompatibleFilters`], [`CounterOverflow`], [`CounterUnderflow`], [`IndexOutOfBounds`], [`SerializationError`] | Graceful recovery at call sites in steady-state code. |
//! | **Internal** | [`InternalError`] | Log and report; indicates a crate bug, not a caller mistake. |
//!
//! [`InvalidParameters`]: BloomCraftError::InvalidParameters
//! [`FalsePositiveRateOutOfBounds`]: BloomCraftError::FalsePositiveRateOutOfBounds
//! [`InvalidItemCount`]: BloomCraftError::InvalidItemCount
//! [`InvalidHashCount`]: BloomCraftError::InvalidHashCount
//! [`InvalidFilterSize`]: BloomCraftError::InvalidFilterSize
//! [`InvalidRange`]: BloomCraftError::InvalidRange
//! [`CapacityExceeded`]: BloomCraftError::CapacityExceeded
//! [`MaxFiltersExceeded`]: BloomCraftError::MaxFiltersExceeded
//! [`UnsupportedOperation`]: BloomCraftError::UnsupportedOperation
//! [`IncompatibleFilters`]: BloomCraftError::IncompatibleFilters
//! [`CounterOverflow`]: BloomCraftError::CounterOverflow
//! [`CounterUnderflow`]: BloomCraftError::CounterUnderflow
//! [`IndexOutOfBounds`]: BloomCraftError::IndexOutOfBounds
//! [`SerializationError`]: BloomCraftError::SerializationError
//! [`InternalError`]: BloomCraftError::InternalError
//!
//! # Forward compatibility
//!
//! `BloomCraftError` is `#[non_exhaustive]`. Match with a wildcard arm to stay
//! forward-compatible across minor releases:
//!
//! ```rust
//! use bloomcraft::BloomCraftError;
//!
//! # fn handle(e: BloomCraftError) {
//! match e {
//!     BloomCraftError::MaxFiltersExceeded { max_filters, current_count } => {
//!         eprintln!("sub-filter limit {max_filters} reached ({current_count})");
//!     }
//!     BloomCraftError::FalsePositiveRateOutOfBounds { fp_rate } => {
//!         eprintln!("invalid FPR: {fp_rate:.6}");
//!     }
//!     _ => eprintln!("{e}"),
//! }
//! # }
//! ```
//!
//! # `?` propagation
//!
//! ```rust
//! use bloomcraft::{Result, BloomCraftError};
//!
//! fn checked_new(n: usize, fp: f64) -> Result<()> {
//!     if n == 0 {
//!         return Err(BloomCraftError::invalid_item_count(n));
//!     }
//!     Ok(())
//! }
//! # assert!(checked_new(100, 0.01).is_ok());
//! # assert!(checked_new(0, 0.01).is_err());
//! ```
//!
//! # Design decisions
//!
//! - **`Clone` + `PartialEq`** — enables testing and error comparison without allocation.
//! - **No `thiserror`** — `Display` is hand-written to keep the dependency footprint small.
//! - **`source()` returns `None`** — error source chains are not preserved; the `message`
//!   field on each variant carries the full diagnostic context. Wrap in `anyhow::Error`
//!   or your own error type if you need root-cause chaining.

#![allow(clippy::module_name_repetitions)]

use std::fmt;

/// Result type alias for BloomCraft operations.
///
/// All fallible functions return [`Result<T>`] where the error variant is
/// [`BloomCraftError`].
///
/// ```rust
/// use bloomcraft::Result;
///
/// fn validate(n: usize) -> Result<()> {
///     if n == 0 {
///         return Err(bloomcraft::BloomCraftError::invalid_item_count(n));
///     }
///     Ok(())
/// }
/// # assert!(validate(1).is_ok());
/// ```
pub type Result<T> = std::result::Result<T, BloomCraftError>;

/// Errors that can occur during Bloom filter operations.
///
/// See the [module-level documentation](self) for the variant taxonomy and
/// forward-compatibility guidance.
///
/// `source()` always returns `None`. Wrapping this type in `anyhow::Error` or
/// your application's error type is the recommended path if you need causal
/// chains.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BloomCraftError {
    /// Invalid filter parameters.
    ///
    /// Returned when constructor arguments don't satisfy the mathematical
    /// constraints required for a functional filter.
    InvalidParameters {
        /// Human-readable description of the problem.
        message: String,
    },

    /// False-positive rate outside `(0, 1)`.
    ///
    /// Values at or outside the open interval `(0, 1)` are mathematically
    /// meaningless — `0` requires infinite memory, `1` accepts everything,
    /// and negatives or values > `1` are nonsensical.
    FalsePositiveRateOutOfBounds {
        /// The rejected false-positive rate.
        fp_rate: f64,
    },

    /// Expected item count is zero.
    ///
    /// A zero item count would cause division by zero or `ln(0)` in parameter
    /// calculations, neither of which is valid.
    InvalidItemCount {
        /// The rejected item count.
        count: usize,
    },

    /// Single-stage filter capacity exceeded.
    ///
    /// The filter has a hard item-count limit and the attempted insertion would
    /// exceed it. For the sub-filter count limit of [`ScalableBloomFilter`] see
    /// [`MaxFiltersExceeded`].
    ///
    /// [`ScalableBloomFilter`]: crate::filters::ScalableBloomFilter
    /// [`MaxFiltersExceeded`]: BloomCraftError::MaxFiltersExceeded
    CapacityExceeded {
        /// Maximum item capacity of the filter.
        capacity: usize,
        /// Number of items the caller attempted to insert.
        attempted: usize,
    },

    /// Sub-filter count limit reached for a [`ScalableBloomFilter`](crate::filters::ScalableBloomFilter).
    ///
    /// Once [`MAX_FILTERS`] (64) sub-filters exist no new ones can be appended.
    /// Further insertions land in the last sub-filter, degrading its FPR beyond
    /// the configured target.
    ///
    /// Configure [`CapacityExhaustedBehavior::Error`] on
    /// [`insert_checked`](crate::filters::ScalableBloomFilter::insert_checked)
    /// to receive this error. The default [`Silent`] behaviour continues
    /// inserting with degraded FPR.
    ///
    /// [`MAX_FILTERS`]: crate::filters::scalable::MAX_FILTERS
    /// [`CapacityExhaustedBehavior::Error`]: crate::filters::scalable::CapacityExhaustedBehavior::Error
    /// [`Silent`]: crate::filters::scalable::CapacityExhaustedBehavior::Silent
    MaxFiltersExceeded {
        /// The hard sub-filter limit (`MAX_FILTERS = 64`).
        max_filters: usize,
        /// Sub-filter count at the time of the attempted growth.
        current_count: usize,
    },

    /// Operation not supported by this filter variant.
    ///
    /// For example, removing items from a standard Bloom filter, which doesn't
    /// track per-item counters.
    UnsupportedOperation {
        /// Name of the attempted operation.
        operation: String,
        /// Name of the filter variant.
        variant: String,
    },

    /// Two filters have incompatible parameters.
    ///
    /// Occurs during merge or union when filters differ in size, hash
    /// configuration, or other critical attributes.
    IncompatibleFilters {
        /// Description of the incompatibility.
        reason: String,
    },

    /// Number of hash functions is out of the allowed range.
    ///
    /// A hash count of zero produces no bit indices; very large values waste
    /// computation and degrade fill rate.
    InvalidHashCount {
        /// The rejected hash count.
        count: usize,
        /// Minimum allowed value.
        min: usize,
        /// Maximum allowed value.
        max: usize,
    },

    /// Bit vector size is invalid.
    ///
    /// The size must be positive and within system memory limits.
    InvalidFilterSize {
        /// The rejected size in bits.
        size: usize,
    },

    /// Serialization or deserialization failed.
    SerializationError {
        /// Description of what failed.
        message: String,
    },

    /// An internal invariant was violated.
    ///
    /// This should never occur in normal use. If it does, it indicates a bug in
    /// BloomCraft. Log the message and file a report — do not attempt to recover.
    InternalError {
        /// Description of the invariant that was violated.
        message: String,
    },

    /// Counter overflow: attempted to increment past the maximum value.
    CounterOverflow {
        /// Maximum value the counter can hold.
        max_value: u64,
    },

    /// Counter underflow: attempted to decrement below zero.
    CounterUnderflow {
        /// Minimum value (always `0` for unsigned counters).
        min_value: u64,
    },

    /// Bit index is at or beyond the vector length.
    IndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// Length of the bit vector.
        length: usize,
    },

    /// Invalid range in a range-based operation (`set_range`, `get_range`, …).
    InvalidRange {
        /// Start index (inclusive).
        start: usize,
        /// End index (exclusive).
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
                write!(f, "Invalid Bloom filter parameters: {message}.")
            }
            Self::FalsePositiveRateOutOfBounds { fp_rate } => {
                write!(
                    f,
                    "False positive rate {fp_rate:.6} is out of bounds. Must be in (0, 1).",
                )
            }
            Self::InvalidItemCount { count } => {
                write!(f, "Invalid item count: {count}. Must be greater than 0.")
            }
            Self::CapacityExceeded {
                capacity,
                attempted,
            } => {
                write!(
                    f,
                    "Filter capacity of {capacity} items exceeded. Attempted to insert {attempted} items.",
                )
            }
            Self::MaxFiltersExceeded {
                max_filters,
                current_count,
            } => {
                write!(
                    f,
                    "ScalableBloomFilter reached the sub-filter limit of {max_filters} \
                     (current: {current_count}). FPR will degrade on further inserts. \
                     See CapacityExhaustedBehavior for options.",
                )
            }
            Self::UnsupportedOperation { operation, variant } => {
                write!(
                    f,
                    "Operation '{operation}' is not supported by {variant} Bloom filter variant.",
                )
            }
            Self::IncompatibleFilters { reason } => {
                write!(
                    f,
                    "Cannot perform operation on incompatible filters: {reason}."
                )
            }
            Self::InvalidHashCount { count, min, max } => {
                write!(
                    f,
                    "Invalid hash function count: {count}. Must be in [{min}, {max}].",
                )
            }
            Self::InvalidFilterSize { size } => {
                write!(
                    f,
                    "Invalid filter size: {size} bits. Must be positive and within memory limits.",
                )
            }
            Self::SerializationError { message } => {
                write!(f, "Serialization error: {message}.")
            }
            Self::InternalError { message } => {
                write!(
                    f,
                    "Internal error (this is a bug in BloomCraft): {message}."
                )
            }
            Self::CounterOverflow { max_value } => {
                write!(
                    f,
                    "Counter overflow: attempted to increment beyond maximum value {max_value}.",
                )
            }
            Self::CounterUnderflow { min_value } => {
                write!(
                    f,
                    "Counter underflow: attempted to decrement below minimum value {min_value}.",
                )
            }
            Self::IndexOutOfBounds { index, length } => {
                write!(
                    f,
                    "Index {index} out of bounds for bit vector of length {length}."
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
                    "Invalid range [{start}..{end}) for bit vector of length {length}: {reason}.",
                )
            }
        }
    }
}

impl std::error::Error for BloomCraftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Source chains are not preserved. Wrap in anyhow::Error or your own
        // error type if you need causal chains.
        None
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for BloomCraftError {
    fn from(e: serde_json::Error) -> Self {
        BloomCraftError::serialization_error(e.to_string())
    }
}

impl BloomCraftError {
    /// Create an [`InvalidParameters`](Self::InvalidParameters) error.
    #[must_use]
    pub fn invalid_parameters(message: impl Into<String>) -> Self {
        Self::InvalidParameters {
            message: message.into(),
        }
    }

    /// Create a [`FalsePositiveRateOutOfBounds`](Self::FalsePositiveRateOutOfBounds) error.
    #[must_use]
    pub fn fp_rate_out_of_bounds(fp_rate: f64) -> Self {
        Self::FalsePositiveRateOutOfBounds { fp_rate }
    }

    /// Create an [`InvalidItemCount`](Self::InvalidItemCount) error.
    #[must_use]
    pub fn invalid_item_count(count: usize) -> Self {
        Self::InvalidItemCount { count }
    }

    /// Create a [`CapacityExceeded`](Self::CapacityExceeded) error.
    ///
    /// For the sub-filter limit of [`ScalableBloomFilter`](crate::filters::ScalableBloomFilter) use
    /// [`max_filters_exceeded`](Self::max_filters_exceeded).
    #[must_use]
    pub fn capacity_exceeded(capacity: usize, attempted: usize) -> Self {
        Self::CapacityExceeded {
            capacity,
            attempted,
        }
    }

    /// Create a [`MaxFiltersExceeded`](Self::MaxFiltersExceeded) error.
    ///
    /// Used by [`ScalableBloomFilter`](crate::filters::ScalableBloomFilter) when appending a new sub-filter would
    /// exceed [`MAX_FILTERS`](crate::filters::scalable::MAX_FILTERS) (64).
    #[must_use]
    pub fn max_filters_exceeded(max_filters: usize, current_count: usize) -> Self {
        Self::MaxFiltersExceeded {
            max_filters,
            current_count,
        }
    }

    /// Create an [`UnsupportedOperation`](Self::UnsupportedOperation) error.
    #[must_use]
    pub fn unsupported_operation(operation: impl Into<String>, variant: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            variant: variant.into(),
        }
    }

    /// Create an [`IncompatibleFilters`](Self::IncompatibleFilters) error.
    #[must_use]
    pub fn incompatible_filters(reason: impl Into<String>) -> Self {
        Self::IncompatibleFilters {
            reason: reason.into(),
        }
    }

    /// Create an [`InvalidHashCount`](Self::InvalidHashCount) error.
    #[must_use]
    pub fn invalid_hash_count(count: usize, min: usize, max: usize) -> Self {
        Self::InvalidHashCount { count, min, max }
    }

    /// Create an [`InvalidFilterSize`](Self::InvalidFilterSize) error.
    #[must_use]
    pub fn invalid_filter_size(size: usize) -> Self {
        Self::InvalidFilterSize { size }
    }

    /// Create a [`SerializationError`](Self::SerializationError).
    #[cfg(feature = "serde")]
    #[must_use]
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    /// Create an [`InternalError`](Self::InternalError).
    ///
    /// Only use this for conditions that indicate a bug in BloomCraft, not
    /// caller errors.
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Create a [`CounterOverflow`](Self::CounterOverflow) error.
    #[must_use]
    pub fn counter_overflow(max_value: u64) -> Self {
        Self::CounterOverflow { max_value }
    }

    /// Create a [`CounterUnderflow`](Self::CounterUnderflow) error.
    #[must_use]
    pub fn counter_underflow(min_value: u64) -> Self {
        Self::CounterUnderflow { min_value }
    }

    /// Create an [`IndexOutOfBounds`](Self::IndexOutOfBounds) error.
    #[must_use]
    pub fn index_out_of_bounds(index: usize, length: usize) -> Self {
        Self::IndexOutOfBounds { index, length }
    }

    /// Create an [`InvalidRange`](Self::InvalidRange) error.
    #[must_use]
    pub fn invalid_range(
        start: usize,
        end: usize,
        length: usize,
        reason: impl Into<String>,
    ) -> Self {
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
        assert!(display.contains("1.500000"));
        assert!(display.contains("out of bounds"));
        assert!(display.contains("(0, 1)"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_fp_rate_special_floats() {
        let _ = format!("{}", BloomCraftError::fp_rate_out_of_bounds(f64::NAN));
        let _ = format!("{}", BloomCraftError::fp_rate_out_of_bounds(f64::INFINITY));
        let _ = format!(
            "{}",
            BloomCraftError::fp_rate_out_of_bounds(f64::NEG_INFINITY)
        );
    }

    #[test]
    fn test_error_display_invalid_item_count() {
        let err = BloomCraftError::invalid_item_count(0);
        let display = format!("{err}");
        assert!(display.contains("0"));
        assert!(display.contains("greater than 0"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_capacity_exceeded() {
        let err = BloomCraftError::capacity_exceeded(1000, 1500);
        let display = format!("{err}");
        assert!(display.contains("1000"));
        assert!(display.contains("1500"));
        assert!(display.contains("exceeded"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_max_filters_exceeded() {
        let err = BloomCraftError::max_filters_exceeded(64, 64);
        let display = format!("{err}");
        assert!(display.contains("64"));
        assert!(display.contains("sub-filter"));
        assert!(display.contains("FPR"));
        assert!(!display.contains("items exceeded"));
    }

    #[test]
    fn test_error_display_unsupported_operation() {
        let err = BloomCraftError::unsupported_operation("remove", "Standard");
        let display = format!("{err}");
        assert!(display.contains("remove"));
        assert!(display.contains("Standard"));
        assert!(display.contains("not supported"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_incompatible_filters() {
        let err = BloomCraftError::incompatible_filters("different sizes");
        let display = format!("{err}");
        assert!(display.contains("incompatible"));
        assert!(display.contains("different sizes"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_invalid_hash_count() {
        let err = BloomCraftError::invalid_hash_count(0, 1, 32);
        let display = format!("{err}");
        assert!(display.contains("0"));
        assert!(display.contains("[1, 32]"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_invalid_filter_size() {
        let err = BloomCraftError::invalid_filter_size(0);
        let display = format!("{err}");
        assert!(display.contains("0 bits"));
        assert!(display.contains("positive"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_index_out_of_bounds() {
        let err = BloomCraftError::index_out_of_bounds(150, 100);
        let display = format!("{err}");
        assert!(display.contains("150"));
        assert!(display.contains("100"));
        assert!(display.contains("out of bounds"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_invalid_range() {
        let err = BloomCraftError::invalid_range(50, 150, 100, "end exceeds length");
        let display = format!("{err}");
        assert!(display.contains("[50..150)"));
        assert!(display.contains("100"));
        assert!(display.contains("end exceeds length"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_counter_overflow() {
        let err = BloomCraftError::counter_overflow(u64::MAX);
        let display = format!("{err}");
        assert!(display.contains("overflow"));
        assert!(display.contains(&u64::MAX.to_string()));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_counter_underflow() {
        let err = BloomCraftError::counter_underflow(0);
        let display = format!("{err}");
        assert!(display.contains("underflow"));
        assert!(display.contains("0"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_internal_error() {
        let err = BloomCraftError::internal_error("impossible state reached");
        let display = format!("{err}");
        assert!(display.contains("Internal error"));
        assert!(display.contains("bug"));
        assert!(display.contains("impossible state reached"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_serialization_error() {
        let err = BloomCraftError::SerializationError {
            message: "unexpected EOF".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("Serialization error"));
        assert!(display.contains("unexpected EOF"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_implements_std_error() {
        let _err: Box<dyn std::error::Error> =
            Box::new(BloomCraftError::invalid_parameters("test"));
    }

    #[test]
    fn test_error_downcast_via_box() {
        let err: Box<dyn std::error::Error> = Box::new(BloomCraftError::invalid_parameters("test"));
        assert!(err.downcast_ref::<BloomCraftError>().is_some());
    }

    #[test]
    fn test_error_source_is_none() {
        use std::error::Error;
        let variants: &[BloomCraftError] = &[
            BloomCraftError::invalid_parameters("x"),
            BloomCraftError::fp_rate_out_of_bounds(1.5),
            BloomCraftError::invalid_item_count(0),
            BloomCraftError::capacity_exceeded(64, 65),
            BloomCraftError::max_filters_exceeded(64, 64),
            BloomCraftError::unsupported_operation("op", "variant"),
            BloomCraftError::incompatible_filters("reason"),
            BloomCraftError::invalid_hash_count(0, 1, 10),
            BloomCraftError::invalid_filter_size(0),
            BloomCraftError::SerializationError {
                message: "test".to_string(),
            },
            BloomCraftError::internal_error("x"),
            BloomCraftError::counter_overflow(u64::MAX),
            BloomCraftError::counter_underflow(0),
            BloomCraftError::index_out_of_bounds(100, 50),
            BloomCraftError::invalid_range(50, 150, 100, "end exceeds length"),
        ];
        for e in variants {
            assert!(
                e.source().is_none(),
                "{e:?} unexpectedly has a source — update source() or add a test exemption"
            );
        }
    }

    #[test]
    fn test_error_clone_and_eq() {
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
    fn test_convenience_constructors_compile() {
        let _ = BloomCraftError::invalid_parameters("test");
        let _ = BloomCraftError::fp_rate_out_of_bounds(1.5);
        let _ = BloomCraftError::invalid_item_count(0);
        let _ = BloomCraftError::capacity_exceeded(100, 200);
        let _ = BloomCraftError::max_filters_exceeded(64, 64);
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

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization_error_constructor() {
        let err = BloomCraftError::serialization_error("bincode: unexpected end of input");
        let display = format!("{err}");
        assert!(display.contains("Serialization error"));
        assert!(display.contains("bincode"));
        assert!(display.ends_with('.'));
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

    #[test]
    fn test_non_exhaustive_wildcard_compiles() {
        let err = BloomCraftError::internal_error("test");
        let _ = match err {
            BloomCraftError::InternalError { .. } => "internal",
            _ => "other",
        };
    }
}
