//! Error types for BloomCraft operations.
//!
//! This module defines [`BloomCraftError`], the single error type used throughout
//! the crate, and the [`Result`] type alias.
//!
//! # Variant taxonomy
//!
//! Variants fall into three categories:
//!
//! - **Construction errors** — returned from constructors and configuration methods.
//!   Handle these at startup, not in steady-state code:
//!   [`InvalidParameters`], [`FalsePositiveRateOutOfBounds`], [`InvalidItemCount`],
//!   [`InvalidHashCount`], [`InvalidFilterSize`], [`InvalidRange`].
//!
//! - **Runtime errors** — returned from operations on live filters; indicate
//!   structural limits the caller may handle gracefully:
//!   [`CapacityExceeded`], [`MaxFiltersExceeded`], [`UnsupportedOperation`],
//!   [`IncompatibleFilters`], [`CounterOverflow`], [`CounterUnderflow`],
//!   [`IndexOutOfBounds`], [`SerializationError`].
//!
//! - **Internal errors** — indicate bugs in BloomCraft, not caller mistakes.
//!   Log and abort if they occur in production: [`InternalError`].
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
//! # Pattern matching with `#[non_exhaustive]`
//!
//! `BloomCraftError` is `#[non_exhaustive]`. New variants may be added in minor
//! versions. Every `match` arm must include a `_` wildcard:
//!
//! ```rust
//! use bloomcraft::BloomCraftError;
//!
//! fn handle(e: BloomCraftError) {
//!     match e {
//!         BloomCraftError::MaxFiltersExceeded { max_filters, current_count } => {
//!             eprintln!("Filter full: {}/{} sub-filters", current_count, max_filters);
//!         }
//!         BloomCraftError::FalsePositiveRateOutOfBounds { fp_rate } => {
//!             eprintln!("Invalid FPR: {fp_rate:.6}");
//!         }
//!         _ => eprintln!("Other error: {e}"),
//!     }
//! }
//! ```
//!
//! # Error propagation
//!
//! ```rust
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

#![allow(clippy::module_name_repetitions)]

use std::fmt;

/// Result type alias for BloomCraft operations.
///
/// All fallible operations return [`Result`] where the error type is [`BloomCraftError`].
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
/// See the [module-level documentation](self) for the variant taxonomy and
/// pattern-matching guidance.
///
/// # Design notes
///
/// - `Clone` + `PartialEq` enable testing and error comparison without allocating.
/// - The enum is `#[non_exhaustive]`: new variants may be added in minor versions.
///   All `match` arms must include a `_` wildcard.
/// - `source()` always returns `None`. Source chains are not preserved; use
///   the `message` field on each variant for full diagnostic context.
/// - The crate does not depend on `thiserror` to keep the dependency footprint
///   minimal. `Display` is implemented by hand.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// # Examples of invalid values
    ///
    /// - ε = 0.0: Would require infinite memory.
    /// - ε = 1.0: Filter accepts everything (useless).
    /// - ε < 0.0: Negative probability (nonsensical).
    /// - ε > 1.0: Probability > 100% (nonsensical).
    FalsePositiveRateOutOfBounds {
        /// The invalid false positive rate that was provided.
        fp_rate: f64,
    },

    /// Expected items count is invalid.
    ///
    /// Occurs when n = 0, which would cause division by zero or `ln(0)`
    /// in parameter calculations.
    InvalidItemCount {
        /// The invalid count that was provided.
        count: usize,
    },

    /// Filter item capacity would be exceeded by an operation.
    ///
    /// Returned when a single-stage filter variant has a hard item-count limit
    /// and an insertion would exceed it. For the [`ScalableBloomFilter`] sub-filter
    /// count limit, see [`MaxFiltersExceeded`].
    ///
    /// [`ScalableBloomFilter`]: crate::filters::ScalableBloomFilter
    /// [`MaxFiltersExceeded`]: BloomCraftError::MaxFiltersExceeded
    CapacityExceeded {
        /// Maximum item capacity of the filter.
        capacity: usize,
        /// Number of items attempted to insert.
        attempted: usize,
    },

    /// The `ScalableBloomFilter` has reached the hard sub-filter limit.
    ///
    /// Once [`MAX_FILTERS`] (64) sub-filters exist, no new ones can be appended.
    /// Subsequent insertions land in the last sub-filter, degrading its FPR
    /// beyond the configured target.
    ///
    /// Configure [`CapacityExhaustedBehavior::Error`] to receive this error from
    /// [`insert_checked`](crate::filters::ScalableBloomFilter::insert_checked).
    /// The default [`Silent`](crate::filters::scalable::CapacityExhaustedBehavior::Silent)
    /// behaviour continues inserting with degraded FPR.
    ///
    /// [`MAX_FILTERS`]: crate::filters::scalable::MAX_FILTERS
    /// [`CapacityExhaustedBehavior::Error`]: crate::filters::scalable::CapacityExhaustedBehavior::Error
    MaxFiltersExceeded {
        /// The hard limit on sub-filter count (`MAX_FILTERS = 64`).
        max_filters: usize,
        /// Sub-filter count at the time of the attempted growth.
        current_count: usize,
    },

    /// Operation requires features not supported by this filter variant.
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
    /// Occurs during merge/union operations when filters have different sizes,
    /// hash functions, or other critical parameters.
    IncompatibleFilters {
        /// Description of the incompatibility.
        reason: String,
    },

    /// Hash function configuration is invalid.
    ///
    /// Occurs if the number of hash functions is 0 or exceeds practical limits.
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
    /// Source chains are not preserved; the `message` field contains the full
    /// diagnostic context available at the error site.
    SerializationError {
        /// Description of what failed.
        message: String,
    },

    /// Internal invariant violated.
    ///
    /// This should never occur in correct usage. If it does, it indicates a bug
    /// in BloomCraft itself. Log and report rather than attempting to recover.
    InternalError {
        /// Description of the invariant that was violated.
        message: String,
    },

    /// Counter overflow: attempted to increment beyond maximum value.
    CounterOverflow {
        /// Maximum value the counter can hold.
        max_value: u64,
    },

    /// Counter underflow: attempted to decrement below minimum value.
    CounterUnderflow {
        /// Minimum value (always 0 for unsigned counters).
        min_value: u64,
    },

    /// Attempted to access or modify a bit at an index >= the vector length.
    IndexOutOfBounds {
        /// The invalid index that was accessed.
        index: usize,
        /// The valid length of the bit vector.
        length: usize,
    },

    /// Invalid range in a range-based operation such as `set_range` or `get_range`.
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
                    "False positive rate {:.6} is out of bounds. Must be in range (0, 1).",
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
            Self::MaxFiltersExceeded { max_filters, current_count } => {
                write!(
                    f,
                    "ScalableBloomFilter reached the sub-filter limit of {} \
                     (current: {}). FPR will degrade on further inserts. \
                     See CapacityExhaustedBehavior for options.",
                    max_filters, current_count
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
                    "Counter overflow: attempted to increment beyond maximum value {}.",
                    max_value
                )
            }
            Self::CounterUnderflow { min_value } => {
                write!(
                    f,
                    "Counter underflow: attempted to decrement below minimum value {}.",
                    min_value
                )
            }
            Self::IndexOutOfBounds { index, length } => {
                write!(
                    f,
                    "Index {} out of bounds for bit vector of length {}.",
                    index, length
                )
            }
            Self::InvalidRange { start, end, length, reason } => {
                write!(
                    f,
                    "Invalid range [{}..{}) for bit vector of length {}: {}.",
                    start, end, length, reason
                )
            }
        }
    }
}

impl std::error::Error for BloomCraftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Source chains are not preserved. The message field on each variant
        // contains the full diagnostic context available at the error site.
        // If you need root-cause chaining, wrap BloomCraftError in anyhow::Error
        // or your application's own error type.
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
    /// Create an `InvalidParameters` error with a formatted message.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::BloomCraftError;
    ///
    /// let err = BloomCraftError::invalid_parameters(
    ///     format!("m={} and k={} would result in degenerate filter", 100, 50)
    /// );
    /// ```
    #[must_use]
    pub fn invalid_parameters(message: impl Into<String>) -> Self {
        Self::InvalidParameters { message: message.into() }
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
    ///
    /// Use this for single-stage filter item-count overflows. For `ScalableBloomFilter`
    /// sub-filter count exhaustion use [`max_filters_exceeded`](Self::max_filters_exceeded).
    #[must_use]
    pub fn capacity_exceeded(capacity: usize, attempted: usize) -> Self {
        Self::CapacityExceeded { capacity, attempted }
    }

    /// Create a `MaxFiltersExceeded` error.
    ///
    /// Used exclusively by [`ScalableBloomFilter`] when appending a new sub-filter
    /// would exceed [`MAX_FILTERS`]. The `max_filters` argument should always be
    /// `MAX_FILTERS = 64`.
    ///
    /// [`ScalableBloomFilter`]: crate::filters::ScalableBloomFilter
    /// [`MAX_FILTERS`]: crate::filters::scalable::MAX_FILTERS
    #[must_use]
    pub fn max_filters_exceeded(max_filters: usize, current_count: usize) -> Self {
        Self::MaxFiltersExceeded { max_filters, current_count }
    }

    /// Create an `UnsupportedOperation` error.
    #[must_use]
    pub fn unsupported_operation(
        operation: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            variant: variant.into(),
        }
    }

    /// Create an `IncompatibleFilters` error.
    #[must_use]
    pub fn incompatible_filters(reason: impl Into<String>) -> Self {
        Self::IncompatibleFilters { reason: reason.into() }
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
        Self::SerializationError { message: message.into() }
    }

    /// Create an `InternalError`.
    ///
    /// Only use this for conditions that indicate bugs in BloomCraft, not caller errors.
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError { message: message.into() }
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
    pub fn invalid_range(
        start: usize,
        end: usize,
        length: usize,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidRange { start, end, length, reason: reason.into() }
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
        // :.6 format — assert on the fixed-decimal form
        assert!(display.contains("1.500000"));
        assert!(display.contains("out of bounds"));
        assert!(display.contains("(0, 1)"));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_error_display_fp_rate_special_floats() {
        // NaN and Inf must not panic the Display impl.
        let _ = format!("{}", BloomCraftError::fp_rate_out_of_bounds(f64::NAN));
        let _ = format!("{}", BloomCraftError::fp_rate_out_of_bounds(f64::INFINITY));
        let _ = format!("{}", BloomCraftError::fp_rate_out_of_bounds(f64::NEG_INFINITY));
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
        // Must NOT say "items exceeded" — that's CapacityExceeded's message.
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
        let err: Box<dyn std::error::Error> =
            Box::new(BloomCraftError::invalid_parameters("test"));
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
            BloomCraftError::SerializationError { message: "test".to_string() },
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
        fn returns_result() -> Result<i32> { Ok(42) }
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
