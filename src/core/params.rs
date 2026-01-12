//! Optimal parameter calculation for Bloom filters.
//!
//! This module implements the mathematical formulas from Burton Bloom's 1970 paper
//! and modern Bloom filter analysis for calculating optimal filter parameters.
//!
//! # Mathematical Background
//!
//! Given:
//! - `n`: Expected number of elements
//! - `ε`: Target false positive rate
//!
//! Optimal parameters:
//! - `m = -n × ln(ε) / (ln 2)²` (bits in filter)
//! - `k = (m/n) × ln 2` (number of hash functions)
//!
//! Expected false positive rate:
//! - `p = (1 - e^(-kn/m))^k`
//!
//! # Performance Considerations
//!
//! Hash function count affects both speed and accuracy:
//! - **Fewer hashes (k=3-5)**: Faster inserts/queries, higher FP rate
//! - **More hashes (k=10-15)**: Slower operations, lower FP rate
//! - **Sweet spot (k=7-10)**: Balanced performance for most use cases
//!
//! Memory vs. speed trade-offs:
//! - Larger filters reduce FP rate but increase memory footprint
//! - Thread-local filters can improve throughput in concurrent scenarios
//!
//! # References
//!
//! - Bloom, Burton H. (1970). "Space/Time Trade-offs in Hash Coding with Allowable Errors"
//! - Kirsch & Mitzenmacher (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter"

#![allow(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{BloomCraftError, Result};
use std::f64::consts::LN_2;

/// Mathematical constant: (ln 2)² ≈ 0.4804530139182014
///
/// Used in optimal bit count calculation to avoid repeated computation.
const LN2_SQUARED: f64 = LN_2 * LN_2;

/// Minimum practical filter size in bits.
///
/// Filters smaller than 8 bits (1 byte) offer negligible utility.
pub const MIN_FILTER_SIZE: usize = 8;

/// Maximum practical number of hash functions.
///
/// Beyond 32 hash functions, the computational cost typically exceeds
/// the marginal improvement in false positive rate. This limit is based on:
/// - Diminishing returns: Each additional hash provides less benefit
/// - Cache efficiency: 32 hashes fit well within typical cache line sizes
pub const MAX_HASH_FUNCTIONS: usize = 32;

/// Minimum number of hash functions.
///
/// At least one hash function is required for any Bloom filter.
pub const MIN_HASH_FUNCTIONS: usize = 1;

/// Maximum reasonable load factor (n/m ratio).
///
/// Beyond this threshold, the filter becomes oversaturated and false
/// positive rates become unacceptably high (approaching 50%+).
const MAX_LOAD_FACTOR: f64 = 2.0;

/// Calculate optimal number of bits for given constraints.
///
/// Implements the formula: `m = -n × ln(ε) / (ln 2)²`
///
/// This formula minimizes the false positive rate for a given number of
/// elements and target false positive probability.
///
/// # Arguments
///
/// * `n` - Expected number of elements to insert (must be > 0)
/// * `fp_rate` - Target false positive rate (must be in range (0, 1))
///
/// # Returns
///
/// * `Ok(usize)` - Optimal number of bits, at least [`MIN_FILTER_SIZE`]
/// * `Err(BloomCraftError)` - If parameters are invalid
///
/// # Errors
///
/// - [`BloomCraftError::InvalidItemCount`] if `n == 0`
/// - [`BloomCraftError::FalsePositiveRateOutOfBounds`] if `fp_rate` not in (0, 1)
/// - [`BloomCraftError::InvalidParameters`] if result exceeds system limits
///
/// # Examples
///
/// ```
/// use bloomcraft::core::params::optimal_bit_count;
///
/// // For 1000 items with 1% false positive rate
/// let bits = optimal_bit_count(1000, 0.01).unwrap();
/// assert!(bits >= 9585 && bits <= 9586); // ≈9585 bits (1.2 KB)
///
/// // For 100K items with 0.1% false positive rate
/// let bits = optimal_bit_count(100_000, 0.001).unwrap();
/// assert!(bits >= 1437758 && bits <= 1437759); // ≈1.44M bits (180 KB)
/// ```
///
/// # Performance
///
/// Time: O(1) - constant time with logarithm and arithmetic operations
pub fn optimal_bit_count(n: usize, fp_rate: f64) -> Result<usize> {
    if n == 0 {
        return Err(BloomCraftError::invalid_item_count(n));
    }

    if fp_rate <= 0.0 || fp_rate >= 1.0 {
        return Err(BloomCraftError::fp_rate_out_of_bounds(fp_rate));
    }

    // Edge case: fp_rate very close to 1 would yield a degenerate filter
    if fp_rate > 0.9999 {
        return Ok(MIN_FILTER_SIZE);
    }

    // Calculate optimal m using formula: m = -n × ln(ε) / (ln 2)²
    let n_f64 = n as f64;
    let numerator = -n_f64 * fp_rate.ln();
    let m = numerator / LN2_SQUARED;

    // Check for overflow before casting to usize
    if m > usize::MAX as f64 {
        return Err(BloomCraftError::invalid_parameters(format!(
            "Calculated filter size {:.0} exceeds system limits (usize::MAX = {})",
            m,
            usize::MAX
        )));
    }

    // Round up to ensure we meet (or exceed) the target FP rate
    let m_ceil = m.ceil();

    // Enforce minimum size
    let m_final = m_ceil.max(MIN_FILTER_SIZE as f64) as usize;

    // Sanity check: prevent unreasonably large allocations
    if m_final > usize::MAX / 2 {
        return Err(BloomCraftError::invalid_parameters(format!(
            "Calculated filter size {} exceeds reasonable bounds. \
             Consider increasing false positive rate or reducing item count.",
            m_final
        )));
    }

    Ok(m_final)
}

/// Calculate optimal number of hash functions.
///
/// Implements the formula: `k = (m/n) × ln 2`
///
/// This formula minimizes the false positive rate for a given filter size
/// and number of elements.
///
/// # Arguments
///
/// * `m` - Filter size in bits (must be > 0)
/// * `n` - Expected number of elements (must be > 0)
///
/// # Returns
///
/// Optimal number of hash functions, clamped to [[`MIN_HASH_FUNCTIONS`], [`MAX_HASH_FUNCTIONS`]]
///
/// # Errors
///
/// - [`BloomCraftError::InvalidFilterSize`] if `m == 0`
/// - [`BloomCraftError::InvalidItemCount`] if `n == 0`
///
/// # Examples
///
/// ```
/// use bloomcraft::core::params::optimal_hash_count;
///
/// // For 9585 bits and 1000 items
/// let k = optimal_hash_count(9585, 1000).unwrap();
/// assert_eq!(k, 7);
///
/// // For 1M bits and 100K items
/// let k = optimal_hash_count(1_000_000, 100_000).unwrap();
/// assert_eq!(k, 7);
/// ```
pub fn optimal_hash_count(m: usize, n: usize) -> Result<usize> {
    if m == 0 {
        return Err(BloomCraftError::invalid_filter_size(m));
    }

    if n == 0 {
        return Err(BloomCraftError::invalid_item_count(n));
    }

    // Calculate optimal k using formula: k = (m/n) × ln 2
    let m_f64 = m as f64;
    let n_f64 = n as f64;
    let k = (m_f64 / n_f64) * LN_2;

    // Round to nearest integer
    let k_rounded = k.round() as usize;

    // Clamp to practical bounds [1, 32]
    let k_final = k_rounded.clamp(MIN_HASH_FUNCTIONS, MAX_HASH_FUNCTIONS);

    Ok(k_final)
}

/// Calculate expected false positive rate for given parameters.
///
/// Implements the formula: `p = (1 - e^(-kn/m))^k`
///
/// This calculates the theoretical false positive probability after inserting
/// `n` elements into a filter of size `m` bits using `k` hash functions.
///
/// # Arguments
///
/// * `m` - Filter size in bits
/// * `n` - Number of elements inserted
/// * `k` - Number of hash functions
///
/// # Returns
///
/// Expected false positive probability in range [0.0, 1.0]
///
/// # Errors
///
/// - [`BloomCraftError::InvalidFilterSize`] if `m == 0`
/// - [`BloomCraftError::InvalidHashCount`] if `k` is outside valid bounds
///
/// # Examples
///
/// ```
/// use bloomcraft::core::params::expected_fp_rate;
///
/// // Calculate FP rate for optimal parameters
/// let m = 9585;
/// let n = 1000;
/// let k = 7;
/// let fp = expected_fp_rate(m, n, k).unwrap();
/// assert!((fp - 0.01).abs() < 0.001); // ≈1%
/// ```
///
/// # Mathematical Note
///
/// The formula assumes:
/// 1. Hash functions are uniformly distributed and independent
/// 2. Filter contains exactly `n` elements
/// 3. Query is for an element **not** in the set
///
/// In practice, enhanced double hashing provides near-perfect independence,
/// making this estimate highly accurate (typically within 5-10%).
pub fn expected_fp_rate(m: usize, n: usize, k: usize) -> Result<f64> {
    if m == 0 {
        return Err(BloomCraftError::invalid_filter_size(m));
    }

    if !(MIN_HASH_FUNCTIONS..=MAX_HASH_FUNCTIONS).contains(&k) {
        return Err(BloomCraftError::invalid_hash_count(
            k,
            MIN_HASH_FUNCTIONS,
            MAX_HASH_FUNCTIONS,
        ));
    }

    // Edge case: empty filter has zero false positive rate
    if n == 0 {
        return Ok(0.0);
    }

    // Calculate using formula: (1 - e^(-kn/m))^k
    let m_f64 = m as f64;
    let n_f64 = n as f64;
    let k_f64 = k as f64;

    let exponent = -(k_f64 * n_f64) / m_f64;
    let prob_bit_zero = exponent.exp();
    let prob_bit_one = 1.0 - prob_bit_zero;

    // Probability all k bits are 1 → false positive
    let fp_rate = prob_bit_one.powf(k_f64);

    // Clamp to [0, 1] to handle floating-point rounding
    Ok(fp_rate.clamp(0.0, 1.0))
}

/// Calculate optimal filter parameters for given constraints.
///
/// Convenience function that combines [`optimal_bit_count`] and
/// [`optimal_hash_count`] to compute both parameters at once.
///
/// # Arguments
///
/// * `n` - Expected number of elements
/// * `fp_rate` - Target false positive rate
///
/// # Returns
///
/// Tuple of `(optimal_bits, optimal_hash_count)`
///
/// # Errors
///
/// Returns error if parameters are invalid (see [`optimal_bit_count`]).
///
/// # Examples
///
/// ```
/// use bloomcraft::core::params::calculate_filter_params;
///
/// let (m, k) = calculate_filter_params(1000, 0.01).unwrap();
/// assert!(m >= 9585 && m <= 9586); // ~9585 bits
/// assert_eq!(k, 7);
/// ```
pub fn calculate_filter_params(n: usize, fp_rate: f64) -> Result<(usize, usize)> {
    let m = optimal_bit_count(n, fp_rate)?;
    let k = optimal_hash_count(m, n)?;
    Ok((m, k))
}

/// Validate that filter parameters are internally consistent.
///
/// Checks that parameters satisfy basic constraints for a functional Bloom filter.
///
/// # Validation Rules
///
/// 1. `m > 0` - Filter must have at least one bit
/// 2. `n > 0` - Expected items must be positive
/// 3. `1 ≤ k ≤ 32` - Hash count within practical bounds
/// 4. `m ≥ k` - Filter size must accommodate hash function count
/// 5. `n/m ≤ 2.0` - Load factor must be reasonable
///
/// # Arguments
///
/// * `m` - Filter size in bits
/// * `n` - Expected number of elements
/// * `k` - Number of hash functions
///
/// # Returns
///
/// * `Ok(())` - Parameters are valid
/// * `Err(BloomCraftError)` - Parameters violate constraints
///
/// # Examples
///
/// ```
/// use bloomcraft::core::params::validate_params;
///
/// // Valid parameters
/// assert!(validate_params(1000, 100, 7).is_ok());
///
/// // Invalid: zero bits
/// assert!(validate_params(0, 100, 7).is_err());
///
/// // Invalid: oversaturated (n/m > 2.0)
/// assert!(validate_params(100, 250, 7).is_err());
/// ```
pub fn validate_params(m: usize, n: usize, k: usize) -> Result<()> {
    if m == 0 {
        return Err(BloomCraftError::invalid_filter_size(m));
    }

    if n == 0 {
        return Err(BloomCraftError::invalid_item_count(n));
    }

    if !(MIN_HASH_FUNCTIONS..=MAX_HASH_FUNCTIONS).contains(&k) {
        return Err(BloomCraftError::invalid_hash_count(
            k,
            MIN_HASH_FUNCTIONS,
            MAX_HASH_FUNCTIONS,
        ));
    }

    // Check for degenerate case: more hash functions than bits
    if m < k {
        return Err(BloomCraftError::invalid_parameters(format!(
            "Filter size ({} bits) must be at least as large as hash count ({}).",
            m, k
        )));
    }

    // Check for oversaturation: beyond this threshold, FP rate explodes
    let load_factor = n as f64 / m as f64;
    if load_factor > MAX_LOAD_FACTOR {
        return Err(BloomCraftError::invalid_parameters(format!(
            "Load factor {:.2} exceeds maximum {:.1}. Filter would have unacceptably high \
             false positive rate (>50%). Increase filter size or reduce item count.",
            load_factor, MAX_LOAD_FACTOR
        )));
    }

    Ok(())
}

/// Calculate bits per element for a given false positive rate.
///
/// Returns the space efficiency metric: how many bits are needed per element
/// to achieve the target false positive rate with optimal parameters.
///
/// Formula: `bits_per_element = -ln(ε) / (ln 2)²`
///
/// # Arguments
///
/// * `fp_rate` - Target false positive rate (must be in (0, 1))
///
/// # Returns
///
/// Bits required per element (as f64)
///
/// # Errors
///
/// Returns error if `fp_rate` is not in range (0, 1).
///
/// # Examples
///
/// ```
/// use bloomcraft::core::params::bits_per_element;
///
/// // 1% FP rate requires ~9.6 bits/element
/// let bpe = bits_per_element(0.01).unwrap();
/// assert!((bpe - 9.6).abs() < 0.1);
///
/// // 0.1% FP rate requires ~14.4 bits/element
/// let bpe = bits_per_element(0.001).unwrap();
/// assert!((bpe - 14.4).abs() < 0.1);
/// ```
pub fn bits_per_element(fp_rate: f64) -> Result<f64> {
    if fp_rate <= 0.0 || fp_rate >= 1.0 {
        return Err(BloomCraftError::fp_rate_out_of_bounds(fp_rate));
    }

    let bpe = -fp_rate.ln() / LN2_SQUARED;
    Ok(bpe)
}

/// Alias for [`optimal_hash_count`] - calculates optimal k (number of hash functions).
///
/// This is a convenience alias using the traditional variable name from Bloom filter literature.
///
/// # Arguments
///
/// * `n` - Expected number of elements
/// * `m` - Filter size in bits
///
/// # Returns
///
/// Optimal number of hash functions
///
/// # Errors
///
/// Returns error if parameters are invalid.
#[inline]
pub fn optimal_k(n: usize, m: usize) -> usize {
    optimal_hash_count(m, n).unwrap_or(7)
}

/// Alias for [`optimal_bit_count`] - calculates optimal m (filter size in bits).
///
/// This is a convenience alias using the traditional variable name from Bloom filter literature.
///
/// # Arguments
///
/// * `n` - Expected number of elements
/// * `fp_rate` - Target false positive rate
///
/// # Returns
///
/// Optimal filter size in bits
///
/// # Panics
///
/// Panics if parameters are invalid.
#[inline]
pub fn optimal_m(n: usize, fp_rate: f64) -> usize {
    optimal_bit_count(n, fp_rate).expect("Invalid parameters for optimal_m")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Theoretical values from Bloom filter formulas
    const EXPECTED_BITS_1000_1PCT: usize = 9585; // -1000 × ln(0.01) / (ln2)²
    const EXPECTED_BITS_1000_0_1PCT: usize = 14377; // -1000 × ln(0.001) / (ln2)²
    const EXPECTED_HASH_9585_1000: usize = 7; // (9585/1000) × ln2 ≈ 6.6 → 7

    #[test]
    fn test_ln2_squared_constant() {
        // Verify the constant matches expected value
        let expected = 0.480_453_013_918_201_4;
        assert!(
            (LN2_SQUARED - expected).abs() < 1e-10,
            "LN2_SQUARED constant incorrect: expected {}, got {}",
            expected,
            LN2_SQUARED
        );
    }

    #[test]
    fn test_optimal_bit_count_1_percent() {
        let m = optimal_bit_count(1000, 0.01).unwrap();
        assert!(
            m >= EXPECTED_BITS_1000_1PCT && m <= EXPECTED_BITS_1000_1PCT + 1,
            "Expected ~{}, got {}",
            EXPECTED_BITS_1000_1PCT,
            m
        );
    }

    #[test]
    fn test_optimal_bit_count_0_1_percent() {
        let m = optimal_bit_count(1000, 0.001).unwrap();
        assert!(
            m >= EXPECTED_BITS_1000_0_1PCT && m <= EXPECTED_BITS_1000_0_1PCT + 1,
            "Expected ~{}, got {}",
            EXPECTED_BITS_1000_0_1PCT,
            m
        );
    }

    #[test]
    fn test_optimal_bit_count_large_n() {
        let m = optimal_bit_count(1_000_000, 0.01).unwrap();
        // Should scale linearly with n
        assert!(m >= 9_585_000 && m <= 9_586_000);
    }

    #[test]
    fn test_optimal_bit_count_zero_items_error() {
        let result = optimal_bit_count(0, 0.01);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BloomCraftError::InvalidItemCount { count: 0 }
        ));
    }

    #[test]
    fn test_optimal_bit_count_invalid_fp_rate_zero() {
        let result = optimal_bit_count(1000, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BloomCraftError::FalsePositiveRateOutOfBounds { fp_rate } if fp_rate == 0.0
        ));
    }

    #[test]
    fn test_optimal_bit_count_invalid_fp_rate_one() {
        let result = optimal_bit_count(1000, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimal_bit_count_invalid_fp_rate_negative() {
        let result = optimal_bit_count(1000, -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimal_bit_count_invalid_fp_rate_greater_than_one() {
        let result = optimal_bit_count(1000, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimal_hash_count_standard() {
        let k = optimal_hash_count(9585, 1000).unwrap();
        assert_eq!(
            k, EXPECTED_HASH_9585_1000,
            "Expected {} hash functions",
            EXPECTED_HASH_9585_1000
        );
    }

    #[test]
    fn test_optimal_hash_count_clamping_max() {
        // Very large m/n ratio should be clamped to MAX_HASH_FUNCTIONS
        let k = optimal_hash_count(100_000, 10).unwrap();
        assert!(k <= MAX_HASH_FUNCTIONS);
    }

    #[test]
    fn test_optimal_hash_count_clamping_min() {
        // Very small m/n ratio should be clamped to MIN_HASH_FUNCTIONS
        let k = optimal_hash_count(10, 100_000).unwrap();
        assert_eq!(k, MIN_HASH_FUNCTIONS);
    }

    #[test]
    fn test_optimal_hash_count_zero_bits_error() {
        let result = optimal_hash_count(0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimal_hash_count_zero_items_error() {
        let result = optimal_hash_count(1000, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_expected_fp_rate_matches_target() {
        let n = 1000;
        let target_fp = 0.01;
        let m = optimal_bit_count(n, target_fp).unwrap();
        let k = optimal_hash_count(m, n).unwrap();

        let actual_fp = expected_fp_rate(m, n, k).unwrap();

        // Should be within 10% of target
        let error = (actual_fp - target_fp).abs() / target_fp;
        assert!(
            error < 0.1,
            "FP rate error {:.2}% exceeds 10%. Expected {}, got {}",
            error * 100.0,
            target_fp,
            actual_fp
        );
    }

    #[test]
    fn test_expected_fp_rate_empty_filter() {
        let fp = expected_fp_rate(1000, 0, 7).unwrap();
        assert_eq!(fp, 0.0, "Empty filter should have 0% FP rate");
    }

    #[test]
    fn test_expected_fp_rate_full_filter() {
        // When n = m (one item per bit), FP rate should be very high
        let fp = expected_fp_rate(1000, 1000, 7).unwrap();
        assert!(fp > 0.5, "Saturated filter should have high FP rate");
    }

    #[test]
    fn test_expected_fp_rate_invalid_zero_bits() {
        let result = expected_fp_rate(0, 1000, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_expected_fp_rate_invalid_hash_count_zero() {
        let result = expected_fp_rate(1000, 100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_expected_fp_rate_invalid_hash_count_too_high() {
        let result = expected_fp_rate(1000, 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_filter_params() {
        let (m, k) = calculate_filter_params(1000, 0.01).unwrap();
        assert!(m >= EXPECTED_BITS_1000_1PCT && m <= EXPECTED_BITS_1000_1PCT + 1);
        assert_eq!(k, EXPECTED_HASH_9585_1000);
    }

    #[test]
    fn test_calculate_filter_params_various_fp_rates() {
        let test_cases = vec![
            (1000, 0.1, 4792, 3),
            (1000, 0.01, 9585, 7),
            (1000, 0.001, 14377, 10),
        ];

        for (n, fp, expected_m, expected_k) in test_cases {
            let (m, k) = calculate_filter_params(n, fp).unwrap();
            assert!(
                m >= expected_m && m <= expected_m + 1,
                "n={}, fp={}: expected m~{}, got {}",
                n,
                fp,
                expected_m,
                m
            );
            assert_eq!(
                k, expected_k,
                "n={}, fp={}: expected k={}, got {}",
                n, fp, expected_k, k
            );
        }
    }

    #[test]
    fn test_validate_params_valid() {
        assert!(validate_params(1000, 100, 7).is_ok());
        assert!(validate_params(10000, 1000, 10).is_ok());
    }

    #[test]
    fn test_validate_params_zero_bits() {
        let result = validate_params(0, 100, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_params_zero_items() {
        let result = validate_params(1000, 0, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_params_invalid_hash_count() {
        let result = validate_params(1000, 100, 0);
        assert!(result.is_err());

        let result = validate_params(1000, 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_params_bits_less_than_hashes() {
        let result = validate_params(5, 100, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_params_high_load_factor() {
        // Load factor > 2.0 should be rejected
        let result = validate_params(100, 250, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_params_allows_moderate_saturation() {
        // Load factor between 1.0 and 2.0 should be allowed
        assert!(validate_params(100, 150, 7).is_ok());
        assert!(validate_params(100, 200, 7).is_ok());
    }

    #[test]
    fn test_bits_per_element() {
        const EXPECTED_BPE_1PCT: f64 = 9.6; // -ln(0.01) / (ln2)²

        let bpe = bits_per_element(0.01).unwrap();
        assert!(
            (bpe - EXPECTED_BPE_1PCT).abs() < 0.1,
            "Expected ~{} bpe, got {}",
            EXPECTED_BPE_1PCT,
            bpe
        );

        const EXPECTED_BPE_0_1PCT: f64 = 14.4; // -ln(0.001) / (ln2)²
        let bpe = bits_per_element(0.001).unwrap();
        assert!(
            (bpe - EXPECTED_BPE_0_1PCT).abs() < 0.1,
            "Expected ~{} bpe, got {}",
            EXPECTED_BPE_0_1PCT,
            bpe
        );
    }

    #[test]
    fn test_bits_per_element_invalid_fp_rate() {
        assert!(bits_per_element(0.0).is_err());
        assert!(bits_per_element(1.0).is_err());
        assert!(bits_per_element(-0.1).is_err());
        assert!(bits_per_element(1.5).is_err());
    }

    #[test]
    fn test_mathematical_consistency() {
        // Verify that optimal_bit_count and bits_per_element are consistent
        let n = 1000;
        let fp_rate = 0.01;

        let m = optimal_bit_count(n, fp_rate).unwrap();
        let bpe = bits_per_element(fp_rate).unwrap();

        let expected_m = (n as f64 * bpe).ceil() as usize;
        assert_eq!(
            m, expected_m,
            "optimal_bit_count and bits_per_element should be consistent"
        );
    }

    #[test]
    fn test_roundtrip_calculation() {
        // Calculate params, then verify expected FP rate matches target
        let n = 10000;
        let target_fp = 0.005;

        let (m, k) = calculate_filter_params(n, target_fp).unwrap();
        let actual_fp = expected_fp_rate(m, n, k).unwrap();

        assert!(
            (actual_fp - target_fp).abs() / target_fp < 0.15,
            "Roundtrip calculation: target {}, got {}",
            target_fp,
            actual_fp
        );
    }

    #[test]
    fn test_optimal_hash_count_various_ratios() {
        let test_cases = vec![
            (1000, 100, 7),  // m/n = 10
            (2000, 100, 14), // m/n = 20
            (500, 100, 3),   // m/n = 5
        ];

        for (m, n, expected_k) in test_cases {
            let k = optimal_hash_count(m, n).unwrap();
            assert_eq!(
                k, expected_k,
                "For m={}, n={}, expected k={}, got {}",
                m, n, expected_k, k
            );
        }
    }

    #[test]
    fn test_optimal_bit_count_various_fp_rates() {
        let n = 1000;

        let test_cases = vec![
            (0.1, 4792),     // -1000 × ln(0.1) / (ln2)²
            (0.01, 9585),    // -1000 × ln(0.01) / (ln2)²
            (0.001, 14377),  // -1000 × ln(0.001) / (ln2)²
            (0.0001, 19170), // -1000 × ln(0.0001) / (ln2)²
        ];

        for (fp_rate, expected_m) in test_cases {
            let m = optimal_bit_count(n, fp_rate).unwrap();
            assert!(
                (m as i32 - expected_m).abs() <= 1,
                "For fp_rate={}, expected ~{}, got {}",
                fp_rate,
                expected_m,
                m
            );
        }
    }

    #[test]
    fn test_validate_params_edge_cases() {
        // Valid edge cases
        assert!(validate_params(1, 1, 1).is_ok());
        assert!(validate_params(100, 1, 1).is_ok());

        // Invalid: m < k
        assert!(validate_params(5, 100, 10).is_err());

        // Invalid: load factor > 2.0
        assert!(validate_params(100, 201, 7).is_err());
    }
}
