//! Bit manipulation utilities and optimizations.
//!
//! This module provides low-level bit manipulation functions optimized for
//! Bloom filter operations. It includes functions for counting bits, finding
//! powers of two, and other bit-level operations.
//!
//! # Performance Notes
//!
//! - Most functions compile to single CPU instructions on modern hardware
//! - `count_ones` uses the `POPCNT` instruction when available
//! - Power-of-two operations are branch-free and constant-time
//!
//! # Usage
//!
//! These utilities are primarily used internally by Bloom filter implementations
//! for operations like:
//! - Calculating optimal filter sizes (nearest power of 2)
//! - Counting set bits in bit vectors
//! - Fast modulo operations with power-of-2 sizes

#![allow(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Count the number of set bits (1s) in a u64 value.
///
/// On modern CPUs, this compiles to the `POPCNT` instruction which
/// executes in 1-3 cycles.
///
/// # Arguments
///
/// * `value` - Value to count bits in
///
/// # Returns
///
/// Number of set bits (0-64)
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::count_ones;
///
/// assert_eq!(count_ones(0b1010), 2);
/// assert_eq!(count_ones(0b1111), 4);
/// assert_eq!(count_ones(0), 0);
/// assert_eq!(count_ones(u64::MAX), 64);
/// ```
#[inline(always)]
#[must_use]
pub const fn count_ones(value: u64) -> u32 {
    value.count_ones()
}

/// Count the number of zero bits in a u64 value.
///
/// # Arguments
///
/// * `value` - Value to count bits in
///
/// # Returns
///
/// Number of zero bits (0-64)
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::count_zeros;
///
/// assert_eq!(count_zeros(0b1010), 62);
/// assert_eq!(count_zeros(0), 64);
/// assert_eq!(count_zeros(u64::MAX), 0);
/// ```
#[inline(always)]
#[must_use]
pub const fn count_zeros(value: u64) -> u32 {
    value.count_zeros()
}

/// Check if a number is a power of two.
///
/// This is a constant-time operation that compiles to just a few instructions.
///
/// # Arguments
///
/// * `n` - Number to check
///
/// # Returns
///
/// `true` if n is a power of two, `false` otherwise
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::is_power_of_two;
///
/// assert!(is_power_of_two(1));
/// assert!(is_power_of_two(2));
/// assert!(is_power_of_two(4));
/// assert!(is_power_of_two(1024));
///
/// assert!(!is_power_of_two(0));
/// assert!(!is_power_of_two(3));
/// assert!(!is_power_of_two(100));
/// ```
#[inline(always)]
#[must_use]
pub const fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Find the next power of two greater than or equal to the given number.
///
/// If the input is already a power of two, it returns the input unchanged.
/// If the input is 0, returns 1.
///
/// # Arguments
///
/// * `n` - Input number
///
/// # Returns
///
/// Next power of two >= n
///
/// # Panics
///
/// Panics if the result would overflow `usize`.
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::next_power_of_two;
///
/// assert_eq!(next_power_of_two(0), 1);
/// assert_eq!(next_power_of_two(1), 1);
/// assert_eq!(next_power_of_two(2), 2);
/// assert_eq!(next_power_of_two(3), 4);
/// assert_eq!(next_power_of_two(5), 8);
/// assert_eq!(next_power_of_two(100), 128);
/// assert_eq!(next_power_of_two(1000), 1024);
/// ```
#[inline]
#[must_use]
pub const fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    
    // If already power of 2, return as-is
    if n & (n - 1) == 0 {
        return n;
    }
    
    // Find next power of 2
    1 << (usize::BITS - (n - 1).leading_zeros())
}

/// Find the previous power of two less than or equal to the given number.
///
/// If the input is already a power of two, it returns the input unchanged.
/// If the input is 0, returns 0.
///
/// # Arguments
///
/// * `n` - Input number
///
/// # Returns
///
/// Previous power of two <= n
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::prev_power_of_two;
///
/// assert_eq!(prev_power_of_two(0), 0);
/// assert_eq!(prev_power_of_two(1), 1);
/// assert_eq!(prev_power_of_two(2), 2);
/// assert_eq!(prev_power_of_two(3), 2);
/// assert_eq!(prev_power_of_two(5), 4);
/// assert_eq!(prev_power_of_two(100), 64);
/// assert_eq!(prev_power_of_two(1000), 512);
/// ```
#[inline]
#[must_use]
pub const fn prev_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    
    // If already power of 2, return as-is
    if n & (n - 1) == 0 {
        return n;
    }
    
    // Find previous power of 2
    1 << (usize::BITS - 1 - n.leading_zeros())
}

/// Round up to the nearest multiple of a given value.
///
/// # Arguments
///
/// * `n` - Number to round
/// * `multiple` - Multiple to round to
///
/// # Returns
///
/// Smallest multiple of `multiple` that is >= n
///
/// # Panics
///
/// Panics if `multiple` is 0.
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::round_up_to_multiple;
///
/// assert_eq!(round_up_to_multiple(10, 8), 16);
/// assert_eq!(round_up_to_multiple(16, 8), 16);
/// assert_eq!(round_up_to_multiple(17, 8), 24);
/// assert_eq!(round_up_to_multiple(100, 64), 128);
/// ```
#[inline]
#[must_use]
pub const fn round_up_to_multiple(n: usize, multiple: usize) -> usize {
    assert!(multiple > 0, "multiple must be greater than 0");
    
    if n == 0 {
        return 0;
    }
    
    ((n + multiple - 1) / multiple) * multiple
}

/// Calculate the number of u64 words needed to store n bits.
///
/// # Arguments
///
/// * `n_bits` - Number of bits
///
/// # Returns
///
/// Number of u64 words needed
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::bits_to_words;
///
/// assert_eq!(bits_to_words(0), 0);
/// assert_eq!(bits_to_words(1), 1);
/// assert_eq!(bits_to_words(64), 1);
/// assert_eq!(bits_to_words(65), 2);
/// assert_eq!(bits_to_words(128), 2);
/// assert_eq!(bits_to_words(129), 3);
/// ```
#[inline]
#[must_use]
pub const fn bits_to_words(n_bits: usize) -> usize {
    (n_bits + 63) / 64
}

/// Calculate the number of bytes needed to store n bits.
///
/// # Arguments
///
/// * `n_bits` - Number of bits
///
/// # Returns
///
/// Number of bytes needed
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::bits_to_bytes;
///
/// assert_eq!(bits_to_bytes(0), 0);
/// assert_eq!(bits_to_bytes(1), 1);
/// assert_eq!(bits_to_bytes(8), 1);
/// assert_eq!(bits_to_bytes(9), 2);
/// assert_eq!(bits_to_bytes(16), 2);
/// ```
#[inline]
#[must_use]
pub const fn bits_to_bytes(n_bits: usize) -> usize {
    (n_bits + 7) / 8
}

/// Get the index of the word containing the given bit.
///
/// # Arguments
///
/// * `bit_index` - Bit index
///
/// # Returns
///
/// Word index (bit_index / 64)
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::word_index;
///
/// assert_eq!(word_index(0), 0);
/// assert_eq!(word_index(63), 0);
/// assert_eq!(word_index(64), 1);
/// assert_eq!(word_index(127), 1);
/// assert_eq!(word_index(128), 2);
/// ```
#[inline(always)]
#[must_use]
pub const fn word_index(bit_index: usize) -> usize {
    bit_index >> 6 // Equivalent to bit_index / 64
}

/// Get the bit offset within a word for the given bit index.
///
/// # Arguments
///
/// * `bit_index` - Bit index
///
/// # Returns
///
/// Bit offset within word (bit_index % 64)
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::bit_offset;
///
/// assert_eq!(bit_offset(0), 0);
/// assert_eq!(bit_offset(1), 1);
/// assert_eq!(bit_offset(63), 63);
/// assert_eq!(bit_offset(64), 0);
/// assert_eq!(bit_offset(65), 1);
/// ```
#[inline(always)]
#[must_use]
pub const fn bit_offset(bit_index: usize) -> usize {
    bit_index & 63 // Equivalent to bit_index % 64
}

/// Create a mask with a single bit set at the given offset.
///
/// # Arguments
///
/// * `offset` - Bit offset (0-63)
///
/// # Returns
///
/// u64 with only the bit at `offset` set
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::bit_mask;
///
/// assert_eq!(bit_mask(0), 0b1);
/// assert_eq!(bit_mask(1), 0b10);
/// assert_eq!(bit_mask(2), 0b100);
/// assert_eq!(bit_mask(63), 1u64 << 63);
/// ```
#[inline(always)]
#[must_use]
pub const fn bit_mask(offset: usize) -> u64 {
    1u64 << (offset & 63)
}

/// Count the total number of set bits in a slice of u64 words.
///
/// This is useful for calculating the current fill rate of a Bloom filter.
///
/// # Arguments
///
/// * `words` - Slice of u64 words
///
/// # Returns
///
/// Total number of set bits across all words
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::count_ones_slice;
///
/// let words = vec![0b1010, 0b1111, 0b0001];
/// assert_eq!(count_ones_slice(&words), 7);
///
/// let empty: Vec<u64> = vec![];
/// assert_eq!(count_ones_slice(&empty), 0);
/// ```
#[inline]
#[must_use]
pub fn count_ones_slice(words: &[u64]) -> usize {
    words.iter().map(|&w| w.count_ones() as usize).sum()
}

/// Calculate the Hamming distance between two bit vectors.
///
/// The Hamming distance is the number of positions at which the
/// corresponding bits are different.
///
/// # Arguments
///
/// * `a` - First bit vector
/// * `b` - Second bit vector
///
/// # Returns
///
/// Hamming distance
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::hamming_distance;
///
/// let a = vec![0b1010, 0b1111];
/// let b = vec![0b1100, 0b1111];
/// assert_eq!(hamming_distance(&a, &b), 2); // Two different bits in first word
/// ```
#[inline]
#[must_use]
pub fn hamming_distance(a: &[u64], b: &[u64]) -> usize {
    assert_eq!(a.len(), b.len(), "Slices must have same length");
    
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

/// Check if all bits in a range are zero.
///
/// # Arguments
///
/// * `words` - Slice of u64 words
/// * `start_bit` - Start bit index (inclusive)
/// * `end_bit` - End bit index (exclusive)
///
/// # Returns
///
/// `true` if all bits in range are zero
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::all_zeros_in_range;
///
/// let words = vec![0b0000_0000, 0b0000_0000];
/// assert!(all_zeros_in_range(&words, 0, 128));
///
/// let words = vec![0b0000_0001, 0b0000_0000];
/// assert!(!all_zeros_in_range(&words, 0, 64));
/// ```
#[must_use]
pub fn all_zeros_in_range(words: &[u64], start_bit: usize, end_bit: usize) -> bool {
    if start_bit >= end_bit {
        return true;
    }
    
    let start_word = word_index(start_bit);
    let end_word = word_index(end_bit.saturating_sub(1));
    
    if start_word >= words.len() {
        return true;
    }
    
    for word_idx in start_word..=end_word.min(words.len().saturating_sub(1)) {
        if words[word_idx] != 0 {
            return false;
        }
    }
    
    true
}

/// Count leading zeros in a u64 value.
///
/// # Arguments
///
/// * `value` - Value to count leading zeros in
///
/// # Returns
///
/// Number of leading zero bits (0-64)
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::leading_zeros;
///
/// assert_eq!(leading_zeros(0), 64);
/// assert_eq!(leading_zeros(1), 63);
/// assert_eq!(leading_zeros(0b1000), 60);
/// assert_eq!(leading_zeros(u64::MAX), 0);
/// ```
#[inline(always)]
#[must_use]
pub const fn leading_zeros(value: u64) -> u32 {
    value.leading_zeros()
}

/// Count trailing zeros in a u64 value.
///
/// # Arguments
///
/// * `value` - Value to count trailing zeros in
///
/// # Returns
///
/// Number of trailing zero bits (0-64)
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::trailing_zeros;
///
/// assert_eq!(trailing_zeros(0), 64);
/// assert_eq!(trailing_zeros(1), 0);
/// assert_eq!(trailing_zeros(0b1000), 3);
/// assert_eq!(trailing_zeros(0b1100), 2);
/// ```
#[inline(always)]
#[must_use]
pub const fn trailing_zeros(value: u64) -> u32 {
    value.trailing_zeros()
}

/// Reverse the bits in a u64 value.
///
/// # Arguments
///
/// * `value` - Value to reverse
///
/// # Returns
///
/// Value with bits reversed
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::reverse_bits;
///
/// assert_eq!(reverse_bits(0b1), 1u64 << 63);
/// assert_eq!(reverse_bits(0b1010), 0b0101 << 60);
/// ```
#[inline]
#[must_use]
pub const fn reverse_bits(value: u64) -> u64 {
    value.reverse_bits()
}

/// Rotate bits left by n positions.
///
/// # Arguments
///
/// * `value` - Value to rotate
/// * `n` - Number of positions to rotate
///
/// # Returns
///
/// Rotated value
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::rotate_left;
///
/// assert_eq!(rotate_left(0b1, 1), 0b10);
/// assert_eq!(rotate_left(0b1001, 2), 0b100100);
/// ```
#[inline(always)]
#[must_use]
pub const fn rotate_left(value: u64, n: u32) -> u64 {
    value.rotate_left(n)
}

/// Rotate bits right by n positions.
///
/// # Arguments
///
/// * `value` - Value to rotate
/// * `n` - Number of positions to rotate
///
/// # Returns
///
/// Rotated value
///
/// # Examples
///
/// ```
/// use bloomcraft::util::bitops::rotate_right;
///
/// assert_eq!(rotate_right(0b10, 1), 0b1);
/// assert_eq!(rotate_right(0b1001, 2), 0b0100_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0010);
/// ```
#[inline(always)]
#[must_use]
pub const fn rotate_right(value: u64, n: u32) -> u64 {
    value.rotate_right(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_ones() {
        assert_eq!(count_ones(0), 0);
        assert_eq!(count_ones(0b1), 1);
        assert_eq!(count_ones(0b1010), 2);
        assert_eq!(count_ones(0b1111), 4);
        assert_eq!(count_ones(u64::MAX), 64);
    }

    #[test]
    fn test_count_zeros() {
        assert_eq!(count_zeros(0), 64);
        assert_eq!(count_zeros(0b1), 63);
        assert_eq!(count_zeros(0b1010), 62);
        assert_eq!(count_zeros(u64::MAX), 0);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(4));
        assert!(!is_power_of_two(5));
        assert!(is_power_of_two(8));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(1023));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(1000), 1024);
        assert_eq!(next_power_of_two(1024), 1024);
    }

    #[test]
    fn test_prev_power_of_two() {
        assert_eq!(prev_power_of_two(0), 0);
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(2), 2);
        assert_eq!(prev_power_of_two(3), 2);
        assert_eq!(prev_power_of_two(4), 4);
        assert_eq!(prev_power_of_two(5), 4);
        assert_eq!(prev_power_of_two(100), 64);
        assert_eq!(prev_power_of_two(1000), 512);
        assert_eq!(prev_power_of_two(1024), 1024);
    }

    #[test]
    fn test_round_up_to_multiple() {
        assert_eq!(round_up_to_multiple(0, 8), 0);
        assert_eq!(round_up_to_multiple(1, 8), 8);
        assert_eq!(round_up_to_multiple(8, 8), 8);
        assert_eq!(round_up_to_multiple(9, 8), 16);
        assert_eq!(round_up_to_multiple(10, 8), 16);
        assert_eq!(round_up_to_multiple(100, 64), 128);
    }

    #[test]
    #[should_panic(expected = "multiple must be greater than 0")]
    fn test_round_up_to_multiple_zero() {
        let _ = round_up_to_multiple(10, 0);
    }

    #[test]
    fn test_bits_to_words() {
        assert_eq!(bits_to_words(0), 0);
        assert_eq!(bits_to_words(1), 1);
        assert_eq!(bits_to_words(64), 1);
        assert_eq!(bits_to_words(65), 2);
        assert_eq!(bits_to_words(128), 2);
        assert_eq!(bits_to_words(129), 3);
    }

    #[test]
    fn test_bits_to_bytes() {
        assert_eq!(bits_to_bytes(0), 0);
        assert_eq!(bits_to_bytes(1), 1);
        assert_eq!(bits_to_bytes(8), 1);
        assert_eq!(bits_to_bytes(9), 2);
        assert_eq!(bits_to_bytes(16), 2);
    }

    #[test]
    fn test_word_index() {
        assert_eq!(word_index(0), 0);
        assert_eq!(word_index(63), 0);
        assert_eq!(word_index(64), 1);
        assert_eq!(word_index(127), 1);
        assert_eq!(word_index(128), 2);
    }

    #[test]
    fn test_bit_offset() {
        assert_eq!(bit_offset(0), 0);
        assert_eq!(bit_offset(1), 1);
        assert_eq!(bit_offset(63), 63);
        assert_eq!(bit_offset(64), 0);
        assert_eq!(bit_offset(65), 1);
    }

    #[test]
    fn test_bit_mask() {
        assert_eq!(bit_mask(0), 1);
        assert_eq!(bit_mask(1), 2);
        assert_eq!(bit_mask(2), 4);
        assert_eq!(bit_mask(63), 1u64 << 63);
    }

    #[test]
    fn test_count_ones_slice() {
        assert_eq!(count_ones_slice(&[]), 0);
        assert_eq!(count_ones_slice(&[0]), 0);
        assert_eq!(count_ones_slice(&[0b1010]), 2);
        assert_eq!(count_ones_slice(&[0b1010, 0b1111]), 6);
        assert_eq!(count_ones_slice(&[u64::MAX, u64::MAX]), 128);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(&[], &[]), 0);
        assert_eq!(hamming_distance(&[0], &[0]), 0);
        assert_eq!(hamming_distance(&[0b1010], &[0b1010]), 0);
        assert_eq!(hamming_distance(&[0b1010], &[0b0101]), 4);
        assert_eq!(hamming_distance(&[0b1111], &[0b0000]), 4);
    }

    #[test]
    #[should_panic(expected = "Slices must have same length")]
    fn test_hamming_distance_different_lengths() {
        let _ = hamming_distance(&[0], &[0, 0]);
    }

    #[test]
    fn test_all_zeros_in_range() {
        let zeros = vec![0u64; 4];
        assert!(all_zeros_in_range(&zeros, 0, 256));
        
        let mixed = vec![0, 0, 1, 0];
        assert!(all_zeros_in_range(&mixed, 0, 128));
        assert!(!all_zeros_in_range(&mixed, 0, 192));
        
        assert!(all_zeros_in_range(&[], 0, 64));
        assert!(all_zeros_in_range(&[0], 100, 50)); // Empty range
    }

    #[test]
    fn test_leading_zeros() {
        assert_eq!(leading_zeros(0), 64);
        assert_eq!(leading_zeros(1), 63);
        assert_eq!(leading_zeros(0b1000), 60);
        assert_eq!(leading_zeros(u64::MAX), 0);
    }

    #[test]
    fn test_trailing_zeros() {
        assert_eq!(trailing_zeros(0), 64);
        assert_eq!(trailing_zeros(1), 0);
        assert_eq!(trailing_zeros(0b1000), 3);
        assert_eq!(trailing_zeros(0b1100), 2);
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0), 0);
        assert_eq!(reverse_bits(1), 1u64 << 63);
        assert_eq!(reverse_bits(u64::MAX), u64::MAX);
    }

    #[test]
    fn test_rotate_left() {
        assert_eq!(rotate_left(0b1, 1), 0b10);
        assert_eq!(rotate_left(0b1, 63), 1u64 << 63);
        assert_eq!(rotate_left(0b1, 64), 0b1); // Full rotation
    }

    #[test]
    fn test_rotate_right() {
        assert_eq!(rotate_right(0b10, 1), 0b1);
        assert_eq!(rotate_right(1u64 << 63, 63), 0b1);
        assert_eq!(rotate_right(0b1, 64), 0b1); // Full rotation
    }
}
