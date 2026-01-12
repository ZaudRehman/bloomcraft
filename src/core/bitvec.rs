//! Lock-free bit vector with atomic operations.
//!
//! This module provides a thread-safe bit vector optimized for Bloom filter operations,
//! using atomic operations for concurrent access without locks.
//!
//! # Overview
//!
//! `BitVec` is a dynamically-sized bit array backed by `Box<[AtomicU64]>`. Each 64-bit
//! word stores 64 bits, providing compact storage with atomic access guarantees.
//!
//! # Thread Safety
//!
//! - `set`: Lock-free, thread-safe with `&self` (uses `Ordering::Release`)
//! - `get`: Lock-free, thread-safe with `&self` (uses `Ordering::Acquire`)
//! - `clear`: Requires exclusive access (`&mut self`, uses `Ordering::Relaxed`)
//!
//! # Memory Ordering
//!
//! This implementation uses Release-Acquire ordering for set/get:
//!
//! - `set` uses `Release`: Ensures all prior writes are visible to threads that observe this bit
//! - `get` uses `Acquire`: Ensures we see all writes that happened-before the corresponding set
//!
//! This prevents false negatives in concurrent insert/query scenarios where one thread
//! inserts an item while another queries for it.
//!
//! # Memory Layout
//!
//! Bits are packed into 64-bit words in little-endian bit order:
//!
//! ```text
//! Word 0: [bit 0][bit 1]...[bit 63]
//! Word 1: [bit 64][bit 65]...[bit 127]
//! Word 2: [bit 128][bit 129]...[bit 191]
//! ```
//!
//! # Performance Characteristics
//!
//! - Space: `⌈n/64⌉ * 8` bytes for `n` bits
//! - `set`: O(1) - single atomic fetch-or
//! - `get`: O(1) - single atomic load + bit test
//! - `count_ones`: O(n/64) - iterates all words, uses CPU POPCNT instruction
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::core::bitvec::BitVec;
//!
//! let bv = BitVec::new(100).unwrap();
//! bv.set(42);
//! assert!(bv.get(42));
//! assert!(!bv.get(43));
//! assert_eq!(bv.count_ones(), 1);
//! ```
//!
//! ## Concurrent Access
//!
//! ```
//! use bloomcraft::core::bitvec::BitVec;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let bv = Arc::new(BitVec::new(1000).unwrap());
//!
//! let handles: Vec<_> = (0..4)
//!     .map(|i| {
//!         let bv = Arc::clone(&bv);
//!         thread::spawn(move || {
//!             for j in 0..250 {
//!                 bv.set(i * 250 + j);
//!             }
//!         })
//!     })
//!     .collect();
//!
//! for h in handles {
//!     h.join().unwrap();
//! }
//!
//! assert_eq!(bv.count_ones(), 1000);
//! ```
//!
//! ## Union and Intersection
//!
//! ```
//! use bloomcraft::core::bitvec::BitVec;
//!
//! let bv1 = BitVec::new(64).unwrap();
//! let bv2 = BitVec::new(64).unwrap();
//!
//! bv1.set(10);
//! bv2.set(20);
//!
//! let union = bv1.union(&bv2).unwrap();
//! assert!(union.get(10));
//! assert!(union.get(20));
//!
//! let intersection = bv1.intersect(&bv2).unwrap();
//! assert!(!intersection.get(10));
//! assert!(!intersection.get(20));
//! ```

use crate::error::{BloomCraftError, Result};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "serde")]
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

/// Lock-free bit vector with atomic operations.
///
/// Provides a fixed-size bit array with atomic operations for concurrent access.
/// Uses `Box<[AtomicU64]>` for storage, where each word holds 64 bits.
///
/// # Type Properties
///
/// - `Send + Sync`: Safe to share across threads (`AtomicU64` is `Send + Sync`)
/// - `Clone`: Creates an independent copy via explicit implementation
/// - `Debug`: Displays internal structure for debugging
/// - `Serde`: Serialization support behind `serde` feature flag
#[derive(Debug)]
pub struct BitVec {
    /// Atomic blocks (words), each storing 64 bits.
    ///
    /// `Box<[AtomicU64]>` provides a fixed-size allocation that's safe to share.
    blocks: Box<[AtomicU64]>,

    /// Total number of bits in the vector.
    len: usize,
}

impl BitVec {
    /// Create a new bit vector with the specified number of bits.
    ///
    /// All bits are initialized to 0. The number of 64-bit blocks allocated is
    /// `⌈num_bits / 64⌉`.
    ///
    /// # Arguments
    ///
    /// * `num_bits` - Number of bits in the vector (must be > 0)
    ///
    /// # Returns
    ///
    /// * `Ok(BitVec)` - New bit vector with all bits set to 0
    /// * `Err(BloomCraftError)` - If `num_bits` is 0
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(1000).unwrap();
    /// assert_eq!(bv.len(), 1000);
    /// assert_eq!(bv.count_ones(), 0);
    /// ```
    #[must_use]
    pub fn new(num_bits: usize) -> Result<Self> {
        if num_bits == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "BitVec size must be greater than 0",
            ));
        }

        let num_blocks = (num_bits + 63) / 64;
        let blocks = (0..num_blocks)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            blocks,
            len: num_bits,
        })
    }

    /// Get the number of bits in the vector.
    ///
    /// # Returns
    ///
    /// Total number of bits (not necessarily a multiple of 64)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(100).unwrap();
    /// assert_eq!(bv.len(), 100);
    /// ```
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the bit vector is empty.
    ///
    /// Since `new` requires `num_bits > 0`, this will always return `false` for a
    /// successfully constructed `BitVec`. Provided for API completeness.
    ///
    /// # Returns
    ///
    /// `false` (a `BitVec` is never empty)
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set a bit to 1 atomically (thread-safe).
    ///
    /// Uses atomic fetch-or with `Ordering::Release` to ensure visibility of this write
    /// to other threads performing Acquire loads.
    ///
    /// This operation is idempotent—setting an already-set bit is safe and has no
    /// additional effect.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index to set (0-based, must be < len)
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`. This is intentional to match standard library indexing
    /// behavior (e.g., `Vec[index]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(64).unwrap();
    /// bv.set(10);
    /// bv.set(10); // Idempotent
    /// assert!(bv.get(10));
    /// ```
    #[inline]
    pub fn set(&self, index: usize) {
        assert!(
            index < self.len,
            "BitVec index out of bounds: index={} len={}",
            index,
            self.len
        );

        let block_idx = index / 64;
        let bit_offset = index % 64;
        let mask = 1u64 << bit_offset;

        // Release ordering: ensures this write is visible to threads doing Acquire loads
        self.blocks[block_idx].fetch_or(mask, Ordering::Release);
    }

    /// Get a bit value atomically (thread-safe).
    ///
    /// Uses atomic load with `Ordering::Acquire` to synchronize with Release stores
    /// from `set`.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index to get (0-based, must be < len)
    ///
    /// # Returns
    ///
    /// * `true` - Bit is set to 1
    /// * `false` - Bit is 0
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(64).unwrap();
    /// assert!(!bv.get(10));
    /// bv.set(10);
    /// assert!(bv.get(10));
    /// ```
    #[must_use]
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        assert!(
            index < self.len,
            "BitVec index out of bounds: index={} len={}",
            index,
            self.len
        );

        let block_idx = index / 64;
        let bit_offset = index % 64;
        let mask = 1u64 << bit_offset;

        // Acquire ordering: synchronizes with Release stores, prevents false negatives
        (self.blocks[block_idx].load(Ordering::Acquire) & mask) != 0
    }

    /// Set a range of bits to a specific value.
    ///
    /// Sets all bits in the range `[start..end)` to the specified value (0 or 1).
    /// This is more efficient than setting bits individually when working with
    /// contiguous ranges.
    ///
    /// # Arguments
    ///
    /// * `start` - Start index (inclusive)
    /// * `end` - End index (exclusive)
    /// * `value` - Value to set (true = 1, false = 0)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `start > end` (invalid range)
    /// - `end > self.len()` (out of bounds)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(100).unwrap();
    /// bv.set_range(10, 20, true);
    ///
    /// assert!(bv.get(10));
    /// assert!(bv.get(19));
    /// assert!(!bv.get(9));
    /// assert!(!bv.get(20));
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This operation is NOT atomic across the entire range. Individual bit sets are
    /// atomic, but the range as a whole is not. Use external synchronization if you
    /// need atomicity across the entire range.
    #[allow(clippy::needless_range_loop)]
    pub fn set_range(&self, start: usize, end: usize, value: bool) {
        // Validate range
        if start > end {
            panic!(
                "set_range: invalid range [{}..{}) - start must be <= end",
                start, end
            );
        }

        if end > self.len {
            panic!(
                "set_range: end index {} out of bounds (length {})",
                end, self.len
            );
        }

        // Early return for empty range
        if start == end {
            return;
        }

        let start_word = start / 64;
        let end_word = (end - 1) / 64; // Inclusive end word

        for word_idx in start_word..=end_word {
            let word_start = if word_idx == start_word {
                start % 64
            } else {
                0
            };

            let word_end = if word_idx == end_word {
                ((end - 1) % 64) + 1 // Convert to exclusive upper bound
            } else {
                64
            };

            // Set/clear each bit in this word
            for bit_idx in word_start..word_end {
                let bit_pos = word_idx * 64 + bit_idx;

                // Verify we're within bounds
                debug_assert!(
                    bit_pos < self.len,
                    "set_range internal error: bit_pos={} >= len={}",
                    bit_pos,
                    self.len
                );

                if value {
                    self.set(bit_pos);
                } else {
                    self.clear_bit(bit_pos);
                }
            }
        }
    }

    /// Clear a single bit to 0 atomically (thread-safe).
    ///
    /// Uses atomic fetch-and with `Ordering::Release` to ensure visibility.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index to clear (0-based, must be < len)
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(64).unwrap();
    /// bv.set(10);
    /// assert!(bv.get(10));
    /// bv.clear_bit(10);
    /// assert!(!bv.get(10));
    /// ```
    #[inline]
    pub fn clear_bit(&self, index: usize) {
        assert!(
            index < self.len,
            "BitVec index out of bounds: index={} len={}",
            index,
            self.len
        );

        let block_idx = index / 64;
        let bit_offset = index % 64;
        let mask = !(1u64 << bit_offset); // Inverted mask for clearing

        // Release ordering: ensures this write is visible to threads doing Acquire loads
        self.blocks[block_idx].fetch_and(mask, Ordering::Release);
    }

    /// Get a range of bits as a vector.
    ///
    /// Returns the values of all bits in the range `[start..end)` as a vector of booleans.
    ///
    /// # Arguments
    ///
    /// * `start` - Start index (inclusive)
    /// * `end` - End index (exclusive)
    ///
    /// # Returns
    ///
    /// Vector of boolean values for the specified range.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `start > end` (invalid range)
    /// - `end > self.len()` (out of bounds)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(100).unwrap();
    /// bv.set(10);
    /// bv.set(11);
    /// bv.set(13);
    ///
    /// let range = bv.get_range(10, 15);
    /// assert_eq!(range, vec![true, true, false, true, false]);
    /// ```
    #[must_use]
    pub fn get_range(&self, start: usize, end: usize) -> Vec<bool> {
        // Validate range
        if start > end {
            panic!(
                "get_range: invalid range [{}..{}) - start must be <= end",
                start, end
            );
        }

        if end > self.len {
            panic!(
                "get_range: end index {} out of bounds (length {})",
                end, self.len
            );
        }

        (start..end).map(|i| self.get(i)).collect()
    }

    /// Clear all bits to 0 (requires exclusive access).
    ///
    /// Since this requires `&mut self`, no other thread can be accessing the bit vector
    /// concurrently. Uses Relaxed ordering since exclusivity provides synchronization.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let mut bv = BitVec::new(64).unwrap();
    /// bv.set(10);
    /// bv.set(20);
    /// assert_eq!(bv.count_ones(), 2);
    ///
    /// bv.clear();
    /// assert_eq!(bv.count_ones(), 0);
    /// ```
    pub fn clear(&mut self) {
        for block in &*self.blocks {
            // Relaxed is safe: `&mut self` guarantees exclusive access
            block.store(0, Ordering::Relaxed);
        }
    }

    /// Count the number of bits set to 1.
    ///
    /// Uses the CPU's POPCNT instruction via `u64::count_ones()` on modern x86-64
    /// processors for efficient counting.
    ///
    /// # Returns
    ///
    /// Number of bits currently set to 1
    ///
    /// # Time Complexity
    ///
    /// O(⌈len/64⌉) - iterates all 64-bit words
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(100).unwrap();
    /// bv.set(0);
    /// bv.set(50);
    /// bv.set(99);
    /// assert_eq!(bv.count_ones(), 3);
    /// ```
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }

    /// Get total memory usage in bytes.
    ///
    /// Includes storage for atomic blocks plus the struct itself.
    ///
    /// # Returns
    ///
    /// Total bytes allocated
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(1000).unwrap();
    /// let bytes = bv.memory_usage();
    /// assert!(bytes >= 128); // At least ⌈1000/64⌉ * 8 bytes
    /// ```
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<AtomicU64>() + std::mem::size_of::<Self>()
    }

    /// Compute the union of two bit vectors (bitwise OR).
    ///
    /// Creates a new `BitVec` where each bit is set if it's set in either input vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Bit vector to union with (must have same length)
    ///
    /// # Returns
    ///
    /// * `Ok(BitVec)` - New bit vector containing the union
    /// * `Err(BloomCraftError)` - If lengths don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv1 = BitVec::new(64).unwrap();
    /// let bv2 = BitVec::new(64).unwrap();
    ///
    /// bv1.set(10);
    /// bv2.set(20);
    ///
    /// let union = bv1.union(&bv2).unwrap();
    /// assert!(union.get(10));
    /// assert!(union.get(20));
    /// ```
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.len != other.len {
            return Err(BloomCraftError::incompatible_filters(format!(
                "BitVec size mismatch: {} vs {}",
                self.len, other.len
            )));
        }

        let result = Self::new(self.len)?;

        for (i, (a, b)) in self.blocks.iter().zip(&*other.blocks).enumerate() {
            let val = a.load(Ordering::Relaxed) | b.load(Ordering::Relaxed);
            result.blocks[i].store(val, Ordering::Relaxed);
        }

        Ok(result)
    }

    /// Compute the intersection of two bit vectors (bitwise AND).
    ///
    /// Creates a new `BitVec` where each bit is set only if it's set in both input vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - Bit vector to intersect with (must have same length)
    ///
    /// # Returns
    ///
    /// * `Ok(BitVec)` - New bit vector containing the intersection
    /// * `Err(BloomCraftError)` - If lengths don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv1 = BitVec::new(64).unwrap();
    /// let bv2 = BitVec::new(64).unwrap();
    ///
    /// bv1.set(10);
    /// bv1.set(20);
    /// bv2.set(10);
    /// bv2.set(30);
    ///
    /// let intersection = bv1.intersect(&bv2).unwrap();
    /// assert!(intersection.get(10));   // In both
    /// assert!(!intersection.get(20));  // Only in bv1
    /// assert!(!intersection.get(30));  // Only in bv2
    /// ```
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.len != other.len {
            return Err(BloomCraftError::incompatible_filters(format!(
                "BitVec size mismatch: {} vs {}",
                self.len, other.len
            )));
        }

        let result = Self::new(self.len)?;

        for (i, (a, b)) in self.blocks.iter().zip(&*other.blocks).enumerate() {
            let val = a.load(Ordering::Relaxed) & b.load(Ordering::Relaxed);
            result.blocks[i].store(val, Ordering::Relaxed);
        }

        Ok(result)
    }

    /// Get the number of 64-bit blocks.
    ///
    /// # Returns
    ///
    /// Number of `AtomicU64` words allocated
    #[must_use]
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Reconstruct bit vector from raw u64 words (for deserialization).
    ///
    /// Creates a new `BitVec` from serialized data. The length is inferred
    /// from the number of u64 words provided.
    ///
    /// # Arguments
    ///
    /// * `raw` - Vector of u64 words
    ///
    /// # Errors
    ///
    /// Returns error if `raw` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::BitVec;
    ///
    /// let mut original = BitVec::new(128).unwrap();
    /// original.set(42);
    ///
    /// let raw = original.to_raw();
    /// let len = original.len();
    /// let restored = BitVec::from_raw(raw, len).unwrap();  // ✓ Correct!
    ///
    /// assert!(restored.get(42));
    /// assert!(!restored.get(43));
    /// ```
    pub fn from_raw(raw: Vec<u64>, len: usize) -> crate::error::Result<Self> {
        if raw.is_empty() {
            return Err(BloomCraftError::invalid_parameters(
                "raw bit vector cannot be empty".to_string(),
            ));
        }

        if len == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "BitVec length must be greater than 0".to_string(),
            ));
        }

        // Calculate required blocks for the given length
        let required_blocks = (len + 63) / 64;
        if raw.len() < required_blocks {
            return Err(BloomCraftError::invalid_parameters(
                format!(
                    "Insufficient blocks: need {} for {} bits, got {}",
                    required_blocks, len, raw.len()
                ),
            ));
        }

        // Convert Vec to Box<[AtomicU64]>
        let blocks: Box<[AtomicU64]> = raw
            .into_iter()
            .map(AtomicU64::new)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self { blocks, len })
    }

    /// Convert bit vector to raw u64 words for serialization.
    ///
    /// Extracts the underlying atomic u64 blocks as plain u64 values.
    /// This is safe because we're only reading the values.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::BitVec;
    ///
    /// let mut bits = BitVec::new(128).unwrap();
    /// bits.set(5);
    /// bits.set(100);
    ///
    /// let raw = bits.to_raw();
    /// assert_eq!(raw.len(), 2); // 128 bits = 2 × 64-bit words
    /// ```
    #[must_use]
    pub fn to_raw(&self) -> Vec<u64> {
        use std::sync::atomic::Ordering;

        self.blocks
            .iter()
            .map(|block| block.load(Ordering::Relaxed))
            .collect()
    }
}

impl Clone for BitVec {
    /// Clone the bit vector.
    ///
    /// Creates an independent copy with the same bit values. Modifications to the clone
    /// do not affect the original.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv1 = BitVec::new(64).unwrap();
    /// bv1.set(10);
    ///
    /// let bv2 = bv1.clone();
    /// assert!(bv2.get(10));
    ///
    /// bv1.set(20);
    /// assert!(!bv2.get(20)); // Independent
    /// ```
    fn clone(&self) -> Self {
        let blocks = self
            .blocks
            .iter()
            .map(|b| AtomicU64::new(b.load(Ordering::Relaxed)))
            .collect();

        Self {
            blocks,
            len: self.len,
        }
    }
}

// Serde support (feature-gated)
#[cfg(feature = "serde")]
impl Serialize for BitVec {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let blocks_data: Vec<u64> = self
            .blocks
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .collect();

        let mut state = serializer.serialize_struct("BitVec", 2)?;
        state.serialize_field("blocks", &blocks_data)?;
        state.serialize_field("len", &self.len)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for BitVec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Blocks,
            Len,
        }

        struct BitVecVisitor;

        impl<'de> Visitor<'de> for BitVecVisitor {
            type Value = BitVec;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("struct BitVec with fields 'blocks' and 'len'")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<BitVec, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut blocks_data: Option<Vec<u64>> = None;
                let mut len: Option<usize> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Blocks => {
                            if blocks_data.is_some() {
                                return Err(de::Error::duplicate_field("blocks"));
                            }
                            blocks_data = Some(map.next_value()?);
                        }
                        Field::Len => {
                            if len.is_some() {
                                return Err(de::Error::duplicate_field("len"));
                            }
                            len = Some(map.next_value()?);
                        }
                    }
                }

                let blocks_data = blocks_data.ok_or_else(|| de::Error::missing_field("blocks"))?;
                let len = len.ok_or_else(|| de::Error::missing_field("len"))?;

                let blocks = blocks_data
                    .into_iter()
                    .map(AtomicU64::new)
                    .collect::<Vec<_>>()
                    .into_boxed_slice();

                Ok(BitVec { blocks, len })
            }
        }

        const FIELDS: &[&str] = &["blocks", "len"];
        deserializer.deserialize_struct("BitVec", FIELDS, BitVecVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bv = BitVec::new(100).unwrap();
        assert_eq!(bv.len(), 100);
        assert_eq!(bv.num_blocks(), 2); // ⌈100/64⌉ = 2
        assert!(!bv.is_empty());
    }

    #[test]
    fn test_new_zero_bits_error() {
        let result = BitVec::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_get() {
        let bv = BitVec::new(128).unwrap();
        assert!(!bv.get(0));

        bv.set(0);
        bv.set(63);
        bv.set(64);
        bv.set(127);

        assert!(bv.get(0));
        assert!(bv.get(63));
        assert!(bv.get(64));
        assert!(bv.get(127));
        assert!(!bv.get(32));
    }

    #[test]
    fn test_set_idempotent() {
        let bv = BitVec::new(64).unwrap();
        bv.set(10);
        bv.set(10);
        bv.set(10);
        assert_eq!(bv.count_ones(), 1);
    }

    #[test]
    fn test_clear() {
        let mut bv = BitVec::new(64).unwrap();
        bv.set(10);
        bv.set(20);
        bv.set(30);
        assert_eq!(bv.count_ones(), 3);

        bv.clear();
        assert_eq!(bv.count_ones(), 0);
        assert!(!bv.get(10));
        assert!(!bv.get(20));
        assert!(!bv.get(30));
    }

    #[test]
    fn test_clear_bit() {
        let bv = BitVec::new(64).unwrap();
        bv.set(10);
        assert!(bv.get(10));

        bv.clear_bit(10);
        assert!(!bv.get(10));
    }

    #[test]
    fn test_count_ones() {
        let bv = BitVec::new(100).unwrap();
        assert_eq!(bv.count_ones(), 0);

        bv.set(0);
        bv.set(50);
        bv.set(99);
        assert_eq!(bv.count_ones(), 3);
    }

    /// ⭐ NEW TEST: Test set_range with valid range
    #[test]
    fn test_set_range_basic() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(10, 20, true);

        // Check bits are set in range
        for i in 10..20 {
            assert!(bv.get(i), "Bit {} should be set", i);
        }

        // Check bits outside range are not set
        assert!(!bv.get(9), "Bit 9 should not be set");
        assert!(!bv.get(20), "Bit 20 should not be set");
    }

    /// ⭐ NEW TEST: Test set_range clearing bits
    #[test]
    fn test_set_range_clear() {
        let bv = BitVec::new(100).unwrap();

        // Set all bits first
        for i in 0..100 {
            bv.set(i);
        }

        // Clear a range
        bv.set_range(40, 60, false);

        // Check cleared range
        for i in 40..60 {
            assert!(!bv.get(i), "Bit {} should be cleared", i);
        }

        // Check outside range is still set
        assert!(bv.get(39), "Bit 39 should still be set");
        assert!(bv.get(60), "Bit 60 should still be set");
    }

    /// ⭐ NEW TEST: Test set_range spanning multiple words
    #[test]
    fn test_set_range_multiple_words() {
        let bv = BitVec::new(200).unwrap();
        bv.set_range(50, 150, true);

        assert_eq!(bv.count_ones(), 100);

        // Check boundaries
        assert!(!bv.get(49));
        assert!(bv.get(50));
        assert!(bv.get(149));
        assert!(!bv.get(150));
    }

    /// ⭐ NEW TEST: Test set_range with empty range
    #[test]
    fn test_set_range_empty() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(50, 50, true); // Empty range

        assert_eq!(bv.count_ones(), 0);
    }

    /// ⭐ NEW TEST: C6 FIX - Test set_range out of bounds (should panic)
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_set_range_out_of_bounds() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(50, 150, true); // 💥 Should panic: end=150 > len=100
    }

    /// ⭐ NEW TEST: C6 FIX - Test set_range with start > end (should panic)
    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn test_set_range_invalid_range() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(60, 50, true); // 💥 Should panic: start > end
    }

    /// ⭐ NEW TEST: Test set_range at boundaries
    #[test]
    fn test_set_range_boundaries() {
        let bv = BitVec::new(100).unwrap();

        // Range starting at 0
        bv.set_range(0, 10, true);
        assert!(bv.get(0));
        assert!(bv.get(9));
        assert!(!bv.get(10));

        // Range ending at len
        bv.set_range(90, 100, true);
        assert!(bv.get(90));
        assert!(bv.get(99));
    }

    /// ⭐ NEW TEST: Test get_range basic functionality
    #[test]
    fn test_get_range_basic() {
        let bv = BitVec::new(100).unwrap();
        bv.set(10);
        bv.set(11);
        bv.set(13);

        let range = bv.get_range(10, 15);
        assert_eq!(range, vec![true, true, false, true, false]);
    }

    /// ⭐ NEW TEST: Test get_range empty range
    #[test]
    fn test_get_range_empty() {
        let bv = BitVec::new(100).unwrap();
        let range = bv.get_range(50, 50);
        assert_eq!(range, Vec::<bool>::new());
    }

    /// ⭐ NEW TEST: C6 FIX - Test get_range out of bounds (should panic)
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_range_out_of_bounds() {
        let bv = BitVec::new(100).unwrap();
        let _ = bv.get_range(50, 150); // 💥 Should panic
    }

    /// ⭐ NEW TEST: C6 FIX - Test get_range with start > end (should panic)
    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn test_get_range_invalid_range() {
        let bv = BitVec::new(100).unwrap();
        let _ = bv.get_range(60, 50); // 💥 Should panic
    }

    #[test]
    fn test_union() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(64).unwrap();

        bv1.set(10);
        bv1.set(20);
        bv2.set(20);
        bv2.set(30);

        let union = bv1.union(&bv2).unwrap();
        assert!(union.get(10));
        assert!(union.get(20));
        assert!(union.get(30));
        assert!(!union.get(40));
        assert_eq!(union.count_ones(), 3);
    }

    #[test]
    fn test_union_size_mismatch() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(128).unwrap();
        assert!(bv1.union(&bv2).is_err());
    }

    #[test]
    fn test_intersect() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(64).unwrap();

        bv1.set(10);
        bv1.set(20);
        bv1.set(30);
        bv2.set(20);
        bv2.set(30);
        bv2.set(40);

        let intersection = bv1.intersect(&bv2).unwrap();
        assert!(!intersection.get(10));
        assert!(intersection.get(20));
        assert!(intersection.get(30));
        assert!(!intersection.get(40));
        assert_eq!(intersection.count_ones(), 2);
    }

    #[test]
    fn test_intersect_size_mismatch() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(128).unwrap();
        assert!(bv1.intersect(&bv2).is_err());
    }

    #[test]
    fn test_clone() {
        let bv1 = BitVec::new(64).unwrap();
        bv1.set(10);
        bv1.set(20);

        let bv2 = bv1.clone();
        assert!(bv2.get(10));
        assert!(bv2.get(20));

        // Verify independence
        bv1.set(30);
        assert!(bv1.get(30));
        assert!(!bv2.get(30));
    }

    #[test]
    fn test_memory_usage() {
        let bv = BitVec::new(1000).unwrap();
        let mem = bv.memory_usage();

        // At least 16 blocks * 8 bytes per block
        assert!(mem >= 128);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_set_out_of_bounds() {
        let bv = BitVec::new(64).unwrap();
        bv.set(64);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let bv = BitVec::new(64).unwrap();
        let _ = bv.get(100);
    }

    #[test]
    fn test_boundary_conditions() {
        let bv = BitVec::new(65).unwrap();

        // Test first bit of first block
        bv.set(0);
        assert!(bv.get(0));

        // Test last bit of first block
        bv.set(63);
        assert!(bv.get(63));

        // Test first bit of second block
        bv.set(64);
        assert!(bv.get(64));

        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let bv = Arc::new(BitVec::new(1000).unwrap());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let bv = Arc::clone(&bv);
                thread::spawn(move || {
                    for j in 0..250 {
                        bv.set(i * 250 + j);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(bv.count_ones(), 1000);
    }

    #[test]
    fn test_from_raw_valid() {
        let blocks = vec![0xFFu64, 0x00u64];
        let bv = BitVec::from_raw(blocks, 128).unwrap();
        assert_eq!(bv.len(), 128);
        assert!(bv.get(0));
        assert!(bv.get(7));
        assert!(!bv.get(64));
    }

    #[test]
    fn test_from_raw_zero_len_error() {
        let blocks = vec![0u64];
        let result = BitVec::from_raw(blocks, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_raw_insufficient_blocks_error() {
        let blocks = vec![0u64];
        let result = BitVec::from_raw(blocks, 128); // Needs 2 blocks
        assert!(result.is_err());
    }

    #[test]
    fn test_to_raw() {
        let bv = BitVec::new(64).unwrap();
        bv.set(0);
        bv.set(63);

        let raw = bv.to_raw();
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0] & 1, 1);
        assert_eq!(raw[0] & (1u64 << 63), 1u64 << 63);
    }

    /// ⭐ NEW TEST: Comprehensive range operations test
    #[test]
    fn test_range_operations_comprehensive() {
        let bv = BitVec::new(256).unwrap();

        // Set multiple ranges
        bv.set_range(0, 10, true);
        bv.set_range(50, 100, true);
        bv.set_range(200, 250, true);

        // Verify total count
        assert_eq!(bv.count_ones(), 10 + 50 + 50);

        // Clear a range
        bv.set_range(75, 85, false);
        assert_eq!(bv.count_ones(), 10 + 50 + 50 - 10);

        // Verify boundaries
        assert!(bv.get(74));
        assert!(!bv.get(75));
        assert!(!bv.get(84));
        assert!(bv.get(85));
    }
}
