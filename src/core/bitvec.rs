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
//! `count_ones` uses `Relaxed` because it is a non-atomic point-in-time snapshot.
//! Concurrent modifications may or may not be reflected. Callers that require a
//! consistent view must provide external synchronisation.
//!
//! `union_inplace` and `intersect_inplace` write with `Release` so concurrent
//! `Acquire` loads from `get` observe the updated bits.
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
//! - `set_range`: O(⌈range/64⌉) - one atomic op per affected word
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

// ── Internal helper ──────────────────────────────────────────────────────────

/// Build a bitmask covering bits `[lo, hi)` within a single 64-bit word.
///
/// `lo` is inclusive, `hi` is exclusive. Both must satisfy `0 <= lo < hi <= 64`.
///
/// # Examples
///
/// - `word_mask(0, 4)`  → `0b0000_1111`
/// - `word_mask(2, 6)`  → `0b0011_1100`
/// - `word_mask(0, 64)` → `u64::MAX`
#[inline]
fn word_mask(lo: usize, hi: usize) -> u64 {
    debug_assert!(lo < hi && hi <= 64, "word_mask: invalid range [{lo},{hi})");
    let count = hi - lo;
    if count == 64 {
        u64::MAX
    } else {
        ((1u64 << count) - 1) << lo
    }
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
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the bit vector is empty.
    ///
    /// Since `new` requires `num_bits > 0`, this always returns `false` for a
    /// successfully constructed `BitVec`. Provided for API completeness.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set a bit to 1 atomically (thread-safe).
    ///
    /// Uses atomic fetch-or with `Ordering::Release` to ensure visibility of this
    /// write to threads performing `Acquire` loads. Idempotent.
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

        self.blocks[block_idx].fetch_or(mask, Ordering::Release);
    }

    /// Get a bit value atomically (thread-safe).
    ///
    /// Uses `Ordering::Acquire` to synchronize with `Release` stores from `set`,
    /// preventing false negatives in concurrent insert/query scenarios.
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

        (self.blocks[block_idx].load(Ordering::Acquire) & mask) != 0
    }

    /// Set a range of bits to a specific value using word-level atomic operations.
    ///
    /// Sets all bits in `[start, end)` to `value`. This is O(⌈range/64⌉) —
    /// one atomic operation per affected 64-bit word, not one per bit.
    ///
    /// # Arguments
    ///
    /// * `start` - Start index (inclusive)
    /// * `end`   - End index (exclusive)
    /// * `value` - `true` sets bits to 1; `false` clears bits to 0
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > self.len()`.
    ///
    /// # Thread Safety
    ///
    /// Each word update is atomic (`Release`), but the range as a whole is not.
    /// Concurrent readers may observe partial updates across word boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(200).unwrap();
    /// bv.set_range(10, 20, true);
    ///
    /// assert!(bv.get(10));
    /// assert!(bv.get(19));
    /// assert!(!bv.get(9));
    /// assert!(!bv.get(20));
    ///
    /// // Crossing a 64-bit word boundary
    /// bv.set_range(60, 70, true);
    /// assert!(bv.get(63));
    /// assert!(bv.get(64));
    /// ```
    pub fn set_range(&self, start: usize, end: usize, value: bool) {
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
        if start == end {
            return;
        }

        let start_word = start / 64;
        let end_word   = (end - 1) / 64; // inclusive

        // FIX: build a word-level mask per affected word and apply in a single
        // atomic op. The original fired one fetch_or/fetch_and per bit (O(n)),
        // which is O(64×) slower than the O(n/64) approach below.
        for word_idx in start_word..=end_word {
            let bit_lo = if word_idx == start_word { start % 64 } else { 0 };
            let bit_hi = if word_idx == end_word   { (end - 1) % 64 + 1 } else { 64 };

            let mask = word_mask(bit_lo, bit_hi);

            if value {
                self.blocks[word_idx].fetch_or(mask, Ordering::Release);
            } else {
                self.blocks[word_idx].fetch_and(!mask, Ordering::Release);
            }
        }
    }

    /// Clear a single bit to 0 atomically (thread-safe).
    ///
    /// Uses atomic fetch-and with `Ordering::Release`.
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
        let mask = !(1u64 << bit_offset);

        self.blocks[block_idx].fetch_and(mask, Ordering::Release);
    }

    /// Get a range of bits as a `Vec<bool>`.
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > self.len()`.
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
    /// `&mut self` guarantees no concurrent access; `Relaxed` ordering is sufficient.
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
            // Relaxed is safe: `&mut self` guarantees exclusive access.
            block.store(0, Ordering::Relaxed);
        }
    }

    /// Count the number of bits set to 1.
    ///
    /// Uses the CPU's POPCNT instruction via `u64::count_ones()`.
    ///
    /// # Consistency Note
    ///
    /// This is a **non-atomic point-in-time snapshot** using `Ordering::Relaxed`.
    /// Concurrent [`set`](Self::set) or [`union_inplace`](Self::union_inplace)
    /// operations may or may not be reflected in the result. Callers that require
    /// a consistent count must provide external synchronisation.
    ///
    /// # Time Complexity
    ///
    /// O(⌈len/64⌉)
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
        // FIX: was Ordering::Acquire. Acquire establishes a happens-before
        // relationship with a specific Release store, which is unnecessary and
        // wasteful here — count_ones is a snapshot, not a synchronisation point.
        // Relaxed is correct and avoids a memory fence on every word.
        self.blocks
            .iter()
            .map(|block| block.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Get total memory usage in bytes.
    ///
    /// Includes storage for atomic blocks plus the struct itself.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<AtomicU64>() + std::mem::size_of::<Self>()
    }

    /// Compute the union of two bit vectors (bitwise OR), returning a new `BitVec`.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] if lengths differ.
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

    /// Compute the intersection of two bit vectors (bitwise AND), returning a new `BitVec`.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] if lengths differ.
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

    /// Reconstruct a `BitVec` from raw `u64` words (for deserialization).
    ///
    /// # Errors
    ///
    /// - `raw` is empty
    /// - `len` is 0
    /// - `raw` does not contain enough words for `len` bits
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::BitVec;
    ///
    /// let original = BitVec::new(128).unwrap();
    /// original.set(42);
    ///
    /// let raw = original.to_raw();
    /// let len = original.len();
    /// let restored = BitVec::from_raw(raw, len).unwrap();
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

        let required_blocks = (len + 63) / 64;
        if raw.len() < required_blocks {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Insufficient blocks: need {} for {} bits, got {}",
                required_blocks, len, raw.len()
            )));
        }

        let blocks: Box<[AtomicU64]> = raw
            .into_iter()
            .map(AtomicU64::new)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self { blocks, len })
    }

    /// Convert bit vector to raw `u64` words for serialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::BitVec;
    ///
    /// let bits = BitVec::new(128).unwrap();
    /// bits.set(5);
    /// bits.set(100);
    ///
    /// let raw = bits.to_raw();
    /// assert_eq!(raw.len(), 2); // 128 bits = 2 × 64-bit words
    /// ```
    #[must_use]
    pub fn to_raw(&self) -> Vec<u64> {
        self.blocks
            .iter()
            .map(|block| block.load(Ordering::Relaxed))
            .collect()
    }

    /// Number of backing `AtomicU64` words.
    ///
    /// Internal use only. Not part of the stable public API.
    #[must_use]
    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Alias for `num_blocks`. Internal use only.
    #[inline]
    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Atomically zero a single backing word by word index.
    ///
    /// Uses `Release` ordering so concurrent `Acquire` loads from [`get`](Self::get)
    /// observe the cleared word.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_blocks()`, consistent with all other indexing in
    /// this type. The original silently ignored out-of-bounds indices, which
    /// masked programmer errors in callers.
    ///
    /// # Internal Use
    ///
    /// Intended for concurrent-clear paths in `StandardBloomFilter` and
    /// `ShardedBloomFilter` where individual word resets are needed without
    /// `&mut self`. External callers should use [`clear`](Self::clear) instead.
    #[inline]
    pub(crate) fn clear_block_atomic(&self, index: usize) {
        assert!(
            index < self.blocks.len(),
            "BitVec::clear_block_atomic: word index {} out of bounds (num_blocks={})",
            index,
            self.blocks.len()
        );
        self.blocks[index].store(0, Ordering::Release);
    }

    /// Software prefetch hint — bring the cache line containing bit `index` into L1.
    ///
    /// On x86_64, emits `PREFETCHT0`. A no-op on other architectures.
    ///
    /// # Internal Use
    ///
    /// Used by `StandardBloomFilter::contains` hot paths to pipeline memory latency.
    // FIX: was `pub`. This is a micro-optimisation detail for filter hot paths,
    // not a contract we want to expose or stabilise.
    #[inline]
    pub(crate) fn prefetch(&self, index: usize) {
        if index >= self.len() {
            return;
        }

        let block_idx = index / 64;
        if block_idx >= self.blocks.len() {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let ptr = &self.blocks[block_idx] as *const _ as *const i8;
                _mm_prefetch(ptr, _MM_HINT_T0);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = self.blocks[block_idx].load(Ordering::Relaxed);
        }
    }

    /// OR `other` into `self` in-place (zero allocation).
    ///
    /// # Memory Ordering
    ///
    /// Each word is written with `Ordering::Release` so concurrent [`get`](Self::get)
    /// calls (which use `Acquire`) observe the updated bits. Reads from `other` use
    /// `Relaxed` — callers must ensure `other` is fully written before calling this.
    ///
    /// # Errors
    ///
    /// Returns error if sizes don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bitvec1 = BitVec::new(1000).unwrap();
    /// let bitvec2 = BitVec::new(1000).unwrap();
    ///
    /// bitvec1.set(10);
    /// bitvec2.set(20);
    ///
    /// bitvec1.union_inplace(&bitvec2).unwrap();
    ///
    /// assert!(bitvec1.get(10));
    /// assert!(bitvec1.get(20));
    /// ```
    pub fn union_inplace(&self, other: &Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "BitVec size mismatch: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let other_val = other.blocks[i].load(Ordering::Relaxed);
            // FIX: was Ordering::Relaxed. Relaxed writes are not guaranteed to be
            // visible to threads performing Acquire loads (e.g. concurrent `get`
            // calls). Release pairs with the Acquire in `get`, ensuring the updated
            // bits are visible. Commutativity/idempotency of OR is irrelevant to
            // memory visibility — those are value properties, not ordering properties.
            block.fetch_or(other_val, Ordering::Release);
        }

        Ok(())
    }

    /// AND `other` into `self` in-place (zero allocation).
    ///
    /// # Memory Ordering
    ///
    /// Same reasoning as [`union_inplace`](Self::union_inplace): writes use
    /// `Release` to pair with the `Acquire` in `get`.
    ///
    /// # Errors
    ///
    /// Returns error if sizes don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bitvec1 = BitVec::new(1000).unwrap();
    /// let bitvec2 = BitVec::new(1000).unwrap();
    ///
    /// bitvec1.set(10);
    /// bitvec1.set(20);
    /// bitvec2.set(10);
    /// bitvec2.set(30);
    ///
    /// bitvec1.intersect_inplace(&bitvec2).unwrap();
    ///
    /// assert!(bitvec1.get(10));  // In both
    /// assert!(!bitvec1.get(20)); // Only in bitvec1
    /// assert!(!bitvec1.get(30)); // Only in bitvec2
    /// ```
    pub fn intersect_inplace(&self, other: &Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "BitVec size mismatch: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let other_val = other.blocks[i].load(Ordering::Relaxed);
            // FIX: was Ordering::Relaxed. Same issue as union_inplace.
            block.fetch_and(other_val, Ordering::Release);
        }

        Ok(())
    }

    /// XOR two bit vectors, returning a new `BitVec` (symmetric difference).
    ///
    /// # Errors
    ///
    /// Returns error if sizes don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bitvec1 = BitVec::new(1000).unwrap();
    /// let bitvec2 = BitVec::new(1000).unwrap();
    ///
    /// bitvec1.set(10);
    /// bitvec1.set(20);
    /// bitvec2.set(10);
    /// bitvec2.set(30);
    ///
    /// let xor_result = bitvec1.xor(&bitvec2).unwrap();
    ///
    /// assert!(!xor_result.get(10)); // In both (cancels out)
    /// assert!(xor_result.get(20));  // Only in bitvec1
    /// assert!(xor_result.get(30));  // Only in bitvec2
    /// ```
    pub fn xor(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "BitVec size mismatch: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        let result = self.clone();

        for (i, block) in result.blocks.iter().enumerate() {
            let other_val = other.blocks[i].load(Ordering::Relaxed);
            let current   = block.load(Ordering::Relaxed);
            block.store(current ^ other_val, Ordering::Relaxed);
        }

        Ok(result)
    }

        // ── Relational predicates ─────────────────────────────────────────────────

    /// Fraction of bits currently set to 1.
    ///
    /// Returns a value in `[0.0, 1.0]`. This is the fill rate of the underlying
    /// bit array. A value above `0.5` indicates the filter is significantly over
    /// its design capacity.
    ///
    /// # Performance
    ///
    /// O(⌈len/64⌉) — same cost as [`count_ones`](Self::count_ones).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let bv = BitVec::new(64).unwrap();
    /// assert_eq!(bv.fill_rate(), 0.0);
    ///
    /// for i in 0..32 { bv.set(i); }
    /// assert!((bv.fill_rate() - 0.5).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        self.count_ones() as f64 / self.len() as f64
    }

    /// Returns `true` if every bit set in `self` is also set in `other`.
    ///
    /// Formally: `(self AND NOT other) == 0`. Equivalently, the set of positions
    /// set in `self` is a subset of those set in `other`.
    ///
    /// # Filter Semantics
    ///
    /// In Bloom filter terms, `A.is_subset_of(B)` means every item whose hash
    /// indices are all set in A also has all those indices set in B. It does NOT
    /// imply A was derived from B — both could have been independently populated.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] if `self.len() != other.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let a = BitVec::new(64).unwrap();
    /// let b = BitVec::new(64).unwrap();
    /// a.set(10);
    /// b.set(10); b.set(20);
    ///
    /// assert!(a.is_subset_of(&b).unwrap());
    /// assert!(!b.is_subset_of(&a).unwrap());
    /// ```
    pub fn is_subset_of(&self, other: &Self) -> Result<bool> {
        if self.len() != other.len() {
            return Err(BloomCraftError::incompatible_filters(format!(
                "BitVec size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }
        for (a, b) in self.blocks.iter().zip(other.blocks.iter()) {
            let av = a.load(Ordering::Relaxed);
            let bv = b.load(Ordering::Relaxed);
            // Any bit set in `a` but not in `b` violates subset.
            if av & !bv != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns `true` if `self` and `other` share no set bits.
    ///
    /// Formally: `(self AND other) == 0`.
    ///
    /// # Filter Semantics
    ///
    /// Disjoint Bloom filters represent item sets that share no hash-index
    /// collisions. True disjointness in item-space is strictly stronger than
    /// bit-level disjointness, but bit-level disjointness is a necessary
    /// condition for item-level disjointness.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] if `self.len() != other.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let a = BitVec::new(64).unwrap();
    /// let b = BitVec::new(64).unwrap();
    /// a.set(10);
    /// b.set(20);
    /// assert!(a.is_disjoint(&b).unwrap());
    ///
    /// b.set(10);
    /// assert!(!a.is_disjoint(&b).unwrap());
    /// ```
    pub fn is_disjoint(&self, other: &Self) -> Result<bool> {
        if self.len() != other.len() {
            return Err(BloomCraftError::incompatible_filters(format!(
                "BitVec size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }
        for (a, b) in self.blocks.iter().zip(other.blocks.iter()) {
            if a.load(Ordering::Relaxed) & b.load(Ordering::Relaxed) != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Compute the Jaccard similarity index between two bit vectors.
    ///
    /// Defined as \( J(A, B) = |A \cap B| / |A \cup B| \). Returns a value in
    /// `[0.0, 1.0]`:
    /// - `1.0` — identical bit patterns
    /// - `0.0` — no bits in common (disjoint)
    ///
    /// Useful for estimating how similar two Bloom filters are before deciding
    /// whether to merge them (e.g., in distributed cache pre-warming scenarios).
    ///
    /// # Edge Case
    ///
    /// Both vectors all-zero → returns `1.0` (vacuously identical).
    ///
    /// # Memory Ordering
    ///
    /// Uses `Relaxed` loads. Callers that require a consistent snapshot across
    /// concurrent mutations must provide external synchronisation.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] if `self.len() != other.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let a = BitVec::new(64).unwrap();
    /// let b = BitVec::new(64).unwrap();
    /// a.set(10); a.set(20);
    /// b.set(10); b.set(30);
    ///
    /// // |A ∩ B| = 1 ({10}), |A ∪ B| = 3 ({10, 20, 30})
    /// let sim = a.jaccard_similarity(&b).unwrap();
    /// assert!((sim - 1.0 / 3.0).abs() < 1e-10);
    /// ```
    pub fn jaccard_similarity(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(BloomCraftError::incompatible_filters(format!(
                "BitVec size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }
        let mut intersection: u64 = 0;
        let mut union: u64 = 0;
        for (a, b) in self.blocks.iter().zip(other.blocks.iter()) {
            let av = a.load(Ordering::Relaxed);
            let bv = b.load(Ordering::Relaxed);
            intersection += (av & bv).count_ones() as u64;
            union += (av | bv).count_ones() as u64;
        }
        if union == 0 {
            return Ok(1.0);
        }
        Ok(intersection as f64 / union as f64)
    }
}

impl Clone for BitVec {
    /// Creates an independent copy. Modifications to the clone do not affect
    /// the original.
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

// ── PartialEq / Eq ───────────────────────────────────────────────────────────

impl PartialEq for BitVec {
    /// Two `BitVec`s are equal iff they have the same length and identical bit patterns.
    ///
    /// Uses `Relaxed` loads. For a consistent comparison across concurrent writes,
    /// provide external synchronisation.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::core::bitvec::BitVec;
    ///
    /// let a = BitVec::new(64).unwrap();
    /// let b = BitVec::new(64).unwrap();
    /// a.set(5);
    /// b.set(5);
    /// assert_eq!(a, b);
    ///
    /// b.set(6);
    /// assert_ne!(a, b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.blocks
            .iter()
            .zip(other.blocks.iter())
            .all(|(a, b)| a.load(Ordering::Relaxed) == b.load(Ordering::Relaxed))
    }
}

impl Eq for BitVec {}

// ── Serde ────────────────────────────────────────────────────────────────────

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
                let mut len: Option<usize>            = None;

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

                let blocks_data = blocks_data
                    .ok_or_else(|| de::Error::missing_field("blocks"))?;
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

// ── Tests ────────────────────────────────────────────────────────────────────

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
        assert!(BitVec::new(0).is_err());
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

    // ── set_range ────────────────────────────────────────────────────────────

    #[test]
    fn test_set_range_basic() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(10, 20, true);

        for i in 10..20 {
            assert!(bv.get(i), "Bit {} should be set", i);
        }
        assert!(!bv.get(9),  "Bit 9 should not be set");
        assert!(!bv.get(20), "Bit 20 should not be set");
    }

    #[test]
    fn test_set_range_clear() {
        let bv = BitVec::new(100).unwrap();
        for i in 0..100 {
            bv.set(i);
        }
        bv.set_range(40, 60, false);
        for i in 40..60 {
            assert!(!bv.get(i), "Bit {} should be cleared", i);
        }
        assert!(bv.get(39), "Bit 39 should still be set");
        assert!(bv.get(60), "Bit 60 should still be set");
    }

    #[test]
    fn test_set_range_multiple_words() {
        let bv = BitVec::new(200).unwrap();
        bv.set_range(50, 150, true);
        assert_eq!(bv.count_ones(), 100);
        assert!(!bv.get(49));
        assert!(bv.get(50));
        assert!(bv.get(149));
        assert!(!bv.get(150));
    }

    /// Verify set_range is O(n/64): a 64-bit range should fire exactly 1 atomic op,
    /// not 64. This test validates correctness of the word-level path.
    #[test]
    fn test_set_range_word_boundary_crossing() {
        // Spans word 0 bits [60,64) and word 1 bits [0,6): total 10 bits, 2 words
        let bv = BitVec::new(200).unwrap();
        bv.set_range(60, 70, true);

        assert!(!bv.get(59));
        assert!(bv.get(60));
        assert!(bv.get(63));
        assert!(bv.get(64));
        assert!(bv.get(69));
        assert!(!bv.get(70));
        assert_eq!(bv.count_ones(), 10);
    }

    #[test]
    fn test_set_range_exactly_one_full_word() {
        let bv = BitVec::new(128).unwrap();
        bv.set_range(64, 128, true);
        for i in 0..64   { assert!(!bv.get(i)); }
        for i in 64..128 { assert!(bv.get(i));  }
        assert_eq!(bv.count_ones(), 64);
    }

    #[test]
    fn test_set_range_empty() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(50, 50, true);
        assert_eq!(bv.count_ones(), 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_set_range_out_of_bounds() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(50, 150, true);
    }

    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn test_set_range_invalid_range() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(60, 50, true);
    }

    #[test]
    fn test_set_range_boundaries() {
        let bv = BitVec::new(100).unwrap();
        bv.set_range(0, 10, true);
        assert!(bv.get(0));
        assert!(bv.get(9));
        assert!(!bv.get(10));

        bv.set_range(90, 100, true);
        assert!(bv.get(90));
        assert!(bv.get(99));
    }

    // ── get_range ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_range_basic() {
        let bv = BitVec::new(100).unwrap();
        bv.set(10);
        bv.set(11);
        bv.set(13);
        let range = bv.get_range(10, 15);
        assert_eq!(range, vec![true, true, false, true, false]);
    }

    #[test]
    fn test_get_range_empty() {
        let bv = BitVec::new(100).unwrap();
        let range = bv.get_range(50, 50);
        assert_eq!(range, Vec::<bool>::new());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_range_out_of_bounds() {
        let bv = BitVec::new(100).unwrap();
        let _ = bv.get_range(50, 150);
    }

    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn test_get_range_invalid_range() {
        let bv = BitVec::new(100).unwrap();
        let _ = bv.get_range(60, 50);
    }

    // ── clear_block_atomic ────────────────────────────────────────────────────

    #[test]
    fn test_clear_block_atomic_zeros_word() {
        let bv = BitVec::new(128).unwrap();
        bv.set_range(0, 64, true);
        assert_eq!(bv.count_ones(), 64);
        bv.clear_block_atomic(0);
        assert_eq!(bv.count_ones(), 0);
    }

    /// Verify clear_block_atomic panics on OOB — consistent with set/get/clear_bit.
    /// The original silently ignored OOB indices.
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_clear_block_atomic_oob_panics() {
        let bv = BitVec::new(64).unwrap(); // 1 word (index 0 only)
        bv.clear_block_atomic(1);          // index 1 does not exist
    }

    // ── union / intersect ─────────────────────────────────────────────────────

    #[test]
    fn test_union() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(64).unwrap();

        bv1.set(10); bv1.set(20);
        bv2.set(20); bv2.set(30);

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

        bv1.set(10); bv1.set(20); bv1.set(30);
        bv2.set(20); bv2.set(30); bv2.set(40);

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

    // ── union_inplace / intersect_inplace ─────────────────────────────────────

    #[test]
    fn test_union_inplace() {
        let bv1 = BitVec::new(1000).unwrap();
        let bv2 = BitVec::new(1000).unwrap();
        bv1.set(10); bv2.set(20);

        bv1.union_inplace(&bv2).unwrap();
        assert!(bv1.get(10) && bv1.get(20));
    }

    #[test]
    fn test_union_inplace_size_mismatch() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(128).unwrap();
        assert!(bv1.union_inplace(&bv2).is_err());
    }

    #[test]
    fn test_intersect_inplace() {
        let bv1 = BitVec::new(1000).unwrap();
        let bv2 = BitVec::new(1000).unwrap();
        bv1.set(10); bv1.set(20);
        bv2.set(10); bv2.set(30);

        bv1.intersect_inplace(&bv2).unwrap();
        assert!(bv1.get(10));
        assert!(!bv1.get(20));
        assert!(!bv1.get(30));
    }

    // ── xor ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_xor() {
        let bv1 = BitVec::new(1000).unwrap();
        let bv2 = BitVec::new(1000).unwrap();
        bv1.set(10); bv1.set(20);
        bv2.set(10); bv2.set(30);

        let result = bv1.xor(&bv2).unwrap();
        assert!(!result.get(10)); // in both — cancels
        assert!(result.get(20));  // only in bv1
        assert!(result.get(30));  // only in bv2
    }

    #[test]
    fn test_xor_size_mismatch() {
        let bv1 = BitVec::new(64).unwrap();
        let bv2 = BitVec::new(128).unwrap();
        assert!(bv1.xor(&bv2).is_err());
    }

    // ── clone ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clone() {
        let bv1 = BitVec::new(64).unwrap();
        bv1.set(10);
        bv1.set(20);

        let bv2 = bv1.clone();
        assert!(bv2.get(10));
        assert!(bv2.get(20));

        bv1.set(30);
        assert!(bv1.get(30));
        assert!(!bv2.get(30)); // independent
    }

    // ── memory_usage ──────────────────────────────────────────────────────────

    #[test]
    fn test_memory_usage() {
        let bv = BitVec::new(1000).unwrap();
        assert!(bv.memory_usage() >= 128); // ⌈1000/64⌉ * 8 = 128 bytes minimum
    }

    // ── OOB panics ────────────────────────────────────────────────────────────

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
        let _ = bv.get(64);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_clear_bit_out_of_bounds() {
        let bv = BitVec::new(64).unwrap();
        bv.clear_bit(64);
    }

    // ── from_raw / to_raw ─────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_via_raw() {
        let original = BitVec::new(128).unwrap();
        original.set(5);
        original.set(100);

        let raw = original.to_raw();
        assert_eq!(raw.len(), 2);

        let restored = BitVec::from_raw(raw, original.len()).unwrap();
        assert!(restored.get(5));
        assert!(restored.get(100));
        assert!(!restored.get(6));
    }

    #[test]
    fn test_from_raw_empty_errors() {
        assert!(BitVec::from_raw(vec![], 64).is_err());
    }

    #[test]
    fn test_from_raw_zero_len_errors() {
        assert!(BitVec::from_raw(vec![0u64], 0).is_err());
    }

    #[test]
    fn test_from_raw_insufficient_blocks_errors() {
        // 1 block = 64 bits; requesting 128 bits requires 2 blocks
        assert!(BitVec::from_raw(vec![0u64], 128).is_err());
    }

    // ── Concurrency ───────────────────────────────────────────────────────────

    #[test]
    fn test_concurrent_set_no_lost_writes() {
        use std::sync::Arc;
        use std::thread;

        let bv = Arc::new(BitVec::new(10_000).unwrap());
        let handles: Vec<_> = (0..8u64)
            .map(|t| {
                let bv = Arc::clone(&bv);
                thread::spawn(move || {
                    for i in 0..100 {
                        bv.set((t * 100 + i) as usize);
                    }
                })
            })
            .collect();

        for h in handles { h.join().unwrap(); }
        assert_eq!(bv.count_ones(), 800);
    }

    #[test]
    fn test_concurrent_union_inplace_all_bits_visible() {
        use std::sync::Arc;
        use std::thread;

        let target = Arc::new(BitVec::new(1000).unwrap());
        let handles: Vec<_> = (0..4usize)
            .map(|t| {
                let target = Arc::clone(&target);
                thread::spawn(move || {
                    let src = BitVec::new(1000).unwrap();
                    src.set(t * 100);
                    target.union_inplace(&src).unwrap();
                })
            })
            .collect();

        for h in handles { h.join().unwrap(); }

        assert!(target.get(0));
        assert!(target.get(100));
        assert!(target.get(200));
        assert!(target.get(300));
    }

        // ── fill_rate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fill_rate_empty() {
        let bv = BitVec::new(100).unwrap();
        assert_eq!(bv.fill_rate(), 0.0);
    }

    #[test]
    fn test_fill_rate_half() {
        let bv = BitVec::new(64).unwrap();
        for i in 0..32 { bv.set(i); }
        assert!((bv.fill_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fill_rate_full() {
        let bv = BitVec::new(64).unwrap();
        for i in 0..64 { bv.set(i); }
        assert!((bv.fill_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_rate_consistent_with_count_ones() {
        let bv = BitVec::new(200).unwrap();
        bv.set_range(50, 100, true);
        let expected = 50.0 / 200.0;
        assert!((bv.fill_rate() - expected).abs() < 1e-10);
    }

    // ── is_subset_of ──────────────────────────────────────────────────────────

    #[test]
    fn test_subset_true() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10);
        b.set(10); b.set(20);
        assert!(a.is_subset_of(&b).unwrap());
    }

    #[test]
    fn test_subset_false_extra_bit() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); a.set(30);
        b.set(10); b.set(20);
        assert!(!a.is_subset_of(&b).unwrap());
    }

    #[test]
    fn test_empty_is_subset_of_any() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        b.set(10);
        assert!(a.is_subset_of(&b).unwrap());
    }

    #[test]
    fn test_every_bitvec_is_subset_of_itself() {
        let a = BitVec::new(64).unwrap();
        a.set(5); a.set(10); a.set(63);
        assert!(a.is_subset_of(&a).unwrap());
    }

    #[test]
    fn test_subset_size_mismatch_errors() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(128).unwrap();
        assert!(a.is_subset_of(&b).is_err());
    }

    #[test]
    fn test_subset_after_union() {
        // After a.union_inplace(b), a must be a superset of b.
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        b.set(10); b.set(20);
        a.union_inplace(&b).unwrap();
        assert!(b.is_subset_of(&a).unwrap());
    }

    // ── is_disjoint ───────────────────────────────────────────────────────────

    #[test]
    fn test_disjoint_true() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); b.set(20);
        assert!(a.is_disjoint(&b).unwrap());
    }

    #[test]
    fn test_disjoint_false_shared_bit() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); a.set(20);
        b.set(20); b.set(30);
        assert!(!a.is_disjoint(&b).unwrap());
    }

    #[test]
    fn test_empty_is_disjoint_with_anything() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        b.set(10); b.set(20);
        assert!(a.is_disjoint(&b).unwrap());
    }

    #[test]
    fn test_disjoint_after_intersect_clears_both() {
        // intersection of two disjoint sets is empty, so the result is disjoint from both.
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); b.set(20);
        let result = a.intersect(&b).unwrap();
        assert!(result.is_disjoint(&a).unwrap());
        assert!(result.is_disjoint(&b).unwrap());
    }

    #[test]
    fn test_disjoint_size_mismatch_errors() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(128).unwrap();
        assert!(a.is_disjoint(&b).is_err());
    }

    // ── jaccard_similarity ────────────────────────────────────────────────────

    #[test]
    fn test_jaccard_both_empty_is_one() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        assert_eq!(a.jaccard_similarity(&b).unwrap(), 1.0);
    }

    #[test]
    fn test_jaccard_identical_is_one() {
        let a = BitVec::new(64).unwrap();
        a.set(10); a.set(20); a.set(30);
        let b = a.clone();
        assert!((a.jaccard_similarity(&b).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_disjoint_is_zero() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); b.set(20);
        assert_eq!(a.jaccard_similarity(&b).unwrap(), 0.0);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); a.set(20);
        b.set(10); b.set(30);
        // |A ∩ B| = 1, |A ∪ B| = 3
        let sim = a.jaccard_similarity(&b).unwrap();
        assert!((sim - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_commutative() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10); a.set(20);
        b.set(20); b.set(30);
        let j_ab = a.jaccard_similarity(&b).unwrap();
        let j_ba = b.jaccard_similarity(&a).unwrap();
        assert!((j_ab - j_ba).abs() < 1e-15);
    }

    #[test]
    fn test_jaccard_size_mismatch_errors() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(128).unwrap();
        assert!(a.jaccard_similarity(&b).is_err());
    }

    #[test]
    fn test_jaccard_subset_implies_gt_zero() {
        // If a ⊂ b and a is non-empty, J(a,b) = |a| / |b| > 0
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(10);
        b.set(10); b.set(20); b.set(30);
        let sim = a.jaccard_similarity(&b).unwrap();
        assert!((sim - 1.0 / 3.0).abs() < 1e-10);
    }

    // ── PartialEq / Eq ────────────────────────────────────────────────────────

    #[test]
    fn test_partial_eq_identical_bits() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(5); b.set(5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_partial_eq_different_bits() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(5); b.set(6);
        assert_ne!(a, b);
    }

    #[test]
    fn test_partial_eq_different_lengths() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(128).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_eq_reflexive() {
        let a = BitVec::new(64).unwrap();
        a.set(5); a.set(63);
        assert_eq!(a, a);
    }

    #[test]
    fn test_eq_symmetric() {
        let a = BitVec::new(64).unwrap();
        let b = BitVec::new(64).unwrap();
        a.set(5); b.set(5);
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_clone_is_eq() {
        let a = BitVec::new(128).unwrap();
        a.set_range(10, 50, true);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_union_result_ge_operands() {
        // u = a ∪ b  ⇒  a ⊆ u  and  b ⊆ u
        let a = BitVec::new(128).unwrap();
        let b = BitVec::new(128).unwrap();
        a.set(10); a.set(20);
        b.set(20); b.set(30);
        let u = a.union(&b).unwrap();
        assert!(a.is_subset_of(&u).unwrap());
        assert!(b.is_subset_of(&u).unwrap());
    }

    #[test]
    fn test_intersection_result_le_operands() {
        // r = a ∩ b  ⇒  r ⊆ a  and  r ⊆ b
        let a = BitVec::new(128).unwrap();
        let b = BitVec::new(128).unwrap();
        a.set(10); a.set(20);
        b.set(20); b.set(30);
        let r = a.intersect(&b).unwrap();
        assert!(r.is_subset_of(&a).unwrap());
        assert!(r.is_subset_of(&b).unwrap());
    }
}
