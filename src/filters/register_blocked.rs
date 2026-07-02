//! Register-blocked Bloom filter — one cache miss per query.
//!
//! [`RegisterBlockedBloomFilter`] partitions its bit array into 512-bit (64-byte)
//! blocks matching one CPU cache line. Every membership test probes only the block
//! selected by the item's hash, guaranteeing **at most one cache miss per query**
//! regardless of the number of hash functions `k`.
//!
//! # Hash strategy
//!
//! 1. **Block selection**: `block_idx = h1 & (num_blocks - 1)` (power-of-two mask).
//! 2. **Intra-block probes**: Kirsch-Mitzenmacher double hashing with
//!    seed `h1.rotate_left(16) ^ (h2 | 1)` and step `h2 | 1`. Forcing bit 0
//!    guarantees the step is odd and coprime with 512, preventing probe collapse
//!    when the hasher returns `h2 = 0`.
//!
//! # Trade-offs
//!
//! Block-localised probes introduce correlation that raises the empirical FPR
//! relative to a standard Bloom filter. The constructor compensates by targeting
//! a lower internal FPR in its bit-count calculation, bounding the realised FPR
//! at full capacity at the cost of additional memory overhead.
//!
//! # Example
//!
//! ```
//! use bloomcraft::filters::RegisterBlockedBloomFilter;
//! use bloomcraft::core::filter::BloomFilter;
//!
//! let mut f = RegisterBlockedBloomFilter::<u64>::new(100_000, 0.01).unwrap();
//! f.insert(&42);
//! assert!(f.contains(&42));
//! ```
//!
//! # References
//!
//! - Putze, F., Sanders, P., & Singler, J. (2007). Cache-, Hash- and Space-Efficient
//!   Bloom Filters. In *Experimental Algorithms* (SEA 2007), Lecture Notes in Computer
//!   Science, vol 4525. Springer, Berlin, Heidelberg.
//! - Lang, H., Neumann, T., Kemper, A., & Boncz, P. (2019). Performance-Optimal Filtering:
//!   Bloom Overtakes Cuckoo at High Throughput. *Proceedings of the 2019 International
//!   Conference on Management of Data (SIGMOD)*.

use std::hash::Hash;
use std::marker::PhantomData;

use crate::core::filter::{BloomFilter, MutableBloomFilter};
use crate::core::params::{optimal_bit_count, optimal_hash_count};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, StdHasher};

// --- Module-level invariants ---
//
// Every `RegisterBlockedBloomFilter` value satisfies the following at all times:
//
//   INV-1: `num_blocks` is a power of two and >= 1.
//   INV-2: `blocks.len() == num_blocks * BLOCK_SIZE_WORDS`.
//   INV-3: `bits_per_block == BLOCK_SIZE_BITS` (always 512, stored for API surface).
//   INV-4: `k` is in [2, 16].
//   INV-5: `target_fpr` is in (0.0, 1.0) and is finite.
//   INV-6: `item_count` equals the number of completed `insert` calls since
//           construction or the last `clear`.
//
// These invariants are established in `with_hasher` and maintained by all
// `&mut self` methods. `debug_assert!` checks guard entry points of methods
// that rely on them.

// --- Public constants ---

/// Block size in bits: one CPU cache line (64 bytes), one AVX-512 register.
pub const BLOCK_SIZE_BITS: usize = 512;

/// Block size in 64-bit words (`BLOCK_SIZE_BITS / 64`).
pub const BLOCK_SIZE_WORDS: usize = 8;

const _: () = assert!(BLOCK_SIZE_BITS == BLOCK_SIZE_WORDS * 64);
const _: () = assert!(BLOCK_SIZE_BITS.is_power_of_two());

// --- Struct definition ---

/// Register-blocked Bloom filter — one cache miss per query.
///
/// # Thread Safety
///
/// `insert` and `clear` require `&mut self`. For concurrent access, wrap in
/// `Arc<Mutex<RegisterBlockedBloomFilter<T>>>` or use
/// [`AtomicScalableBloomFilter`](crate::filters::AtomicScalableBloomFilter)
/// if scalable growth is also required.
#[derive(Debug, Clone)]
pub struct RegisterBlockedBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone + Default,
{
    /// Flat `num_blocks × 8` u64 backing storage.
    blocks: Vec<u64>,

    /// Number of 512-bit blocks. Always a power of two (INV-1).
    num_blocks: usize,

    /// Always 512. Cached field for `dyn BloomFilter` access.
    bits_per_block: usize,

    /// Hash probes per operation. Clamped to [2, 16] (INV-4).
    k: usize,

    /// The configured hasher instance.
    hasher: H,

    /// The `expected_items` value supplied at construction.
    expected_items: usize,

    /// User-supplied FPR target. Internally scaled by `fpr / 2.5` (INV-5).
    target_fpr: f64,

    /// Insertion counter including duplicates. Not unique-item count (INV-6).
    item_count: usize,

    _phantom: PhantomData<T>,
}

// --- Associated constants ---

impl<T, H> RegisterBlockedBloomFilter<T, H>
where
    H: BloomHasher + Clone + Default,
{
    /// Block size in bits (512 = one cache line).
    pub const BLOCK_SIZE_BITS: usize = BLOCK_SIZE_BITS;

    /// Block size in 64-bit words (8).
    pub const BLOCK_SIZE_WORDS: usize = BLOCK_SIZE_WORDS;
}

// --- Construction ---

impl<T, H> RegisterBlockedBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Create a new filter with the default hasher. Delegates to [`with_hasher`](Self::with_hasher).
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::InvalidItemCount`] if `expected_items == 0`, or
    /// [`BloomCraftError::FalsePositiveRateOutOfBounds`] if `fpr ∉ (0.0, 1.0)`.
    ///
    /// # Example
    ///
    /// ```
    /// use bloomcraft::filters::RegisterBlockedBloomFilter;
    /// use bloomcraft::core::filter::BloomFilter;
    /// let f = RegisterBlockedBloomFilter::<String>::new(10_000, 0.01).unwrap();
    /// assert_eq!(f.expected_items(), 10_000);
    /// ```
    pub fn new(expected_items: usize, fpr: f64) -> Result<Self> {
        Self::with_hasher(expected_items, fpr, H::default())
    }

    /// Create a new filter with an explicit hasher. Prefer [`new`](Self::new) for the default hasher.
    ///
    /// Block count is rounded up to the next power of two for bitmask block
    /// selection. This may allocate more bits than `optimal_bit_count` requires,
    /// lowering the achieved FPR below `target_fpr / 2.5` at full capacity.
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new).
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Result<Self> {
        if expected_items == 0 {
            return Err(BloomCraftError::invalid_item_count(expected_items));
        }
        if fpr <= 0.0 || fpr >= 1.0 {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fpr));
        }

        // Compensate for blocking overhead; empirical FPR ≤ 3× target at capacity.
        let adjusted_fpr = fpr / 2.5;

        let total_bits = optimal_bit_count(expected_items, adjusted_fpr)?;

        // Power-of-two for bitmask block selection.
        let raw_blocks = total_bits.div_ceil(BLOCK_SIZE_BITS);
        let num_blocks = raw_blocks.next_power_of_two().max(1);

        let actual_total_bits = num_blocks * BLOCK_SIZE_BITS;
        let k = optimal_hash_count(actual_total_bits, expected_items)?.clamp(2, 16);

        let total_words = num_blocks * BLOCK_SIZE_WORDS;
        let blocks = vec![0u64; total_words];

        // Verify invariants.
        debug_assert!(num_blocks.is_power_of_two(), "INV-1 violated");
        debug_assert!(num_blocks >= 1, "INV-1 violated");
        debug_assert_eq!(
            blocks.len(),
            num_blocks * BLOCK_SIZE_WORDS,
            "INV-2 violated"
        );
        debug_assert!((2..=16).contains(&k), "INV-4 violated");
        debug_assert!(fpr > 0.0 && fpr < 1.0 && fpr.is_finite(), "INV-5 violated");

        Ok(Self {
            blocks,
            num_blocks,
            bits_per_block: BLOCK_SIZE_BITS,
            k,
            hasher,
            expected_items,
            target_fpr: fpr,
            item_count: 0,
            _phantom: PhantomData,
        })
    }

    // --- Accessors ---

    /// Number of 512-bit blocks. Always a power of two.
    #[inline]
    pub const fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Block size in bits (512). Instance method for `dyn` access.
    #[inline]
    pub const fn bits_per_block(&self) -> usize {
        self.bits_per_block
    }

    /// User-supplied FPR target at construction (not the internal `fpr / 2.5`).
    /// Empirical FPR at capacity ≤ `3 × target_fpr()`.
    #[inline]
    pub const fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    // --- Hash & block access ---

    /// Hash `item` → `(h1, h2)` via the configured hasher. Callers apply `h2 | 1`.
    #[inline]
    fn hash_item(&self, item: &T) -> (u64, u64) {
        self.hasher.hash_item(item)
    }

    /// Select block index from `h1`. Mask, not modulo — `num_blocks` is a power of two.
    #[inline]
    fn hash_to_block(&self, h1: u64) -> usize {
        debug_assert!(self.num_blocks.is_power_of_two(), "INV-1 violated");
        (h1 as usize) & (self.num_blocks - 1)
    }

    /// Read-only access to block `block_idx` (8 u64 words).
    #[inline]
    fn block(&self, block_idx: usize) -> &[u64] {
        debug_assert!(block_idx < self.num_blocks);
        let start = block_idx * BLOCK_SIZE_WORDS;
        &self.blocks[start..start + BLOCK_SIZE_WORDS]
    }

    /// Mutable access to block `block_idx` (8 u64 words).
    #[inline]
    fn block_mut(&mut self, block_idx: usize) -> &mut [u64] {
        debug_assert!(block_idx < self.num_blocks);
        let start = block_idx * BLOCK_SIZE_WORDS;
        &mut self.blocks[start..start + BLOCK_SIZE_WORDS]
    }
}

// --- BloomFilter<T> ---

impl<T, H> BloomFilter<T> for RegisterBlockedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    /// Insert `item` into the filter.
    #[inline]
    fn insert(&mut self, item: &T) {
        let (h1, h2) = self.hash_item(item);
        // h2 | 1 forces an odd KM step, coprime with 512 (see module doc).
        let h2 = h2 | 1;

        let block_idx = self.hash_to_block(h1);
        let k = self.k;
        let block = self.block_mut(block_idx);

        let mut hash = h1.rotate_left(16) ^ h2;
        for _ in 0..k {
            let bit_idx = (hash as usize) & (BLOCK_SIZE_BITS - 1);
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            block[word_idx] |= 1u64 << bit_offset;
            hash = hash.wrapping_add(h2);
        }

        self.item_count += 1;
    }

    /// Query whether `item` may be in the filter.
    ///
    /// Returns `false` when `item` was definitely not inserted; `true` if probably
    /// inserted (false positives are possible). Hash sequence matches [`insert`](Self::insert);
    /// returns early on the first zero bit.
    #[inline]
    fn contains(&self, item: &T) -> bool {
        let (h1, h2) = self.hash_item(item);
        let h2 = h2 | 1;

        let block_idx = self.hash_to_block(h1);
        let block = self.block(block_idx);

        let mut hash = h1.rotate_left(16) ^ h2;
        for _ in 0..self.k {
            let bit_idx = (hash as usize) & (BLOCK_SIZE_BITS - 1);
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;

            if block[word_idx] & (1u64 << bit_offset) == 0 {
                return false;
            }
            hash = hash.wrapping_add(h2);
        }

        true
    }

    /// Zero all bits and reset `item_count`. Filter geometry is unchanged.
    fn clear(&mut self) {
        self.blocks.fill(0u64);
        self.item_count = 0;

        debug_assert_eq!(self.count_set_bits(), 0);
        debug_assert_eq!(self.item_count, 0);
    }

    /// Whether the filter is empty (`item_count == 0`). O(1).
    #[inline]
    fn is_empty(&self) -> bool {
        self.item_count == 0
    }

    /// Insertion counter (including duplicates). See [`estimate_count`](Self::estimate_count) for unique approximation.
    #[inline]
    fn len(&self) -> usize {
        self.item_count
    }

    /// Theoretical FPR at current fill level using the standard Bloom formula.
    ///
    /// ```text
    /// FPR = (1 − exp(−k × n / m))^k
    /// ```
    ///
    /// The formula assumes independent bit positions; blocking introduces
    /// correlation that makes the empirical FPR ~2.5× higher. Construction
    /// over-provisions to keep the empirical FPR ≤ `3 × target_fpr`.
    ///
    /// Returns `0.0` when `item_count == 0`.
    fn false_positive_rate(&self) -> f64 {
        if self.item_count == 0 {
            return 0.0;
        }

        let n = self.item_count as f64;
        let m = self.bit_count() as f64;
        let k = self.k as f64;

        let fill_rate = 1.0_f64 - (-k * n / m).exp();
        fill_rate.powi(self.k as i32)
    }

    /// The `expected_items` value supplied at construction.
    #[inline]
    fn expected_items(&self) -> usize {
        self.expected_items
    }

    /// Total number of bits in the filter: `num_blocks × 512`.
    #[inline]
    fn bit_count(&self) -> usize {
        self.num_blocks * self.bits_per_block
    }

    /// Number of hash probes per operation (`k`). In range `[2, 16]`.
    #[inline]
    fn hash_count(&self) -> usize {
        self.k
    }

    /// Estimate distinct item count via the standard Bloom cardinality estimator.
    ///
    /// ```text
    /// n̂ = −(m / k) × ln(1 − X / m)
    /// ```
    ///
    /// where `X` = set bits, `m = bit_count()`, `k = hash_count()`.
    /// Blocking makes bit distribution non-uniform; empirical error is 10–15%
    /// at 20–70% fill. Clamps `X` to `m - num_blocks` near saturation to avoid `+∞`.
    ///
    /// Returns `0` when no bits are set.
    fn estimate_count(&self) -> usize {
        let set_bits: usize = self.blocks.iter().map(|w| w.count_ones() as usize).sum();

        if set_bits == 0 {
            return 0;
        }

        let m = self.bit_count() as f64;
        let k = self.k as f64;

        // Clamp to prevent log(0) when the filter approaches saturation.
        // We clamp X to (m - num_blocks) rather than (m - 1) because we want
        // no individual block's contribution to the average to reach 512/512,
        // which would make the per-block ln(0) undefined.
        let max_x = m - self.num_blocks as f64;
        let x = (set_bits as f64).min(max_x);

        let estimated = -(m / k) * (1.0_f64 - x / m).ln();
        estimated.max(0.0) as usize
    }

    /// Total set bits across all blocks.
    fn count_set_bits(&self) -> usize {
        self.blocks.iter().map(|w| w.count_ones() as usize).sum()
    }
}

// --- MutableBloomFilter<T> ---

/// Marker: mutation requires `&mut self`. Wrap in `Arc<Mutex<...>>` for concurrent access.
impl<T, H> MutableBloomFilter<T> for RegisterBlockedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    type Filter = RegisterBlockedBloomFilter<u64>;

    // --- Construction: valid inputs ---

    #[test]
    fn construction_succeeds_with_valid_params() {
        let f = Filter::new(1_000, 0.01);
        assert!(f.is_ok(), "expected Ok, got {:?}", f.err());
    }

    #[test]
    fn construction_stores_expected_items_and_target_fpr() {
        let f = Filter::new(5_000, 0.05).unwrap();
        assert_eq!(f.expected_items(), 5_000);
        assert_eq!(f.target_fpr(), 0.05);
    }

    #[test]
    fn construction_with_hasher_produces_identical_structure() {
        let f1 = Filter::new(1_000, 0.01).unwrap();
        let f2 = RegisterBlockedBloomFilter::<u64>::with_hasher(1_000, 0.01, StdHasher::default())
            .unwrap();
        assert_eq!(f1.num_blocks(), f2.num_blocks());
        assert_eq!(f1.hash_count(), f2.hash_count());
        assert_eq!(f1.bit_count(), f2.bit_count());
    }

    // --- Construction: error paths ---

    #[test]
    fn construction_rejects_zero_expected_items() {
        let err = Filter::new(0, 0.01).unwrap_err();
        assert!(
            matches!(err, BloomCraftError::InvalidItemCount { .. }),
            "expected InvalidItemCount, got: {:?}",
            err
        );
    }

    #[test]
    fn construction_rejects_fpr_of_zero() {
        let err = Filter::new(1_000, 0.0).unwrap_err();
        assert!(matches!(
            err,
            BloomCraftError::FalsePositiveRateOutOfBounds { .. }
        ));
    }

    #[test]
    fn construction_rejects_fpr_of_one() {
        let err = Filter::new(1_000, 1.0).unwrap_err();
        assert!(matches!(
            err,
            BloomCraftError::FalsePositiveRateOutOfBounds { .. }
        ));
    }

    #[test]
    fn construction_rejects_fpr_above_one() {
        let err = Filter::new(1_000, 1.5).unwrap_err();
        assert!(matches!(
            err,
            BloomCraftError::FalsePositiveRateOutOfBounds { .. }
        ));
    }

    #[test]
    fn construction_rejects_fpr_negative() {
        let err = Filter::new(1_000, -0.01).unwrap_err();
        assert!(matches!(
            err,
            BloomCraftError::FalsePositiveRateOutOfBounds { .. }
        ));
    }

    // --- Structural invariants ---

    #[test]
    fn num_blocks_is_power_of_two_for_all_capacities() {
        for &cap in &[
            1usize, 2, 10, 99, 100, 1_000, 9_999, 10_000, 100_000, 1_000_000,
        ] {
            let f = Filter::new(cap, 0.01).expect("valid params");
            let nb = f.num_blocks();
            assert!(
                nb.is_power_of_two(),
                "cap={cap}: num_blocks={nb} not a power of two"
            );
            assert!(nb >= 1, "cap={cap}: num_blocks must be >= 1");
        }
    }

    #[test]
    fn bit_count_equals_num_blocks_times_block_size() {
        for &cap in &[100usize, 10_000, 1_000_000] {
            let f = Filter::new(cap, 0.01).unwrap();
            assert_eq!(
                f.bit_count(),
                f.num_blocks() * Filter::BLOCK_SIZE_BITS,
                "bit_count invariant violated for cap={cap}"
            );
        }
    }

    #[test]
    fn backing_vec_length_equals_num_blocks_times_block_words() {
        let f = Filter::new(10_000, 0.01).unwrap();
        assert_eq!(f.blocks.len(), f.num_blocks() * Filter::BLOCK_SIZE_WORDS);
    }

    #[test]
    fn bits_per_block_is_always_512() {
        for &cap in &[100usize, 50_000] {
            let f = Filter::new(cap, 0.01).unwrap();
            assert_eq!(f.bits_per_block(), 512);
        }
    }

    #[test]
    fn hash_count_is_within_clamped_range() {
        for &cap in &[100usize, 10_000, 1_000_000] {
            for &fpr in &[0.5_f64, 0.1, 0.01, 0.001, 0.0001] {
                let f = Filter::new(cap, fpr).unwrap();
                let k = f.hash_count();
                assert!(
                    (2..=16).contains(&k),
                    "k={k} out of [2, 16] for cap={cap}, fpr={fpr}"
                );
            }
        }
    }

    #[test]
    fn associated_constants_match_instance_methods() {
        let f = Filter::new(1_000, 0.01).unwrap();
        assert_eq!(f.bits_per_block(), Filter::BLOCK_SIZE_BITS);
        assert_eq!(Filter::BLOCK_SIZE_BITS, BLOCK_SIZE_BITS);
        assert_eq!(Filter::BLOCK_SIZE_WORDS, BLOCK_SIZE_WORDS);
    }

    // --- Correctness: no false negatives ---

    #[test]
    fn no_false_negatives_at_small_scale() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        let items: Vec<u64> = (0..1_000).collect();
        for &item in &items {
            f.insert(&item);
        }
        for &item in &items {
            assert!(f.contains(&item), "false negative for item={item}");
        }
    }

    #[test]
    fn no_false_negatives_at_capacity() {
        let n = 10_000usize;
        let mut f = Filter::new(n, 0.01).unwrap();
        for i in 0u64..n as u64 {
            f.insert(&i);
        }
        for i in 0u64..n as u64 {
            assert!(f.contains(&i), "false negative at capacity for item={i}");
        }
    }

    #[test]
    fn no_false_negatives_at_large_scale() {
        let n = 50_000usize;
        let mut f = Filter::new(n, 0.001).unwrap();
        for i in 0u64..n as u64 {
            f.insert(&i);
        }
        for i in 0u64..n as u64 {
            assert!(f.contains(&i), "false negative for item={i}");
        }
    }

    // --- Correctness: false positive rate ---

    #[test]
    fn empirical_fpr_within_three_times_target_at_1pct() {
        let n = 10_000usize;
        let target_fpr = 0.01_f64;
        let mut f = Filter::new(n, target_fpr).unwrap();
        for i in 0u64..n as u64 {
            f.insert(&i);
        }
        let probe_count = 10_000usize;
        let false_positives: usize = (n as u64..n as u64 + probe_count as u64)
            .filter(|i| f.contains(i))
            .count();
        let empirical = false_positives as f64 / probe_count as f64;
        assert!(
            empirical < 3.0 * target_fpr,
            "empirical FPR {empirical:.4} exceeds 3× target {target_fpr:.4}"
        );
    }

    #[test]
    fn empirical_fpr_within_three_times_target_at_01pct() {
        let n = 20_000usize;
        let target_fpr = 0.001_f64;
        let mut f = Filter::new(n, target_fpr).unwrap();
        for i in 0u64..n as u64 {
            f.insert(&i);
        }
        let probe_count = 50_000usize;
        let false_positives: usize = (n as u64..n as u64 + probe_count as u64)
            .filter(|i| f.contains(i))
            .count();
        let empirical = false_positives as f64 / probe_count as f64;
        assert!(
            empirical < 3.0 * target_fpr,
            "empirical FPR {empirical:.5} exceeds 3× target {target_fpr:.5}"
        );
    }

    // --- State transitions ---

    #[test]
    fn empty_filter_reports_is_empty_true() {
        let f = Filter::new(1_000, 0.01).unwrap();
        assert!(f.is_empty());
    }

    #[test]
    fn empty_filter_reports_len_zero() {
        let f = Filter::new(1_000, 0.01).unwrap();
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn empty_filter_returns_false_for_all_contains() {
        let f = Filter::new(1_000, 0.01).unwrap();
        for i in 0u64..200 {
            assert!(!f.contains(&i), "empty filter returned true for {i}");
        }
    }

    #[test]
    fn empty_filter_false_positive_rate_is_zero() {
        let f = Filter::new(1_000, 0.01).unwrap();
        assert_eq!(f.false_positive_rate(), 0.0);
    }

    #[test]
    fn insert_transitions_is_empty_to_false() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        f.insert(&1u64);
        assert!(!f.is_empty());
    }

    #[test]
    fn len_equals_number_of_insert_calls() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        for i in 0u64..100 {
            f.insert(&i);
            assert_eq!(f.len(), (i + 1) as usize);
        }
    }

    #[test]
    fn len_counts_duplicate_inserts() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        f.insert(&42u64);
        f.insert(&42u64);
        f.insert(&42u64);
        assert_eq!(f.len(), 3);
        assert!(f.contains(&42u64));
    }

    #[test]
    fn is_empty_and_len_are_consistent_after_insert_and_clear() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
        f.insert(&1u64);
        assert!(!f.is_empty());
        assert_eq!(f.len(), 1);
        f.clear();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn clear_zeroes_all_bits_and_resets_count() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        for i in 0u64..100 {
            f.insert(&i);
        }
        assert_eq!(f.len(), 100);
        assert!(!f.is_empty());
        f.clear();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
        assert_eq!(f.count_set_bits(), 0);
    }

    #[test]
    fn clear_makes_all_previously_inserted_items_not_found() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        let items: Vec<u64> = (0..100).collect();
        for &item in &items {
            f.insert(&item);
        }
        f.clear();
        for &item in &items {
            assert!(!f.contains(&item), "item {item} found after clear");
        }
    }

    #[test]
    fn false_positive_rate_increases_monotonically_with_fill() {
        // This test validates the mathematical property of the FPR formula:
        // (1 - exp(-k*n/m))^k is strictly increasing in n for fixed k and m.
        // It also guards against accidental formula changes.
        let mut f = Filter::new(10_000, 0.01).unwrap();
        let mut prev_fpr = f.false_positive_rate();
        assert_eq!(prev_fpr, 0.0);
        for i in 0u64..10_000 {
            f.insert(&i);
            if i % 1_000 == 999 {
                let fpr = f.false_positive_rate();
                assert!(
                    fpr >= prev_fpr,
                    "FPR decreased from {prev_fpr:.6} to {fpr:.6} at i={i}"
                );
                prev_fpr = fpr;
            }
        }
    }

    #[test]
    fn false_positive_rate_is_zero_after_clear() {
        let mut f = Filter::new(10_000, 0.01).unwrap();
        for i in 0u64..5_000 {
            f.insert(&i);
        }
        assert!(f.false_positive_rate() > 0.0);
        f.clear();
        assert_eq!(f.false_positive_rate(), 0.0);
    }

    // --- Bit-count accounting ---

    #[test]
    fn count_set_bits_zero_on_empty_filter() {
        let f = Filter::new(1_000, 0.01).unwrap();
        assert_eq!(f.count_set_bits(), 0);
    }

    #[test]
    fn count_set_bits_nonzero_after_insert() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        f.insert(&1u64);
        assert!(f.count_set_bits() > 0);
    }

    #[test]
    fn count_set_bits_never_exceeds_bit_count() {
        let mut f = Filter::new(1_000, 0.01).unwrap();
        for i in 0u64..5_000 {
            f.insert(&i);
        }
        assert!(f.count_set_bits() <= f.bit_count());
    }

    // --- estimate_count ---

    #[test]
    fn estimate_count_zero_on_empty_filter() {
        let f = Filter::new(1_000, 0.01).unwrap();
        assert_eq!(f.estimate_count(), 0);
    }

    #[test]
    fn estimate_count_within_50_percent_at_half_capacity() {
        let n = 10_000usize;
        let mut f = Filter::new(n, 0.01).unwrap();
        let insert_count = n / 2;
        for i in 0u64..insert_count as u64 {
            f.insert(&i);
        }
        let estimate = f.estimate_count();
        let lower = insert_count / 2;
        let upper = insert_count * 2;
        assert!(
            estimate >= lower && estimate <= upper,
            "estimate_count {estimate} far from {insert_count} (expected [{lower}, {upper}])"
        );
    }

    // --- Clone ---

    #[test]
    fn clone_produces_independent_instance() {
        let mut original = Filter::new(1_000, 0.01).unwrap();
        original.insert(&10u64);
        original.insert(&20u64);
        let mut cloned = original.clone();
        cloned.insert(&999u64);
        cloned.clear();
        assert_eq!(original.len(), 2);
        assert!(original.contains(&10u64));
        assert!(original.contains(&20u64));
        assert!(cloned.is_empty());
    }

    #[test]
    fn clone_has_identical_structural_params() {
        let original = Filter::new(5_000, 0.005).unwrap();
        let cloned = original.clone();
        assert_eq!(original.num_blocks(), cloned.num_blocks());
        assert_eq!(original.bit_count(), cloned.bit_count());
        assert_eq!(original.hash_count(), cloned.hash_count());
        assert_eq!(original.bits_per_block(), cloned.bits_per_block());
        assert_eq!(original.target_fpr(), cloned.target_fpr());
        assert_eq!(original.expected_items(), cloned.expected_items());
    }

    // --- String type ---

    #[test]
    fn works_with_string_items() {
        let mut f = RegisterBlockedBloomFilter::<String>::new(500, 0.01).unwrap();
        let words = ["hello", "world", "rust", "bloom", "filter"];
        for word in &words {
            f.insert(&word.to_string());
        }
        for word in &words {
            assert!(f.contains(&word.to_string()), "false negative for '{word}'");
        }
    }

    // --- Send + Sync ---

    #[test]
    fn filter_satisfies_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Filter>();
        assert_send_sync::<RegisterBlockedBloomFilter<String>>();
    }

    // --- Hash routing correctness ---

    #[test]
    fn hash_to_block_always_in_range() {
        let f = Filter::new(10_000, 0.01).unwrap();
        let num_blocks = f.num_blocks();
        let test_hashes = [
            0u64,
            1,
            u64::MAX,
            u64::MAX / 2,
            0xDEAD_BEEF_CAFE_BABE,
            0x0101_0101_0101_0101,
            0xFFFF_FFFF_0000_0000,
        ];
        for &h in &test_hashes {
            let idx = f.hash_to_block(h);
            assert!(
                idx < num_blocks,
                "hash_to_block({h:#x}) = {idx} >= num_blocks={num_blocks}"
            );
        }
    }

    #[test]
    fn block_selection_uses_low_bits_of_h1() {
        let f = Filter::new(10_000, 0.01).unwrap();
        let mask = f.num_blocks() - 1;
        for h in 0u64..1_000 {
            let expected = (h as usize) & mask;
            let actual = f.hash_to_block(h);
            assert_eq!(actual, expected, "h={h}");
        }
    }

    // --- h2 = 0 regression: probe collapse guard ---

    /// A mock hasher that always returns `(1, 0)` — used to verify the `h2 | 1`
    /// guard prevents KM probe collapse when `h2 = 0`.
    ///
    /// All items hash to the same block (`h1 = 1`, `block_idx = 1 & mask`) and
    /// would collapse to the same bit without the guard. This is an artificial
    /// pathological case that validates a specific correctness property; it does
    /// not represent realistic hasher behaviour.
    #[derive(Debug, Clone, Default)]
    struct ZeroH2Hasher;

    impl BloomHasher for ZeroH2Hasher {
        fn name(&self) -> &'static str {
            "ZeroH2Hasher"
        }

        fn hash_bytes(&self, _bytes: &[u8]) -> u64 {
            0
        }

        fn hash_item<T: Hash>(&self, _item: &T) -> (u64, u64) {
            (1, 0)
        }
    }

    #[test]
    fn h2_eq_zero_produces_distinct_probes() {
        // Without `h2 | 1`: h2 = 0, step = 0, all k probes land on the same bit.
        // With `h2 | 1`: h2 = 1, step = 1, probes are at positions
        //   seed & 511, (seed+1) & 511, (seed+2) & 511, ..., (seed+k-1) & 511
        // which are k distinct positions (for k <= 512).
        let mut f =
            RegisterBlockedBloomFilter::<u64, ZeroH2Hasher>::with_hasher(100, 0.01, ZeroH2Hasher)
                .unwrap();
        let k = f.hash_count();
        assert!(k >= 2, "k={k} must be >= 2");

        f.insert(&42u64);

        let set = f.count_set_bits();
        assert!(
            set >= 2,
            "probe collapse detected: {set} bit(s) set for k={k}, expected >= 2"
        );
        assert!(
            f.contains(&42u64),
            "item not found after insertion with ZeroH2Hasher"
        );
    }

    // --- Over-capacity insertion ---

    #[test]
    fn over_capacity_insertion_no_false_negatives() {
        let n = 1_000usize;
        let mut f = Filter::new(n, 0.01).unwrap();
        let over = 5 * n;
        for i in 0u64..over as u64 {
            f.insert(&i);
        }
        for i in 0u64..over as u64 {
            assert!(
                f.contains(&i),
                "false negative at over-capacity for item={i}"
            );
        }
    }

    // --- Clear + reinsert ---

    #[test]
    fn clear_and_reinsert_no_false_negatives() {
        let mut f = Filter::new(5_000, 0.01).unwrap();
        let items: Vec<u64> = (0..5_000).collect();
        for &item in &items {
            f.insert(&item);
        }
        f.clear();
        for &item in &items {
            assert!(!f.contains(&item), "item {item} found after clear");
        }
        for &item in &items {
            f.insert(&item);
        }
        for &item in &items {
            assert!(
                f.contains(&item),
                "false negative after reinsert for item={item}"
            );
        }
    }
}
