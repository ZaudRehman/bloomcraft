//! Dynamically growing Bloom filter for unbounded datasets with bounded false positive rates.
//!
//! This module implements the Scalable Bloom Filter described by Almeida et al. (2007),
//! extended with HyperLogLog++ cardinality estimation, adaptive FPR tightening, and
//! production-grade observability.
//!
//! # Algorithm
//!
//! A `ScalableBloomFilter` is a sequence of standard Bloom filters that grows on demand:
//!
//! ```text
//! ScalableBloomFilter = [Filter₀, Filter₁, Filter₂, ..., Filterₗ₋₁]
//!
//! Where:
//!   capacity(Filterᵢ) = initial_capacity × sⁱ       (geometric growth factor s)
//!   FPR(Filterᵢ)      = target_fpr × rⁱ             (tightening ratio r)
//!   compound FPR      ≤ target_fpr / (1 - r)         (convergent series bound)
//!
//! Insert: always into the last filter; grow when fill ratio ≥ fill_threshold
//! Query:  iterate filters in configured order; return true on first positive
//! ```
//!
//! Items are inserted exclusively into the current (last) filter. When that filter's
//! fill ratio reaches `fill_threshold` (default 0.5, the proven optimal per Almeida 2007),
//! a new filter is appended with tighter FPR and larger capacity. Items already in earlier
//! filters are never rehashed or migrated.
//!
//! # Parameter Selection
//!
//! The three parameters that govern correctness and performance are:
//!
//! | Parameter | Field | Recommended | Effect |
//! |-----------|-------|-------------|--------|
//! | Fill threshold (p) | `fill_threshold` | **0.5** | Proven optimal; do not lower below 0.45 without invalidating the FPR bound |
//! | Tightening ratio (r) | `error_ratio` | **0.8–0.9** | r = 0.5 gives integer k increments but wastes ~2× space; r = 0.9 is better for large growth |
//! | Growth factor (s) | `GrowthStrategy` | **Geometric(2.0)** | s = 2 keeps capacities as powers of two; s = 4 halves stage count for very large datasets |
//!
//! The compound FPR is bounded by `target_fpr / (1 - r)`. If you set `r = 0.9` you must
//! set `target_fpr = desired_fpr × 0.1` so the bound converges to `desired_fpr`.
//!
//! # Performance
//!
//! **Hit queries** short-circuit on the first matching filter. With `QueryStrategy::Reverse`
//! (default), the newest filter — where recent items live — is checked first, making
//! recent-item hits O(k).
//!
//! **Miss queries** must traverse all l filters before returning false. The cost grows
//! super-linearly with depth because each successive filter requires more hash evaluations
//! (k₀ + i·log₂(r⁻¹) hashes for filter i, per Almeida 2007). At depth 8 with r = 0.5
//! and k₀ = 7, a definitive miss costs 84 hash evaluations.
//!
//! **Batch operations** use a segment-based loop: `insert_batch` computes how many
//! items fit in the current sub-filter before the next growth trigger, bulk-inserts
//! that segment with a single growth check at the boundary, then grows once and
//! repeats. This eliminates the per-item overhead of `should_grow()`, `last_mut()`,
//! and `filter_nonempty` assignment. Prefer `insert_batch` for bulk ingestion of
//! ≥ 500 items; below that threshold the segmentation overhead is measurable
//! and a plain `insert` loop is equally fast.
//!
//! # Thread Safety
//!
//! `ScalableBloomFilter` is single-threaded: `insert` and `clear` require `&mut self`.
//! `contains` is safe to call concurrently via `&self` because it only reads through
//! the atomic bit-array operations in `StandardBloomFilter`. For concurrent writes,
//! use [`AtomicScalableBloomFilter`].
//!
//! # Examples
//!
//! ## Basic usage
//!
//! ```rust
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(1_000, 0.01)?;
//!
//! filter.insert(&"alice");
//! filter.insert(&"bob");
//!
//! assert!(filter.contains(&"alice"));
//! assert!(!filter.contains(&"carol")); // true negative (no false negatives ever)
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! ## Growing across many items
//!
//! ```rust
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?;
//!
//! for i in 0..100_000u64 {
//!     filter.insert(&i);
//! }
//!
//! // The filter grew automatically; FPR is still bounded near target
//! assert!(filter.filter_count() > 1);
//! assert!(filter.estimate_fpr() < 0.05);
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! ## Choosing a growth strategy
//!
//! ```rust
//! use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};
//!
//! // Geometric 2x growth (default): logarithmic stage count, good for most workloads
//! let mut filter = ScalableBloomFilter::<u64>::with_strategy(
//!     1_000, 0.01, 0.5, GrowthStrategy::Geometric(2.0),
//! )?;
//!
//! // Adaptive: self-tuning tightening ratio based on observed fill rates
//! let mut adaptive = ScalableBloomFilter::<u64>::with_strategy(
//!     1_000, 0.01, 0.5,
//!     GrowthStrategy::Adaptive { initial_ratio: 0.5, min_ratio: 0.3, max_ratio: 0.9 },
//! )?;
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! ## Unique-item estimation
//!
//! ```rust
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1_000, 0.01)?
//!     .with_cardinality_tracking();
//!
//! for _ in 0..3 {
//!     for i in 0..10_000i32 { filter.insert(&i); }
//! }
//!
//! assert_eq!(filter.len(), 30_000);                     // total insertions
//! let unique = filter.estimate_unique_count();
//! assert!((unique as i64 - 10_000).abs() < 500);        // ±2% of 10,000
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! # References
//!
//! - Almeida, P. S., Baquero, C., Preguiça, N., & Hutchison, D. (2007).
//!   "Scalable Bloom Filters." *Information Processing Letters*, 101(6), 255–261.
//! - Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007).
//!   "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm."
//!   *DMTCS Proceedings*, AH, 127–146.

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::filter::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ── CONSTANTS ────────────────────────────────────────────────────────────────

/// Hard upper bound on the number of sub-filters.
///
/// At geometric 2× growth from an initial capacity of 1, 64 stages provide
/// theoretical capacity for 2⁶⁴ − 1 items. In practice `initial_capacity`
/// is always ≥ 1, so this bound is never a real constraint before hardware
/// memory limits apply.
pub const MAX_FILTERS: usize = 64;

/// Floor on any sub-filter's FPR to prevent floating-point underflow.
///
/// `target_fpr × r^i` converges toward zero as i grows. Clamping at 1e-15
/// keeps FPR calculations finite without meaningful loss of accuracy —
/// a 1e-15 false positive rate is already unmeasurable in practice.
const MIN_FPR: f64 = 1e-15;

/// Default fill ratio at which the current sub-filter triggers growth.
///
/// 0.5 (50%) is the mathematically optimal trigger point per Almeida et al. (2007),
/// maximising the number of items stored per bit for any given FPR and filter size.
const DEFAULT_FILL_THRESHOLD: f64 = 0.5;

/// How many filters below `MAX_FILTERS` before `is_near_capacity()` returns true.
const CAPACITY_WARNING_THRESHOLD: usize = 5;

/// HyperLogLog++ precision parameter b, giving m = 2^b registers.
///
/// b = 14 → 16,384 registers → standard error ≈ 1.04 / √16384 ≈ 0.81%.
const HLL_PRECISION: u8 = 14;
const HLL_REGISTER_COUNT: usize = 1 << HLL_PRECISION;
const HLL_REGISTER_MASK: u64 = (HLL_REGISTER_COUNT - 1) as u64;

/// HyperLogLog++ bias-correction constant α_∞ with small-m correction term.
/// See Flajolet et al. (2007), §4, equation (3).
const ALPHA_INF: f64 = 0.7213 / (1.0 + 1.079 / HLL_REGISTER_COUNT as f64);
const SMALL_RANGE_THRESHOLD: f64 = (5.0 / 2.0) * HLL_REGISTER_COUNT as f64;
const LARGE_RANGE_THRESHOLD: f64 = (1u64 << 32) as f64 / 30.0;

/// Maximum retained growth events to bound `growth_history` memory.
///
/// At 128 entries, the deque occupies roughly 128 × ~56 bytes ≈ 7 KB — negligible
/// relative to the bit arrays, but still bounded against pathological workloads
/// that trigger continuous growth.
const MAX_GROWTH_HISTORY: usize = 128;


// ── INTERNAL DETERMINISTIC HASHER ────────────────────────────────────────────
//
// Used exclusively by `HyperLogLog::add`. `std::hash::DefaultHasher` uses
// SipHash-1-3 with per-process randomised keys, making it non-deterministic
// across process restarts and Rust versions — a hard requirement violation for
// HLL register assignment. FNV-1a is keyless, stable, and fast for short byte
// streams. The MurmurHash3 fmix64 finalizer is applied on top to satisfy HLL's
// geometric distribution assumption (FNV-1a alone correlates high bits for
// sequential integer inputs, causing >10% estimation error without mixing).
//
// This type is intentionally private. It is never exposed through the public API.
struct InternalHasher(u64);

impl InternalHasher {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:        u64 = 0x0000_0100_0000_01B3;

    #[inline]
    fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    /// Hash any `Hash`-implementing value to a `u64`. Zero allocation.
    ///
    /// Applies FNV-1a byte-by-byte, then passes the result through the
    /// MurmurHash3 fmix64 finalizer for avalanche. `finish()` deliberately
    /// omits fmix64 — the Bloom double-hashing path has its own mixing.
    #[inline]
    fn hash_one<T: Hash>(item: &T) -> u64 {
        let mut h = Self::new();
        item.hash(&mut h);
        Self::fmix64(h.0)
    }

    /// MurmurHash3 64-bit finalizer.
    ///
    /// Converts FNV-1a output into a uniformly distributed value via three
    /// xor-shift-multiply rounds. The constants are the verified MurmurHash3
    /// magic numbers; substituting others breaks the avalanche property.
    #[inline]
    fn fmix64(mut k: u64) -> u64 {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51afd7ed558ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
        k ^= k >> 33;
        k
    }
}

impl std::hash::Hasher for InternalHasher {
    #[inline]
    fn finish(&self) -> u64 { self.0 }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
}


// ── TYPE DEFINITIONS ─────────────────────────────────────────────────────────

/// Controls how sub-filter capacities scale as the filter grows.
///
/// The choice of growth strategy affects both the total memory footprint and the
/// number of stages at a given dataset size. Almeida et al. (2007) recommend
/// `Geometric(2.0)` as the practical default.
///
/// # Space overhead relative to a single optimal filter
///
/// | Strategy | Overhead |
/// |----------|----------|
/// | `Constant` | Unbounded — stages grow linearly |
/// | `Geometric(2.0)` | ~2× |
/// | `Geometric(4.0)` | ~1.33×, half as many stages |
/// | `Adaptive` | Between 1.33× and 2×, self-tuned |
/// | `Bounded` | Bounded per-filter; reduces waste for skewed workloads |
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GrowthStrategy {
    /// Every sub-filter has the same capacity as the initial one.
    ///
    /// This is almost never the right choice: stage count grows linearly with
    /// dataset size, making miss queries increasingly expensive. Use only when
    /// dataset size is known to be small and bounded.
    Constant,

    /// Each successive sub-filter is `scale` times the capacity of the first.
    ///
    /// `scale` must be > 1.0 or growth stalls. The paper's recommendation is
    /// `scale = 2.0`; use `scale = 4.0` for datasets spanning many orders of
    /// magnitude to reduce stage count at the cost of ~33% higher memory overhead.
    Geometric(f64),

    /// Geometric 2× growth with a per-stage FPR tightening ratio that adapts
    /// to the observed fill rate of the just-completed sub-filter.
    ///
    /// If a filter's fill rate at growth time significantly exceeds
    /// `fill_threshold`, the tightening ratio contracts toward `min_ratio`
    /// (tighter per-filter FPR, more bits). If it falls significantly below,
    /// the ratio relaxes toward `max_ratio` (looser per-filter FPR, fewer bits).
    ///
    /// # Constraints
    ///
    /// - `0.0 < min_ratio ≤ initial_ratio ≤ max_ratio < 1.0`
    Adaptive {
        /// Starting tightening ratio (also the reset value after `clear`).
        initial_ratio: f64,
        /// Minimum tightening ratio (most aggressive FPR reduction per stage).
        min_ratio: f64,
        /// Maximum tightening ratio (least aggressive FPR reduction per stage).
        max_ratio: f64,
    },

    /// Geometric growth capped at a per-filter size limit.
    ///
    /// Useful when individual filter allocations must not exceed a fixed budget
    /// (e.g. to stay within a single NUMA node's memory). Once `max_filter_size`
    /// is reached, all subsequent sub-filters have equal capacity, degrading to
    /// `Constant` growth semantics for the tail of the sequence.
    Bounded {
        /// Geometric growth factor applied until `max_filter_size` is reached.
        scale: f64,
        /// Maximum number of items any single sub-filter may be sized for.
        max_filter_size: usize,
    },
}

impl Default for GrowthStrategy {
    fn default() -> Self {
        Self::Geometric(2.0)
    }
}

/// What happens when the filter reaches [`MAX_FILTERS`] and cannot grow further.
///
/// Capacity exhaustion is a structural limit, not a transient error: once
/// `MAX_FILTERS` sub-filters exist, no new ones can be added regardless of fill rate.
/// Insertions that would require a new filter instead go into the last sub-filter,
/// increasing its FPR beyond the configured target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CapacityExhaustedBehavior {
    /// Continue inserting into the saturated last filter.
    ///
    /// The FPR degrades silently but no error is returned. Appropriate for
    /// workloads where data loss is worse than a degraded FPR.
    Silent,

    /// Return [`Err(BloomCraftError::MaxFiltersExceeded)`](crate::error::BloomCraftError::MaxFiltersExceeded)
    /// from `insert_checked`.
    ///
    /// `insert` (the infallible variant) ignores this setting and behaves as
    /// `Silent`. Use `insert_checked` consistently if you want error propagation.
    Error,

    /// Panic immediately.
    ///
    /// Only available in debug builds. Use in tests to surface capacity
    /// exhaustion as a hard failure rather than a silently degraded filter.
    #[cfg(debug_assertions)]
    Panic,
}

impl Default for CapacityExhaustedBehavior {
    fn default() -> Self {
        Self::Silent
    }
}

/// Iteration order used when querying sub-filters.
///
/// The order affects hit latency and miss latency differently:
///
/// - **Hit latency**: depends on which filter contains the item. `Reverse` finds
///   recently-inserted items in O(k); `Forward` makes old items fast and new items slow.
/// - **Miss latency**: always O(k × l) regardless of order, where l is the number of
///   sub-filters and k is the hash count of the first sub-filter (later filters have
///   higher k, so actual cost is Σᵢ kᵢ).
///
/// For workloads where queries skew toward recently-inserted items — which covers the
/// majority of real-world use cases — `Reverse` is the correct choice. `Forward` is
/// appropriate only when queries are uniformly distributed across the full history or
/// skew toward old items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QueryStrategy {
    /// Check sub-filters oldest-first (filter 0 → filter l−1).
    Forward,

    /// Check sub-filters newest-first (filter l−1 → filter 0). **Default.**
    ///
    /// Achieves O(k) hit latency for recently-inserted items. Recommended for
    /// any workload following a recency bias (e.g. deduplication, session tracking,
    /// cache membership).
    Reverse,
}

impl Default for QueryStrategy {
    fn default() -> Self {
        Self::Reverse
    }
}


// ── HYPERLOGLOG++ ─────────────────────────────────────────────────────────────

/// HyperLogLog++ cardinality estimator with sparse/dense hybrid representation.
///
/// Estimates the number of distinct items in a stream using O(m) memory, where
/// m = 2^`HLL_PRECISION` = 16,384 registers. Each register stores a single byte,
/// giving a base memory cost of 16 KB in dense mode.
///
/// The sketch uses a sparse representation (a `HashMap<u16, u8>`) until the number
/// of non-zero registers exceeds `sparse_threshold` (m/4 = 4,096), at which point
/// it converts to the dense `[u8; m]` array. This optimises the common case of
/// estimating small cardinalities (< 4,096 unique items) at the cost of a one-time
/// conversion.
///
/// # Accuracy
///
/// Standard error ≈ 1.04 / √m ≈ 0.81% for this precision. The implementation
/// applies all three HLL++ range corrections (small-range linear counting,
/// large-range 32-bit correction, mid-range raw estimate) from Flajolet et al. (2007).
///
/// # Hash independence
///
/// Uses `InternalHasher` (FNV-1a + fmix64) for register assignment. This is
/// intentionally independent of the `BloomHasher` used for membership testing:
/// the two paths have different uniformity requirements and must not share state.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Box<[u8; HLL_REGISTER_COUNT]>,
    // Sparse representation for low cardinalities. `None` once converted to dense.
    sparse: Option<std::collections::HashMap<u16, u8>>,
    // Register count above which the sparse map is converted to the dense array.
    sparse_threshold: usize,
}

impl HyperLogLog {
    /// Create a new, empty HyperLogLog++ sketch.
    ///
    /// Starts in sparse mode; converts to dense automatically when the number of
    /// distinct registers observed exceeds `m / 4`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bloomcraft::filters::scalable::HyperLogLog;
    /// let hll = HyperLogLog::new();
    /// assert_eq!(hll.estimate(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            registers: Box::new([0; HLL_REGISTER_COUNT]),
            sparse: Some(std::collections::HashMap::new()),
            sparse_threshold: HLL_REGISTER_COUNT / 4,
        }
    }

    /// Record an item in the sketch.
    ///
    /// Hashes `item` with `InternalHasher`, selects the register via the low
    /// `HLL_PRECISION` bits, and updates the register with the leading-zero count
    /// of the remaining bits if it exceeds the current value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bloomcraft::filters::scalable::HyperLogLog;
    /// let mut hll = HyperLogLog::new();
    /// for i in 0u64..1_000 { hll.add(&i); }
    /// let est = hll.estimate();
    /// assert!((est as i64 - 1_000).abs() < 50);
    /// ```
    pub fn add<T: Hash>(&mut self, item: &T) {
        let hash = InternalHasher::hash_one(item);
        let register_idx = (hash & HLL_REGISTER_MASK) as usize;
        let remaining = hash >> HLL_PRECISION;
        let leading_zeros: u8 = if remaining == 0 {
            (64u8 - HLL_PRECISION) + 1
        } else {
            remaining.leading_zeros() as u8 - HLL_PRECISION + 1
        };
        if let Some(ref mut sparse) = self.sparse {
            let current = sparse.get(&(register_idx as u16)).copied().unwrap_or(0);
            if leading_zeros > current {
                sparse.insert(register_idx as u16, leading_zeros);
            }
            if sparse.len() > self.sparse_threshold {
                self.convert_to_dense();
            }
        } else if leading_zeros > self.registers[register_idx] {
            self.registers[register_idx] = leading_zeros;
        }
    }

    /// Estimate the number of distinct items added to this sketch.
    ///
    /// Returns 0 for an empty sketch. Applies the HLL++ range corrections:
    /// small-range linear counting, mid-range raw harmonic mean estimate,
    /// and large-range 32-bit correction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bloomcraft::filters::scalable::HyperLogLog;
    /// let mut hll = HyperLogLog::new();
    /// for i in 0u64..10_000 { hll.add(&i); }
    /// let est = hll.estimate();
    /// assert!((est as i64 - 10_000).abs() < 200); // within 2%
    /// ```
    #[must_use]
    pub fn estimate(&self) -> usize {
        if let Some(ref sparse) = self.sparse {
            return self.estimate_sparse(sparse);
        }
        self.estimate_dense()
    }

    /// Merge another sketch into this one, taking the element-wise maximum of registers.
    ///
    /// After merging, `self.estimate()` approximates the cardinality of the union
    /// of the two sets. This operation is correct for all combinations of sparse and
    /// dense internal representations and does not allocate on the hot path when
    /// both sketches are dense.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bloomcraft::filters::scalable::HyperLogLog;
    /// let mut a = HyperLogLog::new();
    /// let mut b = HyperLogLog::new();
    /// for i in 0u64..500 { a.add(&i); }
    /// for i in 500u64..1_000 { b.add(&i); }
    /// a.merge(&b);
    /// let est = a.estimate();
    /// assert!((est as i64 - 1_000).abs() < 30);
    /// ```
    pub fn merge(&mut self, other: &Self) {
        match (&mut self.sparse, &other.sparse) {
            (Some(self_sparse), Some(other_sparse)) => {
                for (&idx, &val) in other_sparse.iter() {
                    let entry = self_sparse.entry(idx).or_insert(0);
                    *entry = (*entry).max(val);
                }                
                if self_sparse.len() > self.sparse_threshold {
                    self.convert_to_dense();
                }
            }
            (Some(_), None) => {
                self.convert_to_dense();
                for i in 0..HLL_REGISTER_COUNT {
                    self.registers[i] = self.registers[i].max(other.registers[i]);
                }
            }
            (None, Some(other_sparse)) => {
                for (&idx, &val) in other_sparse.iter() {
                    let i = idx as usize;
                    self.registers[i] = self.registers[i].max(val);
                }
            }
            (None, None) => {
                for i in 0..HLL_REGISTER_COUNT {
                    self.registers[i] = self.registers[i].max(other.registers[i]);
                }
            }
        }
    }

    // Promote the sparse HashMap into the dense byte array and drop the map.
    fn convert_to_dense(&mut self) {
        if let Some(sparse) = self.sparse.take() {
            for (idx, val) in sparse.iter() {
                self.registers[*idx as usize] = *val;
            }
        }
    }

    fn estimate_sparse(&self, sparse: &std::collections::HashMap<u16, u8>) -> usize {
        if sparse.is_empty() {
            return 0;
        }

        // For very small cardinalities the sparse map size is itself a good estimate.
        if sparse.len() < 50 {
            return sparse.len();
        }

        let mut sum = 0.0;
        let mut zero_count = HLL_REGISTER_COUNT;

        for i in 0..HLL_REGISTER_COUNT {
            let val = sparse.get(&(i as u16)).copied().unwrap_or(0);
            if val > 0 {
                zero_count -= 1;
            }
            sum += 2f64.powi(-(val as i32));
        }

        let raw_estimate = ALPHA_INF * (HLL_REGISTER_COUNT as f64).powi(2) / sum;

        if raw_estimate <= SMALL_RANGE_THRESHOLD && zero_count > 0 {
            (HLL_REGISTER_COUNT as f64 * (HLL_REGISTER_COUNT as f64 / zero_count as f64).ln()) as usize
        } else {
            raw_estimate as usize
        }
    }

    fn estimate_dense(&self) -> usize {
        let mut sum = 0.0;
        let mut zero_count = 0;

        for &register in self.registers.iter() {
            if register == 0 {
                zero_count += 1;
            }
            sum += 2f64.powi(-(register as i32));
        }

        let raw_estimate = ALPHA_INF * (HLL_REGISTER_COUNT as f64).powi(2) / sum;

        if raw_estimate <= SMALL_RANGE_THRESHOLD {
            if zero_count > 0 {
                (HLL_REGISTER_COUNT as f64 * (HLL_REGISTER_COUNT as f64 / zero_count as f64).ln()) as usize
            } else {
                raw_estimate as usize
            }
        } else if raw_estimate <= LARGE_RANGE_THRESHOLD {
            raw_estimate as usize
        } else {
            let corrected = -((1u64 << 32) as f64) * (1.0 - raw_estimate / (1u64 << 32) as f64).ln();
            corrected as usize
        }
    }

    /// Returns the total heap memory consumed by this sketch in bytes.
    ///
    /// In dense mode: 16,384 bytes for the register array plus `size_of::<Self>()`.
    /// In sparse mode: register array is allocated but zeroed; the HashMap consumes
    /// approximately 3 bytes per entry.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() 
            + HLL_REGISTER_COUNT
            + self.sparse.as_ref().map(|s| s.capacity() * 3).unwrap_or(0)
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}


// ── GROWTH EVENT ──────────────────────────────────────────────────────────────

// Recorded each time a new sub-filter is added or the filter is cleared.
// Retained in a bounded deque for observability; not part of the public API.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GrowthEvent {
    timestamp: u64,      // Unix seconds at the time of growth
    filter_index: usize, // Index of the newly created sub-filter
    capacity: usize,     // Configured item capacity of the new sub-filter
    fpr: f64,            // Configured FPR of the new sub-filter
    total_items: usize,  // Total items in the SBF when growth was triggered
}


// ── HEALTH METRICS ────────────────────────────────────────────────────────────

/// A snapshot of runtime state for monitoring and diagnostics.
///
/// Returned by [`ScalableBloomFilter::health_metrics`]. All fields are computed
/// at call time; there is no background collection.
///
/// # FPR values
///
/// `estimated_fpr` uses the complement rule (1 − ∏(1 − pᵢ)), which is the
/// tightest achievable bound. `max_fpr` uses the union bound (∑ pᵢ), which is
/// always ≥ `estimated_fpr` and suitable for conservative capacity planning.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalableHealthMetrics {
    /// Number of sub-filters currently allocated.
    pub filter_count: usize,

    /// Sum of `expected_items` across all sub-filters.
    pub total_capacity: usize,

    /// Total calls to `insert` or `insert_batch` since creation or last `clear`.
    /// Counts duplicates.
    pub total_items: usize,

    /// Compound FPR using the complement rule: `1 − ∏(1 − pᵢ)`.
    pub estimated_fpr: f64,

    /// Compound FPR upper bound using the union rule: `∑ pᵢ`.
    pub max_fpr: f64,

    /// Configured `target_fpr` of the first sub-filter.
    pub target_fpr: f64,

    /// Current FPR tightening ratio. For `Adaptive` growth, this may differ
    /// from the value passed to the constructor.
    pub current_error_ratio: f64,

    /// Fill ratio of the current (most recently created) sub-filter.
    /// Triggers growth when it reaches `fill_threshold`.
    pub current_fill_rate: f64,

    /// Mean fill ratio across all sub-filters.
    pub avg_fill_rate: f64,

    /// Total heap memory in bytes: sub-filter bit arrays + HLL sketches + bookkeeping.
    pub memory_bytes: usize,

    /// Number of additional sub-filters that can be created before hitting `MAX_FILTERS`.
    pub remaining_growth: usize,

    /// Number of growth events recorded (capped at `MAX_GROWTH_HISTORY`).
    pub growth_events: usize,

    /// Active iteration order for membership queries.
    pub query_strategy: QueryStrategy,
}

impl fmt::Display for ScalableHealthMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ScalableBloomFilter Health Metrics")?;
        writeln!(f, "==================================")?;
        writeln!(f, "Filters:          {}", self.filter_count)?;
        writeln!(f, "Total capacity:   {}", self.total_capacity)?;
        writeln!(f, "Total items:      {}", self.total_items)?;
        writeln!(f, "Estimated FPR:    {:.4}%", self.estimated_fpr * 100.0)?;
        writeln!(f, "Max FPR (bound):  {:.4}%", self.max_fpr * 100.0)?;
        writeln!(f, "Target FPR:       {:.4}%", self.target_fpr * 100.0)?;
        writeln!(f, "Error ratio:      {:.3}", self.current_error_ratio)?;
        writeln!(f, "Current fill:     {:.1}%", self.current_fill_rate * 100.0)?;
        writeln!(f, "Avg fill:         {:.1}%", self.avg_fill_rate * 100.0)?;
        writeln!(f, "Memory usage:     {} bytes", self.memory_bytes)?;
        writeln!(f, "Remaining growth: {} filters", self.remaining_growth)?;
        writeln!(f, "Growth events:    {}", self.growth_events)?;

        writeln!(f, "Query strategy:   {:?}", self.query_strategy)?;
        Ok(())
    }
}


// ── QUERY TRACING (feature-gated) ───────────────────────────────────────────

#[cfg(feature = "trace")]
pub mod trace {
    use super::*;
    use std::time::{Duration, Instant};

    /// A complete execution trace of a single `contains_traced` call.
    ///
    /// Captures per-filter timing, hash counts, fill rates, and whether
    /// the query terminated early due to a positive match. Intended for
    /// offline analysis of slow or unexpected queries; not suitable for
    /// production hot paths due to `Instant::now()` overhead.
    #[derive(Debug, Clone)]
    pub struct QueryTrace {
        /// Wall-clock duration from entry to exit of `contains_traced`.
        pub total_duration: Duration,
        /// Per-filter execution details, in the order filters were checked.
        pub filter_traces: Vec<FilterTrace>,
        /// True if the query returned early on a positive match.
        pub early_terminated: bool,
        /// Index of the sub-filter that produced a positive result, if any.
        pub matched_filter: Option<usize>,
        /// Total bit positions examined across all sub-filters.
        pub total_bits_checked: usize,
        /// String representation of the active `QueryStrategy`.
        pub strategy: String,
    }

    /// Execution trace for a single sub-filter within a `QueryTrace`.
    #[derive(Debug, Clone)]
    pub struct FilterTrace {
        /// Sub-filter index (position in the `filters` vector).
        pub index: usize,
        /// Time spent inside this sub-filter's `contains` call.
        pub duration: Duration,
        /// Whether this sub-filter returned a positive result.
        pub matched: bool,
        /// Number of hash functions evaluated (equals `hash_count()` for this filter).
        pub hashes_checked: usize,
        /// Number of bit positions checked (same as `hashes_checked` for standard filters).
        pub bits_checked: usize,
        /// Fill ratio of this sub-filter at query time.
        pub fill_rate: f64,
    }

    impl QueryTrace {
        #[must_use]
        pub fn new() -> Self {
            Self {
                total_duration: Duration::ZERO,
                filter_traces: Vec::new(),
                early_terminated: false,
                matched_filter: None,
                total_bits_checked: 0,
                strategy: String::from("unknown"),
            }
        }

        /// Format the trace as a multi-line human-readable string.
        ///
        /// Includes total duration, early-termination status, matched filter,
        /// and a per-filter breakdown of timing, fill rate, and bit count.
        #[must_use]
        pub fn format_detailed(&self) -> String {
            let mut output = String::new();
            output.push_str(&format!("Query Trace ({})\n", self.strategy));
            output.push_str(&format!("Total duration: {:?}\n", self.total_duration));
            output.push_str(&format!("Early terminated: {}\n", self.early_terminated));
            output.push_str(&format!("Matched filter: {:?}\n", self.matched_filter));
            output.push_str(&format!("Total bits checked: {}\n", self.total_bits_checked));
            output.push_str("Filters checked:\n");
            for ft in &self.filter_traces {
                output.push_str(&format!(
                    "  [{}] {:?} | matched: {} | fill: {:.1}% | bits: {}\n",
                    ft.index,
                    ft.duration,
                    ft.matched,
                    ft.fill_rate * 100.0,
                    ft.bits_checked
                ));
            }
            output
        }
    }

    impl Default for QueryTrace {
        fn default() -> Self {
            Self::new()
        }
    }

    pub struct QueryTraceBuilder {
        trace: QueryTrace,
        start_time: Instant,
    }

    impl QueryTraceBuilder {
        #[must_use]
        pub fn new(strategy: &str) -> Self {
            let mut trace = QueryTrace::new();
            trace.strategy = strategy.to_string();
            Self { trace, start_time: Instant::now() }
        }

        pub fn record_filter(
            &mut self,
            index: usize,
            matched: bool,
            hashes_checked: usize,
            bits_checked: usize,
            fill_rate: f64,
            start: Instant,
        ) {
            self.trace.filter_traces.push(FilterTrace {
                index,
                duration: start.elapsed(),
                matched,
                hashes_checked,
                bits_checked,
                fill_rate,
            });
            self.trace.total_bits_checked += bits_checked;
            if matched {
                self.trace.matched_filter = Some(index);
            }
        }

        #[must_use]
        pub fn finish(mut self) -> QueryTrace {
            self.trace.total_duration = self.start_time.elapsed();
            self.trace
        }
    }
}

#[cfg(feature = "trace")]
pub use trace::{QueryTrace, QueryTraceBuilder};

// ── MAIN STRUCT ───────────────────────────────────────────────────────────────

/// A dynamically growing Bloom filter with bounded false positive rate.
///
/// Maintains a sequence of [`StandardBloomFilter`] sub-filters. Insertions
/// always target the last sub-filter; a new sub-filter is appended when the
/// current one's fill ratio exceeds [`fill_threshold`]. Queries iterate sub-filters
/// in the configured [`QueryStrategy`] order and return on the first positive.
///
/// # Invariants
///
/// - `filters` is never empty; there is always at least one sub-filter.
/// - `filters.len() == filter_nonempty.len()` at all times.
/// - `filter_nonempty[i]` is `true` iff at least one item has been inserted into
///   sub-filter `i` since the last `clear`. A `true` flag with a zero fill rate
///   is a bug.
/// - `total_items` equals the number of successful calls to `insert` / `insert_batch`
///   since creation or last `clear`, counting duplicates.
/// - `items_in_current_filter` and `current_filter_threshold` are transient: they
///   carry `#[serde(skip)]` and are reconstructed lazily on the first insert after
///   deserialization.
///
/// # Type Parameters
///
/// - `T`: Item type. Must implement [`Hash`].
/// - `H`: Hash strategy. Must implement [`BloomHasher`] + [`Clone`] + [`Default`].
///   Defaults to [`StdHasher`] (SipHash-1-3, adequate for most use cases).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "H: BloomHasher + Clone + Default",
        deserialize = "H: BloomHasher + Clone + Default"
    ))
)]

pub struct ScalableBloomFilter<T, H = StdHasher>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Ordered sequence of sub-filters. Index 0 is the oldest; the last is current.
    filters: Vec<StandardBloomFilter<T, H>>,

    /// Parallel bool vector: `filter_nonempty[i]` is set to `true` on the first
    /// insert into `filters[i]`, and reset to `false` by `clear`. Used to skip
    /// filters that have never received items (relevant only during the brief window
    /// after a new sub-filter is appended and before the first item lands in it).
    ///
    /// Carries `#[serde(skip)]`; reconstructed by `recalibrate_grow_state()`.
    #[cfg_attr(feature = "serde", serde(skip))]
    filter_nonempty: Vec<bool>,

    /// Item capacity passed to each initial sub-filter construction.
    initial_capacity: usize,

    /// FPR configured for sub-filter 0. Sub-filter i uses `target_fpr × error_ratio^i`.
    target_fpr: f64,

    /// FPR tightening ratio r. For `Adaptive` growth, this field mutates over time.
    error_ratio: f64,

    /// Sub-filter capacity growth strategy.
    growth: GrowthStrategy,

    /// Fill ratio at which the current sub-filter triggers creation of the next.
    ///
    /// Mathematically optimal at 0.5 per Almeida et al. (2007). Values below ~0.45
    /// cause sub-filters to be replaced before reaching their designed capacity,
    /// invalidating the FPR convergence bound. Validated at construction; mutable
    /// via `set_fill_threshold`.
    fill_threshold: f64,

    /// Hash strategy instance, shared across all sub-filters.
    ///
    /// Carries `#[serde(skip)]`; sub-filters re-initialise with `H::default()` after
    /// deserialization. This is safe if and only if `H::default()` produces
    /// **deterministic** hash outputs — i.e., two independent `H::default()` instances
    /// hash the same input to the same value, both within the same process and across
    /// process restarts.
    #[cfg_attr(feature = "serde", serde(skip))]
    hasher: H,

    /// Total insertions since creation or last `clear`. Counts duplicates.
    total_items: usize,

    /// Insertions into the current (last) sub-filter since it was created.
    ///
    /// Maintained as an O(1) counter to avoid scanning the bit array on every
    /// insert. Reset to 0 each time `try_add_filter` creates a new sub-filter.
    ///
    /// Carries `#[serde(skip)]`; sentinel value 0 with a non-empty `filters`
    /// vec triggers `recalibrate_grow_state()` on the next insert.
    #[cfg_attr(feature = "serde", serde(skip))]
    items_in_current_filter: usize,

    /// Pre-computed item count at which the current sub-filter triggers growth.
    ///
    /// Equals `⌈capacity × fill_threshold⌉`. Compared against
    /// `items_in_current_filter` in `should_grow()`, making growth detection O(1).
    ///
    /// Sentinel value 0 (set by `#[serde(skip)]` after deserialization) causes
    /// `should_grow()` to fall back to the O(m/64) `fill_rate()` path exactly once,
    /// after which `recalibrate_grow_state()` restores the O(1) path.
    #[cfg_attr(feature = "serde", serde(skip))]
    current_filter_threshold: usize,

    /// Behaviour when the filter reaches `MAX_FILTERS` and cannot grow.
    capacity_behavior: CapacityExhaustedBehavior,

    /// Iteration order for `contains`, `contains_batch`, and `contains_with_provenance`.
    query_strategy: QueryStrategy,

    /// Bounded ring buffer of growth events for observability.
    /// Capped at `MAX_GROWTH_HISTORY` entries; oldest entries are evicted first.
    #[cfg_attr(feature = "serde", serde(skip))]
    growth_history: std::collections::VecDeque<GrowthEvent>,

    /// Per-filter HyperLogLog++ sketches for cardinality estimation.
    /// Only populated when `track_cardinality` is true.
    #[cfg_attr(feature = "serde", serde(skip))]
    cardinality_sketches: Vec<HyperLogLog>,

    /// Whether `insert` feeds items into `cardinality_sketches`.
    track_cardinality: bool,

    _phantom: PhantomData<T>,
}

impl<T, H> Clone for ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    fn clone(&self) -> Self {
        Self {
            filters: self.filters.clone(),
            filter_nonempty: self.filter_nonempty.clone(),
            initial_capacity: self.initial_capacity,
            target_fpr: self.target_fpr,
            error_ratio: self.error_ratio,
            growth: self.growth,
            fill_threshold: self.fill_threshold,
            hasher: self.hasher.clone(),
            total_items: self.total_items,
            items_in_current_filter: self.items_in_current_filter,
            current_filter_threshold: self.current_filter_threshold,
            capacity_behavior: self.capacity_behavior,
            query_strategy: self.query_strategy,
            growth_history: self.growth_history.clone(),
            cardinality_sketches: self.cardinality_sketches.clone(),
            track_cardinality: self.track_cardinality,
            _phantom: PhantomData,
        }
    }
}

// ── CONSTRUCTORS ──────────────────────────────────────────────────────────────

impl<T> ScalableBloomFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a scalable Bloom filter with default configuration.
    ///
    /// Defaults:
    /// - Growth strategy: `Geometric(2.0)` — each sub-filter is twice the capacity of the first
    /// - Error tightening ratio: 0.5 — each sub-filter has half the FPR of the previous
    /// - Fill threshold: 0.5 — growth triggers at 50% fill (optimal per Almeida 2007)
    /// - Query strategy: `Reverse` — newest filter checked first
    /// - Capacity exhausted: `Silent` — continues inserting into the saturated last filter
    ///
    /// The compound FPR is bounded by `target_fpr / (1 − 0.5) = 2 × target_fpr`.
    /// If you require the compound FPR to stay within `target_fpr`, set
    /// `target_fpr` to half your budget, or use `with_strategy` with a higher ratio
    /// and correspondingly adjusted initial FPR.
    ///
    /// # Errors
    ///
    /// - `initial_capacity == 0` → [`BloomCraftError::InvalidItemCount`]
    /// - `target_fpr ∉ (0.0, 1.0)` → [`BloomCraftError::FalsePositiveRateOutOfBounds`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(10_000, 0.01)?;
    /// filter.insert(&42u64);
    /// assert!(filter.contains(&42u64));
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn new(initial_capacity: usize, target_fpr: f64) -> Result<Self> {
        Self::with_hasher(initial_capacity, target_fpr, StdHasher::new())
    }

    /// Create a scalable Bloom filter with an explicit growth strategy and tightening ratio.
    ///
    /// `error_ratio` is ignored when `growth` is `Adaptive`; use the `initial_ratio`
    /// field of the `Adaptive` variant instead.
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new), plus:
    /// - `error_ratio ∉ (0.0, 1.0)` for non-Adaptive strategies
    /// - Adaptive constraint violations (see [`GrowthStrategy::Adaptive`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};
    ///
    /// // r = 0.9: compound FPR ≤ target_fpr / 0.1 = 10 × target_fpr.
    /// // Set target_fpr = 0.001 to keep compound FPR ≤ 0.01.
    /// let mut filter = ScalableBloomFilter::<u64>::with_strategy(
    ///     10_000, 0.001, 0.9, GrowthStrategy::Geometric(2.0),
    /// )?;
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn with_strategy(
        initial_capacity: usize,
        target_fpr: f64,
        error_ratio: f64,
        growth: GrowthStrategy,
    ) -> Result<Self> {
        Self::with_strategy_and_hasher(
            initial_capacity,
            target_fpr,
            error_ratio,
            growth,
            StdHasher::new(),
        )
    }
}

impl<T, H> ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Create a scalable Bloom filter with a custom hash strategy.
    ///
    /// Uses all other defaults from [`new`](ScalableBloomFilter::new). Prefer this
    /// when you need a non-default `BloomHasher` (e.g. xxHash for throughput-critical paths).
    ///
    /// # Errors
    ///
    /// Same as [`new`](ScalableBloomFilter::new).
    #[must_use]
    pub fn with_hasher(initial_capacity: usize, target_fpr: f64, hasher: H) -> Result<Self> {
        Self::with_strategy_and_hasher(
            initial_capacity,
            target_fpr,
            0.5,
            GrowthStrategy::default(),
            hasher,
        )
    }

    /// Create a scalable Bloom filter with full control over all parameters.
    ///
    /// This is the canonical constructor; all other constructors delegate here.
    ///
    /// # Parameters
    ///
    /// - `initial_capacity`: target item count for sub-filter 0. Must be ≥ 1.
    /// - `target_fpr`: FPR target for sub-filter 0, in `(0.0, 1.0)`. The compound
    ///   FPR is bounded by `target_fpr / (1 − error_ratio)`.
    /// - `error_ratio`: FPR tightening ratio r, in `(0.0, 1.0)`. Ignored for
    ///   `Adaptive` growth; use `initial_ratio` in that variant instead.
    /// - `growth`: capacity growth strategy.
    /// - `hasher`: hash strategy instance, cloned for each sub-filter.
    ///
    /// # Errors
    ///
    /// - `initial_capacity == 0` → [`BloomCraftError::InvalidItemCount`]
    /// - `target_fpr ∉ (0.0, 1.0)` → [`BloomCraftError::FalsePositiveRateOutOfBounds`]
    /// - `error_ratio ∉ (0.0, 1.0)` (non-Adaptive) → [`BloomCraftError::InvalidParameters`]
    /// - `Adaptive` constraints violated → [`BloomCraftError::InvalidParameters`]
    /// - Initial sub-filter allocation fails → propagated from [`StandardBloomFilter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};
    /// use bloomcraft::hash::StdHasher;
    ///
    /// let filter = ScalableBloomFilter::<u64>::with_strategy_and_hasher(
    ///     10_000,
    ///     0.001,
    ///     0.9,
    ///     GrowthStrategy::Geometric(2.0),
    ///     StdHasher::new(),
    /// )?;
    ///
    /// assert_eq!(filter.filter_count(), 1);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn with_strategy_and_hasher(
        initial_capacity: usize,
        target_fpr: f64,
        error_ratio: f64,
        growth: GrowthStrategy,
        hasher: H,
    ) -> Result<Self> {
        if initial_capacity == 0 {
            return Err(BloomCraftError::invalid_item_count(initial_capacity));
        }
        if target_fpr <= 0.0 || target_fpr >= 1.0 {
            return Err(BloomCraftError::fp_rate_out_of_bounds(target_fpr));
        }

        // Validate error_ratio and growth-specific parameters
        match growth {
            GrowthStrategy::Adaptive {
                initial_ratio,
                min_ratio,
                max_ratio,
            } => {
                // Validate min_ratio
                if min_ratio <= 0.0 || min_ratio >= 1.0 {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "Adaptive growth min_ratio must be in (0.0, 1.0), got {}",
                        min_ratio
                    )));
                }

                // Validate max_ratio
                if max_ratio <= 0.0 || max_ratio >= 1.0 {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "Adaptive growth max_ratio must be in (0.0, 1.0), got {}",
                        max_ratio
                    )));
                }

                // Validate ordering: min_ratio ≤ initial_ratio ≤ max_ratio
                if min_ratio > initial_ratio || initial_ratio > max_ratio {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "Adaptive growth requires min_ratio ({}) ≤ initial_ratio ({}) ≤ max_ratio ({})",
                        min_ratio, initial_ratio, max_ratio
                    )));
                }
            }
            _ => {
                // For non-Adaptive strategies, validate error_ratio
                if error_ratio <= 0.0 || error_ratio >= 1.0 {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "error_ratio must be in (0.0, 1.0), got {}",
                        error_ratio
                    )));
                }
            }
        }

        // Create filter with validated parameters
        let mut filter = Self {
            filters: Vec::new(),
            filter_nonempty: Vec::new(),
            initial_capacity,
            target_fpr,
            error_ratio,
            growth,
            fill_threshold: DEFAULT_FILL_THRESHOLD,
            hasher: hasher.clone(),
            total_items: 0,
            items_in_current_filter: 0,
            current_filter_threshold: 0,
            capacity_behavior: CapacityExhaustedBehavior::default(),
            query_strategy: QueryStrategy::default(),
            growth_history: std::collections::VecDeque::new(),
            cardinality_sketches: Vec::new(),
            track_cardinality: false,
            _phantom: PhantomData,
        };

        // Create initial filter (propagate errors)
        filter.try_add_filter()?;

        Ok(filter)
    }

    
    // ── BUILDER-STYLE CONFIGURATION ───────────────────────────────────────────

    /// Set the behaviour when `MAX_FILTERS` is reached and the filter cannot grow.
    ///
    /// See [`CapacityExhaustedBehavior`] for semantics of each variant. The default
    /// is [`Silent`](CapacityExhaustedBehavior::Silent).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::{ScalableBloomFilter, CapacityExhaustedBehavior};
    ///
    /// let filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?
    ///     .with_capacity_behavior(CapacityExhaustedBehavior::Error);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn with_capacity_behavior(mut self, behavior: CapacityExhaustedBehavior) -> Self {
        self.capacity_behavior = behavior;
        self
    }

    /// Set the sub-filter iteration order for membership queries.
    ///
    /// See [`QueryStrategy`] for the performance trade-offs. The default is
    /// [`Reverse`](QueryStrategy::Reverse).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::{ScalableBloomFilter, QueryStrategy};
    ///
    /// // Use Forward if queries are uniformly distributed across the full history.
    /// let filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?
    ///     .with_query_strategy(QueryStrategy::Forward);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn with_query_strategy(mut self, strategy: QueryStrategy) -> Self {
        self.query_strategy = strategy;
        self
    }

    /// Enable HyperLogLog++ cardinality tracking.
    ///
    /// Attaches one `HyperLogLog` sketch per sub-filter. Each sketch uses
    /// approximately 16 KB in dense mode and ~12 bytes per unique register in
    /// sparse mode. The total overhead is at most `filter_count × 16 KB`.
    ///
    /// Once enabled, `estimate_unique_count()` returns a ±2% estimate of
    /// distinct items rather than the raw insertion count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?
    ///     .with_cardinality_tracking();
    ///
    /// for i in 0u64..1_000 { filter.insert(&i); }
    /// for i in 0u64..1_000 { filter.insert(&i); } // duplicates
    ///
    /// assert_eq!(filter.len(), 2_000);             // total insertions
    /// let unique = filter.estimate_unique_count();
    /// assert!((unique as i64 - 1_000).abs() < 30);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn with_cardinality_tracking(mut self) -> Self {
        self.track_cardinality = true;
        self.cardinality_sketches = vec![HyperLogLog::new()];
        self
    }


    // ── INTERNAL FILTER MANAGEMENT ────────────────────────────────────────────

    /// Allocate and append a new sub-filter.
    ///
    /// Computes capacity and FPR for the next stage, constructs a
    /// `StandardBloomFilter`, records a `GrowthEvent`, and resets the
    /// `items_in_current_filter` counter and `current_filter_threshold`.
    ///
    /// Called by `insert_checked` when `should_grow()` returns true, and once
    /// during construction to create the initial sub-filter.
    fn try_add_filter(&mut self) -> Result<()> {
        let filter_index = self.filters.len();

        if filter_index >= MAX_FILTERS {
            return Err(BloomCraftError::max_filters_exceeded(
                MAX_FILTERS,
                filter_index,
            ));
        }

        // Calculate capacity with adaptive growth support
        let capacity = self.calculate_next_capacity(filter_index)?;

        // Calculate error rate with adaptive support
        let fpr = self.calculate_next_fpr(filter_index);

        // Create and add new filter
        let new_filter = StandardBloomFilter::with_hasher(capacity, fpr, self.hasher.clone());
        self.filters.push(new_filter?);
        self.filter_nonempty.push(false);

        // Add HLL sketch if tracking cardinality
        if self.track_cardinality {
            self.cardinality_sketches.push(HyperLogLog::new());
        }

        // Record growth event
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.growth_history.push_back(GrowthEvent {
            timestamp,
            filter_index,
            capacity,
            fpr,
            total_items: self.total_items,
        });

        // Limit growth history size
        if self.growth_history.len() > MAX_GROWTH_HISTORY {
            self.growth_history.pop_front();
        }

        self.items_in_current_filter = 0;
        self.current_filter_threshold = ((capacity as f64) * self.fill_threshold).ceil() as usize;
        self.current_filter_threshold = self.current_filter_threshold.max(1);

        Ok(())
    }


    /// Compute the item capacity for the sub-filter at `filter_index`.
    ///
    /// Checks for integer overflow before converting from `f64`. Returns
    /// `Err(InvalidParameters)` if the computed value would overflow `usize`
    /// or produce a nonsensical result (e.g. wraparound under geometric growth).
    ///
    /// The `Adaptive` variant always uses a fixed 2× scale for capacity;
    /// the `initial_ratio` / `min_ratio` / `max_ratio` fields belong to the
    /// FPR domain and have no effect on filter sizing.
    fn calculate_next_capacity(&self, filter_index: usize) -> Result<usize> {
        const MAX_CAP: f64 = usize::MAX as f64;

        let capacity = match self.growth {
            GrowthStrategy::Constant => self.initial_capacity,

            GrowthStrategy::Geometric(scale) => {
                if filter_index == 0 {
                    self.initial_capacity
                } else {
                    let scale_log    = scale.ln();
                    let max_safe_exp = (MAX_CAP.ln() - (self.initial_capacity as f64).ln())
                        / scale_log;
                    if filter_index as f64 >= max_safe_exp {
                        return Err(BloomCraftError::invalid_parameters(format!(
                            "Filter index {} would cause capacity overflow (max safe: {:.1})",
                            filter_index, max_safe_exp
                        )));
                    }
                    let computed = self.initial_capacity as f64 * scale.powi(filter_index as i32);
                    if computed > MAX_CAP || !computed.is_finite() {
                        return Err(BloomCraftError::invalid_parameters(format!(
                            "Computed capacity {:.2e} exceeds usize::MAX", computed
                        )));
                    }
                    let new_cap = computed as usize;
                    if new_cap < self.initial_capacity {
                        return Err(BloomCraftError::invalid_parameters(
                            "Capacity calculation resulted in wraparound",
                        ));
                    }
                    new_cap
                }
            }

            GrowthStrategy::Adaptive { .. } => {
                if filter_index == 0 {
                    self.initial_capacity
                } else {
                    const SCALE: f64 = 2.0;
                    let max_safe_exp = (MAX_CAP.ln() - (self.initial_capacity as f64).ln())
                        / SCALE.ln();
                    if filter_index as f64 >= max_safe_exp {
                        return Err(BloomCraftError::invalid_parameters(format!(
                            "Adaptive filter index {} would cause capacity overflow",
                            filter_index
                        )));
                    }
                    let computed = self.initial_capacity as f64 * SCALE.powi(filter_index as i32);
                    if computed > MAX_CAP || !computed.is_finite() {
                        return Err(BloomCraftError::invalid_parameters(
                            "Adaptive capacity overflow",
                        ));
                    }
                    let new_cap = computed as usize;
                    if new_cap < self.initial_capacity {
                        return Err(BloomCraftError::invalid_parameters(
                            "Adaptive capacity wraparound",
                        ));
                    }
                    new_cap
                }
            }

            GrowthStrategy::Bounded { scale, max_filter_size } => {
                if filter_index == 0 {
                    self.initial_capacity
                } else {
                    let computed = self.initial_capacity as f64 * scale.powi(filter_index as i32);
                    if computed > MAX_CAP || !computed.is_finite() {
                        return Err(BloomCraftError::invalid_parameters(
                            "Bounded capacity calculation overflow",
                        ));
                    }
                    (computed as usize).min(max_filter_size)
                }
            }
        };

        Ok(capacity)
    }

    /// Compute the FPR for the sub-filter at `filter_index`.
    ///
    /// For `Adaptive` growth, compares the fill rate of the just-completed filter
    /// against `fill_threshold` and adjusts `self.error_ratio` up or down by 10%,
    /// clamped to `[min_ratio, max_ratio]`. This mutation is intentional: the
    /// adaptive ratio is the filter's authoritative state, not a transient value.
    ///
    /// Result is clamped to `[MIN_FPR, 1.0]` to prevent floating-point edge cases.
    fn calculate_next_fpr(&mut self, filter_index: usize) -> f64 {
        let ratio = match self.growth {
            GrowthStrategy::Adaptive { initial_ratio: _, min_ratio, max_ratio } => {
                // Adapt based on actual vs predicted fill rate
                if filter_index > 0 && !self.filters.is_empty() {
                    let last_filter = &self.filters[filter_index - 1];
                    let actual_fill = last_filter.fill_rate();

                    // If fill rate exceeded threshold, tighten (reduce ratio)
                    if actual_fill > self.fill_threshold * 1.2 {
                        self.error_ratio = (self.error_ratio * 0.9).max(min_ratio);
                    } else if actual_fill < self.fill_threshold * 0.8 {
                        // If well below threshold, relax (increase ratio)
                        self.error_ratio = (self.error_ratio * 1.1).min(max_ratio);
                    }
                }
                self.error_ratio
            }
            _ => self.error_ratio,
        };

        // Clamp exponent to prevent underflow
        const MAX_SAFE_EXP: i32 = 1000;
        let safe_index = (filter_index as i32).min(MAX_SAFE_EXP);
        
        let raw_fpr = self.target_fpr * ratio.powi(safe_index);
        
        // Ensure FPR never goes to zero or below MIN_FPR
        raw_fpr.max(MIN_FPR).min(1.0)
    }

    /// Returns true if the current sub-filter has reached its growth threshold.
    ///
    /// In steady state, this is an O(1) integer comparison:
    /// `items_in_current_filter >= current_filter_threshold`.
    ///
    /// The sentinel path (`current_filter_threshold == 0`) fires exactly once after
    /// serde deserialization and performs the O(m/64) `fill_rate()` scan to determine
    /// the correct answer before `recalibrate_grow_state()` is called.
    #[inline]
    fn should_grow(&self) -> bool {
        if self.filters.is_empty() {
            #[cfg(debug_assertions)]
            eprintln!("[ScalableBloomFilter] WARNING: No filters exist, forcing growth");
            return true;
        }

        if self.current_filter_threshold == 0 {
            // Post-deserialization sentinel: fall back to fill_rate() once.
            return self
                .filters
                .last()
                .map(|f| f.fill_rate() >= self.fill_threshold)
                .unwrap_or(true);
        }

        let should = self.items_in_current_filter >= self.current_filter_threshold;

        #[cfg(debug_assertions)]
        if should {
            let current = self.filters.last().unwrap();
            eprintln!(
                "[ScalableBloomFilter] Growing: filter {} reached {} items (threshold {}), capacity {}, items {}",
                self.filters.len(),
                self.items_in_current_filter,
                self.current_filter_threshold,
                current.expected_items(),
                current.len()
            );
        }

        should
    }

    /// Rebuild `items_in_current_filter`, `current_filter_threshold`, and
    /// `filter_nonempty` from durable state after serde deserialization.
    ///
    /// Both counter fields carry `#[serde(skip)]` and are zero-initialised after
    /// deserialization. The zero value of `current_filter_threshold` is the sentinel
    /// that triggers this function on the next insert. It fires at most once per
    /// deserialized instance and incurs an O(m/64) `fill_rate()` scan per filter —
    /// acceptable as a one-time reconstruction cost.
    fn recalibrate_grow_state(&mut self) {
        if let Some(current) = self.filters.last() {
            let fill = current.fill_rate();
            let estimated = (fill * current.expected_items() as f64).round() as usize;
            self.items_in_current_filter = estimated;
            self.current_filter_threshold = ((current.expected_items() as f64) * self.fill_threshold)
                .ceil() as usize;
            self.current_filter_threshold = self.current_filter_threshold.max(1);
        }

        self.filter_nonempty = self.filters
            .iter()
            .map(|f| f.fill_rate() > 0.0)
            .collect();
    }

    /// Assert the `filter_nonempty` invariant in debug builds.
    ///
    /// Verifies that every `filter_nonempty[i] == true` corresponds to a filter
    /// with a non-zero fill rate. The converse (false flag, non-zero fill) is not
    /// checked because `clear()` resets flags synchronously while bit-zeroing is
    /// a separate operation — a transient inconsistency is valid immediately after
    /// clearing.
    ///
    /// Called at the entry to `contains` in debug builds to verify the invariant
    /// on the read path. May also be called explicitly after any state-mutating
    /// operation during testing. Never called in release builds.
    #[cfg(debug_assertions)]
    fn assert_nonempty_invariant(&self) {
        assert_eq!(
            self.filters.len(),
            self.filter_nonempty.len(),
            "filter_nonempty length {} != filters length {}",
            self.filter_nonempty.len(),
            self.filters.len()
        );
        for (i, (filter, &flag)) in self.filters.iter()
            .zip(self.filter_nonempty.iter())
            .enumerate()
        {
            if flag {
                debug_assert!(
                    filter.fill_rate() > 0.0,
                    "filter_nonempty[{i}] is true but filter {i} has fill_rate 0.0 — invariant violated"
                );
            }
        }
    }
}


// ── CORE OPERATIONS ────────────────────────────────────────────────────────

impl<T, H> ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Insert an item, returning an error if capacity is exhausted and the
    /// configured behaviour is [`CapacityExhaustedBehavior::Error`].
    ///
    /// On success, the item is inserted into the current sub-filter,
    /// `total_items` and `items_in_current_filter` are incremented, and
    /// `filter_nonempty[last]` is set to `true`. If the current sub-filter
    /// has reached its fill threshold, a new one is appended before insertion.
    ///
    /// For infallible insertion with degraded FPR on capacity exhaustion,
    /// use [`insert`](Self::insert).
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::CapacityExceeded`] if the filter has reached
    ///   `MAX_FILTERS` and `capacity_behavior` is `Error`.
    /// - [`BloomCraftError::InternalError`] if no filters exist (should be
    ///   unreachable in normal operation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::{ScalableBloomFilter, CapacityExhaustedBehavior};
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(10, 0.01)?
    ///     .with_capacity_behavior(CapacityExhaustedBehavior::Error);
    ///
    /// for i in 0u64..100_000 {
    ///     if let Err(e) = filter.insert_checked(&i) {
    ///         eprintln!("Capacity exhausted at {} items: {}", i, e);
    ///         break;
    ///     }
    /// }
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    pub fn insert_checked(&mut self, item: &T) -> Result<()> {
        // Lazy recalibration after serde deserialization. The sentinel value 0 for
        // current_filter_threshold is set by #[serde(skip)]; this branch fires once.
        if self.current_filter_threshold == 0 && !self.filters.is_empty() {
            self.recalibrate_grow_state();
        }

        if self.should_grow() {
            match self.try_add_filter() {
                Ok(()) => {}
                Err(e) => {
                    match self.capacity_behavior {
                        CapacityExhaustedBehavior::Silent => {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[ScalableBloomFilter] WARNING: Cannot grow: {}. Continuing with degraded FPR.",
                                e
                            );
                        }
                        CapacityExhaustedBehavior::Error => {
                            return Err(e);
                        }
                        #[cfg(debug_assertions)]
                        CapacityExhaustedBehavior::Panic => {
                            panic!("Capacity exhausted: {}", e);
                        }
                    }
                }
            }
        }

        if let Some(current) = self.filters.last_mut() {
            current.insert(item);
            self.total_items += 1;
            self.items_in_current_filter += 1;

            let current_idx = self.filters.len() - 1;
            self.filter_nonempty[current_idx] = true;

            if self.track_cardinality {
                if let Some(sketch) = self.cardinality_sketches.last_mut() {
                    sketch.add(item);
                }
            }

            Ok(())
        } else {
            Err(BloomCraftError::internal_error("No filters available"))
        }
    }

    /// Insert an item using the configured capacity-exhaustion behaviour.
    ///
    /// Silently discards the error from `insert_checked` if capacity is exhausted
    /// and `capacity_behavior` is `Silent` (the default). For explicit error
    /// handling, use [`insert_checked`](Self::insert_checked).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(1_000, 0.01)?;
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    pub fn insert(&mut self, item: &T) {
        let _ = self.insert_checked(item);
    }

    /// Insert a slice of items in bulk.
    ///
    /// Functionally equivalent to calling `insert_checked` for each item, but with
    /// reduced overhead: the `current_filter_threshold == 0` sentinel check runs
    /// once before the loop rather than on every item.
    ///
    /// Note: unlike a fully amortised batch implementation, this still calls
    /// `should_grow()` per item. Growth checks are O(1) comparisons; the
    /// meaningful saving comes from avoiding the sentinel path overhead.
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidParameters`] if adding `items.len()` to
    ///   `total_items` would overflow `usize`. This check runs before any
    ///   insertion, making it an atomic pre-condition.
    /// - Propagates capacity errors from `try_add_filter` if `capacity_behavior`
    ///   is not `Silent`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?;
    /// let items: Vec<u64> = (0..10_000).collect();
    /// filter.insert_batch(&items)?;
    /// assert!(filter.contains(&9_999));
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    pub fn insert_batch(&mut self, items: &[T]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        if self.current_filter_threshold == 0 && !self.filters.is_empty() {
            self.recalibrate_grow_state();
        }

        self.total_items
        .checked_add(items.len())
        .ok_or_else(|| {
            BloomCraftError::invalid_parameters(format!(
                "Batch insert of {} items would overflow total_items counter \
                 (current: {}, max: {})",
                items.len(),
                self.total_items,
                usize::MAX,
            ))
        })?;

        #[cfg(debug_assertions)]
        let original_total = self.total_items;

        let mut remaining = items;

        while !remaining.is_empty() {
            if self.should_grow() {
                if self.is_at_max_capacity() {
                    match self.capacity_behavior {
                        CapacityExhaustedBehavior::Silent => {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[ScalableBloomFilter] WARNING: MAX_FILTERS ({}) reached. \
                                Inserting into saturated filter; FPR is no longer bounded.",
                                MAX_FILTERS
                            );
                        }
                        CapacityExhaustedBehavior::Error => {
                            return Err(BloomCraftError::max_filters_exceeded(
                                MAX_FILTERS,
                                self.filters.len(),
                            ));
                        }
                        #[cfg(debug_assertions)]
                        CapacityExhaustedBehavior::Panic => {
                            panic!(
                                "ScalableBloomFilter::insert_batch: MAX_FILTERS ({}) reached. \
                                Configure CapacityExhaustedBehavior::Error for non-panicking \
                                exhaustion handling.",
                                MAX_FILTERS
                            ); 
                        }
                    }
                } else {
                    self.try_add_filter()?;
                }
            }

            let space = self.current_filter_threshold
                .saturating_sub(self.items_in_current_filter);
            let seg_len = remaining.len().min(space.max(1));

            let (segment, rest) = remaining.split_at(seg_len);
            remaining = rest;

            let current_idx = self.filters.len() - 1;

            {
                let filter = &mut self.filters[current_idx];
                for item in segment {
                    filter.insert(item);
                }
            }

            self.filter_nonempty[current_idx] = true;
            self.items_in_current_filter += seg_len;
            self.total_items += seg_len;

            if self.track_cardinality {
                if let Some(sketch) = self.cardinality_sketches.get_mut(current_idx) {
                    for item in segment {
                        sketch.add(item);
                    }
                }
            }
        }
        
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.total_items,
            original_total + items.len(),
            "insert_batch total_items drift: expected {}, got {}. \
            A loop iteration incremented total_items by a value other than seg_len.",
            original_total + items.len(),
            self.total_items,
        );

        Ok(())
    }

    /// Test whether `item` is a member of the filter.
    ///
    /// Iterates sub-filters in the configured [`QueryStrategy`] order and returns
    /// `true` on the first positive result. Returns `false` only after all sub-filters
    /// have been checked.
    ///
    /// **No false negatives**: any item passed to `insert` or `insert_batch` will
    /// always be found by `contains`.
    ///
    /// **False positives**: an item never inserted may return `true` with probability
    /// at most `estimate_fpr()`.
    ///
    /// # Performance
    ///
    /// - Best case O(k₀) — `Reverse` strategy + item in the newest sub-filter
    /// - Average case O(Σᵢ₌₀^{l/2} kᵢ) — item in the middle sub-filter
    /// - Worst case O(Σᵢ₌₀^{l-1} kᵢ) — item absent, all sub-filters checked;
    ///   for l sub-filters with r = 0.5 and k₀ hash functions,
    ///   this equals l·k₀ + l(l−1)/2 hash evaluations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(100, 0.01)?;
    /// filter.insert(&7u64);
    ///
    /// assert!(filter.contains(&7));   // guaranteed true
    /// assert!(!filter.contains(&99)); // true negative (99 was never inserted)
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn contains(&self, item: &T) -> bool {
        #[cfg(debug_assertions)]
        self.assert_nonempty_invariant();

        match self.query_strategy {
            QueryStrategy::Forward => {
                for (filter, &nonempty) in self.filters.iter()
                    .zip(self.filter_nonempty.iter())
                {
                    if !nonempty { continue; }
                    if filter.contains(item) { return true; }
                }
            }
            QueryStrategy::Reverse => {
                for (filter, &nonempty) in self.filters.iter()
                    .zip(self.filter_nonempty.iter())
                    .rev()
                {
                    if !nonempty { continue; }
                    if filter.contains(item) { return true; }
                }
            }
        }
        false
    }

    /// Test membership for each item in `items`, returning one `bool` per item.
    ///
    /// Equivalent to `items.iter().map(|i| self.contains(i)).collect()`.
    /// Each query is independent; no cross-item cache effects are exploited.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(100, 0.01)?;
    /// filter.insert_batch(&[1u64, 2, 3])?;
    ///
    /// let results = filter.contains_batch(&[1u64, 2, 3, 4, 5]);
    /// assert_eq!(results, vec![true, true, true, false, false]);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Test membership and identify which sub-filter produced the positive result.
    ///
    /// Returns `(true, Some(index))` when the item matches sub-filter `index`, or
    /// `(false, None)` when all sub-filters return negative. The returned index is
    /// always the original absolute position in `filters` regardless of query strategy.
    ///
    /// Useful for diagnosing query patterns or confirming that recent inserts
    /// land in the expected filter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(10, 0.01)?;
    /// for i in 0u64..100 { filter.insert(&i); }
    ///
    /// let (found, idx) = filter.contains_with_provenance(&50u64);
    /// assert!(found);
    /// println!("50 is in sub-filter {}", idx.unwrap());
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn contains_with_provenance(&self, item: &T) -> (bool, Option<usize>) {
        match self.query_strategy {
            QueryStrategy::Forward => {
                for (idx, (filter, &nonempty)) in self.filters.iter()
                    .zip(self.filter_nonempty.iter())
                    .enumerate()
                {
                    if !nonempty { continue; }
                    if filter.contains(item) { return (true, Some(idx)); }
                }
            }
            QueryStrategy::Reverse => {
                // enumerate().rev() preserves original indices — correct provenance.
                for (idx, (filter, &nonempty)) in self.filters.iter()
                    .zip(self.filter_nonempty.iter())
                    .enumerate()
                    .rev()
                {
                    if !nonempty { continue; }
                    if filter.contains(item) { return (true, Some(idx)); }
                }
            }
        }
        (false, None)
    }


    /// Test membership with a detailed execution trace.
    ///
    /// Records per-filter latency, hash counts, fill rates, and early-termination
    /// status. Intended for offline debugging; the `Instant::now()` calls add
    /// measurable overhead and should not be used on production hot paths.
    ///
    /// Only available with the `trace` feature enabled.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// #[cfg(feature = "trace")]
    /// {
    ///     let (result, trace) = filter.contains_traced(&42u64);
    ///     println!("{}", trace.format_detailed());
    /// }
    /// ```
    #[cfg(feature = "trace")]
    #[must_use]
    pub fn contains_traced(&self, item: &T) -> (bool, QueryTrace) {
        use std::time::Instant;

        let strategy_name = format!("{:?}", self.query_strategy);
        let mut builder = QueryTraceBuilder::new(&strategy_name);

        // Static dispatch — no heap allocation on the hot query path.
        match self.query_strategy {
            QueryStrategy::Forward => {
                for (idx, filter) in self.filters.iter().enumerate() {
                    let start   = Instant::now();
                    let matched = filter.contains(item);
                    builder.record_filter(
                        idx, matched,
                        filter.hash_count(),
                        filter.hash_count(),
                        filter.fill_rate(),
                        start,
                    );
                    if matched {
                        return (true, builder.finish());
                    }
                }
            }
            QueryStrategy::Reverse => {
                for (idx, filter) in self.filters.iter().enumerate().rev() {
                    let start   = Instant::now();
                    let matched = filter.contains(item);
                    builder.record_filter(
                        idx, matched,
                        filter.hash_count(),
                        filter.hash_count(),
                        filter.fill_rate(),
                        start,
                    );
                    if matched {
                        return (true, builder.finish());
                    }
                }
            }
        }
        (false, builder.finish())
    }

    /// Reset the filter to its initial state, returning an error if the initial
    /// sub-filter cannot be recreated.
    ///
    /// Drops all existing sub-filters and their allocations, resets all counters
    /// and bookkeeping to zero, and creates a single fresh sub-filter sized to
    /// `initial_capacity` / `target_fpr`. For `Adaptive` growth, `error_ratio`
    /// is also reset to `initial_ratio`.
    ///
    /// Prefer this over `clear` in production code so that filter creation
    /// failures (e.g. out-of-memory) can be handled rather than panicked.
    ///
    /// # Errors
    ///
    /// - Any error from [`StandardBloomFilter::with_hasher`] (typically OOM).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?;
    /// for i in 0u64..5_000 { filter.insert(&i); }
    /// assert!(filter.filter_count() > 1);
    ///
    /// filter.clear_checked()?;
    ///
    /// assert_eq!(filter.filter_count(), 1);
    /// assert_eq!(filter.len(), 0);
    /// assert!(!filter.contains(&0u64));
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    pub fn clear_checked(&mut self) -> Result<()> {
        let capacity = self.calculate_next_capacity(0)?;

        let fpr = self.target_fpr;

        let replacement = StandardBloomFilter::with_hasher(
            capacity,
            fpr,
            self.hasher.clone(),
        )?;

        let new_sketch = self.track_cardinality.then(HyperLogLog::new);

        if let GrowthStrategy::Adaptive { initial_ratio, .. } = self.growth {
            self.error_ratio = initial_ratio;
        }

        self.filters.clear();
        self.filter_nonempty.clear();
        self.cardinality_sketches.clear();
        self.growth_history.clear();

        self.total_items = 0;
        self.items_in_current_filter = 0;
        self.current_filter_threshold =
            ((capacity as f64) * self.fill_threshold).ceil() as usize;
        self.current_filter_threshold = self.current_filter_threshold.max(1);

        self.filters.push(replacement);
        self.filter_nonempty.push(false);

        if let Some(sketch) = new_sketch {
            self.cardinality_sketches.push(sketch);
        }

        // Record the reset as a growth event at index 0.
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.growth_history.push_back(GrowthEvent {
            timestamp:    ts,
            filter_index: 0,
            capacity,
            fpr,
            total_items:  0,
        });

        Ok(())
    }

    /// Reset the filter to its initial state.
    ///
    /// Delegates to [`clear_checked`](Self::clear_checked). Panics if the initial
    /// sub-filter cannot be recreated (e.g. OOM). Use
    /// [`clear_checked`](Self::clear_checked) in production code where allocation
    /// failures must be handled gracefully rather than aborted.
    ///
    /// # Panics
    ///
    /// Panics if [`clear_checked`](Self::clear_checked) returns an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(100, 0.01)?;
    /// filter.insert(&1u64);
    /// filter.clear();
    /// assert_eq!(filter.len(), 0);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    pub fn clear(&mut self) {
        self.clear_checked()
            .expect("ScalableBloomFilter::clear() failed to recreate initial filter")
    }
}


// ── ANALYTICS ─────────────────────────────────────────────────────────────────

impl<T, H> ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Predict the compound false positive rate at a hypothetical future item count.
    ///
    /// Projects forward by simulating how many additional sub-filters would be
    /// required to accommodate `future_items` items under the current growth strategy.
    /// Returns the compound FPR using the complement rule across all projected filters.
    ///
    /// This is a planning tool: use it to decide whether the configured `target_fpr`
    /// and `initial_capacity` will remain acceptable at expected dataset sizes before
    /// you hit them in production.
    ///
    /// # Parameters
    ///
    /// - `future_items`: the projected total insertion count to evaluate. Values
    ///   less than or equal to the current `total_items` return the current FPR.
    ///
    /// # Notes
    ///
    /// The projection assumes all subsequent insertions use the current `error_ratio`
    /// and growth strategy. For `Adaptive` growth, the ratio is frozen at its current
    /// value for the projection.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?;
    /// for i in 0u64..5_000 { filter.insert(&i); }
    ///
    /// println!("FPR at 10K:  {:.4}%", filter.predict_fpr(10_000) * 100.0);
    /// println!("FPR at 100K: {:.4}%", filter.predict_fpr(100_000) * 100.0);
    /// println!("FPR at 1M:   {:.4}%", filter.predict_fpr(1_000_000) * 100.0);
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn predict_fpr(&self, at_item_count: usize) -> f64 {
        if at_item_count <= self.total_items {
            return self.estimate_fpr();
        }

        // Walk the capacity sequence to determine how many sub-filters are needed.
        let mut cumulative        = 0usize;
        let mut estimated_filters = 0usize;

        loop {
            if cumulative >= at_item_count || estimated_filters >= MAX_FILTERS {
                break;
            }
            let cap = self
                .calculate_next_capacity(estimated_filters)
                .unwrap_or(self.initial_capacity);
            let usable = ((cap as f64) * self.fill_threshold) as usize;
            cumulative = cumulative.saturating_add(usable.max(1));
            estimated_filters += 1;
        }

        let n = estimated_filters.max(self.filters.len());

        // Complement rule: FPR_total = 1 - prod(1 - FPR_i)
        let mut product = 1.0f64;
        for i in 0..n {
            let fpr_i = (self.target_fpr * self.error_ratio.powi(i as i32)).max(MIN_FPR);
            product *= 1.0 - fpr_i;
        }
        1.0 - product
    }

    /// Return the individual FPR and its fraction of the compound FPR for each sub-filter.
    ///
    /// Returns a `Vec` of `(filter_index, individual_fpr, fractional_contribution)`
    /// tuples in index order (oldest first). Contributions sum to 1.0.
    ///
    /// Identifies which sub-filters are driving the overall FPR — typically the
    /// oldest filters with the loosest per-filter FPR. Useful when tuning
    /// `error_ratio`: if the first sub-filter accounts for >90% of the total FPR,
    /// a higher ratio (tighter tightening) is warranted.
    ///
    /// Returns an empty `Vec` if there are no sub-filters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(100, 0.01)?;
    /// for i in 0u64..1_000 { filter.insert(&i); }
    ///
    /// for (idx, fpr, share) in filter.filter_fpr_breakdown() {
    ///     println!("Filter {idx}: FPR = {:.4}%, share = {:.1}%",
    ///         fpr * 100.0, share * 100.0);
    /// }
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn filter_fpr_breakdown(&self) -> Vec<(usize, f64, f64)> {
        let total_fpr = self.estimate_fpr();
        let n = self.filters.len();

        self.filters
            .iter()
            .enumerate()
            .map(|(idx, filter)| {
                let individual_fpr = filter.estimate_fpr();
                let contribution = if total_fpr > 0.0 {
                    individual_fpr / total_fpr
                } else if n > 0 {
                    1.0 / n as f64
                } else {
                    0.0
                };
                (idx, individual_fpr, contribution)
            })
            .collect()
    }

    /// Compute the compound false positive rate using the complement rule.
    ///
    /// The result is `1 − ∏(1 − pᵢ)` where `pᵢ` is the configured FPR of
    /// sub-filter i. This is the tightest valid FPR bound; it is always ≤
    /// the union bound returned by [`max_fpr`](Self::max_fpr).
    ///
    /// Returns 0.0 for an empty filter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?;
    /// for i in 0u64..10_000 { filter.insert(&i); }
    ///
    /// let fpr = filter.estimate_fpr_exact();
    /// assert!(fpr > 0.0 && fpr < 0.1); // strictly better than 10%
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn estimate_fpr_exact(&self) -> f64 {
        if self.filters.is_empty() {
            return 0.0;
        }
        1.0 - self
            .filters
            .iter()
            .map(|f| 1.0 - f.estimate_fpr())
            .product::<f64>()
    }

    /// Alias for [`estimate_fpr_exact`](Self::estimate_fpr_exact).
    ///
    /// Provided for API ergonomics alongside `max_fpr` and `predict_fpr`.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        self.estimate_fpr_exact()
    }

    /// Compute the compound FPR using the union (additive) bound.
    ///
    /// The result is `∑ pᵢ` where `pᵢ` is the configured FPR of sub-filter i.
    /// This is always ≥ `estimate_fpr()` and is appropriate for conservative
    /// capacity planning where you need a safe upper bound.
    ///
    /// For Almeida 2007's convergent-series bound with ratio r:
    /// `max_fpr ≤ target_fpr / (1 − r)`. For r = 0.5, the bound is 2 × target_fpr.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?;
    /// for i in 0u64..5_000 { filter.insert(&i); }
    ///
    /// // Union bound is always ≥ complement-rule estimate.
    /// assert!(filter.max_fpr() >= filter.estimate_fpr());
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn max_fpr(&self) -> f64 {
        self.filters.iter().map(|f| f.estimate_fpr()).sum()
    }

    /// Estimate the number of distinct items inserted since creation or last `clear`.
    ///
    /// Requires [`with_cardinality_tracking`](Self::with_cardinality_tracking) to have
    /// been called during construction. If cardinality tracking is not enabled, returns
    /// `total_items` (insertion count including duplicates) as a fallback.
    ///
    /// When enabled, merges the HyperLogLog sketches from all sub-filters and returns
    /// the combined estimate. Accuracy is ±~1% for cardinalities above ~1,000; for
    /// small sets (< 50 unique items) the sparse mode returns the exact count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<u64> = ScalableBloomFilter::new(1_000, 0.01)?
    ///     .with_cardinality_tracking();
    ///
    /// for _ in 0..5 {
    ///     for i in 0u64..10_000 { filter.insert(&i); }
    /// }
    ///
    /// assert_eq!(filter.len(), 50_000);                    // total insertions
    /// let unique = filter.estimate_unique_count();
    /// assert!((unique as i64 - 10_000).abs() < 300);      // ±3% at 10K
    /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
    /// ```
    #[must_use]
    pub fn estimate_unique_count(&self) -> usize {
        if !self.track_cardinality || self.cardinality_sketches.is_empty() {
            return self.total_items; // Fallback to total inserts
        }

        // Merge all HLL sketches
        let mut merged = HyperLogLog::new();
        for sketch in &self.cardinality_sketches {
            merged.merge(sketch);
        }
        merged.estimate()
    }

    /// Get cardinality estimation error bound
    ///
    /// HyperLogLog++ theoretical error: ±1.04 / sqrt(m) where m = 16384
    #[must_use]
    pub fn cardinality_error_bound(&self) -> f64 {
        1.04 / (HLL_REGISTER_COUNT as f64).sqrt()
    }

    /// Get comprehensive health metrics
    ///
    /// Provides 13 metrics for production monitoring.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ScalableBloomFilter;
    ///
    /// let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01).unwrap();
    /// for i in 0..5000 {
    ///     filter.insert(&i);
    /// }
    ///
    /// let metrics = filter.health_metrics();
    /// println!("{}", metrics);
    /// ```
    #[must_use]
    pub fn health_metrics(&self) -> ScalableHealthMetrics {
        let avg_fill_rate = if !self.filters.is_empty() {
            self.filters.iter().map(|f| f.fill_rate()).sum::<f64>() / self.filters.len() as f64
        } else {
            0.0
        };

        ScalableHealthMetrics {
            filter_count: self.filters.len(),
            total_capacity: self.total_capacity(),
            total_items: self.total_items,
            estimated_fpr: self.estimate_fpr(),
            max_fpr: self.max_fpr(),
            target_fpr: self.target_fpr,
            current_error_ratio: self.error_ratio,
            current_fill_rate: self.current_fill_rate(),
            avg_fill_rate,
            memory_bytes: self.memory_usage(),
            remaining_growth: self.remaining_growth_capacity(),
            growth_events: self.growth_history.len(),
            query_strategy: self.query_strategy,
        }
    }

    // ── ACCESSORS ─────────────────────────────────────────────────────────────────

    /// Get the number of sub-filters
    #[must_use]
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Get total capacity across all filters
    #[must_use]
    pub fn total_capacity(&self) -> usize {
        self.filters.iter().map(|f| f.expected_items()).sum()
    }

    /// Get total items inserted (counts duplicates)
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_items
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_items == 0
    }

    /// Get current filter fill rate
    #[must_use]
    pub fn current_fill_rate(&self) -> f64 {
        self.filters
            .last()
            .map(|f| f.fill_rate())
            .unwrap_or(0.0)
    }

    /// Get aggregate fill rate across all filters
    #[must_use]
    pub fn aggregate_fill_rate(&self) -> f64 {
        if self.filters.is_empty() {
            return 0.0;
        }

        let total_bits: usize = self.filters.iter().map(|f| f.size()).sum();
        let set_bits: usize = self.filters.iter().map(|f| f.count_set_bits()).sum();

        if total_bits == 0 {
            0.0
        } else {
            set_bits as f64 / total_bits as f64
        }
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.filters.iter().map(|f| f.memory_usage()).sum::<usize>()
            + std::mem::size_of::<Self>()
            + self.cardinality_sketches.iter().map(|h| h.memory_usage()).sum::<usize>()
    }

    /// Check if at maximum capacity
    #[must_use]
    pub fn is_at_max_capacity(&self) -> bool {
        self.filters.len() >= MAX_FILTERS
    }

    /// Check if near maximum capacity
    #[must_use]
    pub fn is_near_capacity(&self) -> bool {
        self.filters.len() + CAPACITY_WARNING_THRESHOLD >= MAX_FILTERS
    }

    /// Get remaining growth capacity
    #[must_use]
    pub fn remaining_growth_capacity(&self) -> usize {
        MAX_FILTERS.saturating_sub(self.filters.len())
    }

    /// Get detailed statistics for each sub-filter
    ///
    /// Returns: `(capacity, fill_rate, fpr)` for each filter
    #[must_use]
    pub fn filter_stats(&self) -> Vec<(usize, f64, f64)> {
        self.filters
            .iter()
            .map(|f| (f.expected_items(), f.fill_rate(), f.estimate_fpr()))
            .collect()
    }

    /// Get growth strategy
    #[must_use]
    pub fn growth_strategy(&self) -> GrowthStrategy {
        self.growth
    }

    /// Get error ratio
    #[must_use]
    pub fn error_ratio(&self) -> f64 {
        self.error_ratio
    }

    /// Get fill threshold
    #[must_use]
    pub fn fill_threshold(&self) -> f64 {
        self.fill_threshold
    }

    /// Set fill threshold
    ///
    /// # Errors
    ///
    /// Returns error if threshold is not in (0.0, 1.0).
    pub fn set_fill_threshold(&mut self, threshold: f64) -> Result<()> {
        if threshold <= 0.0 || threshold >= 1.0 {
            return Err(BloomCraftError::invalid_parameters(
                format!("fill_threshold must be in (0.0, 1.0), got {}", threshold)
            ));
        }
        self.fill_threshold = threshold;
        Ok(())
    }

    /// Get target FPR
    #[must_use]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Get initial capacity
    #[must_use]
    pub fn initial_capacity(&self) -> usize {
        self.initial_capacity
    }

    /// Returns the total number of items inserted since creation or last clear.
    ///
    /// This is an O(1) read of an internal counter. It counts all insertions
    /// including duplicates, not unique items. For unique-item estimation,
    /// use [`estimate_unique_count`].
    #[must_use]
    #[inline]
    pub fn total_items(&self) -> usize {
        self.total_items
    }
}


// ── TRAIT IMPLEMENTATIONS ─────────────────────────────────────────────────────

/// Implement the core BloomFilter trait
impl<T, H> BloomFilter<T> for ScalableBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn insert(&mut self, item: &T) {
        ScalableBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        ScalableBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        ScalableBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        ScalableBloomFilter::len(self)
    }

    fn is_empty(&self) -> bool {
        ScalableBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.total_capacity()
    }

    fn bit_count(&self) -> usize {
        self.filters.iter().map(|f| f.bit_count()).sum()
    }

    fn hash_count(&self) -> usize {
        self.filters
            .first()
            .map(|f| f.hash_count())
            .unwrap_or(0)
    }

    fn count_set_bits(&self) -> usize {
        self.filters.iter().map(|f| f.count_set_bits()).sum()
    }
}

/// Display implementation for debugging
impl<T, H> fmt::Display for ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ScalableBloomFilter {{ filters: {}, capacity: {}, items: {}, fill: {:.1}%, est_fpr: {:.4}% }}",
            self.filter_count(),
            self.total_capacity(),
            self.len(),
            self.current_fill_rate() * 100.0,
            self.estimate_fpr() * 100.0
        )
    }
}

/// Extend trait for ergonomic bulk inserts
impl<T, H> std::iter::Extend<T> for ScalableBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(&item);
        }
    }
}

/// FromIterator trait for creating from iterators
impl<T> std::iter::FromIterator<T> for ScalableBloomFilter<T>
where
    T: Hash,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let estimated_count = lower.max(100);
        
        let mut filter = Self::new(estimated_count, 0.01)
            .expect("ScalableBloomFilter::from_iter: failed to create filter");
        
        for item in iter {
            filter.insert(&item);
        }
        
        filter
    }
}

// ── CONCURRENT VARIANT: AtomicScalableBloomFilter ─────────────────────────────

/// Lock-minimised concurrent scalable Bloom filter.
///
/// # Architecture
///
/// [`AtomicScalableBloomFilter`] wraps a growable sequence of [`ShardedFilter`]
/// instances behind an `Arc<AtomicScalableInner>`. Each [`ShardedFilter`] is
/// itself a fixed array of independent [`StandardBloomFilter`] shards, routed
/// by the upper bits of each item's hash.
///
/// ```text
/// AtomicScalableBloomFilter<T>
///   └─ Arc<AtomicScalableInner>
///         ├─ RwLock<Vec<Arc<ShardedFilter>>>   ← filter list
///         ├─ [AtomicBool; MAX_FILTERS]          ← nonempty flags
///         ├─ CacheAligned<AtomicUsize>          ← current_filter index
///         ├─ CacheAligned<AtomicUsize>          ← total_items counter
///         ├─ CacheAligned<AtomicBool>           ← growth_in_progress flag
///         ├─ CacheAligned<AtomicUsize>          ← check_interval threshold
///         └─ ConcurrentConfig                  ← immutable after construction
///
/// ShardedFilter<T>
///   └─ Vec<StandardBloomFilter>   (len == shard_count, typically 8–16)
/// ```
///
/// # Concurrency model
///
/// | Operation | Mechanism | Notes |
/// |---|---|---|
/// | `contains` | RwLock (read) for filter-list snapshot | Fully concurrent with other reads and inserts |
/// | `insert` | RwLock (read) + per-filter lock-free atomic bit-set | Different shards run in parallel |
/// | `insert_batch` | As above, batched per shard for cache locality | Write lock released between shard buckets |
/// | `clear` | RwLock (write) | Serialises all concurrent inserts; alloc done before lock |
/// | Growth | `growth_in_progress` CAS + RwLock (write) for ~10 ns | Alloc runs outside write lock |
///
/// # Memory ordering
///
/// Hot-path atomics use `Acquire`/`Release` or `Relaxed` deliberately:
///
/// - `current_filter`: `Release` on write, `Acquire` on read — any thread
///   loading a filter index is guaranteed to observe the filter it points to.
/// - `total_items`, `check_interval`: `Relaxed` reads/writes — slight staleness
///   is acceptable; a missed growth trigger is caught on the next insert.
/// - `growth_in_progress`: `AcqRel`/`Relaxed` compare-exchange — full ordering
///   on the winning thread, no guarantees needed on the losing threads.
/// - `filter_nonempty`: `Relaxed` — a false negative here causes a redundant
///   full scan, not a correctness failure.
///
/// # False positive rate
///
/// Under `Geometric(2.0)` growth, the compound FPR across `n` sub-filters is:
///
/// ```text
/// FPR_total = 1 − ∏(1 − p · rⁱ)  for i = 0..n−1
/// ```
///
/// where `p = target_fpr` and `r = error_ratio`. With `p = 0.01`, `r = 0.5`,
/// this converges to approximately `0.02` (2×p) regardless of `n`.
///
/// # Examples
///
/// ## Basic concurrent usage
///
/// ```rust
/// use bloomcraft::filters::AtomicScalableBloomFilter;
/// use std::sync::Arc;
/// use std::thread;
///
/// let filter = Arc::new(AtomicScalableBloomFilter::<u64>::new(10_000, 0.01)?);
///
/// let handles: Vec<_> = (0u64..8)
///     .map(|thread_id| {
///         let f = Arc::clone(&filter);
///         thread::spawn(move || {
///             for i in 0..1_000 {
///                 f.insert(&(thread_id * 1_000 + i));
///             }
///         })
///     })
///     .collect();
///
/// for h in handles { h.join().unwrap(); }
///
/// assert_eq!(filter.len(), 8_000);
/// for i in 0u64..8_000 {
///     assert!(filter.contains(&i), "false negative at {}", i);
/// }
/// # Ok::<(), bloomcraft::error::BloomCraftError>(())
/// ```
///
/// ## URL deduplication pipeline
///
/// ```rust
/// use bloomcraft::filters::AtomicScalableBloomFilter;
/// use std::sync::Arc;
///
/// let seen: Arc<AtomicScalableBloomFilter<String>> =
///     Arc::new(AtomicScalableBloomFilter::new(1_000_000, 0.001)?);
///
/// // Multiple crawler threads share the same filter.
/// let is_new = !seen.contains(&"https://example.com".to_string());
/// if is_new {
///     seen.insert(&"https://example.com".to_string());
/// }
/// # Ok::<(), bloomcraft::error::BloomCraftError>(())
/// ```
#[cfg(feature = "concurrent")]
pub mod concurrent {
    use super::*;
    use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
    use std::sync::{Arc, RwLock};

    // ── CACHE-LINE PADDING CONSTANTS ────────────────────────────────────────────

    /// Target cache line size (64 bytes on x86-64, ARM64, RISC-V)
    const CACHE_LINE_SIZE: usize = 64;

    /// Size of CacheAligned<AtomicUsize> wrapper
    /// 
    /// On 64-bit systems: AtomicUsize is 8 bytes, CacheAligned adds alignment
    /// but doesn't change size. Total padded to 64 bytes.
    const CACHE_ALIGNED_USIZE_SIZE: usize = std::mem::size_of::<CacheAligned<AtomicUsize>>();

    /// Size of CacheAligned<AtomicBool> wrapper
    /// 
    /// On all systems: AtomicBool is 1 byte, CacheAligned adds alignment.
    /// Total padded to 64 bytes.
    const CACHE_ALIGNED_BOOL_SIZE: usize = std::mem::size_of::<CacheAligned<AtomicBool>>();

    /// Padding after CacheAligned<AtomicUsize> to reach 64 bytes
    const PAD_AFTER_USIZE: usize = CACHE_LINE_SIZE - CACHE_ALIGNED_USIZE_SIZE;

    /// Padding after CacheAligned<AtomicBool> to reach 64 bytes
    const PAD_AFTER_BOOL: usize = CACHE_LINE_SIZE - CACHE_ALIGNED_BOOL_SIZE;
    
    // ── CacheAligned ─────────────────────────────────────────────────────────

    /// Forces the wrapped value onto its own 64-byte cache line.
    ///
    /// `#[repr(align(64))]` guarantees that no other field shares the same
    /// cache line as `T`. Combined with the explicit `_pad*` fields in
    /// [`AtomicScalableInner`], this eliminates false sharing between the
    /// hot-path atomics on multi-core systems.
    ///
    /// Use [`std::ops::Deref`] to access the inner value directly.
    #[repr(align(64))]
    struct CacheAligned<T> {
        value: T,
    }

    impl<T> CacheAligned<T> {
        fn new(value: T) -> Self {
            Self { value }
        }
    }

    impl<T> std::ops::Deref for CacheAligned<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.value
        }
    }

    // ── ShardedFilter ─────────────────────────────────────────────────────────

    /// A fixed-capacity Bloom filter sharded across `shard_count` independent
    /// [`StandardBloomFilter`] instances.
    ///
    /// Items are routed to shards by the lower bits of `InternalHasher::hash_one`,
    /// which must agree exactly with the routing used by `contains`. A mismatch
    /// between insert-time and query-time routing produces false negatives.
    ///
    /// Sharding buys two things:
    ///
    /// 1. **Write parallelism**: concurrent inserts to different shards run
    ///    simultaneously with zero contention.  `StandardBloomFilter::insert`
    ///    uses `AtomicU64::fetch_or` internally — no wrapper lock required.
    ///
    /// 2. **Cache locality**: each shard's bit array fits in fewer cache lines
    ///    than a monolithic filter of the same total capacity.  Under a uniform
    ///    hash distribution, each shard carries `1 / shard_count` of the total
    ///    load, so its bit array is `shard_count`× smaller.
    ///
    /// # Invariants
    ///
    /// - `shards.len() == shard_count` at all times after construction.
    /// - All shards were created with the same `fpr` and `hasher`.
    /// - `design_fpr` equals the `fpr` passed to `new`; it is returned as a
    ///   baseline when no shards exist (degenerate case).
    struct ShardedFilter<T, H>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default + Send + Sync,
    {
        /// Independent sub-filters, one per shard slot.
        ///
        /// Direct field access is intentional: `insert_batch` routes items into
        /// pre-computed shard indices and writes via `insert_into_shard` to avoid
        /// a redundant hash recomputation on the hot path.
        shards: Vec<StandardBloomFilter<T, H>>,

        /// Number of shards. Cached separately so callers do not need to read
        /// `shards.len()` when the vec is borrowed.
        shard_count: usize,

        /// The FPR this filter was configured for, returned by `estimate_fpr`
        /// when `shards` is empty.
        design_fpr: f64,
    }

    impl<T, H> ShardedFilter<T, H>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default,
    {
        /// Allocate `shard_count` independent sub-filters, each sized for
        /// `ceil(capacity / shard_count)` items at `fpr`.
        ///
        /// This is the only allocation site for a `ShardedFilter`. It is
        /// intentionally expensive — it zeroes up to `shard_count` bit arrays —
        /// and **must not** run while holding the `filters` write-lock in
        /// [`AtomicScalableInner`]. See [`AtomicScalableBloomFilter::perform_growth`]
        /// for the correct call site (Phase 2, outside the lock).
        ///
        /// Ceiling division ensures every item has a valid shard slot regardless
        /// of how `capacity` divides by `shard_count`.
        ///
        /// # Errors
        ///
        /// Propagates any error from [`StandardBloomFilter::with_hasher`], most
        /// commonly `InvalidParameters` if `capacity == 0` or `fpr` is out of range.
        fn new(capacity: usize, fpr: f64, shard_count: usize, hasher: H) -> Result<Self> {
            let per_shard_capacity = (capacity + shard_count - 1) / shard_count;

            let shards = (0..shard_count)
                .map(|_| StandardBloomFilter::with_hasher(per_shard_capacity, fpr, hasher.clone()))
                .collect::<Result<Vec<_>>>()?;

            Ok(Self {
                shards,
                shard_count,
                design_fpr: fpr,
            })
        }

        /// Insert `item` into its designated shard.
        ///
        /// The shard index is derived from `InternalHasher::hash_one(item)`.
        /// This is lock-free: `StandardBloomFilter::insert` uses
        /// `AtomicU64::fetch_or` with `Release` ordering internally.
        #[inline]
        fn insert(&self, item: &T) {
            let shard_idx = InternalHasher::hash_one(item) as usize % self.shard_count;
            self.shards[shard_idx].insert(item);
        }

        /// Insert `item` into the pre-computed shard at `shard_idx`.
        ///
        /// Used by [`AtomicScalableBloomFilter::insert_batch`] to avoid
        /// recomputing the shard hash for items that have already been bucketed.
        ///
        /// # Invariant
        ///
        /// The caller must guarantee:
        /// `shard_idx == InternalHasher::hash_one(item) as usize % self.shard_count`
        ///
        /// Violating this invariant causes false negatives — an item inserted at
        /// the wrong shard will never be found by `contains`.
        #[inline]
        fn insert_into_shard(&self, shard_idx: usize, item: &T) {
            self.shards[shard_idx].insert(item);
        }

        /// Test whether `item` may be in this filter.
        ///
        /// Lock-free. `StandardBloomFilter::contains` uses `AtomicU64::load`
        /// with `Acquire` ordering internally — all `fetch_or` writes that
        /// preceded this load on any thread are visible here.
        #[inline]
        fn contains(&self, item: &T) -> bool {
            let shard_idx = InternalHasher::hash_one(item) as usize % self.shard_count;
            self.shards[shard_idx].contains(item)
        }

        /// Aggregate fill rate across all shards.
        ///
        /// Computed from raw bit counts rather than by averaging per-shard fill
        /// rates. When shards carry slightly unequal load due to hash skew, the
        /// raw-count aggregate is more accurate than the arithmetic mean of
        /// per-shard values.
        fn fill_rate(&self) -> f64 {
            if self.shards.is_empty() {
                return 0.0;
            }
            let total_bits: usize = self.shards.iter().map(|s| s.bit_count()).sum();
            let set_bits: usize = self.shards.iter().map(|s| s.count_set_bits()).sum();
            if total_bits == 0 {
                return 0.0;
            }
            set_bits as f64 / total_bits as f64
        }

        /// Total expected items across all shards.
        fn expected_items(&self) -> usize {
            self.shards.iter().map(|s| s.expected_items()).sum()
        }

        /// Memory footprint: struct overhead + all shard bit arrays.
        fn memory_usage(&self) -> usize {
            std::mem::size_of::<Self>()
                + self.shards.iter().map(|s| s.memory_usage()).sum::<usize>()
        }

        /// Raw bit-level statistics: `(total_bits, set_bits, utilization_pct)`.
        ///
        /// Reads `count_set_bits()` directly from each shard's atomic counters
        /// rather than deriving from `fill_rate()`. Avoids the rounding error
        /// introduced by `(fill_rate * bit_count) as usize`.
        fn bit_statistics(&self) -> (usize, usize, f64) {
            let total_bits: usize = self.shards.iter().map(|s| s.bit_count()).sum();
            let set_bits: usize = self.shards.iter().map(|s| s.count_set_bits()).sum();
            let utilization = if total_bits > 0 {
                set_bits as f64 / total_bits as f64 * 100.0
            } else {
                0.0
            };
            (total_bits, set_bits, utilization)
        }

        /// Estimated FPR based on the actual fill state of each shard.
        ///
        /// Per-shard FPRs are averaged rather than combined via the complement
        /// rule. This is correct because `InternalHasher` distributes items
        /// uniformly across shards — each shard carries equal expected load and
        /// its FPR is an equally-weighted sample of the whole.
        ///
        /// Returns `design_fpr` when `shards` is empty (degenerate state that
        /// cannot occur after a successful `new`).
        pub fn estimate_fpr(&self) -> f64 {
            if self.shards.is_empty() {
                return self.design_fpr;
            }
            let fpr_sum: f64 = self.shards.iter().map(|s| s.estimate_fpr()).sum();
            fpr_sum / self.shard_count as f64
        }
    }

    // ── AtomicScalableBloomFilter (public handle) ─────────────────────────────

    /// A concurrent, automatically-growing Bloom filter.
    ///
    /// The public handle is a thin `Arc` wrapper around [`AtomicScalableInner`].
    /// Cloning is O(1) — it increments the reference count, not the bit arrays.
    /// All clones share the same underlying filter state.
    ///
    /// See the [module-level documentation](self) for the concurrency model,
    /// memory ordering guarantees, and FPR analysis.
    pub struct AtomicScalableBloomFilter<T, H = StdHasher>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default + Send + Sync,
    {
        inner: Arc<AtomicScalableInner<T, H>>,
    }

    // ── AtomicScalableInner (shared state) ────────────────────────────────────

    /// Shared state for all clones of an [`AtomicScalableBloomFilter`].
    ///
    /// Each hot-path atomic is placed on its own 64-byte cache line via
    /// `CacheAligned<T>` plus explicit `_pad*` arrays. On a 16-core system
    /// where all threads hammer the same filter, unpadded atomics would cause
    /// the owning cache line to bounce between cores on every access, degrading
    /// throughput by 3–10×.
    ///
    /// # Field ordering
    ///
    /// Fields are ordered from most-frequently-read to least-frequently-read:
    /// `current_filter` and `check_interval` are read on every insert;
    /// `config` and `growth_in_progress` are read only on growth events.
    struct AtomicScalableInner<T, H>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default + Send + Sync,
    {
        /// Sequence of sub-filters with fine-grained locking
        /// 
        /// Each filter is independently lockable, allowing parallel inserts
        /// to different filters. The Vec itself is protected by RwLock.
        filters: RwLock<Vec<Arc<ShardedFilter<T, H>>>>,

        /// Tracks which sub-filters are non-empty
        filter_nonempty: [AtomicBool; MAX_FILTERS],

        /// Index of current filter for inserts
        current_filter: CacheAligned<AtomicUsize>,
        /// Cache-line padding (ensures `current_filter` has exclusive cache line)
        _pad1: [u8; PAD_AFTER_USIZE],

        /// Total items inserted (Relaxed ordering - exact count not critical)
        total_items: CacheAligned<AtomicUsize>,
        /// Cache-line padding (ensures `total_items` has exclusive cache line)
        _pad2: [u8; PAD_AFTER_USIZE],

        /// Growth coordination flag
        growth_in_progress: CacheAligned<AtomicBool>,
        /// Cache-line padding (ensures `growth_in_progress` has exclusive cache line)
        _pad3: [u8; PAD_AFTER_BOOL],

        /// Cached check interval (updated on growth)
        check_interval: CacheAligned<AtomicUsize>,
        
        /// Immutable configuration
        config: ConcurrentConfig<H>,
    }

    // ── ConcurrentConfig ─────────────────────────────────────────────────────

    /// Immutable configuration for [`AtomicScalableBloomFilter`].
    ///
    /// All fields are truly immutable after construction except `error_ratio`,
    /// which is stored as `AtomicU64` (f64 bits) so that `Adaptive` growth can
    /// update it from `&self`. The filters write-lock in `perform_growth`
    /// provides the happens-before edge that makes `Relaxed` atomic stores
    /// on `error_ratio` safe to read from any thread that subsequently acquires
    /// the filters read-lock.
    struct ConcurrentConfig<H>
    where
        H: BloomHasher + Clone + Default,
    {
        /// Item count for the zeroth sub-filter.
        initial_capacity: usize,

        /// FPR target for the zeroth sub-filter; tightened by `error_ratio`
        /// per subsequent filter.
        target_fpr: f64,

        /// FPR tightening ratio, encoded as `f64::to_bits()` in an `AtomicU64`
        /// to allow mutation from `&self` during `Adaptive` growth events.
        ///
        /// Read via `error_ratio()`. Written via `store_error_ratio()`, which
        /// must only be called while holding the filters write-lock.
        error_ratio: std::sync::atomic::AtomicU64,

        /// Fraction of a sub-filter's capacity that triggers growth.
        fill_threshold: f64,

        /// Growth strategy applied when computing the capacity and FPR of each
        /// new sub-filter.
        growth_strategy: GrowthStrategy,

        /// Hash function instance. Cloned once per `ShardedFilter::new` call
        /// and once per shard within it. Must be `Clone`.
        hasher: H,

        /// Number of shards per [`ShardedFilter`].
        ///
        /// Determined at construction via `optimal_shard_count()` and immutable
        /// thereafter. Changing shard count mid-life would break the routing
        /// invariant between insert and contains.
        shard_count: usize,
    }

    impl<H: BloomHasher + Clone + Default> ConcurrentConfig<H> {
        /// Load the current `error_ratio` as `f64`.
        ///
        /// `Relaxed` ordering is correct: callers either hold the filters
        /// write-lock (which provides the happens-before edge) or are computing
        /// an approximate FPR projection where staleness is acceptable.
        #[inline]
        fn error_ratio(&self) -> f64 {
            f64::from_bits(self.error_ratio.load(Ordering::Relaxed))
        }

        /// Store a new `error_ratio`.
        ///
        /// # Safety
        ///
        /// Must only be called while holding the filters write-lock.
        /// The lock provides the happens-before edge that makes this `Relaxed`
        /// store visible to any thread that subsequently acquires the read-lock.
        #[inline]
        fn store_error_ratio(&self, v: f64) {
            self.error_ratio.store(v.to_bits(), Ordering::Relaxed);
        }

        #[inline]
        fn fill_threshold(&self) -> f64 { self.fill_threshold }
    }

    // ── Shard count heuristic ─────────────────────────────────────────────────

    /// Return the optimal shard count for the current system.
    ///
    /// Uses the number of logical CPUs, capped at 16. Beyond 16 shards the
    /// per-shard capacity drops below the point where the hash function can
    /// distribute items uniformly, and the coordination overhead of routing
    /// outweighs the parallelism benefit.
    ///
    /// Falls back to 8 if `available_parallelism` is unavailable (WASM, some
    /// embedded targets).
    fn optimal_shard_count() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get().min(16))
            .unwrap_or(8)
    }

    // ── Constructors (StdHasher specialisation) ───────────────────────────────
    impl<T> AtomicScalableBloomFilter<T, StdHasher>
    where
        T: Hash + Send + Sync,
    {
        /// Create a concurrent scalable filter with default settings.
        ///
        /// Uses `StdHasher`, `Geometric(2.0)` growth, `error_ratio = 0.5`,
        /// and `fill_threshold = 0.5`. The shard count is chosen automatically
        /// via [`optimal_shard_count`].
        ///
        /// # Errors
        ///
        /// - `InvalidItemCount` if `initial_capacity == 0`.
        /// - `FalsePositiveRateOutOfBounds` if `target_fpr` is not in (0.0, 1.0).
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<u64> =
        ///     AtomicScalableBloomFilter::new(100_000, 0.01)?;
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        #[must_use]
        pub fn new(initial_capacity: usize, target_fpr: f64) -> Result<Self> {
            Ok(Self::with_hasher(initial_capacity, target_fpr, StdHasher::new())?)
        }

        /// Create a concurrent scalable filter with an explicit growth strategy.
        ///
        /// `error_ratio` controls how aggressively the FPR is tightened per
        /// sub-filter: a value of `0.5` halves the FPR at each generation.
        /// Valid range is (0.0, 1.0).
        ///
        /// # Errors
        ///
        /// See [`with_strategy_and_hasher`](Self::with_strategy_and_hasher).
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::{AtomicScalableBloomFilter, GrowthStrategy};
        ///
        /// let filter: AtomicScalableBloomFilter<String> =
        ///     AtomicScalableBloomFilter::with_strategy(
        ///         50_000,
        ///         0.001,
        ///         0.5,
        ///         GrowthStrategy::Geometric(2.0),
        ///     )?;
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        #[must_use]
        pub fn with_strategy(
            initial_capacity: usize,
            target_fpr: f64,
            error_ratio: f64,
            growth_strategy: GrowthStrategy,
        ) -> Result<Self> {
            Self::with_strategy_and_hasher(
                initial_capacity,
                target_fpr,
                error_ratio,
                growth_strategy,
                StdHasher::new(),
            )
        }

        /// Pre-allocate all sub-filters needed to hold `estimated_total_items`.
        ///
        /// Eliminates growth events — and their associated write-lock
        /// contention — during a high-throughput insert phase when the total
        /// item count is known in advance. Each sub-filter is fully allocated
        /// and zeroed at construction time, amortising allocation cost before
        /// concurrent inserts begin.
        ///
        /// The growth sequence assumes `Geometric(2.0)` regardless of the
        /// configured strategy. For other strategies, call
        /// [`new`](Self::new) and let the filter grow organically.
        ///
        /// # Errors
        ///
        /// - `InvalidParameters` if `estimated_total_items <= initial_capacity`.
        /// - Any error from `new` or the internal pre-allocation loop.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// // Pre-size for 10M items so inserts never trigger growth.
        /// let filter: AtomicScalableBloomFilter<u64> =
        ///     AtomicScalableBloomFilter::with_preallocated(100_000, 0.01, 10_000_000)?;
        ///
        /// // Parallel inserts proceed without any write-lock contention.
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        pub fn with_preallocated(
            initial_capacity: usize,
            target_fpr: f64,
            estimated_total_items: usize,
        ) -> Result<Self> {
            if estimated_total_items == 0 {
                return Err(BloomCraftError::invalid_parameters(
                    "estimated_total_items must be greater than 0",
                ));
            }

            let filter = Self::new(initial_capacity, target_fpr)?;

            if estimated_total_items <= initial_capacity {
                return Ok(filter);
            }

            let estimated_filters = {
                let mut count = 1usize;
                let mut capacity_sum = initial_capacity;
                while capacity_sum < estimated_total_items && count < MAX_FILTERS {
                    capacity_sum += initial_capacity * 2_usize.pow(count as u32);
                    count += 1;
                }
                count.min(MAX_FILTERS)
            };

            let mut new_filters: Vec<Arc<ShardedFilter<T, StdHasher>>> =
                Vec::with_capacity(estimated_filters.saturating_sub(1));

            for i in 1..estimated_filters {
                let capacity = filter.calculate_next_capacity(i)?;
                let fpr = filter.calculate_next_fpr(i);
                let f = Arc::new(ShardedFilter::new(
                    capacity,
                    fpr,
                    filter.inner.config.shard_count,
                    filter.inner.config.hasher.clone(),
                )?);
                new_filters.push(f);
            }
            
            {
                let mut filters = filter.inner.filters.write().unwrap();
                for f in new_filters {
                    if filters.len() < MAX_FILTERS {
                        filters.push(f);
                    }
                }
            }

            Ok(filter)
        }
    }

    // ── Constructors (generic) ────────────────────────────────────────────────

    impl<T, H> AtomicScalableBloomFilter<T, H>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default + Send + Sync,
    {
        /// Create a concurrent scalable filter with a custom hasher.
        ///
        /// Uses `Geometric(2.0)` growth and `error_ratio = 0.5`.
        ///
        /// # Errors
        ///
        /// See [`with_strategy_and_hasher`](Self::with_strategy_and_hasher).
        #[must_use]
        pub fn with_hasher(initial_capacity: usize, target_fpr: f64, hasher: H) -> Result<Self> {
            Self::with_strategy_and_hasher(
                initial_capacity,
                target_fpr,
                0.5,
                GrowthStrategy::Geometric(2.0),
                hasher,
            )
        }

        /// Create a concurrent scalable filter with full control over all parameters.
        ///
        /// This is the canonical constructor; all other constructors delegate here.
        ///
        /// # Parameters
        ///
        /// - `initial_capacity` — Expected item count for the first sub-filter.
        ///   Must be > 0.
        /// - `target_fpr` — False positive rate for the first sub-filter.
        ///   Must be in (0.0, 1.0).
        /// - `error_ratio` — Per-generation FPR tightening factor.
        ///   Must be in (0.0, 1.0). A value of `0.5` halves the FPR at each
        ///   growth event; `0.9` tightens slowly.
        /// - `growth_strategy` — Controls how sub-filter capacity scales.
        ///   [`GrowthStrategy::Geometric(2.0)`] doubles capacity each generation.
        /// - `hasher` — Hash function instance. Cloned once per sub-filter.
        ///
        /// # Errors
        ///
        /// - [`BloomCraftError::InvalidItemCount`] if `initial_capacity == 0`.
        /// - [`BloomCraftError::FalsePositiveRateOutOfBounds`] if `target_fpr`
        ///   is not in (0.0, 1.0).
        /// - [`BloomCraftError::InvalidParameters`] if `error_ratio` is not in
        ///   (0.0, 1.0).
        /// - Any allocation error from [`ShardedFilter::new`].
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::{AtomicScalableBloomFilter, GrowthStrategy};
        /// use bloomcraft::hash::hasher::StdHasher;
        ///
        /// let filter: AtomicScalableBloomFilter<String> =
        ///     AtomicScalableBloomFilter::with_strategy_and_hasher(
        ///         10_000,
        ///         0.005,
        ///         0.5,
        ///         GrowthStrategy::Adaptive {
        ///             initial_ratio: 0.5,
        ///             min_ratio: 0.1,
        ///             max_ratio: 0.9,
        ///         },
        ///         StdHasher::new(),
        ///     )?;
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        #[must_use]
        pub fn with_strategy_and_hasher(
            initial_capacity: usize,
            target_fpr: f64,
            error_ratio: f64,
            growth_strategy: GrowthStrategy,
            hasher: H,
        ) -> Result<Self> {
            if initial_capacity == 0 {
                return Err(BloomCraftError::invalid_item_count(initial_capacity));
            }

            if target_fpr <= 0.0 || target_fpr >= 1.0 {
                return Err(BloomCraftError::fp_rate_out_of_bounds(target_fpr));
            }

            if error_ratio <= 0.0 || error_ratio >= 1.0 {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "error_ratio must be in (0.0, 1.0), got {}",
                    error_ratio
                )));
            }

            let shard_count = optimal_shard_count();

            let initial_filter = Arc::new(ShardedFilter::new(
                initial_capacity,
                target_fpr,
                shard_count,
                hasher.clone(),
            )?);

            let initial_check_interval = (initial_capacity as f64 * DEFAULT_FILL_THRESHOLD) as usize;

            let config = ConcurrentConfig {
                initial_capacity,
                target_fpr,
                error_ratio: std::sync::atomic::AtomicU64::new(error_ratio.to_bits()),
                fill_threshold: DEFAULT_FILL_THRESHOLD,
                growth_strategy,
                hasher,
                shard_count,
            };

            let inner = Arc::new(AtomicScalableInner {
                filters: RwLock::new(vec![initial_filter]),
                filter_nonempty: std::array::from_fn(|_| AtomicBool::new(false)),
                current_filter: CacheAligned::new(AtomicUsize::new(0)),
                _pad1: [0; PAD_AFTER_USIZE], 
                total_items: CacheAligned::new(AtomicUsize::new(0)),
                _pad2: [0; PAD_AFTER_USIZE],
                growth_in_progress: CacheAligned::new(AtomicBool::new(false)),
                _pad3: [0; PAD_AFTER_BOOL],
                check_interval: CacheAligned::new(AtomicUsize::new(initial_check_interval)),
                config,
            });

            Ok(Self { inner })
        }

        // ── Core operations ───────────────────────────────────────────────────

        /// Insert `item` into the filter.
        ///
        /// # Concurrency
        ///
        /// Multiple threads may call `insert` simultaneously:
        ///
        /// - The active sub-filter's shard array is lock-free
        ///   (`AtomicU64::fetch_or` with `Release`). Concurrent inserts to
        ///   the same shard are safe and produce no lost updates.
        /// - Inserts do not block concurrent `contains` calls.
        /// - If a growth event fires mid-insert, `try_grow` is called by the
        ///   first thread to cross the threshold; all others continue inserting
        ///   into the current filter until the new one is activated.
        ///
        /// # Growth
        ///
        /// Growth is triggered when `total_items >= check_interval`. Only one
        /// thread performs the growth (via `growth_in_progress` CAS). Growth
        /// allocates outside the write lock; the write lock is held only for
        /// the Vec::push and atomic index advance (~10 ns).
        pub fn insert(&self, item: &T) {
            // Retry loop to handle growth without data loss
            loop {
                let current_idx = self.inner.current_filter.load(Ordering::Acquire);
                
                let filter = {
                    let filters = self.inner.filters.read().unwrap();
                    filters.get(current_idx).map(Arc::clone)
                };
                
                // Try to insert into current filter
                if let Some(sharded_filter) = filter {
                    sharded_filter.insert(item);

                    self.inner.filter_nonempty[current_idx].store(true, Ordering::Relaxed);
                    break;
                }
                
                // Growth may have occurred, retry
                std::thread::yield_now();
            }

            self.inner.total_items.fetch_add(1, Ordering::Relaxed);

            let total = self.inner.total_items.load(Ordering::Relaxed);
            let next_threshold = self.inner.check_interval.load(Ordering::Acquire);
            
            if total >= next_threshold {
                self.try_grow();
            }
        }

        /// Test whether `item` may have been inserted.
        ///
        /// Returns `false` only if `item` was definitely not inserted.
        /// Returns `true` if `item` was inserted, or with probability
        /// ≤ `estimate_fpr()` if it was not (false positive).
        ///
        /// # Concurrency
        ///
        /// Completely non-blocking. Acquires the read-lock for the duration of
        /// the filter-list snapshot (~10 ns), then iterates in reverse order
        /// with lock-free `AtomicU64::load` reads. Concurrent inserts,
        /// other contains calls, and growth events never block this call.
        ///
        /// # Iteration order
        ///
        /// Filters are checked newest-first. Under typical workloads where
        /// recently inserted items are queried more often, this provides
        /// early-exit on the first (and freshest) sub-filter for the majority
        /// of hits, minimising average latency.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<u64> =
        ///     AtomicScalableBloomFilter::new(1_000, 0.01)?;
        ///
        /// filter.insert(&42);
        ///
        /// assert!(filter.contains(&42));   // guaranteed true
        /// assert!(!filter.contains(&999)); // false with probability ≥ 1 − FPR
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        #[must_use]
        #[inline]
        pub fn contains(&self, item: &T) -> bool {
            let filters = self.inner.filters.read().unwrap();

            for (idx, filter) in filters.iter().enumerate().rev() {
                if !self.inner.filter_nonempty[idx].load(Ordering::Relaxed) {
                    continue;
                }
                if filter.contains(item) {
                    return true;
                }
            }

            false
        }

        /// Insert a slice of items with shard-grouped access for cache locality.
        ///
        /// # Performance vs `insert` in a loop
        ///
        /// A loop of `insert` calls acquires one read-lock per item and
        /// computes the shard index twice (once to route, once inside
        /// `ShardedFilter::insert`). `insert_batch` groups items by shard
        /// upfront, then writes each shard group sequentially:
        ///
        /// - Sequential writes to one shard's `AtomicU64` words hit L1/L2
        ///   cache repeatedly — 3–4× faster than random access across all shards.
        /// - Read-lock acquisitions are O(shard_count) per batch, not O(n).
        /// - Growth checks fire O(shard_count) times per batch regardless of n.
        ///
        /// # Concurrency
        ///
        /// The read-lock is held only for the `Arc::clone` of the target filter
        /// (~10 ns per shard bucket) and released before bit-setting begins.
        /// Growth write-locks can proceed between shard buckets rather than
        /// being blocked for the full batch duration.
        ///
        /// # Errors
        ///
        /// Returns `InvalidParameters` if the batch would overflow `total_items`
        /// (`usize::MAX`). The filter is not modified on error.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<u64> =
        ///     AtomicScalableBloomFilter::new(10_000, 0.01)?;
        ///
        /// let items: Vec<u64> = (0..1_000).collect();
        /// filter.insert_batch(&items)?;
        ///
        /// assert_eq!(filter.len(), 1_000);
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        pub fn insert_batch(&self, items: &[T]) -> Result<()> {
            if items.is_empty() {
                return Ok(());
            }

            // Validate before touching any state.
            let current = self.inner.total_items.load(Ordering::Relaxed);
            current
                .checked_add(items.len())
                .ok_or_else(|| {
                    BloomCraftError::invalid_parameters(format!(
                        "Batch insert of {} items would overflow counter (current: {})",
                        items.len(),
                        current
                    ))
                })?;

            // ── PHASE 1: GROUP BY SHARD ──────────────────────────────────────────────
            //
            // Routing must match ShardedFilter::insert/contains exactly.
            // shard_count is immutable; no lock required.
            let shard_count = self.inner.config.shard_count;

            let mut shard_buckets: Vec<Vec<&T>> = vec![Vec::new(); shard_count];
            for item in items {
                let shard_idx = InternalHasher::hash_one(item) as usize % shard_count;
                shard_buckets[shard_idx].push(item);
            }

            // ── PHASE 2: INSERT PER SHARD BUCKET ────────────────────────────────────
            for (shard_idx, bucket) in shard_buckets.iter().enumerate() {
                if bucket.is_empty() {
                    continue;
                }

                let current_idx = self.inner.current_filter.load(Ordering::Acquire);

                let filter = {
                    let filters = self.inner.filters.read().unwrap();
                    filters.get(current_idx).map(Arc::clone)
                };

                if let Some(sharded) = filter {
                    for item in bucket {
                        // Lock-free. AtomicU64::fetch_or inside StandardBloomFilter.
                        sharded.insert_into_shard(shard_idx, item);
                    }
                    // Set only when we actually inserted — not unconditionally.
                    self.inner.filter_nonempty[current_idx].store(true, Ordering::Relaxed);
                }

                // Bump counter by bucket size and check growth threshold.
                let total = self.inner
                    .total_items
                    .fetch_add(bucket.len(), Ordering::Relaxed)
                    + bucket.len();

                if total >= self.inner.check_interval.load(Ordering::Acquire) {
                    self.try_grow();
                }
            }

            Ok(())
        }

        /// Test a slice of items, returning one `bool` per item.
        ///
        /// Each item is checked independently via [`contains`](Self::contains).
        /// The result vec is in the same order as `items`.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<u64> =
        ///     AtomicScalableBloomFilter::new(100, 0.01)?;
        /// filter.insert_batch(&[1u64, 2, 3])?;
        ///
        /// let results = filter.contains_batch(&[1u64, 2, 3, 4, 5]);
        /// assert_eq!(results, vec![true, true, true, false, false]);
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        #[must_use]
        pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
            items.iter().map(|item| self.contains(item)).collect()
        }

        /// Reset the filter to its post-construction state (fallible).
        ///
        /// # Correctness guarantee
        ///
        /// The replacement filter is allocated **before** acquiring the write
        /// lock. If the allocation fails, `self` is left completely intact
        /// and the error is returned without modifying any state.
        ///
        /// # Concurrency contract
        ///
        /// The write lock serialises all concurrent inserts. A concurrent
        /// `contains` call that overlaps with `clear` may observe the
        /// pre-clear or post-clear filter state. Both are correct: a
        /// `contains` reading the post-clear (empty) state returns `false`
        /// for any item, which is a valid (conservative) answer.
        ///
        /// # Performance note
        ///
        /// Under N concurrent readers, `clear` must wait for all N read guards
        /// to be dropped before the write lock is granted. At 8 concurrent
        /// readers the observed overhead is ~162× versus an uncontested clear.
        /// If your workload rotates windows frequently under high read
        /// concurrency, prefer a double-buffer pattern over calling `clear`
        /// on a live filter.
        ///
        /// # Errors
        ///
        /// Propagates any allocation error from [`ShardedFilter::new`].
        /// On error, the filter is unchanged.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<i32> =
        ///     AtomicScalableBloomFilter::new(100, 0.01)?;
        /// filter.insert(&42);
        /// filter.clear_checked()?;
        ///
        /// assert!(filter.is_empty());
        /// assert!(!filter.contains(&42));
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        pub fn clear_checked(&self) -> Result<()> {
            // Allocate before acquiring any lock. On OOM the filter is intact.
            let replacement = Arc::new(ShardedFilter::new(
                self.inner.config.initial_capacity,
                self.inner.config.target_fpr,
                self.inner.config.shard_count,
                self.inner.config.hasher.clone(),
            )?);

            // Unwrap is correct: a poisoned lock means a panic inside the
            // critical section, which is an unrecoverable program state.
            let mut filters = self.inner.filters.write().unwrap();

            // All fallible work is done. Safe to mutate state from here.
            filters.clear();
            filters.push(replacement);

            self.inner.current_filter.store(0, Ordering::Release);
            self.inner.total_items.store(0, Ordering::Release);

            for flag in &self.inner.filter_nonempty {
                flag.store(false, Ordering::Relaxed);
            }

            if let GrowthStrategy::Adaptive { initial_ratio, .. } =
                self.inner.config.growth_strategy
            {
                self.inner.config.store_error_ratio(initial_ratio);
            }

            let initial_check_interval = (self.inner.config.initial_capacity as f64
                * self.inner.config.fill_threshold) as usize;
            self.inner
                .check_interval
                .store(initial_check_interval.max(1), Ordering::Release);

            drop(filters);

            Ok(())
        }

        /// Reset the filter to its post-construction state (infallible).
        ///
        /// Panics if the initial replacement filter cannot be allocated (e.g. OOM).
        /// Use [`clear_checked`](Self::clear_checked) when the error must be
        /// propagated rather than treated as fatal.
        ///
        /// # Panics
        ///
        /// Panics if [`clear_checked`](Self::clear_checked) returns an error.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<i32> =
        ///     AtomicScalableBloomFilter::new(100, 0.01)?;
        /// filter.insert(&42);
        /// filter.clear();
        ///
        /// assert!(filter.is_empty());
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        pub fn clear(&self) {
            self.clear_checked()
                .expect("AtomicScalableBloomFilter::clear() failed to recreate initial filter")
        }

        // ── Growth ────────────────────────────────────────────────────────────

        /// Attempt to trigger a growth event if the filter is ready to grow.
        ///
        /// Uses a double-checked locking pattern:
        ///
        /// 1. Cheap `Relaxed` load to bail early if growth is already running.
        /// 2. `AcqRel`/`Relaxed` CAS to elect exactly one thread to grow.
        /// 3. Elected thread calls `perform_growth` and releases the flag.
        ///
        /// All other threads that arrive concurrently return immediately. They
        /// continue inserting into the current sub-filter without waiting.
        /// This means no insert is ever blocked waiting for growth to complete.
        fn try_grow(&self) {
            if self.inner.growth_in_progress.load(Ordering::Relaxed) {
                return;
            }

            if self.inner.growth_in_progress
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                return;
            }

            let result = self.perform_growth();

            // Release ordering: the Relaxed stores in perform_growth are
            // visible to any thread that subsequently observes this store
            // as `false` and acquires the read lock.
            self.inner.growth_in_progress.store(false, Ordering::Release);

            if let Err(e) = result {
                #[cfg(debug_assertions)]
                eprintln!("[AtomicScalableBloomFilter] Growth failed: {}", e);
            }
        }

        /// Perform one growth step, advancing `current_filter` to a new sub-filter.
        ///
        /// # Phase ordering
        ///
        /// Growth is split into three phases to minimise write-lock hold time:
        ///
        /// ```text
        /// Phase 1  read lock    Fill-rate check + pre-alloc probe       ~10 ns
        /// Phase 2  no lock      ShardedFilter::new — zeroes bit arrays  potentially ms
        /// Phase 3  write lock   Vec::push + two atomic stores           ~10 ns
        /// ```
        ///
        /// Allocation (Phase 2) deliberately runs without any lock. At filter
        /// depth 6 with `Geometric(2.0)` and `initial_capacity = 1_000`, a
        /// single `ShardedFilter::new` zeroes ~600 KB–1.2 MB of bit arrays.
        /// Running that under the write lock would stall every concurrent
        /// insert for the entire zeroing duration.
        ///
        /// Because `growth_in_progress` guarantees only one thread enters this
        /// function at a time, the Vec length is stable between Phase 2 and
        /// Phase 3. The defensive double-check in Phase 3 guards against future
        /// refactors that weaken that invariant.
        ///
        /// # Pre-allocated fast path
        ///
        /// If `with_preallocated` already built the next sub-filter, Phase 2 is
        /// skipped entirely. Growth reduces to two atomic stores (~10 ns total),
        /// which is fast enough to be invisible in latency distributions.
        fn perform_growth(&self) -> Result<()> {
            // ── PHASE 1: Read lock ────────────────────────────────────────────
            let (current_idx, have_preallocated) = {
                let filters = self.inner.filters.read().unwrap();
                let idx = self.inner.current_filter.load(Ordering::Acquire);

                let next_index = idx + 1;
                let have_preallocated = next_index < filters.len();

                if !have_preallocated && filters.len() >= MAX_FILTERS {
                    self.inner.check_interval.store(usize::MAX, Ordering::Release);
                    return Err(BloomCraftError::capacity_exceeded(MAX_FILTERS, filters.len()));
                }

                (idx, have_preallocated)
            };

            if have_preallocated {
                let filters = self.inner.filters.write().unwrap();

                if self.inner.current_filter.load(Ordering::Relaxed) != current_idx {
                    return Ok(());
                }

                let next_index = current_idx + 1;
                let usable = (filters[next_index].expected_items() as f64
                    * self.inner.config.fill_threshold()) as usize;
                let cur_total = self.inner.total_items.load(Ordering::Relaxed);
                self.inner.check_interval.store(
                    cur_total.saturating_add(usable.max(1)),
                    Ordering::Release,
                );
                self.inner.current_filter.store(next_index, Ordering::Release);

                #[cfg(debug_assertions)]
                eprintln!(
                    "[AtomicScalableBloomFilter] Advanced to pre-allocated filter {} (no allocation)",
                    next_index
                );

                return Ok(());
            }

            // ── PHASE 2: Allocate BEFORE write lock ──────────────────────────────────
            let filter_index = self.inner.filters.read().unwrap().len();

            let capacity = self.calculate_next_capacity(filter_index)?;
            let fpr = self.calculate_next_fpr(filter_index);

            let new_filter = Arc::new(ShardedFilter::new(
                capacity,
                fpr,
                self.inner.config.shard_count,
                self.inner.config.hasher.clone(),
            )?);

            // ── PHASE 3: Write lock ───────────────────────────────────────────────────
            let next_threshold;
            {
                let mut filters = self.inner.filters.write().unwrap();

                if self.inner.current_filter.load(Ordering::Relaxed) != current_idx {
                    return Ok(());
                }

                filters.push(new_filter);
                self.inner.current_filter.store(filter_index, Ordering::Release);

                // ── Adaptive FPR tuning ───────────────────────────────────────
                //
                // Runs under the write lock so that `store_error_ratio` is
                // sequenced with filter creation. Any thread that subsequently
                // reads `current_filter` (Acquire) also observes the updated
                // error_ratio via the filters lock's happens-before edge.
                if let GrowthStrategy::Adaptive { min_ratio, max_ratio, .. } =
                    self.inner.config.growth_strategy
                {
                    let just_filled_idx = filters.len().saturating_sub(2);
                    if let Some(filled) = filters.get(just_filled_idx) {
                        let actual_fill = filled.fill_rate();
                        let current = self.inner.config.error_ratio();
                        let updated = if actual_fill > self.inner.config.fill_threshold() * 1.2 {
                            (current * 0.9).max(min_ratio)
                        } else if actual_fill < self.inner.config.fill_threshold() * 0.8 {
                            (current * 1.1).min(max_ratio)
                        } else {
                            current
                        };
                        self.inner.config.store_error_ratio(updated);
                    }
                }

                // ── THRESHOLD UPDATE ─────────────────────────────────────────────────
                let new_filter_usable = filters
                    .last()
                    .map(|f| (f.expected_items() as f64 * self.inner.config.fill_threshold()) as usize)
                    .unwrap_or(self.inner.config.initial_capacity);
                let cur_total = self.inner.total_items.load(Ordering::Relaxed);
                next_threshold = cur_total.saturating_add(new_filter_usable.max(1));
                self.inner.check_interval.store(next_threshold, Ordering::Release);
            }

            #[cfg(debug_assertions)]
            eprintln!(
                "[AtomicScalableBloomFilter] Grew to {} filters (capacity: {}, FPR: {:.6}, interval: {})",
                filter_index + 1,
                capacity,
                fpr,
                next_threshold
            );

            Ok(())
        }

        /// Compute the bit-array capacity for the sub-filter at `filter_index`.
        ///
        /// Delegates to the configured [`GrowthStrategy`]. Validates that the
        /// computed capacity does not overflow `usize` before casting.
        ///
        /// # Errors
        ///
        /// Returns `InvalidParameters` if the geometric or adaptive sequence
        /// would overflow `usize::MAX` at the given index. In practice this
        /// cannot occur within the `MAX_FILTERS = 64` limit unless
        /// `initial_capacity` is unreasonably large.
        fn calculate_next_capacity(&self, filter_index: usize) -> Result<usize> {
            const MAX_CAPACITY: f64 = usize::MAX as f64;

            let capacity = match self.inner.config.growth_strategy {
                GrowthStrategy::Constant => self.inner.config.initial_capacity,

                GrowthStrategy::Geometric(scale) => {
                    if filter_index == 0 {
                        self.inner.config.initial_capacity
                    } else {
                        let scale_log = scale.ln();
                        let max_safe_exp = (MAX_CAPACITY.ln() - (self.inner.config.initial_capacity as f64).ln()) / scale_log;

                        if filter_index as f64 >= max_safe_exp {
                            return Err(BloomCraftError::invalid_parameters(
                                format!("Filter index {} would cause capacity overflow (max safe: {:.1})",
                                    filter_index, max_safe_exp)
                            ));
                        }

                        let growth_factor = scale.powi(filter_index as i32);
                        let computed = self.inner.config.initial_capacity as f64 * growth_factor;

                        if computed > MAX_CAPACITY || !computed.is_finite() {
                            return Err(BloomCraftError::invalid_parameters(
                                format!("Computed capacity {:.2e} exceeds usize::MAX", computed)
                            ));
                        }

                        let new_capacity = computed as usize;

                        if new_capacity < self.inner.config.initial_capacity {
                            return Err(BloomCraftError::invalid_parameters(
                                "Capacity calculation resulted in overflow"
                            ));
                        }

                        new_capacity
                    }
                }

                GrowthStrategy::Bounded { scale, max_filter_size } => {
                    if filter_index == 0 {
                        self.inner.config.initial_capacity
                    } else {
                        let computed = self.inner.config.initial_capacity as f64 * scale.powi(filter_index as i32);
                        
                        if computed > MAX_CAPACITY || !computed.is_finite() {
                            return Err(BloomCraftError::invalid_parameters(
                                "Bounded capacity calculation overflow"
                            ));
                        }
                        
                        let geometric = computed as usize;
                        geometric.min(max_filter_size)
                    }
                }

                GrowthStrategy::Adaptive { .. } => {
                    if filter_index == 0 {
                        self.inner.config.initial_capacity
                    } else {
                        const SCALE: f64 = 2.0;
                        let max_safe_exp = (MAX_CAPACITY.ln()
                            - (self.inner.config.initial_capacity as f64).ln())
                            / SCALE.ln();
                        if filter_index as f64 >= max_safe_exp {
                            return Err(BloomCraftError::invalid_parameters(format!(
                                "Adaptive filter index {} would cause capacity overflow",
                                filter_index
                            )));
                        }
                        let computed = self.inner.config.initial_capacity as f64
                            * SCALE.powi(filter_index as i32);
                        if computed > MAX_CAPACITY || !computed.is_finite() {
                            return Err(BloomCraftError::invalid_parameters(
                                "Adaptive capacity overflow",
                            ));
                        }
                        let new_cap = computed as usize;
                        if new_cap < self.inner.config.initial_capacity {
                            return Err(BloomCraftError::invalid_parameters(
                                "Adaptive capacity wraparound",
                            ));
                        }
                        new_cap
                    }
                }
            };

            Ok(capacity)
        }

        /// Calculate FPR for next filter
        fn calculate_next_fpr(&self, filter_index: usize) -> f64 {
            let ratio = self.inner.config.error_ratio();
            
            const MAX_SAFE_EXP: i32 = 1000;
            let safe_index = (filter_index as i32).min(MAX_SAFE_EXP);
            
            let raw_fpr = self.inner.config.target_fpr * ratio.powi(safe_index);
            
            raw_fpr.max(MIN_FPR).min(1.0)
        }

        // ── Accessors ─────────────────────────────────────────────────────────

        /// Number of active sub-filters (including the current one).
        #[must_use]
        pub fn filter_count(&self) -> usize {
            self.inner.filters.read().unwrap().len()
        }

        /// Total items inserted since the last `clear`.
        ///
        /// Uses `Relaxed` ordering — the count may be up to a few inserts
        /// stale on a heavily contended filter. This is a monitoring value,
        /// not a correctness-critical count.
        #[must_use]
        pub fn len(&self) -> usize {
            self.inner.total_items.load(Ordering::Relaxed)
        }

        /// Returns `true` if no items have been inserted since the last `clear`.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Total item capacity of all sub-filters.
        #[must_use]
        pub fn total_capacity(&self) -> usize {
            let filters = self.inner.filters.read().unwrap();
            filters
                .iter()
                .map(|f| f.expected_items())
                .sum()
        }

        /// Check if at maximum capacity
        #[must_use]
        pub fn is_at_max_capacity(&self) -> bool {
            self.filter_count() >= MAX_FILTERS
        }

        /// Check if near maximum capacity
        #[must_use]
        pub fn is_near_capacity(&self) -> bool {
            self.filter_count() + CAPACITY_WARNING_THRESHOLD >= MAX_FILTERS
        }

        /// Estimated compound FPR based on the actual fill state of each sub-filter.
        ///
        /// Computed via the complement rule:
        ///
        /// ```text
        /// FPR_total = 1 − ∏ (1 − FPR_i)   for i in 0..n
        /// ```
        ///
        /// More accurate than the theoretical formula because it reflects actual
        /// fill rates rather than expected ones. Not suitable for hot paths —
        /// acquires the read lock and reads atomic counters from every shard of
        /// every sub-filter.
        #[must_use]
        pub fn estimate_fpr(&self) -> f64 {
            let filters = self.inner.filters.read().unwrap();
            
            let product: f64 = filters
                .iter()
                .map(|f| 1.0 - f.estimate_fpr())
                .product();
            
            1.0 - product
        }

        /// Total memory footprint in bytes.
        ///
        /// Includes the struct overhead and all shard bit arrays. Does not
        /// account for `Arc`/`RwLock` metadata or OS-level page table overhead.
        #[must_use]
        pub fn memory_usage(&self) -> usize {
            let filters = self.inner.filters.read().unwrap();
            filters.iter().map(|f| f.memory_usage()).sum::<usize>()
                + std::mem::size_of::<Self>()
        }

        /// Fill rate of the currently-active sub-filter (0.0 – 1.0).
        ///
        /// When this approaches `fill_threshold` (default 0.5), the next
        /// insert will trigger a growth event.
        #[must_use]
        pub fn current_fill_rate(&self) -> f64 {
            let filters = self.inner.filters.read().unwrap();
            let current_idx = self.inner.current_filter.load(Ordering::Acquire);

            filters
                .get(current_idx)
                .map(|f| f.fill_rate())
                .unwrap_or(0.0)
        }

        /// Bit-level statistics aggregated across all sub-filters.
        ///
        /// Returns `(total_bits, set_bits, utilization_percent)`.
        ///
        /// Note: `total_bits` grows with each sub-filter addition. Asserting
        /// `total_bits` is constant across inserts is only valid if no growth
        /// event fires during the measurement window. Use a single-sub-filter
        /// setup (e.g. `initial_capacity` >> item count) if you need to observe
        /// a stable bit count.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use bloomcraft::filters::AtomicScalableBloomFilter;
        ///
        /// let filter: AtomicScalableBloomFilter<i32> =
        ///     AtomicScalableBloomFilter::new(10_000, 0.01)?;
        ///
        /// for i in 0..1_000 { filter.insert(&i); }
        ///
        /// let (total, set, utilization) = filter.bit_statistics();
        /// println!("{}/{} bits set ({:.1}%)", set, total, utilization);
        /// # Ok::<(), bloomcraft::error::BloomCraftError>(())
        /// ```
        #[must_use]
        pub fn bit_statistics(&self) -> (usize, usize, f64) {
            let filters = self.inner.filters.read().unwrap();
            
            let mut total_bits = 0;
            let mut set_bits = 0;
            
            for filter in filters.iter() {
                let (t, s, _) = filter.bit_statistics();
                total_bits += t;
                set_bits += s;
            }
            
            let utilization = if total_bits > 0 {
                (set_bits as f64 / total_bits as f64) * 100.0
            } else {
                0.0
            };
            
            (total_bits, set_bits, utilization)
        }

        /// Number of shards per sub-filter.
        ///
        /// Determined at construction via [`optimal_shard_count`] and immutable
        /// thereafter. Changing the shard count mid-life would invalidate the
        /// routing invariant between `insert` and `contains`.
        #[must_use]
        pub fn shard_count(&self) -> usize {
            self.inner.config.shard_count
        }
    }

    // ── Trait implementations ─────────────────────────────────────────────────

    /// Cloning is O(1): it increments the `Arc` reference count.
    ///
    /// All clones share the same underlying filter state. To get an independent
    /// deep copy, use `ScalableBloomFilter` (single-threaded) or construct a
    /// new `AtomicScalableBloomFilter` and replay inserts.
    impl<T, H> Clone for AtomicScalableBloomFilter<T, H>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default + Send + Sync,
    {
        fn clone(&self) -> Self {
            Self {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    impl<T, H> fmt::Display for AtomicScalableBloomFilter<T, H>
    where
        T: Hash + Send + Sync,
        H: BloomHasher + Clone + Default + Send + Sync,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "AtomicScalableBloomFilter {{ filters: {}, capacity: {}, items: {}, est_fpr: {:.4}% }}",
                self.filter_count(),
                self.total_capacity(),
                self.len(),
                self.estimate_fpr() * 100.0
            )
        }
    }
}

#[cfg(feature = "concurrent")]
pub use concurrent::AtomicScalableBloomFilter;


// TEST SUITE

#[cfg(test)]
mod tests {
    use super::*;

    // BASIC FUNCTIONALITY TESTS

    #[test]
    fn test_new() {
        let filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01).unwrap();
        assert_eq!(filter.filter_count(), 1);
        assert!(filter.is_empty());
        assert_eq!(filter.len(), 0);
        assert_eq!(filter.total_capacity(), 1000);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01).unwrap();
        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_no_false_negatives() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();
        let items: Vec<i32> = (0..1000).collect();

        for item in &items {
            filter.insert(item);
        }

        for item in &items {
            assert!(
                filter.contains(item),
                "False negative for {} (filter depth: {})",
                item,
                filter.filter_count()
            );
        }
    }

    #[test]
    fn test_clear() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        assert!(!filter.is_empty());
        assert!(filter.filter_count() > 1);

        filter.clear();

        assert!(filter.is_empty());
        assert_eq!(filter.filter_count(), 1);
        assert!(!filter.contains(&42));
    }

    // GROWTH TESTS

    #[test]
    fn test_automatic_growth() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();
        assert_eq!(filter.filter_count(), 1);

        for i in 0..100 {
            filter.insert(&i);
        }

        assert!(
            filter.filter_count() > 1,
            "Filter should have grown, count: {}",
            filter.filter_count()
        );

        // Verify all items still present
        for i in 0..100 {
            assert!(filter.contains(&i));
        }
    }

    #[test]
    fn test_geometric_growth() {
        let mut filter = ScalableBloomFilter::with_strategy(
            10, 
            0.01, 
            0.5, 
            GrowthStrategy::Geometric(2.0)
        ).unwrap();

        for i in 0..200 {
            filter.insert(&i);
        }

        let stats = filter.filter_stats();
        if stats.len() >= 2 {
            let ratio = stats[1].0 as f64 / stats[0].0 as f64;
            assert!(
                ratio > 1.5 && ratio < 2.5,
                "Growth ratio should be ~2.0, got {}",
                ratio
            );
        }
    }

    #[test]
    fn test_constant_growth() {
        let mut filter = ScalableBloomFilter::with_strategy(
            10, 
            0.01, 
            0.5, 
            GrowthStrategy::Constant
        ).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let stats = filter.filter_stats();
        if stats.len() >= 2 {
            assert_eq!(stats[0].0, stats[1].0, "All filters should have same capacity");
        }
    }

    #[test]
    fn test_bounded_growth() {
        let mut filter = ScalableBloomFilter::with_strategy(
            100,
            0.01,
            0.5,
            GrowthStrategy::Bounded {
                scale: 2.0,
                max_filter_size: 500,
            }
        ).unwrap();

        for i in 0..2000 {
            filter.insert(&i);
        }

        // No filter should exceed max_filter_size
        for (capacity, _, _) in filter.filter_stats() {
            assert!(capacity <= 500, "Filter capacity {} exceeds max 500", capacity);
        }
    }

    #[test]
    fn test_negative_query_fast_path() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();

        for i in 0..1000 {
            filter.insert(&i);
        }

        // Negative query should be fast
        assert!(!filter.contains(&99999));

        // Positive query should work
        assert!(filter.contains(&500));
    }

    #[test]
    fn test_reverse_iteration() {
        let mut filter = ScalableBloomFilter::new(10, 0.01)
            .unwrap()
            .with_query_strategy(QueryStrategy::Reverse);

        for i in 0..100 {
            filter.insert(&i);
        }

        // Recent items should be found (in newest filters)
        assert!(filter.contains(&99));
        assert!(filter.contains(&0));
    }

    #[test]
    fn test_forward_iteration() {
        let mut filter = ScalableBloomFilter::new(10, 0.01)
            .unwrap()
            .with_query_strategy(QueryStrategy::Forward);

        for i in 0..100 {
            filter.insert(&i);
        }

        // Should work with forward iteration too
        assert!(filter.contains(&0));
        assert!(filter.contains(&99));
    }

    #[test]
    fn test_predict_fpr() {
        let mut filter = ScalableBloomFilter::<i32>::new(100, 0.01).unwrap();

        for i in 0..200 {
            filter.insert(&i);
        }

        let fpr_1k = filter.predict_fpr(1000);
        let fpr_10k = filter.predict_fpr(10000);

        assert!(fpr_1k > 0.0, "FPR at 1K should be > 0");
        assert!(fpr_10k > fpr_1k, "FPR should increase with scale");
        assert!(fpr_10k < 0.1, "FPR should stay reasonable");
    }

    #[test]
    fn test_fpr_breakdown() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let breakdown = filter.filter_fpr_breakdown();
        assert!(!breakdown.is_empty());

        for (_idx, individual_fpr, contribution) in breakdown {
            assert!(individual_fpr >= 0.0 && individual_fpr <= 1.0);
            assert!(contribution >= 0.0 && contribution <= 1.0);
        }
    }

    #[test]
    fn test_capacity_exhausted_error() {
        let mut filter = ScalableBloomFilter::with_strategy(
            10,
            0.01,
            0.5,
            GrowthStrategy::Bounded {
                scale: 1.5,
                max_filter_size: 50,
            }
        ).unwrap().with_capacity_behavior(CapacityExhaustedBehavior::Error);

        let mut exhausted = false;
        for i in 0..100_000 {
            match filter.insert_checked(&i) {
                Err(BloomCraftError::MaxFiltersExceeded { max_filters, current_count }) => {
                    exhausted = true;
                    assert_eq!(max_filters, MAX_FILTERS);
                    assert_eq!(current_count, MAX_FILTERS);
                    break;
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
                Ok(_) => {}
            }
        }

        assert!(exhausted, 
            "Should have reached MAX_FILTERS, got {} filters", 
            filter.filter_count());
    }

    #[test]
    fn test_contains_with_provenance() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let (found, filter_idx) = filter.contains_with_provenance(&50);
        assert!(found);
        assert!(filter_idx.is_some());

        let (not_found, no_idx) = filter.contains_with_provenance(&9999);
        assert!(!not_found);
        assert!(no_idx.is_none());
    }

    #[test]
    fn test_adaptive_growth() {
        let mut filter = ScalableBloomFilter::with_strategy(
            100,
            0.01,
            0.5,
            GrowthStrategy::Adaptive {
                initial_ratio: 0.5,
                min_ratio: 0.3,
                max_ratio: 0.9,
            }
        ).unwrap();

        for i in 0..1000 {
            filter.insert(&i);
        }

        // Error ratio should have adapted
        let final_ratio = filter.error_ratio();
        assert!(final_ratio >= 0.3 && final_ratio <= 0.9);
    }

    #[test]
    fn test_cardinality_tracking() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01).unwrap()
            .with_cardinality_tracking();

        // Insert 1000 unique items
        for i in 0..1000 {
            filter.insert(&i);
        }

        // Insert 1000 duplicates
        for i in 0..1000 {
            filter.insert(&i);
        }

        assert_eq!(filter.len(), 2000); // Total insertions

        let unique_count = filter.estimate_unique_count();
        let error = (unique_count as f64 - 1000.0).abs() / 1000.0;

        assert!(error < 0.05, "Cardinality error {:.2}% exceeds 5%", error * 100.0);
    }

    #[test]
    fn test_cardinality_error_bound() {
        let filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01).unwrap()
            .with_cardinality_tracking();

        let error_bound = filter.cardinality_error_bound();
        assert!(error_bound > 0.0 && error_bound < 0.02); // Should be ~0.008
    }

    #[test]
    fn test_health_metrics() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();

        for i in 0..500 {
            filter.insert(&i);
        }

        let metrics = filter.health_metrics();

        assert!(metrics.filter_count > 1);
        assert_eq!(metrics.total_items, 500);
        assert!(metrics.estimated_fpr > 0.0);
        assert!(metrics.estimated_fpr < 0.1);
        assert!(metrics.current_fill_rate >= 0.0 && metrics.current_fill_rate <= 1.0);
        assert!(metrics.memory_bytes > 0);
    }

    #[test]
    fn test_health_metrics_display() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(500, 0.01).unwrap();

        for i in 0..2000 {
            filter.insert(&i);
        }

        let metrics = filter.health_metrics();
        let display = format!("{}", metrics);

        assert!(display.contains("ScalableBloomFilter Health Metrics"));
        assert!(display.contains("Filters:"));
        assert!(display.contains("Estimated FPR:"));
    }

    // BATCH OPERATIONS TESTS

    #[test]
    fn test_insert_batch() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();
        let items: Vec<i32> = (0..1000).collect();

        let _ = filter.insert_batch(&items);

        assert_eq!(filter.len(), 1000);
        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();
        let _ = filter.insert_batch(&[1, 2, 3]);

        let results = filter.contains_batch(&[1, 2, 3, 4, 5]);
        assert_eq!(results, vec![true, true, true, false, false]);
    }

    // TRAIT IMPLEMENTATION TESTS

    #[test]
    fn test_display_trait() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01).unwrap();

        for i in 0..500 {
            filter.insert(&i);
        }

        let display = format!("{}", filter);

        assert!(display.contains("ScalableBloomFilter"));
        assert!(display.contains("filters:"));
        assert!(display.contains("capacity:"));
        assert!(display.contains("items:"));
    }

    #[test]
    fn test_extend_trait() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();
        filter.extend(0..50);

        assert_eq!(filter.len(), 50);
        assert!(filter.contains(&25));
        assert!(!filter.contains(&100));
    }

    #[test]
    fn test_from_iterator() {
        let filter: ScalableBloomFilter<i32> = (0..100).collect();

        assert_eq!(filter.len(), 100);
        assert!(filter.contains(&50));
        assert!(!filter.contains(&200));
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter: ScalableBloomFilter<&str> = ScalableBloomFilter::new(100, 0.01).unwrap();

        BloomFilter::insert(&mut filter, &"test");
        assert!(BloomFilter::contains(&filter, &"test"));
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    // FPR AND ACCURACY TESTS

    #[test]
    fn test_estimate_fpr_vs_max_fpr() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();

        for i in 0..200 {
            filter.insert(&i);
        }

        let max_fpr = filter.max_fpr();
        let actual_fpr = filter.estimate_fpr();

        // Union bound should always be >= actual FPR
        assert!(
            max_fpr >= actual_fpr - 1e-10,
            "max_fpr ({}) should be >= actual_fpr ({})",
            max_fpr,
            actual_fpr
        );
    }

    #[test]
    fn test_fpr_increases_with_growth() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();

        let initial_fpr = filter.estimate_fpr();

        for i in 0..1000 {
            filter.insert(&i);
        }

        let final_fpr = filter.estimate_fpr();

        assert!(final_fpr >= initial_fpr, "FPR should not decrease with growth");
    }

    // CAPACITY AND LIMITS TESTS

    #[test]
    fn test_capacity_monitoring() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();

        assert!(!filter.is_at_max_capacity());
        assert!(!filter.is_near_capacity());
        assert_eq!(filter.remaining_growth_capacity(), MAX_FILTERS - 1);

        for i in 0..1_000 {
            filter.insert(&i);
            if filter.is_at_max_capacity() {
                assert_eq!(filter.remaining_growth_capacity(), 0);
                break;
            }
        }

        assert!(filter.filter_count() > 1);
    }

    #[test]
    fn test_max_filters_limit_is_enforced() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::with_strategy(
            1,
            0.01,
            0.5,
            GrowthStrategy::Constant,
        )
        .unwrap();

        // Push items until well past the limit.
        for i in 0..=MAX_FILTERS as i32 {
            filter.insert(&i);
        }

        // The filter must never exceed the hard cap.
        assert_eq!(
            filter.filter_count(),
            MAX_FILTERS,
            "filter_count exceeded MAX_FILTERS"
        );
    }

    #[test]
    fn test_fpr_degradation_at_capacity() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1, 0.01).unwrap();

        let initial_fpr = filter.estimate_fpr();

        // Fill to MAX_FILTERS
        for i in 0..10_000 {
            filter.insert(&i);
            if filter.is_at_max_capacity() {
                break;
            }
        }

        // Continue inserting beyond capacity
        let start_over_capacity = 10_000;
        for i in start_over_capacity..start_over_capacity + 5_000 {
            filter.insert(&i);
        }

        let final_fpr = filter.estimate_fpr();

        // FPR should have increased due to saturation
        assert!(
            final_fpr > initial_fpr,
            "FPR should degrade at capacity: initial={}, final={}",
            initial_fpr,
            final_fpr
        );
    }

    // STRESS AND LARGE-SCALE TESTS

    #[test]
    fn test_large_scale_insertion() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();

        // Insert 10,000 items
        for i in 0..10_000 {
            filter.insert(&i);
        }

        assert_eq!(filter.len(), 10_000);
        assert!(filter.filter_count() > 1);

        // Verify all items present (no false negatives)
        for i in 0..10_000 {
            assert!(filter.contains(&i), "False negative for {}", i);
        }

        // Check FPR is reasonable
        let fpr = filter.estimate_fpr();
        assert!(fpr < 0.05, "FPR {} is too high", fpr);
    }

    // ACCESSORS AND GETTERS TESTS

    #[test]
    fn test_accessors() {
        let filter: ScalableBloomFilter<i32> = ScalableBloomFilter::with_strategy(
            500,
            0.02,
            0.4,
            GrowthStrategy::Geometric(3.0),
        ).unwrap();

        assert_eq!(filter.initial_capacity(), 500);
        assert_eq!(filter.target_fpr(), 0.02);
        assert_eq!(filter.error_ratio(), 0.4);
        assert_eq!(filter.growth_strategy(), GrowthStrategy::Geometric(3.0));
        assert_eq!(filter.fill_threshold(), DEFAULT_FILL_THRESHOLD);
    }

    #[test]
    fn test_set_fill_threshold() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();

        let _ = filter.set_fill_threshold(0.8);
        assert_eq!(filter.fill_threshold(), 0.8);
    }

    #[test]
    fn test_invalid_fill_threshold() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();
        let result = filter.set_fill_threshold(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_stats() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let stats = filter.filter_stats();
        assert!(!stats.is_empty());

        for (capacity, fill_rate, fpr) in stats {
            assert!(capacity > 0);
            assert!(fill_rate >= 0.0 && fill_rate <= 1.0);
            assert!(fpr >= 0.0 && fpr <= 1.0);
        }
    }

    #[test]
    fn test_memory_usage() {
        let filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(1000, 0.01).unwrap();
        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_clone() {
        let mut filter1 = ScalableBloomFilter::new(100, 0.01).unwrap();
        filter1.insert(&"test");

        let filter2 = filter1.clone();

        assert!(filter2.contains(&"test"));
        assert_eq!(filter1.filter_count(), filter2.filter_count());
        assert_eq!(filter1.len(), filter2.len());
    }

    // EDGE CASES

    #[test]
    fn test_current_vs_aggregate_fill_rate() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01).unwrap();

        for i in 0..300 {
            filter.insert(&i);
        }

        let current = filter.current_fill_rate();
        let aggregate = filter.aggregate_fill_rate();

        assert!(current >= 0.0 && current <= 1.0);
        assert!(aggregate >= 0.0 && aggregate <= 1.0);

        if filter.filter_count() > 1 {
            assert!(current > 0.0);
            assert!(aggregate > 0.0);
        }
    }

    #[test]
    fn test_growth_strategy_default() {
        let strategy = GrowthStrategy::default();
        assert_eq!(strategy, GrowthStrategy::Geometric(2.0));
    }

    #[test]
    fn test_fpr_precision_clamp() {
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::with_strategy(
            10, 
            0.01, 
            0.1, 
            GrowthStrategy::Geometric(2.0)
        ).unwrap();

        // Trigger multiple growths
        for i in 0..10_000 {
            filter.insert(&i);
        }

        // FPR should never be less than MIN_FPR
        let fpr = filter.estimate_fpr();
        assert!(fpr >= MIN_FPR);
    }

    #[cfg(feature = "concurrent")]
    #[test]
    fn test_bit_statistics() {
        let filter: AtomicScalableBloomFilter<i32> =
            AtomicScalableBloomFilter::new(1_000, 0.01).unwrap();

        let (total, set, utilization) = filter.bit_statistics();
        assert!(total > 0, "Expected allocated bits, got 0");
        assert_eq!(set, 0, "No bits should be set before any insert");
        assert_eq!(utilization, 0.0);

        for i in 0..200 {
            filter.insert(&i);
        }

        let (total2, set2, utilization2) = filter.bit_statistics();

        assert_eq!(
            total2, total,
            "Total bits changed unexpectedly — a growth event fired. \
            Reduce insert count or increase initial_capacity."
        );
        assert!(set2 > 0, "Expected set bits after 200 inserts, got 0");
        assert!(
            utilization2 > 0.0 && utilization2 < 100.0,
            "Utilization {:.2} out of (0, 100) range",
            utilization2
        );

        for i in 200..600 {
            filter.insert(&i);
        }
        let (total3, set3, _) = filter.bit_statistics();
        assert!(
            total3 >= total2,
            "Total bits decreased after growth: {} < {}",
            total3,
            total2
        );
        assert!(set3 > set2, "Set bits did not increase after more inserts");
    }
}