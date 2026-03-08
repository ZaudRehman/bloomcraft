//! Standard Bloom filter implementation.
//!
//! This module provides the canonical Bloom filter: a space-efficient probabilistic
//! set membership structure with lock-free concurrent access, enhanced double hashing,
//! and bitwise set algebra.
//!
//! # Algorithm
//!
//! A Bloom filter maps each inserted element to `k` bit positions derived from two
//! independent hash values. A membership query returns `false` only when at least one
//! of the `k` positions is unset — a guaranteed non-member. When all `k` positions
//! are set the element is *probably* a member; false positives are possible, false
//! negatives are not (for standard insert/query; see [`StandardBloomFilter::intersect`]
//! for the exception).
//!
//! # Mathematical Foundation
//!
//! Given expected item count `n` and target false positive rate `p`:
//!
//! ```text
//! m = −n × ln(p) / (ln 2)²    (filter size in bits)
//! k = (m / n) × ln 2           (number of hash functions)
//!
//! p_actual = (1 − e^(−kn/m))^k (actual FPR at n insertions)
//! ```
//!
//! At 1% FPR: m ≈ 9.6 bits per element, k ≈ 7.
//!
//! # Hashing Architecture
//!
//! The hashing pipeline has two stages:
//!
//! 1. **Item → bytes** — `T::hash()` writes its canonical byte representation into a
//!    [`HashBytes`] accumulator (a `std::hash::Hasher` backed by a 128-byte inline
//!    buffer). No `DefaultHasher` compression occurs; `H` receives the full item data.
//! 2. **Bytes → (h1, h2)** — The accumulated bytes are passed to
//!    `H::hash_bytes_pair`, producing two independent 64-bit values used for
//!    enhanced double hashing: `g_i = h1 + i·h2 + i(i+1)/2 (mod m)`.
//!
//! This preserves the independence assumption of enhanced double hashing at the
//! [`BloomHasher`] boundary, not merely at the structural level.
//!
//! # Trait Implementations
//!
//! | Trait | Receiver | Purpose |
//! |---|---|---|
//! | [`BloomFilter`] | `&mut self` insert, `&self` query | Single-threaded contract |
//! | [`ConcurrentBloomFilter`] | `&self` for all operations | Lock-free, safe on `Arc` |
//! | [`MergeableBloomFilter`] | `&mut self` | In-place bitwise OR / AND |
//!
//! ## `union` / `intersect` disambiguation
//!
//! Two variants of each operation exist simultaneously on `StandardBloomFilter`:
//!
//! | Call syntax | Receiver | Returns | Semantics |
//! |---|---|---|---|
//! | `filter.union(&other)` | `&self` | `Result<Self>` | Constructive — new filter |
//! | `MergeableBloomFilter::union(&mut filter, &other)` | `&mut self` | `Result<()>` | In-place OR |
//! | `filter.intersect(&other)` | `&self` | `Result<Self>` | Constructive — new filter |
//! | `MergeableBloomFilter::intersect(&mut filter, &other)` | `&mut self` | `Result<()>` | In-place AND |
//!
//! Inherent methods win dot-call resolution. Use UFCS to reach the in-place trait
//! variants:
//!
//! ```ignore
//! use bloomcraft::core::MergeableBloomFilter;
//! MergeableBloomFilter::union(&mut f1, &f2)?;
//! ```
//!
//! # Intersection and False Negatives
//!
//! Bitwise AND clears bits not present in both operands, breaking the no-false-negatives
//! guarantee for items present in only one source filter. See
//! [`StandardBloomFilter::intersect`] for the full contract.
//!
//! # Concurrency Model
//!
//! All insert and query methods take `&self` and are safe for concurrent use on an
//! `Arc<StandardBloomFilter<T, H>>` with no external synchronisation. Bit writes use
//! `fetch_or` with `Release` ordering; reads use `load` with `Acquire` ordering,
//! establishing a happens-before edge between an insert and any subsequent query that
//! observes its effects.
//!
//! Operations that require `&mut self` — `clear`, `MergeableBloomFilter::union`,
//! `MergeableBloomFilter::intersect` — must be serialised externally (e.g.,
//! `RwLock`) if called alongside concurrent inserts or queries.
//!
//! # Serialisation
//!
//! Enable the `serde` feature. The bit array is serialised verbatim; the hasher is
//! excluded and reconstructed via `H::default()` on deserialisation. Filters
//! serialised with a process-local `StdHasher` seed produce false negatives when
//! loaded in a different process. Use `WyHasher` (`wyhash` feature) for stable
//! cross-process serialisation.
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors."
//!   *Commun. ACM*, 13(7), 422–426.
//! - Kirsch, A., & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a
//!   Better Bloom Filter." *ESA 2006, LNCS 4168*, 456–467.

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::bitvec::BitVec;
use crate::core::filter::{BloomFilter, ConcurrentBloomFilter, MergeableBloomFilter};
use crate::core::params::{optimal_k, optimal_m};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, StdHasher};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "wyhash")]
use crate::hash::WyHasher;

use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// fast_reduce
// ─────────────────────────────────────────────────────────────────────────────

/// Maps `val` into `[0, m)` using Lemire's multiply-shift range reduction.
///
/// Computes `⌊val × m / 2⁶⁴⌋` via a single 128-bit multiply, replacing the
/// `val % m` integer division in the hot-path index recurrence. The bias relative
/// to true modular reduction is at most 1 in 2⁶⁴ — below the FPR measurement
/// noise floor for any practical filter size.
///
/// **Performance.** On x86-64: `div r64` costs ~20–40 cycles; `mul r64` (with
/// 128-bit result) costs ~4 cycles. At k = 7 this saves roughly 100–280 cycles
/// per insert or query.
///
/// Reference: Lemire (2019), "Fast Random Integer Generation in an Interval,"
/// *ACM Trans. Model. Comput. Simul.* 29(1).
#[inline(always)]
fn fast_reduce(val: u64, m: u64) -> usize {
    ((val as u128 * m as u128) >> 64) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// HashBytes
// ─────────────────────────────────────────────────────────────────────────────

/// A byte-accumulating [`std::hash::Hasher`] that captures the canonical byte
/// stream emitted by `T::hash()` without compressing it to a `u64`.
///
/// # Motivation
///
/// The naive approach — feeding `T` through `DefaultHasher` and taking
/// `finish().to_le_bytes()` — compresses the entire item to 8 bytes before
/// [`BloomHasher`] ever sees it. Both `h1` and `h2` are then derived from the
/// same scalar, weakening the independence assumption of enhanced double hashing.
///
/// `HashBytes` implements `Hasher` as a plain byte sink. Every `write` call from
/// `T::hash()` — field bytes, length prefixes, enum discriminants — is stored
/// verbatim. `H::hash_bytes_pair` then receives the complete item representation.
///
/// # Allocation Behaviour
///
/// A 128-byte inline buffer handles all primitive types and strings up to ~112
/// characters without any heap allocation. Larger inputs spill to a `Vec<u8>` on
/// first overflow; only one allocation ever occurs per item.
///
/// # Correctness
///
/// The correctness of the hashing pipeline depends on `T`'s `Hash` implementation
/// writing a canonical and injective byte representation. A `Hash` impl that
/// writes nothing maps all instances of that type to the same filter positions;
/// this is a violation of the `Hash` contract, not a filter defect.
pub struct HashBytes {
    /// Inline storage for the common case. Valid bytes occupy `[0..len]`.
    inline: [u8; 128],
    /// Number of bytes written to `inline`. Meaningful only while `spill` is empty.
    len: usize,
    /// Heap overflow buffer. Non-empty only after `inline` was exhausted.
    spill: Vec<u8>,
}

impl HashBytes {
    #[inline]
    fn new() -> Self {
        Self {
            inline: [0u8; 128],
            len: 0,
            spill: Vec::new(),
        }
    }

    /// Returns the accumulated bytes as a contiguous slice.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        if self.spill.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }
}

impl std::hash::Hasher for HashBytes {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if self.spill.is_empty() {
            let remaining = self.inline.len() - self.len;
            if bytes.len() <= remaining {
                // Fast path: append into the inline buffer without any allocation.
                self.inline[self.len..self.len + bytes.len()].copy_from_slice(bytes);
                self.len += bytes.len();
                return;
            }
            // Spill path: migrate inline bytes to heap, then append the overflow.
            self.spill.reserve(self.len + bytes.len());
            self.spill.extend_from_slice(&self.inline[..self.len]);
        }
        self.spill.extend_from_slice(bytes);
    }

    /// Satisfies the `Hasher` trait contract. Not used by the Bloom filter pipeline;
    /// the accumulated bytes are consumed directly via [`as_bytes`](Self::as_bytes).
    #[inline]
    fn finish(&self) -> u64 {
        0
    }
}

/// Drives `T::hash()` into a fresh [`HashBytes`] accumulator and returns it.
///
/// This is the single entry point for item hashing. The returned [`HashBytes`]
/// is passed to `H::hash_bytes_pair` to derive the two base hash values.
#[inline]
fn collect_hash_bytes<T: Hash>(item: &T) -> HashBytes {
    let mut collector = HashBytes::new();
    item.hash(&mut collector);
    collector
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterHealth
// ─────────────────────────────────────────────────────────────────────────────

/// Operational health classification for a [`StandardBloomFilter`].
///
/// Returned by [`StandardBloomFilter::health_check`]. The three states reflect
/// fill rate and FPR degradation relative to the filter's design targets, and
/// are intended for production monitoring and rotation planning.
///
/// ## State Boundaries
///
/// | State | Fill rate | FPR relative to target |
/// |---|---|---|
/// | `Healthy` | < 50% | < 2× |
/// | `Degraded` | < 70% | < 5× |
/// | `Critical` | ≥ 70% **or** | ≥ 5× |
///
/// A `Healthy` filter always satisfies `!BloomFilter::is_saturated()`.
///
/// # Examples
///
/// ```
/// use bloomcraft::filters::{StandardBloomFilter, FilterHealth};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let filter = StandardBloomFilter::<u32>::new(1_000, 0.01)?;
/// for i in 0..500 { filter.insert(&i); }
///
/// match filter.health_check() {
///     FilterHealth::Healthy { fill_rate, .. } => {
///         assert!(fill_rate < 0.5);
///     }
///     FilterHealth::Degraded { recommendation, .. } => eprintln!("[WARN] {recommendation}"),
///     FilterHealth::Critical { recommendation, .. } => eprintln!("[CRIT] {recommendation}"),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FilterHealth {
    /// Filter is operating within healthy parameters.
    Healthy {
        /// Fraction of bits currently set, in `[0.0, 1.0]`.
        fill_rate: f64,
        /// Estimated false positive rate derived from the current fill rate.
        current_fpr: f64,
        /// Unique item estimate (Bloom 1970 cardinality estimator).
        estimated_items: usize,
    },

    /// Fill rate or FPR is elevated but the filter remains operational.
    ///
    /// Plan rotation or expansion before the filter reaches `Critical`.
    Degraded {
        /// Fraction of bits currently set.
        fill_rate: f64,
        /// Estimated false positive rate.
        current_fpr: f64,
        /// `current_fpr / target_fpr`.
        fpr_ratio: f64,
        /// Unique item estimate.
        estimated_items: usize,
        /// Human-readable remediation recommendation.
        recommendation: &'static str,
    },

    /// Fill rate or FPR has exceeded safe operating limits.
    ///
    /// The false positive rate is severely degraded. Replace or reset the filter
    /// immediately.
    Critical {
        /// Fraction of bits currently set.
        fill_rate: f64,
        /// Estimated false positive rate.
        current_fpr: f64,
        /// `current_fpr / target_fpr`.
        fpr_ratio: f64,
        /// Unique item estimate.
        estimated_items: usize,
        /// Human-readable remediation recommendation.
        recommendation: &'static str,
    },
}

impl FilterHealth {
    /// Returns the fill rate field from whichever variant is active.
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        match self {
            Self::Healthy { fill_rate, .. }
            | Self::Degraded { fill_rate, .. }
            | Self::Critical { fill_rate, .. } => *fill_rate,
        }
    }

    /// Returns the estimated false positive rate from whichever variant is active.
    #[must_use]
    pub fn current_fpr(&self) -> f64 {
        match self {
            Self::Healthy { current_fpr, .. }
            | Self::Degraded { current_fpr, .. }
            | Self::Critical { current_fpr, .. } => *current_fpr,
        }
    }

    /// Returns the estimated unique item count from whichever variant is active.
    #[must_use]
    pub fn estimated_items(&self) -> usize {
        match self {
            Self::Healthy { estimated_items, .. }
            | Self::Degraded { estimated_items, .. }
            | Self::Critical { estimated_items, .. } => *estimated_items,
        }
    }

    /// Returns `true` if this snapshot represents a `Healthy` filter.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy { .. })
    }

    /// Returns `true` if this snapshot represents a `Degraded` filter.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::Degraded { .. })
    }

    /// Returns `true` if this snapshot represents a `Critical` filter.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Critical { .. })
    }
}

impl std::fmt::Display for FilterHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterHealth::Healthy { fill_rate, current_fpr, estimated_items } => write!(
                f,
                "[OK] Healthy: Fill {:.1}%, FPR {:.4}, Items ~{}",
                fill_rate * 100.0,
                current_fpr,
                estimated_items
            ),
            FilterHealth::Degraded { fill_rate, fpr_ratio, recommendation, .. } => write!(
                f,
                "[WARN] Degraded: Fill {:.1}%, FPR {:.1}× target — {}",
                fill_rate * 100.0,
                fpr_ratio,
                recommendation
            ),
            FilterHealth::Critical { fill_rate, fpr_ratio, recommendation, .. } => write!(
                f,
                "[CRIT] Critical: Fill {:.1}%, FPR {:.1}× target — {}",
                fill_rate * 100.0,
                fpr_ratio,
                recommendation
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-operation counters for a [`StandardBloomFilter`].
///
/// Available when the crate is built with the `metrics` feature. All fields are
/// populated by the filter's insert and query methods and are reset by
/// [`StandardBloomFilter::clear`].
///
/// These counters reflect *logical* operations, not individual atomic bit
/// accesses; one `insert` call increments `total_inserts` by one regardless of `k`.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Default)]
pub struct FilterMetrics {
    /// Total calls to `insert` or `insert_batch` since construction or last clear.
    pub total_inserts: usize,
    /// Total calls to `contains` or `contains_batch` since construction or last clear.
    pub total_queries: usize,
    /// Queries that returned `true` (members and false positives combined).
    pub query_hits: usize,
    /// Queries that returned `false` (guaranteed non-members).
    pub query_misses: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// StandardBloomFilter
// ─────────────────────────────────────────────────────────────────────────────

/// A standard Bloom filter with automatic parameter derivation and lock-free
/// concurrent access.
///
/// # Type Parameters
///
/// * `T` — Element type. Must implement [`Hash`].
/// * `H` — Hash function. Must implement [`BloomHasher`] + [`Clone`]. Defaults
///   to [`StdHasher`] when constructed via [`new`](Self::new).
///
/// # Construction
///
/// The primary constructor [`new`](Self::new) accepts an expected item count `n`
/// and a target false positive rate `p`, deriving optimal `m` and `k`
/// automatically. Use [`with_params`](Self::with_params) when you need to specify
/// `m` and `k` explicitly, or [`with_hasher`](Self::with_hasher) to supply a
/// custom or seeded hash function.
///
/// # Thread Safety
///
/// All insert and query methods take `&self` and are safe for concurrent use on
/// `Arc<Self>`. The underlying [`BitVec`] performs bit writes via
/// `AtomicU64::fetch_or` (`Release`) and reads via `AtomicU64::load` (`Acquire`).
///
/// The following operations require `&mut self` and must be externally serialised
/// if called alongside concurrent inserts or queries:
/// - [`clear`](Self::clear)
/// - [`MergeableBloomFilter::union`]
/// - [`MergeableBloomFilter::intersect`]
///
/// # Serialisation
///
/// With the `serde` feature, the bit array and structural parameters are
/// serialised. The hasher is excluded and reconstructed via `H::default()` on
/// deserialisation. See the module documentation for cross-process stability
/// considerations.
///
/// # Examples
///
/// ```
/// use bloomcraft::filters::StandardBloomFilter;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let filter = StandardBloomFilter::<String>::new(100_000, 0.01)?;
///
/// filter.insert(&"alice@example.com".to_string());
///
/// assert!( filter.contains(&"alice@example.com".to_string()));
/// assert!(!filter.contains(&"carol@example.com".to_string()));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StandardBloomFilter<T, H = StdHasher>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// The underlying bit array. Atomic word-level access enables lock-free concurrency.
    bitvec: BitVec,
    /// Number of hash functions (k).
    k: usize,
    /// Hash function instance. Excluded from serialisation; restored via `H::default()`.
    #[cfg_attr(feature = "serde", serde(skip, default = "H::default"))]
    hasher: H,
    /// Item count the filter was sized for (`n`). Zero when constructed via
    /// `with_params` or `from_parts`.
    expected_items: usize,
    /// Target false positive rate the filter was sized for. Zero when constructed
    /// via `with_params` or `from_parts`.
    target_fpr: f64,
    /// Set to `true` on the first insert. Used by `is_empty` to avoid an O(m)
    /// popcount scan on the common cold-start path.
    #[cfg_attr(feature = "serde", serde(skip, default))]
    has_inserts: AtomicBool,
    _phantom: PhantomData<T>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Clone
// ─────────────────────────────────────────────────────────────────────────────

impl<T, H> Clone for StandardBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bitvec: self.bitvec.clone(),
            k: self.k,
            hasher: self.hasher.clone(),
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            // AtomicBool does not implement Clone; load and re-wrap manually.
            has_inserts: AtomicBool::new(self.has_inserts.load(Ordering::Relaxed)),
            _phantom: PhantomData,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor — T only (H = StdHasher)
// ─────────────────────────────────────────────────────────────────────────────

impl<T> StandardBloomFilter<T>
where
    T: Hash,
{
    /// Creates a filter sized for `expected_items` elements at the given false
    /// positive rate, using the default hasher ([`StdHasher`]).
    ///
    /// Optimal `m` and `k` are derived from the standard Bloom filter formulae.
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidItemCount`] if `expected_items == 0`.
    /// - [`BloomCraftError::FalsePositiveRateOutOfBounds`] if `fpr` ∉ (0, 1).
    /// - [`BloomCraftError::InvalidParameters`] if the derived parameters exceed
    ///   implementation limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let filter = StandardBloomFilter::<u64>::new(100_000, 0.01)?;
    /// assert!(filter.size() > 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(expected_items: usize, fpr: f64) -> Result<Self> {
        Self::with_hasher(expected_items, fpr, StdHasher::new())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructors + full API — T + H
// ─────────────────────────────────────────────────────────────────────────────

impl<T, H> StandardBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Creates a filter with a custom hasher, sized for `expected_items` at `fpr`.
    ///
    /// Prefer this constructor when you need a seeded or non-default hasher
    /// (e.g., for cross-process stability or domain separation). `m` and `k`
    /// are derived automatically.
    ///
    /// # Errors
    ///
    /// Same conditions as [`new`](Self::new).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let filter = StandardBloomFilter::<u64>::with_hasher(
    ///     10_000,
    ///     0.01,
    ///     StdHasher::with_seed(0xDEAD_BEEF),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Result<Self> {
        if expected_items == 0 {
            return Err(BloomCraftError::invalid_item_count(0));
        }
        if !(0.0 < fpr && fpr < 1.0) {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fpr));
        }
        let m = optimal_m(expected_items, fpr)?;
        let k = optimal_k(expected_items, m)?;
        Ok(Self {
            bitvec: BitVec::new(m)?,
            k,
            hasher,
            expected_items,
            target_fpr: fpr,
            has_inserts: AtomicBool::new(false),
            _phantom: PhantomData,
        })
    }

    /// Creates a filter with explicit bit-array size `m` and hash count `k`,
    /// bypassing automatic parameter derivation.
    ///
    /// Use this when integrating with an external system that dictates filter
    /// dimensions, or when benchmarking specific parameter combinations.
    /// `expected_items` and `target_fpr` are recorded as zero; methods that
    /// depend on them (e.g., [`health_check`](Self::health_check)) will report
    /// a `fpr_ratio` of `1.0`.
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidFilterSize`] if `m == 0`.
    /// - [`BloomCraftError::InvalidHashCount`] if `k == 0` or `k > 32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Matches the parameters of a filter created with new(1000, 0.01).
    /// let filter = StandardBloomFilter::<u64>::with_params(9_585, 7, StdHasher::new())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_params(m: usize, k: usize, hasher: H) -> Result<Self> {
        if m == 0 {
            return Err(BloomCraftError::invalid_filter_size(0));
        }
        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }
        Ok(Self {
            bitvec: BitVec::new(m)?,
            k,
            hasher,
            expected_items: 0,
            target_fpr: 0.0,
            has_inserts: AtomicBool::new(false),
            _phantom: PhantomData,
        })
    }

    /// Reconstructs a filter from a raw [`BitVec`] and hash count `k`.
    ///
    /// Intended for custom deserialisation paths where the bit array has been
    /// stored and retrieved independently of the standard `serde` feature.
    /// The hasher is initialised via `H::default()`.
    ///
    /// `expected_items` and `target_fpr` are set to zero; callers that need
    /// accurate capacity metadata should store and restore those values separately.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::InvalidHashCount`] if `k` is outside `[1, 32]`.
    pub fn from_parts(bits: BitVec, k: usize) -> Result<Self>
    where
        H: Default,
    {
        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }
        Ok(Self {
            expected_items: 0,
            target_fpr: 0.0,
            k,
            bitvec: bits,
            hasher: H::default(),
            has_inserts: AtomicBool::new(false),
            _phantom: PhantomData,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Structural accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns the total number of bits allocated (`m`).
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.bitvec.len()
    }

    /// Returns the number of hash functions (`k`).
    #[must_use]
    #[inline]
    pub fn hash_count(&self) -> usize {
        self.k
    }

    /// Alias for [`hash_count`](Self::hash_count).
    #[must_use]
    #[inline]
    pub fn num_hashes(&self) -> usize {
        self.k
    }

    /// Returns the expected item count this filter was sized for.
    ///
    /// Returns `0` for filters constructed via [`with_params`](Self::with_params)
    /// or [`from_parts`](Self::from_parts).
    #[must_use]
    #[inline]
    pub fn expected_items(&self) -> usize {
        self.expected_items
    }

    /// Returns the target false positive rate this filter was sized for.
    ///
    /// Returns `0.0` for filters constructed via [`with_params`](Self::with_params)
    /// or [`from_parts`](Self::from_parts).
    #[must_use]
    #[inline]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Observability
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns the number of bits currently set to 1 (popcount of the bit array).
    ///
    /// Runs in O(⌈m/64⌉). This is the raw bit count, not the number of inserted
    /// items — multiple items can share bits, and a single item sets `k` bits
    /// (some of which may already be set). For an estimated unique item count use
    /// [`estimate_cardinality`](Self::estimate_cardinality).
    ///
    /// This method backs [`BloomFilter::count_set_bits`], which in turn drives
    /// the trait's provided `estimate_count`, `fill_rate`, and `is_saturated`
    /// implementations.
    #[must_use]
    #[inline]
    pub fn count_set_bits(&self) -> usize {
        self.bitvec.count_ones()
    }

    /// Returns the popcount of the bit array.
    ///
    /// This is an alias for [`count_set_bits`](Self::count_set_bits) provided
    /// for ergonomic consistency with Rust collection conventions. It does **not**
    /// return the number of inserted items.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.count_set_bits()
    }

    /// Returns the fraction of bits currently set, in `[0.0, 1.0]`.
    ///
    /// Equivalent to `count_set_bits() as f64 / size() as f64`. Also available
    /// without importing any trait as a mirror of [`BloomFilter::fill_rate`].
    #[must_use]
    #[inline]
    pub fn fill_rate(&self) -> f64 {
        self.count_set_bits() as f64 / self.size() as f64
    }

    /// Returns `true` when the fill rate exceeds 50%.
    ///
    /// At this point the actual false positive rate has risen above the design
    /// target. Equivalent to [`BloomFilter::is_saturated`].
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool {
        self.fill_rate() > 0.5
    }

    /// Returns `true` if no items have been inserted since construction or the
    /// last [`clear`](Self::clear). O(1).
    ///
    /// This check is based on an insertion flag rather than a bit-array scan.
    /// After a constructive [`union`](Self::union) with a non-empty filter, this
    /// method returns `false` because the flag is propagated from the source.
    /// A union of two empty filters correctly returns `true`.
    ///
    /// For bit-precise emptiness after deserialisation use
    /// `count_set_bits() == 0`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.has_inserts.load(Ordering::Relaxed)
    }

    /// Estimates the current false positive rate from the observed fill rate.
    ///
    /// The estimate inverts the fill probability to recover an approximate item
    /// count `n̂`, then applies the standard FPR formula:
    ///
    /// ```text
    /// n̂          = −(m/k) × ln(1 − X/m)
    /// p_estimated = (1 − e^(−k·n̂/m))^k
    /// ```
    ///
    /// where `X` is `count_set_bits()`. Returns `0.0` on an empty filter and
    /// `1.0` when every bit is set.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let set_bits = self.count_set_bits();
        if set_bits == 0 {
            return 0.0;
        }
        let m = self.size() as f64;
        let k = self.k as f64;
        let fill_rate = set_bits as f64 / m;
        if fill_rate >= 1.0 {
            return 1.0;
        }
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();
        let exponent = -k * estimated_n / m;
        (1.0 - exponent.exp()).powf(k)
    }

    /// Alias for [`estimate_fpr`](Self::estimate_fpr).
    #[must_use]
    #[inline]
    pub fn estimated_fp_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    /// Estimates the number of distinct items currently in the filter.
    ///
    /// Uses the Bloom (1970) cardinality estimator, which inverts the expected
    /// fill probability:
    ///
    /// ```text
    /// n̂ = −(m/k) × ln(1 − X/m)
    /// ```
    ///
    /// Typical accuracy is ±5% for fill rates in `[0.1, 0.7]`. Returns `0` on an
    /// empty filter and [`usize::MAX`] when the bit array is fully saturated.
    /// Never panics.
    #[must_use]
    pub fn estimate_cardinality(&self) -> usize {
        let set_bits = self.count_set_bits();
        if set_bits == 0 {
            return 0;
        }
        if set_bits >= self.size() {
            return usize::MAX;
        }
        let m = self.size() as f64;
        let k = self.k as f64;
        let fill_rate = set_bits as f64 / m;
        (-(m / k) * (1.0 - fill_rate).ln()).max(0.0) as usize
    }

    /// Returns the approximate heap memory consumed by this filter, in bytes.
    ///
    /// The [`BitVec`] heap allocation is measured precisely. Fixed-size scalar
    /// fields (`k`, `expected_items`, `target_fpr`) are exact. The hasher `H` is
    /// measured via `size_of::<H>()`, which covers only its stack footprint;
    /// hashers with internal heap allocations will be undercounted.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bitvec.memory_usage()
            + std::mem::size_of::<usize>() * 2  // k, expected_items
            + std::mem::size_of::<f64>()         // target_fpr
            + std::mem::size_of::<H>()           // hasher (stack only)
            + std::mem::size_of::<AtomicBool>()
    }

    /// Returns the raw bit array as a `Vec<u64>`.
    ///
    /// Intended for custom serialisation. Pair with [`from_parts`](Self::from_parts)
    /// to reconstruct the filter.
    #[must_use]
    pub fn raw_bits(&self) -> Vec<u64> {
        self.bitvec.to_raw()
    }

    /// Returns the name of the hasher backing this filter.
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }

    /// Returns the index generation strategy used by this filter.
    ///
    /// Always [`IndexingStrategy::EnhancedDouble`](crate::hash::IndexingStrategy).
    #[must_use]
    pub fn hash_strategy(&self) -> crate::hash::IndexingStrategy {
        crate::hash::IndexingStrategy::EnhancedDouble
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Core operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Inserts `item` into the filter.
    ///
    /// Lock-free. Safe to call concurrently from multiple threads on a shared
    /// `Arc<Self>`.
    ///
    /// The operation is idempotent: inserting the same item twice does not change
    /// the bit array or increase the false positive rate beyond the first insert.
    ///
    /// This method is the shared implementation for both [`BloomFilter::insert`]
    /// (which takes `&mut self` for single-threaded callers) and
    /// [`ConcurrentBloomFilter::insert_concurrent`] (which takes `&self`).
    #[inline]
    pub fn insert(&self, item: &T) {
        let hb = collect_hash_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(hb.as_bytes());
        self.set_indices(h1, h2);
        if !self.has_inserts.load(Ordering::Relaxed) {
            self.has_inserts.store(true, Ordering::Relaxed);
        }
    }

    /// Tests whether `item` is in the filter.
    ///
    /// Lock-free. Safe for concurrent use.
    ///
    /// Returns `false` only when `item` is a guaranteed non-member. Returns
    /// `true` for all members and for false positives; the false positive rate
    /// is bounded by the filter's design FPR when the insertion count is at or
    /// below `expected_items`.
    ///
    /// This method is the shared implementation for both [`BloomFilter::contains`]
    /// and [`ConcurrentBloomFilter::contains_concurrent`].
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        let hb = collect_hash_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(hb.as_bytes());
        self.all_indices_set(h1, h2)
    }

    /// Inserts all items in `items` into the filter.
    ///
    /// Lock-free. Safe for concurrent use on a shared `Arc<Self>`.
    ///
    /// Prefer [`insert_batch_ref`](Self::insert_batch_ref) when you hold
    /// references rather than owned values, to avoid the per-item copy.
    pub fn insert_batch(&self, items: &[T]) {
        for item in items {
            let hb = collect_hash_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(hb.as_bytes());
            self.set_indices(h1, h2);
        }
        if !items.is_empty() && !self.has_inserts.load(Ordering::Relaxed) {
            self.has_inserts.store(true, Ordering::Relaxed);
        }
    }

    /// Inserts all items via references, avoiding per-element copies.
    ///
    /// Lock-free. Safe for concurrent use.
    ///
    /// Use this variant when items are not owned by the caller. The semantics are
    /// identical to [`insert_batch`](Self::insert_batch).
    pub fn insert_batch_ref(&self, items: &[&T]) {
        for item in items {
            let hb = collect_hash_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(hb.as_bytes());
            self.set_indices(h1, h2);
        }
        if !items.is_empty() && !self.has_inserts.load(Ordering::Relaxed) {
            self.has_inserts.store(true, Ordering::Relaxed);
        }
    }

    /// Tests membership for each item in `items`.
    ///
    /// Returns a `Vec<bool>` where `result[i]` is the membership result for
    /// `items[i]`. Lock-free. The result vector is pre-allocated to `items.len()`.
    ///
    /// Prefer [`contains_batch_ref`](Self::contains_batch_ref) when holding
    /// references to avoid per-item copies.
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        let mut results = Vec::with_capacity(items.len());
        for item in items {
            let hb = collect_hash_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(hb.as_bytes());
            results.push(self.all_indices_set(h1, h2));
        }
        results
    }

    /// Tests membership for each item via references.
    ///
    /// Zero-copy equivalent of [`contains_batch`](Self::contains_batch). Lock-free.
    #[must_use]
    pub fn contains_batch_ref(&self, items: &[&T]) -> Vec<bool> {
        let mut results = Vec::with_capacity(items.len());
        for item in items {
            let hb = collect_hash_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(hb.as_bytes());
            results.push(self.all_indices_set(h1, h2));
        }
        results
    }

    /// Clears all bits and resets the insertion flag.
    ///
    /// Requires exclusive access. Must not be called concurrently with any insert
    /// or query operation without external synchronisation (e.g., `RwLock`).
    /// After a `clear`, all `contains` calls return `false` and `is_empty`
    /// returns `true`.
    pub fn clear(&mut self) {
        self.bitvec.clear();
        self.has_inserts.store(false, Ordering::Relaxed);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Hot-path index generation
    //
    // The enhanced double hashing recurrence:
    //
    //   g_i = h1 + i·h2 + i(i+1)/2  (mod m)
    //
    // is computed incrementally. Let val[0] = h1, step[0] = h2. Then:
    //
    //   step[i] = step[i-1] + 1
    //   val[i]  = val[i-1] + step[i]    (all arithmetic wrapping u64)
    //
    // Expanding: val[i] = h1 + Σ_{j=1}^{i}(h2 + j) = h1 + i·h2 + i(i+1)/2 ✓
    //
    // Per iteration: two wrapping additions + one multiply-shift (fast_reduce).
    // No heap allocation at any call site.
    // ─────────────────────────────────────────────────────────────────────────

    /// Sets the `k` bit positions derived from `(h1, h2)` using the inline
    /// enhanced double hashing recurrence.
    #[inline(always)]
    fn set_indices(&self, h1: u64, h2: u64) {
        let m = self.bitvec.len() as u64;
        let mut val = h1;
        let mut step = h2;
        self.bitvec.set(fast_reduce(val, m));
        for _ in 1..self.k {
            step = step.wrapping_add(1);
            val = val.wrapping_add(step);
            self.bitvec.set(fast_reduce(val, m));
        }
    }

    /// Returns `true` iff all `k` bit positions derived from `(h1, h2)` are set.
    ///
    /// Short-circuits on the first unset bit; the recurrence is never advanced
    /// past that point.
    #[inline(always)]
    fn all_indices_set(&self, h1: u64, h2: u64) -> bool {
        let m = self.bitvec.len() as u64;
        let mut val = h1;
        let mut step = h2;
        if !self.bitvec.get(fast_reduce(val, m)) {
            return false;
        }
        for _ in 1..self.k {
            step = step.wrapping_add(1);
            val = val.wrapping_add(step);
            if !self.bitvec.get(fast_reduce(val, m)) {
                return false;
            }
        }
        true
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Constructive set algebra
    //
    // These inherent methods return a new filter and take `&self`, coexisting
    // with the MergeableBloomFilter trait methods that modify `self` in place
    // and take `&mut self`. Inherent methods win dot-call resolution; use UFCS
    // for the trait variants:
    //
    //   use bloomcraft::core::MergeableBloomFilter;
    //   MergeableBloomFilter::union(&mut f1, &f2)?;
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns a new filter whose bit array is the bitwise OR of `self` and `other`.
    ///
    /// Both source filters are left unmodified. The result contains every item
    /// present in either source; no false negatives are introduced. The FPR of
    /// the result reflects the combined fill rate of both sources.
    ///
    /// For the **in-place** variant use UFCS with [`MergeableBloomFilter`]:
    ///
    /// ```ignore
    /// use bloomcraft::core::MergeableBloomFilter;
    /// MergeableBloomFilter::union(&mut f1, &f2)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`], including the actual `m`
    /// and `k` values, if `self.size() != other.size()` or `self.hash_count() !=
    /// other.hash_count()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let f1 = StandardBloomFilter::<String>::new(1_000, 0.01)?;
    /// let f2 = StandardBloomFilter::<String>::new(1_000, 0.01)?;
    /// f1.insert(&"alice".to_string());
    /// f2.insert(&"bob".to_string());
    ///
    /// let combined = f1.union(&f2)?;
    /// assert!(combined.contains(&"alice".to_string()));
    /// assert!(combined.contains(&"bob".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.size() != other.size() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "size mismatch: self.m={} vs other.m={}",
                    self.size(),
                    other.size()
                ),
            });
        }
        if self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "hash count mismatch: self.k={} vs other.k={}",
                    self.k, other.k
                ),
            });
        }
        let mut result = self.clone();
        result.bitvec = self.bitvec.union(&other.bitvec)?;

        // Propagate the insertion flag: the result is non-empty if either source was.
        let has = self.has_inserts.load(Ordering::Relaxed)
            || other.has_inserts.load(Ordering::Relaxed);
        result.has_inserts.store(has, Ordering::Relaxed);
        Ok(result)
    }

    /// Returns a new filter whose bit array is the bitwise AND of `self` and `other`.
    ///
    /// Both source filters are left unmodified.
    ///
    /// # ⚠ False Negatives
    ///
    /// **This operation breaks the no-false-negatives guarantee.**
    ///
    /// Bitwise AND clears any bit not set in both filters. Items present in only
    /// one source will have some or all of their `k` bits cleared, causing
    /// subsequent `contains` calls to return `false` for them. This is an inherent
    /// consequence of Bloom filter intersection — not a bug.
    ///
    /// Use the result only to approximate membership in the **overlap** of two
    /// known sets (e.g., items replicated across two nodes). Never rely on it as a
    /// general-purpose filter over items from a single source.
    ///
    /// For the **in-place** variant use UFCS with [`MergeableBloomFilter`]:
    ///
    /// ```ignore
    /// use bloomcraft::core::MergeableBloomFilter;
    /// MergeableBloomFilter::intersect(&mut f1, &f2)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] if `size()` or
    /// `hash_count()` differ between the two filters.
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.size() != other.size() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "size mismatch: self.m={} vs other.m={}",
                    self.size(),
                    other.size()
                ),
            });
        }
        if self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "hash count mismatch: self.k={} vs other.k={}",
                    self.k, other.k
                ),
            });
        }
        let mut result = self.clone();
        result.bitvec = self.bitvec.intersect(&other.bitvec)?;
        Ok(result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Health check
    // ─────────────────────────────────────────────────────────────────────────

    /// Classifies the operational health of this filter.
    ///
    /// Computes the current fill rate and estimated FPR, compares them against
    /// the design targets, and returns a [`FilterHealth`] variant. See
    /// [`FilterHealth`] for the exact boundaries between states.
    ///
    /// A filter classified [`Healthy`](FilterHealth::Healthy) always satisfies
    /// `!BloomFilter::is_saturated()`.
    ///
    /// **Note**: This method calls [`count_set_bits`](Self::count_set_bits)
    /// internally, which is O(⌈m/64⌉). Avoid calling it in hot paths; prefer a
    /// background sampling interval (e.g., a metrics scrape every few seconds).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let filter = StandardBloomFilter::<u32>::new(1_000, 0.01)?;
    /// for i in 0..100 { filter.insert(&i); }
    ///
    /// println!("{}", filter.health_check()); // "[OK] Healthy: Fill 9.8%, FPR 0.0001, Items ~100"
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn health_check(&self) -> FilterHealth {
        let fill_rate = self.fill_rate();
        let current_fpr = self.estimate_fpr();
        let estimated_items = self.estimate_cardinality();
        let fpr_ratio = if self.target_fpr > 0.0 {
            current_fpr / self.target_fpr
        } else {
            // No design FPR recorded (constructed via with_params/from_parts).
            1.0
        };

        if fill_rate < 0.5 && fpr_ratio < 2.0 {
            FilterHealth::Healthy {
                fill_rate,
                current_fpr,
                estimated_items,
            }
        } else if fill_rate < 0.7 && fpr_ratio < 5.0 {
            FilterHealth::Degraded {
                fill_rate,
                current_fpr,
                fpr_ratio,
                estimated_items,
                recommendation: "Consider creating a new filter soon",
            }
        } else {
            FilterHealth::Critical {
                fill_rate,
                current_fpr,
                fpr_ratio,
                estimated_items,
                recommendation: "URGENT: Replace filter immediately — FPR is severely degraded",
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// impl BloomFilter
// ─────────────────────────────────────────────────────────────────────────────

impl<T, H> BloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    /// Inserts `item` via the single-threaded `BloomFilter` trait contract.
    ///
    /// The `&mut self` receiver satisfies the trait's ownership model for
    /// single-threaded callers. Internally delegates to the lock-free inherent
    /// `insert(&self)`; exclusive access is not operationally required here.
    fn insert(&mut self, item: &T) {
        StandardBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        StandardBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        StandardBloomFilter::clear(self);
    }

    /// Returns the popcount of the bit array.
    ///
    /// Not the insertion count. Use [`estimate_count`](BloomFilter::estimate_count)
    /// for a unique-item estimate.
    fn len(&self) -> usize {
        self.count_set_bits()
    }

    fn is_empty(&self) -> bool {
        StandardBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.size()
    }

    fn hash_count(&self) -> usize {
        self.k
    }

    /// Returns the raw popcount of the bit array.
    ///
    /// O(⌈m/64⌉). Backs the trait's provided `estimate_count`, `fill_rate`, and
    /// `is_saturated` default implementations.
    fn count_set_bits(&self) -> usize {
        StandardBloomFilter::count_set_bits(self)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// impl ConcurrentBloomFilter
// ─────────────────────────────────────────────────────────────────────────────

impl<T, H> ConcurrentBloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    /// Lock-free insert. Safe for concurrent use on a shared `Arc<Self>`.
    #[inline]
    fn insert_concurrent(&self, item: &T) {
        StandardBloomFilter::insert(self, item);
    }

    fn insert_batch_concurrent(&self, items: &[T]) {
        StandardBloomFilter::insert_batch(self, items);
    }

    fn insert_batch_ref_concurrent(&self, items: &[&T]) {
        StandardBloomFilter::insert_batch_ref(self, items);
    }

    /// Lock-free membership query. Safe for concurrent use on a shared `Arc<Self>`.
    ///
    /// Returns `false` only for guaranteed non-members.
    #[inline]
    fn contains_concurrent(&self, item: &T) -> bool {
        StandardBloomFilter::contains(self, item)
    }

    /// Batch concurrent membership query.
    ///
    /// Overrides the trait default to delegate to the optimised inherent
    /// [`contains_batch`](StandardBloomFilter::contains_batch), which
    /// pre-allocates the result vector and short-circuits on the first unset bit
    /// per item, rather than calling `contains_concurrent` in a loop.
    fn contains_batch_concurrent(&self, items: &[T]) -> Vec<bool> {
        StandardBloomFilter::contains_batch(self, items)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// impl MergeableBloomFilter
// ─────────────────────────────────────────────────────────────────────────────

/// In-place merge operations for [`StandardBloomFilter`].
///
/// Two filters are **compatible** when `size()` (m) and `hash_count()` (k) are
/// equal. Hasher type and seed are not checked at runtime; callers must ensure
/// both filters were constructed with the same `H` and seed to guarantee that
/// hash positions correspond.
///
/// For constructive (non-mutating) alternatives that return a new filter, see the
/// inherent [`union`](StandardBloomFilter::union) and
/// [`intersect`](StandardBloomFilter::intersect) methods. Inherent methods win
/// dot-call resolution; use UFCS to reach these in-place variants:
///
/// ```ignore
/// use bloomcraft::core::MergeableBloomFilter;
/// MergeableBloomFilter::union(&mut f1, &f2)?;
/// MergeableBloomFilter::intersect(&mut f1, &f2)?;
/// ```
impl<T, H> MergeableBloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    /// Returns `true` if `self` and `other` can be safely merged.
    ///
    /// Compatibility requires equal `size()` and `hash_count()`. Hasher type
    /// and seed are not verified.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::core::MergeableBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let f1 = StandardBloomFilter::<u64>::new(1_000, 0.01)?;
    /// let f2 = StandardBloomFilter::<u64>::new(1_000, 0.01)?;
    /// let f3 = StandardBloomFilter::<u64>::new(5_000, 0.01)?;
    ///
    /// assert!( f1.is_compatible(&f2));
    /// assert!(!f1.is_compatible(&f3));
    /// # Ok(())
    /// # }
    /// ```
    fn is_compatible(&self, other: &Self) -> bool {
        self.size() == other.size() && self.k == other.k
    }

    /// Modifies `self` in place via bitwise OR with `other`.
    ///
    /// After this call, `self.contains(x)` returns `true` for every `x` that
    /// was present in either source filter before the merge. The no-false-negatives
    /// and bounded-FPR guarantees are preserved; the resulting FPR reflects the
    /// combined fill rate of both operands. `other` is not modified.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] with actual `m` and `k`
    /// values if `!self.is_compatible(other)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::core::MergeableBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut f1 = StandardBloomFilter::<String>::new(1_000, 0.01)?;
    /// let     f2 = StandardBloomFilter::<String>::new(1_000, 0.01)?;
    /// f1.insert(&"alice".to_string());
    /// f2.insert(&"bob".to_string());
    ///
    /// MergeableBloomFilter::union(&mut f1, &f2)?;
    ///
    /// assert!(f1.contains(&"alice".to_string()));
    /// assert!(f1.contains(&"bob".to_string()));
    /// # Ok(())
    /// # }
    /// ```
    fn union(&mut self, other: &Self) -> Result<()> {
        if !self.is_compatible(other) {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "union requires equal dimensions: self(m={}, k={}) vs other(m={}, k={})",
                    self.size(), self.k,
                    other.size(), other.k,
                ),
            });
        }
        self.bitvec = self.bitvec.union(&other.bitvec)?;
        if other.has_inserts.load(Ordering::Relaxed) {
            self.has_inserts.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Modifies `self` in place via bitwise AND with `other`.
    ///
    /// # ⚠ False Negatives
    ///
    /// **This operation breaks the no-false-negatives guarantee.**
    ///
    /// Bitwise AND clears any bit not set in both operands. Items present in only
    /// one source filter will have some or all of their `k` bits cleared, causing
    /// `contains` to return `false` for them. This is an inherent property of
    /// Bloom filter intersection.
    ///
    /// Use the result only to approximate membership in the **overlap** of two
    /// known sets. Never rely on it as a general-purpose filter over items from a
    /// single source. `other` is not modified.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::IncompatibleFilters`] with actual `m` and `k`
    /// values if `!self.is_compatible(other)`.
    fn intersect(&mut self, other: &Self) -> Result<()> {
        if !self.is_compatible(other) {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "intersect requires equal dimensions: self(m={}, k={}) vs other(m={}, k={})",
                    self.size(), self.k,
                    other.size(), other.k,
                ),
            });
        }
        self.bitvec = self.bitvec.intersect(&other.bitvec)?;
        Ok(())
    }

    // union_many() and intersect_many() are provided by the trait defaults.
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::filter::{BloomFilter, ConcurrentBloomFilter, MergeableBloomFilter};
    use std::sync::Arc;
    use std::thread;
    use std::hash::Hasher;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_basic() {
        let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(1000, 0.01).unwrap();
        assert!(filter.size() > 0);
        assert!(filter.hash_count() > 0);
        assert_eq!(filter.expected_items(), 1000);
        assert_eq!(filter.target_fpr(), 0.01);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_new_various_sizes() {
        let small  = StandardBloomFilter::<u64>::new(10,        0.01).unwrap();
        let medium = StandardBloomFilter::<u64>::new(10_000,    0.01).unwrap();
        let large  = StandardBloomFilter::<u64>::new(1_000_000, 0.01).unwrap();
        assert!(small.size() < medium.size());
        assert!(medium.size() < large.size());
    }

    #[test]
    fn test_new_various_fpr() {
        let high   = StandardBloomFilter::<u64>::new(1000, 0.1  ).unwrap();
        let medium = StandardBloomFilter::<u64>::new(1000, 0.01 ).unwrap();
        let low    = StandardBloomFilter::<u64>::new(1000, 0.001).unwrap();
        assert!(high.size()  < medium.size());
        assert!(medium.size() < low.size());
        assert!(high.hash_count() <= medium.hash_count());
        assert!(medium.hash_count() <= low.hash_count());
    }

    #[test]
    fn test_with_params() {
        let filter = StandardBloomFilter::<u64>::with_params(10_000, 7, StdHasher::new()).unwrap();
        assert_eq!(filter.size(), 10_000);
        assert_eq!(filter.hash_count(), 7);
    }

    #[test]
    fn test_with_hasher() {
        let filter = StandardBloomFilter::<u64>::with_hasher(
            1000, 0.01, StdHasher::with_seed(42),
        ).unwrap();
        assert!(filter.size() > 0);
    }

    #[test]
    fn test_new_zero_items() {
        let result = StandardBloomFilter::<u64>::new(0, 0.01);
        assert!(matches!(result.unwrap_err(), BloomCraftError::InvalidItemCount { .. }));
    }

    #[test] fn test_new_invalid_fpr_zero()     { assert!(StandardBloomFilter::<u64>::new(1000,  0.0 ).is_err()); }
    #[test] fn test_new_invalid_fpr_one()      { assert!(StandardBloomFilter::<u64>::new(1000,  1.0 ).is_err()); }
    #[test] fn test_new_invalid_fpr_negative() { assert!(StandardBloomFilter::<u64>::new(1000, -0.01).is_err()); }
    #[test] fn test_new_invalid_fpr_over_one() { assert!(StandardBloomFilter::<u64>::new(1000,  1.5 ).is_err()); }

    #[test] fn test_with_params_zero_size()    { assert!(StandardBloomFilter::<u64>::with_params(   0, 7,  StdHasher::new()).is_err()); }
    #[test] fn test_with_params_zero_hashes()  { assert!(StandardBloomFilter::<u64>::with_params(1000, 0,  StdHasher::new()).is_err()); }
    #[test] fn test_with_params_excess_hashes(){ assert!(StandardBloomFilter::<u64>::with_params(1000, 33, StdHasher::new()).is_err()); }

    // ── HashBytes pipeline ────────────────────────────────────────────────────

    #[test]
    fn hash_bytes_inline_roundtrip() {
        let mut hb = HashBytes::new();
        hb.write(b"hello world");
        assert_eq!(hb.as_bytes(), b"hello world");
        assert!(hb.spill.is_empty(), "must not spill for short input");
    }

    #[test]
    fn hash_bytes_spill_roundtrip() {
        let data = vec![0xABu8; 256];
        let mut hb = HashBytes::new();
        hb.write(&data);
        assert_eq!(hb.as_bytes(), data.as_slice());
        assert!(!hb.spill.is_empty(), "must spill for 256-byte input");
    }

    #[test]
    fn hash_bytes_multi_write_inline() {
        let mut hb = HashBytes::new();
        hb.write(b"foo");
        hb.write(b"bar");
        assert_eq!(hb.as_bytes(), b"foobar");
        assert!(hb.spill.is_empty());
    }

    #[test]
    fn hash_bytes_inline_to_spill_transition() {
        let first  = vec![0u8; 100];
        let second = vec![1u8; 100]; // 200 bytes total exceeds the 128-byte inline buffer
        let mut hb = HashBytes::new();
        hb.write(&first);
        hb.write(&second);
        let expected: Vec<u8> = first.iter().chain(second.iter()).copied().collect();
        assert_eq!(hb.as_bytes(), expected.as_slice());
    }

    #[test]
    fn distinct_items_produce_distinct_hash_positions() {
        let filter = StandardBloomFilter::<u64>::new(1_000_000, 0.0001).unwrap();
        filter.insert(&0u64);
        filter.insert(&u64::MAX);
        assert!(filter.contains(&0u64));
        assert!(filter.contains(&u64::MAX));
        assert!(filter.count_set_bits() > 0);
    }

    #[test]
    fn string_items_with_common_prefix_are_distinguished() {
        let empty = StandardBloomFilter::<String>::new(100_000, 0.0001).unwrap();
        assert!(!empty.contains(&"abc".to_string()));
        assert!(!empty.contains(&"abcd".to_string()));

        let filter = StandardBloomFilter::<String>::new(100_000, 0.0001).unwrap();
        filter.insert(&"abc".to_string());
        assert!(filter.contains(&"abc".to_string()));
    }

    #[test]
    fn seeded_hasher_consistent_across_instances() {
        let f1 = StandardBloomFilter::<u64>::with_hasher(1000, 0.01, StdHasher::with_seed(99)).unwrap();
        let f2 = StandardBloomFilter::<u64>::with_hasher(1000, 0.01, StdHasher::with_seed(99)).unwrap();
        f1.insert(&42u64);
        let u = f1.union(&f2).unwrap();
        assert!(u.contains(&42u64));
    }

    // ── Insert / contains ────────────────────────────────────────────────────

    #[test]
    fn test_insert_and_contains() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        filter.insert(&"hello".to_string());
        assert!( filter.contains(&"hello".to_string()));
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_insert_multiple() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        for i in 0..50 { filter.insert(&i); }
        for i in 0..50 { assert!(filter.contains(&i), "item {i} must be present"); }
    }

    #[test]
    fn test_no_false_negatives() {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let items: Vec<u64> = (0..1000).collect();
        for &item in &items { filter.insert(&item); }
        for &item in &items {
            assert!(filter.contains(&item), "false negative for item {item}");
        }
    }

    #[test]
    fn test_different_types() {
        let f1 = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        f1.insert(&"test".to_string());
        assert!(f1.contains(&"test".to_string()));

        let f2 = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        f2.insert(&42);
        assert!(f2.contains(&42));

        let f3 = StandardBloomFilter::<[u8; 8]>::new(100, 0.01).unwrap();
        let bytes = 3.14f64.to_le_bytes();
        f3.insert(&bytes);
        assert!(f3.contains(&bytes));
    }

    #[test]
    fn test_empty_string() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert(&String::new());
        assert!(filter.contains(&String::new()));
    }

    #[test]
    fn test_duplicate_inserts_idempotent() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        filter.insert(&42);
        let after_first = filter.count_set_bits();
        filter.insert(&42);
        assert!(after_first > 0, "inserting an item must set at least one bit");
        assert_eq!(
            after_first, filter.count_set_bits(),
            "re-inserting the same item must not change the bit count"
        );
    }

    #[test]
    fn test_very_small_fpr() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.0001).unwrap();
        for i in 0..10 { filter.insert(&i); }
        for i in 0..10 { assert!(filter.contains(&i)); }
    }

    #[test]
    fn test_extreme_load() {
        // Intentionally overload a small filter to verify graceful degradation.
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        for i in 0..10_000 { filter.insert(&i); }
        assert!(filter.fill_rate() > 0.9);
        for i in 0..100 { assert!(filter.contains(&i)); }
    }

    // ── Batch operations ──────────────────────────────────────────────────────

    #[test]
    fn test_insert_batch_no_false_negatives() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        let items  = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        filter.insert_batch(&items);
        for item in &items { assert!(filter.contains(item)); }
    }

    #[test]
    fn test_insert_batch_empty() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert_batch(&[]);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_batch_large() {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let items: Vec<u64> = (0..5000).collect();
        filter.insert_batch(&items);
        for &item in &items { assert!(filter.contains(&item)); }
    }

    #[test]
    fn test_contains_batch() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        filter.insert_batch(&items);
        let results = filter.contains_batch(&[
            "a".to_string(), "b".to_string(), "x".to_string(),
        ]);
        assert_eq!(results.len(), 3);
        assert!( results[0]);
        assert!( results[1]);
        assert!(!results[2]);
    }

    #[test]
    fn test_contains_batch_all_present() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let items: Vec<u64> = (0..100).collect();
        filter.insert_batch(&items);
        assert!(filter.contains_batch(&items).iter().all(|&r| r));
    }

    #[test]
    fn test_contains_batch_empty_slice() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        assert!(filter.contains_batch(&[]).is_empty());
    }

    // ── Popcount / fill rate / saturation ────────────────────────────────────

    #[test]
    fn test_count_set_bits_zero_on_empty() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.count_set_bits(), 0);
    }

    #[test]
    fn test_count_set_bits_increases_after_insert() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let before = filter.count_set_bits();
        filter.insert(&42);
        assert!(filter.count_set_bits() > before);
    }

    #[test]
    fn test_count_set_bits_bounded_by_bit_count() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        for i in 0..10_000u64 { filter.insert(&i); }
        assert!(filter.count_set_bits() <= filter.size());
    }

    #[test]
    fn test_fill_rate_zero_on_empty() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.fill_rate(), 0.0);
    }

    #[test]
    fn test_fill_rate_in_unit_range() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { filter.insert(&i); }
        let rate = filter.fill_rate();
        assert!((0.0..=1.0).contains(&rate), "fill_rate={rate} out of [0, 1]");
    }

    #[test]
    fn test_estimate_fpr_empty() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.estimate_fpr(), 0.0);
    }

    #[test]
    fn test_estimate_fpr_increases_with_load() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        let fpr_0 = filter.estimate_fpr();
        for i in 0..50u64  { filter.insert(&i); }
        let fpr_50 = filter.estimate_fpr();
        for i in 50..100u64 { filter.insert(&i); }
        let fpr_100 = filter.estimate_fpr();
        assert!(fpr_0 < fpr_50);
        assert!(fpr_50 < fpr_100);
    }

    #[test]
    fn test_estimate_cardinality() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { filter.insert(&i); }
        let estimated = filter.estimate_cardinality();
        assert!(
            estimated >= 80 && estimated <= 120,
            "estimated {estimated} items, expected ~100"
        );
    }

    #[test]
    fn test_is_empty() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        assert!(filter.is_empty());
        filter.insert(&42);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_is_full_transitions() {
        let filter = StandardBloomFilter::<u64>::new(10, 0.01).unwrap();
        assert!(!filter.is_full());
        for i in 0..1000u64 { filter.insert(&i); }
        assert!(filter.is_full());
    }

    #[test]
    fn test_memory_usage() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let usage = filter.memory_usage();
        let min_expected = filter.size() / 8;
        assert!(usage >= min_expected);
    }

    // ── Clear ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear() {
        let mut filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert(&"hello".to_string());
        filter.clear();
        assert!(filter.is_empty());
        assert!(!filter.contains(&"hello".to_string()));
        assert_eq!(filter.count_set_bits(), 0);
    }

    #[test]
    fn test_clear_idempotent() {
        let mut filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        filter.insert(&42);
        filter.clear();
        filter.clear();
        assert!(filter.is_empty());
    }

    // ── BloomFilter trait ─────────────────────────────────────────────────────

    #[test]
    fn trait_count_set_bits_zero_on_empty() {
        let f = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(BloomFilter::count_set_bits(&f), 0);
    }

    #[test]
    fn trait_count_set_bits_increases_after_insert() {
        let mut f = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let before = BloomFilter::count_set_bits(&f);
        BloomFilter::insert(&mut f, &42u64);
        assert!(BloomFilter::count_set_bits(&f) > before);
    }

    #[test]
    fn trait_fill_rate_matches_inherent() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { filter.insert(&i); }
        let inherent  = filter.fill_rate();
        let via_trait = BloomFilter::fill_rate(&filter);
        assert!((inherent - via_trait).abs() < f64::EPSILON);
    }

    #[test]
    fn trait_estimate_count_reasonable() {
        let mut f = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..1_000u64 { BloomFilter::insert(&mut f, &i); }
        let est = BloomFilter::estimate_count(&f);
        assert!(est >= 800 && est <= 1200, "estimate_count={est} expected ~1000");
    }

    #[test]
    fn trait_is_saturated_false_when_lightly_loaded() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100 { filter.insert(&i); }
        assert!(!BloomFilter::is_saturated(&filter));
    }

    // ── ConcurrentBloomFilter ─────────────────────────────────────────────────

    #[test]
    fn test_contains_concurrent_basic() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        filter.insert(&"hello".to_string());
        assert!( filter.contains_concurrent(&"hello".to_string()));
        assert!(!filter.contains_concurrent(&"goodbye".to_string()));
    }

    #[test]
    fn test_contains_concurrent_no_false_negatives() {
        let filter = Arc::new(StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap());
        for i in 0..500u64 { filter.insert_concurrent(&i); }
        for i in 0..500u64 {
            assert!(filter.contains_concurrent(&i), "false negative for {i}");
        }
    }

    #[test]
    fn test_contains_concurrent_multithreaded() {
        let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap());

        let writers: Vec<_> = (0..4u64).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                for i in 0..1000u64 { f.insert_concurrent(&(tid * 1000 + i)); }
            })
        }).collect();
        for h in writers { h.join().unwrap(); }

        let readers: Vec<_> = (0..4u64).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                for i in 0..1000u64 {
                    assert!(
                        f.contains_concurrent(&(tid * 1000 + i)),
                        "false negative at {}", tid * 1000 + i
                    );
                }
            })
        }).collect();
        for h in readers { h.join().unwrap(); }
    }

    #[test]
    fn test_contains_batch_concurrent_matches_individual() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let items: Vec<u64> = (0..100).collect();
        for &item in &items { filter.insert_concurrent(&item); }
        let batch = filter.contains_batch_concurrent(&items);
        for (i, &result) in batch.iter().enumerate() {
            assert_eq!(result, filter.contains_concurrent(&items[i]));
        }
    }

    #[test]
    fn test_concurrent_inserts_no_false_negatives() {
        let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap());
        let num_threads = 8usize;
        let per_thread  = 1000usize;

        let handles: Vec<_> = (0..num_threads).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                for i in 0..per_thread {
                    f.insert_concurrent(&((tid * per_thread + i) as u64));
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }

        for tid in 0..num_threads {
            for i in 0..per_thread {
                let item = (tid * per_thread + i) as u64;
                assert!(filter.contains(&item), "false negative for {item}");
            }
        }
    }

    #[test]
    fn test_concurrent_mixed_read_write() {
        let filter = Arc::new(StandardBloomFilter::<u64>::new(50_000, 0.01).unwrap());
        for i in 0..1000u64 { filter.insert(&i); }

        let handles: Vec<_> = (0..4u64).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                if tid % 2 == 0 {
                    for i in 1000..2000u64 { f.insert(&(i + tid * 10_000)); }
                } else {
                    // Pre-inserted items must always be present regardless of
                    // concurrent writes from even-numbered threads.
                    for i in 0..1000u64 { assert!(f.contains(&i)); }
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    }

    #[test]
    fn test_concurrent_batch_operations() {
        let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap());

        let handles: Vec<_> = (0..4u64).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                let items: Vec<u64> = (tid * 1000..(tid + 1) * 1000).collect();
                f.insert_batch(&items);
            })
        }).collect();
        for h in handles { h.join().unwrap(); }

        for i in 0..4000u64 { assert!(filter.contains(&i)); }
    }

    // ── MergeableBloomFilter::is_compatible ──────────────────────────────────

    #[test]
    fn test_is_compatible_same_params() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert!(f1.is_compatible(&f2));
    }

    #[test]
    fn test_is_compatible_reflexive() {
        let f = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert!(f.is_compatible(&f));
    }

    #[test]
    fn test_is_compatible_different_size() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(5000, 0.01).unwrap();
        assert!(!f1.is_compatible(&f2));
    }

    #[test]
    fn test_is_compatible_different_k() {
        let f1 = StandardBloomFilter::<u64>::with_params(1000, 5, StdHasher::new()).unwrap();
        let f2 = StandardBloomFilter::<u64>::with_params(1000, 7, StdHasher::new()).unwrap();
        assert!(!f1.is_compatible(&f2));
    }

    // ── MergeableBloomFilter::union ───────────────────────────────────────────

    #[test]
    fn test_union_contains_both_sources() {
        let f1 = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        f1.insert(&"alice".to_string());
        f2.insert(&"bob".to_string());

        let mut merged = f1.clone();
        MergeableBloomFilter::union(&mut merged, &f2).unwrap();

        assert!(merged.contains(&"alice".to_string()));
        assert!(merged.contains(&"bob".to_string()));
    }

    #[test]
    fn test_union_no_false_negatives_after_merge() {
        let f1 = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..500u64    { f1.insert(&i); }
        for i in 500..1000u64 { f2.insert(&i); }

        let mut base = f1.clone();
        MergeableBloomFilter::union(&mut base, &f2).unwrap();

        for i in 0..1000u64 {
            assert!(base.contains(&i), "false negative at {i} after union");
        }
    }

    #[test]
    fn test_union_fill_rate_monotone() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..50u64  { f1.insert(&i); }
        for i in 50..100u64 { f2.insert(&i); }

        let before = f1.fill_rate();
        let mut merged = f1.clone();
        MergeableBloomFilter::union(&mut merged, &f2).unwrap();

        assert!(merged.fill_rate() >= before);
    }

    #[test]
    fn test_union_incompatible_size() {
        let mut f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let     f2 = StandardBloomFilter::<u64>::new(5000, 0.01).unwrap();
        assert!(MergeableBloomFilter::union(&mut f1, &f2).is_err());
    }

    #[test]
    fn test_union_incompatible_k() {
        let mut f1 = StandardBloomFilter::<u64>::with_params(1000, 5, StdHasher::new()).unwrap();
        let     f2 = StandardBloomFilter::<u64>::with_params(1000, 7, StdHasher::new()).unwrap();
        assert!(MergeableBloomFilter::union(&mut f1, &f2).is_err());
    }

    #[test]
    fn test_union_with_empty_source() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { f1.insert(&i); }

        let mut merged = f1.clone();
        MergeableBloomFilter::union(&mut merged, &f2).unwrap();

        // Merging with an empty filter must not discard any existing members.
        for i in 0..100u64 {
            assert!(merged.contains(&i), "item {i} lost after union with empty filter");
        }
    }

    #[test]
    fn test_union_of_two_empty_filters() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let mut merged = f1.clone();
        MergeableBloomFilter::union(&mut merged, &f2).unwrap();
        assert!(merged.is_empty());
        assert_eq!(merged.count_set_bits(), 0);
    }

    #[test]
    fn test_union_is_symmetric_in_bit_content() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..50u64  { f1.insert(&i); }
        for i in 50..100u64 { f2.insert(&i); }

        let mut a = f1.clone();
        MergeableBloomFilter::union(&mut a, &f2).unwrap();

        let mut b = f2.clone();
        MergeableBloomFilter::union(&mut b, &f1).unwrap();

        // OR is commutative; both merged filters must have identical bit counts.
        assert_eq!(a.count_set_bits(), b.count_set_bits());
    }

    // ── Inherent constructive union ───────────────────────────────────────────

    #[test]
    fn test_inherent_union_leaves_sources_unchanged() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..50u64  { f1.insert(&i); }
        for i in 50..100u64 { f2.insert(&i); }

        let before_f1 = f1.count_set_bits();
        let before_f2 = f2.count_set_bits();

        let _combined = f1.union(&f2).unwrap();

        assert_eq!(f1.count_set_bits(), before_f1, "f1 must not be mutated by constructive union");
        assert_eq!(f2.count_set_bits(), before_f2, "f2 must not be mutated by constructive union");
    }

    #[test]
    fn test_inherent_union_contains_all_items() {
        let f1 = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        f1.insert(&"alice".to_string());
        f2.insert(&"bob".to_string());

        let combined = f1.union(&f2).unwrap();
        assert!(combined.contains(&"alice".to_string()));
        assert!(combined.contains(&"bob".to_string()));
    }

    #[test]
    fn test_inherent_union_propagates_nonempty_flag() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        f2.insert(&1u64);

        let combined = f1.union(&f2).unwrap();
        assert!(!combined.is_empty(), "union of empty+nonempty must not report is_empty");
    }

    #[test]
    fn test_inherent_union_both_empty_stays_empty() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let combined = f1.union(&f2).unwrap();
        assert!(combined.is_empty());
    }

    #[test]
    fn test_inherent_union_incompatible_size() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(5000, 0.01).unwrap();
        assert!(f1.union(&f2).is_err());
    }

    // ── MergeableBloomFilter::intersect ──────────────────────────────────────

    #[test]
    fn test_intersect_shared_items_remain_findable() {
        let f1 = StandardBloomFilter::<u64>::new(10_000, 0.001).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(10_000, 0.001).unwrap();

        // Items 0..100 are in both filters.
        for i in 0..200u64 { f1.insert(&i); }
        for i in 0..100u64 { f2.insert(&i); }

        let result = f1.intersect(&f2).unwrap();

        // Items only in f1 (100..200) must not reliably appear; items in both
        // (0..100) should still be present at the intersection FPR.
        for i in 0..100u64 {
            assert!(
                result.contains(&i),
                "item {i} present in both sources must survive intersection"
            );
        }
    }

    #[test]
    fn test_intersect_reduces_fill_rate() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { f1.insert(&i); }
        for i in 50..150u64 { f2.insert(&i); }

        let result = f1.intersect(&f2).unwrap();
        assert!(
            result.count_set_bits() <= f1.count_set_bits(),
            "intersection must not have more set bits than either source"
        );
        assert!(
            result.count_set_bits() <= f2.count_set_bits()
        );
    }

    #[test]
    fn test_intersect_leaves_sources_unchanged() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..50u64  { f1.insert(&i); }
        for i in 25..75u64 { f2.insert(&i); }

        let before_f1 = f1.count_set_bits();
        let before_f2 = f2.count_set_bits();

        let _result = f1.intersect(&f2).unwrap();

        assert_eq!(f1.count_set_bits(), before_f1, "f1 must not be mutated by constructive intersect");
        assert_eq!(f2.count_set_bits(), before_f2, "f2 must not be mutated by constructive intersect");
    }

    #[test]
    fn test_intersect_incompatible_size() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(5000, 0.01).unwrap();
        assert!(f1.intersect(&f2).is_err());
    }

    #[test]
    fn test_intersect_incompatible_k() {
        let f1 = StandardBloomFilter::<u64>::with_params(1000, 5, StdHasher::new()).unwrap();
        let f2 = StandardBloomFilter::<u64>::with_params(1000, 7, StdHasher::new()).unwrap();
        assert!(f1.intersect(&f2).is_err());
    }

    #[test]
    fn test_intersect_empty_with_nonempty_yields_subset_of_empty() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { f2.insert(&i); }

        let result = f1.intersect(&f2).unwrap();
        // AND of all-zero with anything is all-zero.
        assert_eq!(result.count_set_bits(), 0);
    }

    #[test]
    fn test_inplace_intersect_modifies_receiver() {
        let f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { f1.insert(&i); }
        for i in 50..150u64 { f2.insert(&i); }

        let before = f1.count_set_bits();
        let mut mutable_f1 = f1.clone();
        MergeableBloomFilter::intersect(&mut mutable_f1, &f2).unwrap();
        assert!(
            mutable_f1.count_set_bits() <= before,
            "in-place intersect must not increase the bit count"
        );
    }

    #[test]
    fn test_inplace_intersect_incompatible() {
        let mut f1 = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let     f2 = StandardBloomFilter::<u64>::new(5000, 0.01).unwrap();
        assert!(MergeableBloomFilter::intersect(&mut f1, &f2).is_err());
    }

    // ── union_many / intersect_many ───────────────────────────────────────────

    #[test]
    fn test_union_many_all_items_present() {
        let mut base = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let filters: Vec<_> = (0..4u64).map(|tid| {
            let f = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
            for i in tid * 250..(tid + 1) * 250 { f.insert(&i); }
            f
        }).collect();

        base.union_many(filters.iter()).unwrap();

        for i in 0..1000u64 {
            assert!(base.contains(&i), "item {i} missing after union_many");
        }
    }

    #[test]
    fn test_union_many_empty_iterator() {
        let mut base = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { base.insert(&i); }
        let before = base.count_set_bits();
        base.union_many(std::iter::empty()).unwrap();
        assert_eq!(base.count_set_bits(), before, "union_many over empty iterator must be a no-op");
    }

    #[test]
    fn test_intersect_many_reduces_to_common_bits() {
        let mut base = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let f2       = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..500u64 { base.insert(&i); }
        for i in 0..500u64 { f2.insert(&i); }

        let before = base.count_set_bits();
        base.intersect_many([&f2].into_iter()).unwrap();

        // Intersecting identical content must not reduce the bit count.
        assert_eq!(base.count_set_bits(), before);
    }

    // ── FilterHealth ─────────────────────────────────────────────────────────

    #[test]
    fn test_health_check_healthy_on_fresh_filter() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert!(filter.health_check().is_healthy());
    }

    #[test]
    fn test_health_check_healthy_when_lightly_loaded() {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..1000u64 { filter.insert(&i); }
        assert!(filter.health_check().is_healthy());
    }

    #[test]
    fn test_health_check_degraded_or_critical_when_overloaded() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        for i in 0..10_000u64 { filter.insert(&i); }
        let health = filter.health_check();
        assert!(
            health.is_degraded() || health.is_critical(),
            "overloaded filter must not report Healthy; got {:?}", health
        );
    }

    #[test]
    fn test_health_check_critical_on_full_filter() {
        let filter = StandardBloomFilter::<u64>::new(10, 0.01).unwrap();
        for i in 0..100_000u64 { filter.insert(&i); }
        assert!(filter.health_check().is_critical());
    }

    #[test]
    fn test_health_check_fill_rate_consistency() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..200u64 { filter.insert(&i); }
        let health = filter.health_check();
        let direct = filter.fill_rate();
        assert!(
            (health.fill_rate() - direct).abs() < 1e-10,
            "health_check fill_rate must match direct fill_rate()"
        );
    }

    #[test]
    fn test_health_check_display() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let display = format!("{}", filter.health_check());
        assert!(
            display.starts_with("[OK]"),
            "empty filter display must start with '[OK]'; got: {display}"
        );
    }

    #[test]
    fn test_health_check_healthy_implies_not_saturated() {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..500u64 { filter.insert(&i); }
        let health = filter.health_check();
        if health.is_healthy() {
            assert!(
                !BloomFilter::is_saturated(&filter),
                "Healthy state must be consistent with !is_saturated()"
            );
        }
    }

    #[test]
    fn test_filter_health_accessors() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { filter.insert(&i); }
        let health = filter.health_check();

        let fill    = health.fill_rate();
        let fpr     = health.current_fpr();
        let items   = health.estimated_items();

        assert!((0.0..=1.0).contains(&fill));
        assert!((0.0..=1.0).contains(&fpr));
        assert!(items > 0);
    }

    // ── Clone ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clone_preserves_bit_array() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { filter.insert(&i); }
        let clone = filter.clone();
        assert_eq!(filter.count_set_bits(), clone.count_set_bits());
        for i in 0..100u64 { assert!(clone.contains(&i)); }
    }

    #[test]
    fn test_clone_is_independent() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        filter.insert(&1u64);
        let mut clone = filter.clone();

        // Inserting into the clone must not affect the original.
        clone.insert(&999u64);
        assert!(!filter.contains(&999u64));

        // Clearing the clone must not affect the original.
        clone.clear();
        assert!(filter.contains(&1u64));
    }

    #[test]
    fn test_clone_has_inserts_flag() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        let empty_clone = filter.clone();
        assert!(empty_clone.is_empty());

        filter.insert(&42u64);
        let nonempty_clone = filter.clone();
        assert!(!nonempty_clone.is_empty());
    }

    // ── from_parts ────────────────────────────────────────────────────────────

    #[test]
    fn test_from_parts_round_trip() {
        let original = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100u64 { original.insert(&i); }

        let bits = original.raw_bits();
        let m    = original.size();
        let k    = original.hash_count();

        let bitvec = BitVec::from_raw(bits, m).unwrap();
        let restored = StandardBloomFilter::<u64, StdHasher>::from_parts(bitvec, k).unwrap();

        // Bit counts must be identical after round-trip.
        assert_eq!(original.count_set_bits(), restored.count_set_bits());
    }

    #[test]
    fn test_from_parts_invalid_k() {
        let bv = BitVec::new(1000).unwrap();
        assert!(StandardBloomFilter::<u64, StdHasher>::from_parts(bv.clone(),  0).is_err());
        assert!(StandardBloomFilter::<u64, StdHasher>::from_parts(bv,         33).is_err());
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hasher_name() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        let name   = filter.hasher_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_hash_strategy() {
        use crate::hash::IndexingStrategy;
        let filter   = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        let strategy = filter.hash_strategy();
        assert_eq!(strategy, IndexingStrategy::EnhancedDouble);
    }

    #[test]
    fn test_target_fpr_accessor() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.05).unwrap();
        assert!((filter.target_fpr() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expected_items_accessor() {
        let filter = StandardBloomFilter::<u64>::new(12_345, 0.01).unwrap();
        assert_eq!(filter.expected_items(), 12_345);
    }

    // ── Fast-reduce ───────────────────────────────────────────────────────────

    #[test]
    fn fast_reduce_within_range() {
        for m in [1u64, 7, 64, 1000, 1 << 20] {
            for val in [0u64, 1, u64::MAX / 2, u64::MAX] {
                let result = fast_reduce(val, m);
                assert!(
                    (result as u64) < m,
                    "fast_reduce({val}, {m}) = {result}, expected < {m}"
                );
            }
        }
    }

    #[test]
    fn fast_reduce_zero_val_is_zero() {
        // ⌊0 × m / 2⁶⁴⌋ = 0 for all m.
        for m in [1u64, 100, u32::MAX as u64] {
            assert_eq!(fast_reduce(0, m), 0);
        }
    }

    // ── Hash-position recurrence matches reference formula ────────────────────

    #[test]
    fn set_indices_matches_enhanced_double_hashing_reference() {
        // Verify that the incremental recurrence inside set_indices/all_indices_set
        // produces the same positions as the reference formula:
        //   g_i = h1 + i·h2 + i(i+1)/2  (mod m), mapped via fast_reduce.
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let m      = filter.size() as u64;
        let k      = filter.hash_count();

        let h1: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let h2: u64 = 0x0123_4567_89AB_CDEF;

        // Reference positions from the closed-form formula.
        let reference: Vec<usize> = (0..k).map(|i| {
            let i = i as u64;
            let pos = h1
                .wrapping_add(i.wrapping_mul(h2))
                .wrapping_add(i.wrapping_mul(i.wrapping_add(1)) / 2);
            fast_reduce(pos, m)
        }).collect();

        // Positions produced by the incremental recurrence.
        let mut incremental = Vec::with_capacity(k);
        let mut val  = h1;
        let mut step = h2;
        incremental.push(fast_reduce(val, m));
        for _ in 1..k {
            step = step.wrapping_add(1);
            val  = val.wrapping_add(step);
            incremental.push(fast_reduce(val, m));
        }

        assert_eq!(
            reference, incremental,
            "incremental recurrence must produce identical positions to the closed-form formula"
        );
    }

    // ── False positive rate ───────────────────────────────────────────────────

    #[test]
    fn test_false_positive_rate_at_design_capacity() {
        // Empirically measure FPR at exactly n insertions and verify it is within
        // a reasonable multiple of the design target. Probabilistic: may rarely
        // flake for pathological hash collisions; tolerated in CI.
        let n       = 10_000usize;
        let target  = 0.01f64;
        let filter  = StandardBloomFilter::<u64>::new(n, target).unwrap();

        for i in 0..n as u64 { filter.insert(&i); }

        let probe_count  = 100_000usize;
        let offset       = 1_000_000u64;
        let false_positives = (0..probe_count as u64)
            .filter(|&i| filter.contains(&(offset + i)))
            .count();

        let measured_fpr = false_positives as f64 / probe_count as f64;
        assert!(
            measured_fpr <= target * 3.0,
            "measured FPR {measured_fpr:.4} exceeds 3× the target {target:.4} at n={n} insertions"
        );
    }

    #[test]
    fn test_estimate_fpr_consistent_with_measured() {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..5_000u64 { filter.insert(&i); }

        let estimated = filter.estimate_fpr();
        assert!(
            estimated < 0.05,
            "estimated FPR {estimated:.4} is implausibly high at 50% fill"
        );
    }

    // ── Serde round-trip ──────────────────────────────────────────────────────

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[test]
        fn serde_round_trip_preserves_bit_array() {
            let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
            filter.insert(&"hello".to_string());
            filter.insert(&"world".to_string());

            let json     = serde_json::to_string(&filter).unwrap();
            let restored: StandardBloomFilter<String> = serde_json::from_str(&json).unwrap();

            assert_eq!(filter.count_set_bits(), restored.count_set_bits());
            assert_eq!(filter.size(),           restored.size());
            assert_eq!(filter.hash_count(),     restored.hash_count());
        }

        #[test]
        fn serde_round_trip_no_false_negatives() {
            let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
            for i in 0..100u64 { filter.insert(&i); }

            let json     = serde_json::to_string(&filter).unwrap();
            let restored: StandardBloomFilter<u64> = serde_json::from_str(&json).unwrap();

            for i in 0..100u64 {
                assert!(
                    restored.contains(&i),
                    "false negative for item {i} after serde round-trip"
                );
            }
        }

        #[test]
        fn serde_round_trip_empty_filter() {
            let filter   = StandardBloomFilter::<u64>::new(500, 0.01).unwrap();
            let json     = serde_json::to_string(&filter).unwrap();
            let restored: StandardBloomFilter<u64> = serde_json::from_str(&json).unwrap();

            assert_eq!(restored.count_set_bits(), 0);
        }

        #[test]
        fn filter_health_serde_round_trip() {
            let health = FilterHealth::Healthy {
                fill_rate:       0.25,
                current_fpr:     0.005,
                estimated_items: 2_500,
            };
            let json     = serde_json::to_string(&health).unwrap();
            let restored: FilterHealth = serde_json::from_str(&json).unwrap();
            assert_eq!(health, restored);
        }
    }

    // ── wyhash feature ────────────────────────────────────────────────────────

    #[cfg(feature = "wyhash")]
    mod wyhash_tests {
        use super::*;

        #[test]
        fn wyhash_filter_no_false_negatives() {
            let filter = StandardBloomFilter::<u64, WyHasher>::with_hasher(
                1000, 0.01, WyHasher::new(),
            ).unwrap();
            for i in 0..100u64 { filter.insert(&i); }
            for i in 0..100u64 {
                assert!(filter.contains(&i), "false negative at {i} with WyHasher");
            }
        }

        #[test]
        fn wyhash_union_no_false_negatives() {
            let f1 = StandardBloomFilter::<u64, WyHasher>::with_hasher(
                1000, 0.01, WyHasher::new(),
            ).unwrap();
            let f2 = StandardBloomFilter::<u64, WyHasher>::with_hasher(
                1000, 0.01, WyHasher::new(),
            ).unwrap();
            for i in 0..50u64  { f1.insert(&i); }
            for i in 50..100u64 { f2.insert(&i); }

            let combined = f1.union(&f2).unwrap();
            for i in 0..100u64 {
                assert!(combined.contains(&i), "false negative at {i} after WyHasher union");
            }
        }
    }
}
