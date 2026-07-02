//! Concurrent, automatically-growing Bloom filter for multi-threaded workloads.
//!
//! This module provides [`AtomicScalableBloomFilter`], a thread-safe variant of
//! [`ScalableBloomFilter`](crate::filters::ScalableBloomFilter) that uses sharding,
//! per-shard lock-free bit operations, and a three-phase growth protocol to
//! support concurrent insert and query with minimal contention.
//!
//! See the [`scalable`](crate::filters::scalable) module for the concurrency model
//! details, FPR analysis, and configuration guidance.
//!
//! For a full architecture diagram, the concurrency model table, and the memory
//! ordering rationale, see the [`AtomicScalableBloomFilter`] struct-level
//! documentation (the `# Architecture`, `# Concurrency model`, and
//! `# Memory ordering` sections).
//!
//! # References
//!
//! - Almeida, P. S., Baquero, C., Preguiça, N., & Hutchison, D. (2007).
//!   "Scalable Bloom Filters." *Information Processing Letters*, 101(6), 255–261.
//! - Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007).
//!   "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm."
//!   *DMTCS Proceedings*, AH, 127–146.

use crate::core::filter::{BloomFilter, SharedBloomFilter};
use crate::error::{BloomCraftError, Result};
use crate::filters::scalable::{
    GrowthStrategy, InternalHasher, CAPACITY_WARNING_THRESHOLD, DEFAULT_FILL_THRESHOLD,
    MAX_FILTERS, MIN_FPR,
};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::fmt;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

// --- CACHE-LINE PADDING CONSTANTS ---

// Target cache line size (64 bytes on x86-64, ARM64, RISC-V)
const CACHE_LINE_SIZE: usize = 64;

// Computed once for struct layout; explanation in `CacheAligned`.
const CACHE_ALIGNED_USIZE_SIZE: usize = std::mem::size_of::<CacheAligned<AtomicUsize>>();

// See `CacheAligned` doc for alignment rationale.
const CACHE_ALIGNED_BOOL_SIZE: usize = std::mem::size_of::<CacheAligned<AtomicBool>>();

// Padding constants derived from the sizes above.
const PAD_AFTER_USIZE: usize = CACHE_LINE_SIZE - CACHE_ALIGNED_USIZE_SIZE;

// Padding after CacheAligned<AtomicBool> (see `CacheAligned` doc).
const PAD_AFTER_BOOL: usize = CACHE_LINE_SIZE - CACHE_ALIGNED_BOOL_SIZE;

// --- CacheAligned ---

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

// --- ShardedFilter ---

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
        let per_shard_capacity = capacity.div_ceil(shard_count);

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
        std::mem::size_of::<Self>() + self.shards.iter().map(|s| s.memory_usage()).sum::<usize>()
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

    /// Total bit count across all shards.
    fn bit_count(&self) -> usize {
        self.shards.iter().map(|s| s.bit_count()).sum()
    }

    /// Total bits set across all shards.
    fn count_set_bits(&self) -> usize {
        self.shards.iter().map(|s| s.count_set_bits()).sum()
    }

    /// Number of hash functions per shard (identical across all shards).
    fn hash_count(&self) -> usize {
        self.shards.first().map(|s| s.hash_count()).unwrap_or(0)
    }

    /// Estimated FPR based on the actual fill state of each shard.
    ///
    /// Per-shard FPRs are averaged rather than combined via the complement
    /// rule. This is correct because `InternalHasher` distributes items
    /// uniformly across shards — each shard carries equal expected load and
    /// its FPR is an equally-weighted sample of the whole.
    ///
    /// **Accuracy note:** The averaged FPR assumes the hash function
    /// distributes items uniformly across shards. Adversarially constructed
    /// inputs or pathological [`Hash`](std::hash::Hash) implementations can
    /// skew per-shard load, causing the averaged FPR to underestimate the
    /// true compound rate. This is a monitoring-grade estimate, not a
    /// cryptographic bound.
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

// --- AtomicScalableBloomFilter (public handle) ---

/// Lock-minimised concurrent scalable Bloom filter.
///
/// The public handle is a thin [`Arc`] wrapper around `AtomicScalableInner`.
/// Cloning is O(1) — it increments the reference count, not the bit arrays.
/// All clones share the same underlying filter state.
///
/// # Architecture
///
/// [`AtomicScalableBloomFilter`] wraps a growable sequence of `ShardedFilter`
/// instances behind an `Arc<AtomicScalableInner>`. Each `ShardedFilter` is
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
pub struct AtomicScalableBloomFilter<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    inner: Arc<AtomicScalableInner<T, H>>,
}

// --- AtomicScalableInner (shared state) ---

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

// --- ConcurrentConfig ---

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
    /// # Precondition
    ///
    /// Must only be called while holding the filters write-lock.
    /// The lock provides the happens-before edge that makes this `Relaxed`
    /// store visible to any thread that subsequently acquires the read-lock.
    #[inline]
    fn store_error_ratio(&self, v: f64) {
        self.error_ratio.store(v.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn fill_threshold(&self) -> f64 {
        self.fill_threshold
    }
}

// --- Shard count heuristic ---

/// Return the optimal shard count for the current system.
///
/// Uses the number of logical CPUs, capped at 16. The cap is a rule of thumb:
/// beyond 16 shards the per-shard capacity drops below the point where the
/// hash function can distribute items uniformly, and the coordination
/// overhead of routing outweighs the parallelism benefit.
///
/// Falls back to 8 if `available_parallelism` is unavailable (WASM, some
/// embedded targets).
fn optimal_shard_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(16))
        .unwrap_or(8)
}

// --- Constructors (StdHasher specialisation) ---
impl<T> AtomicScalableBloomFilter<T, StdHasher>
where
    T: Hash + Send + Sync,
{
    /// Create a concurrent scalable filter with default settings.
    ///
    /// Uses `StdHasher`, `Geometric(2.0)` growth, `error_ratio = 0.5`,
    /// and `fill_threshold = 0.5`. The shard count is chosen automatically
    /// via `optimal_shard_count`.
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
    pub fn new(initial_capacity: usize, target_fpr: f64) -> Result<Self> {
        Self::with_hasher(initial_capacity, target_fpr, StdHasher::new())
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
                let term = initial_capacity
                    .saturating_mul(2usize.checked_pow(count as u32).unwrap_or(usize::MAX));
                capacity_sum = capacity_sum.saturating_add(term);
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

// --- Constructors (generic) ---

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
    /// `initial_capacity` is the expected item count for the first sub-filter
    /// (must be > 0). `target_fpr` is the false positive rate for the first
    /// sub-filter (must be in (0.0, 1.0)). `error_ratio` controls per-generation
    /// FPR tightening (must be in (0.0, 1.0)); a value of `0.5` halves the FPR
    /// at each growth event, `0.9` tightens slowly. `growth_strategy` controls
    /// how sub-filter capacity scales — see [`GrowthStrategy`] for variants.
    /// `hasher` is the hash function instance, cloned once per sub-filter.
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidItemCount`] if `initial_capacity == 0`.
    /// - [`BloomCraftError::FalsePositiveRateOutOfBounds`] if `target_fpr`
    ///   is not in (0.0, 1.0).
    /// - [`BloomCraftError::InvalidParameters`] if `error_ratio` is not in
    ///   (0.0, 1.0).
    /// - Any allocation error from `ShardedFilter::new`.
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
    pub fn with_strategy_and_hasher(
        initial_capacity: usize,
        target_fpr: f64,
        error_ratio: f64,
        growth_strategy: GrowthStrategy,
        hasher: H,
    ) -> Result<Self> {
        growth_strategy.validate()?;

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

    // --- Core operations ---

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
    /// **Note:** Under concurrent insert, the overflow check is best-effort:
    /// another thread may increment `total_items` between the check and the
    /// actual `fetch_add`, allowing the counter to wrap. Single-threaded
    /// callers are fully protected.
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
        current.checked_add(items.len()).ok_or_else(|| {
            BloomCraftError::invalid_parameters(format!(
                "Batch insert of {} items would overflow counter (current: {})",
                items.len(),
                current
            ))
        })?;

        // --- PHASE 1: GROUP BY SHARD ---
        //
        // Routing must match ShardedFilter::insert/contains exactly.
        // shard_count is immutable; no lock required.
        let shard_count = self.inner.config.shard_count;

        let mut shard_buckets: Vec<Vec<&T>> = vec![Vec::new(); shard_count];
        for item in items {
            let shard_idx = InternalHasher::hash_one(item) as usize % shard_count;
            shard_buckets[shard_idx].push(item);
        }

        // --- PHASE 2: INSERT PER SHARD BUCKET ---
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
            let total = self
                .inner
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
    /// Propagates any allocation error from `ShardedFilter::new`.
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

        if let GrowthStrategy::Adaptive { initial_ratio, .. } = self.inner.config.growth_strategy {
            self.inner.config.store_error_ratio(initial_ratio);
        }

        let initial_check_interval =
            (self.inner.config.initial_capacity as f64 * self.inner.config.fill_threshold) as usize;
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

    // --- Growth ---

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

        if self
            .inner
            .growth_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let result = self.perform_growth();

        // Release ordering: the Relaxed stores in perform_growth are
        // visible to any thread that subsequently observes this store
        // as `false` and acquires the read lock.
        self.inner
            .growth_in_progress
            .store(false, Ordering::Release);

        if let Err(_e) = result {
            #[cfg(debug_assertions)]
            eprintln!("[AtomicScalableBloomFilter] Growth failed: {}", _e);
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
        // --- PHASE 1: Read lock ---
        let (current_idx, have_preallocated) = {
            let filters = self.inner.filters.read().unwrap();
            let idx = self.inner.current_filter.load(Ordering::Acquire);

            let next_index = idx + 1;
            let have_preallocated = next_index < filters.len();

            if !have_preallocated && filters.len() >= MAX_FILTERS {
                self.inner
                    .check_interval
                    .store(usize::MAX, Ordering::Release);
                return Err(BloomCraftError::capacity_exceeded(
                    MAX_FILTERS,
                    filters.len(),
                ));
            }

            (idx, have_preallocated)
        };

        if have_preallocated {
            let filters = self.inner.filters.read().unwrap();

            if self.inner.current_filter.load(Ordering::Relaxed) != current_idx {
                return Ok(());
            }

            let next_index = current_idx + 1;
            let usable = (filters[next_index].expected_items() as f64
                * self.inner.config.fill_threshold()) as usize;
            let cur_total = self.inner.total_items.load(Ordering::Relaxed);
            self.inner
                .check_interval
                .store(cur_total.saturating_add(usable.max(1)), Ordering::Release);
            self.inner
                .current_filter
                .store(next_index, Ordering::Release);

            #[cfg(debug_assertions)]
            eprintln!(
                "[AtomicScalableBloomFilter] Advanced to pre-allocated filter {} (no allocation)",
                next_index
            );

            return Ok(());
        }

        // --- PHASE 2: Allocate BEFORE write lock ---
        let filter_index = self.inner.filters.read().unwrap().len();

        let capacity = self.calculate_next_capacity(filter_index)?;
        let fpr = self.calculate_next_fpr(filter_index);

        let new_filter = Arc::new(ShardedFilter::new(
            capacity,
            fpr,
            self.inner.config.shard_count,
            self.inner.config.hasher.clone(),
        )?);

        // --- PHASE 3: Write lock ---
        let next_threshold;
        {
            let mut filters = self.inner.filters.write().unwrap();

            if self.inner.current_filter.load(Ordering::Relaxed) != current_idx {
                return Ok(());
            }

            filters.push(new_filter);
            self.inner
                .current_filter
                .store(filter_index, Ordering::Release);

            // --- Adaptive FPR tuning ---
            //
            // Runs under the write lock so that `store_error_ratio` is
            // sequenced with filter creation. Any thread that subsequently
            // reads `current_filter` (Acquire) also observes the updated
            // error_ratio via the filters lock's happens-before edge.
            if let GrowthStrategy::Adaptive {
                min_ratio,
                max_ratio,
                ..
            } = self.inner.config.growth_strategy
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

            // --- THRESHOLD UPDATE ---
            let new_filter_usable = filters
                .last()
                .map(|f| (f.expected_items() as f64 * self.inner.config.fill_threshold()) as usize)
                .unwrap_or(self.inner.config.initial_capacity);
            let cur_total = self.inner.total_items.load(Ordering::Relaxed);
            next_threshold = cur_total.saturating_add(new_filter_usable.max(1));
            self.inner
                .check_interval
                .store(next_threshold, Ordering::Release);
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
                    let max_safe_exp = (MAX_CAPACITY.ln()
                        - (self.inner.config.initial_capacity as f64).ln())
                        / scale_log;

                    if filter_index as f64 >= max_safe_exp {
                        return Err(BloomCraftError::invalid_parameters(format!(
                            "Filter index {} would cause capacity overflow (max safe: {:.1})",
                            filter_index, max_safe_exp
                        )));
                    }

                    let growth_factor = scale.powi(filter_index as i32);
                    let computed = self.inner.config.initial_capacity as f64 * growth_factor;

                    if computed > MAX_CAPACITY || !computed.is_finite() {
                        return Err(BloomCraftError::invalid_parameters(format!(
                            "Computed capacity {:.2e} exceeds usize::MAX",
                            computed
                        )));
                    }

                    let new_capacity = computed as usize;

                    if new_capacity < self.inner.config.initial_capacity {
                        return Err(BloomCraftError::invalid_parameters(
                            "Capacity calculation resulted in overflow",
                        ));
                    }

                    new_capacity
                }
            }

            GrowthStrategy::Bounded {
                scale,
                max_filter_size,
            } => {
                if filter_index == 0 {
                    self.inner.config.initial_capacity
                } else {
                    let max_safe_exp = (MAX_CAPACITY.ln()
                        - (self.inner.config.initial_capacity as f64).ln())
                        / scale.ln();

                    if filter_index as f64 >= max_safe_exp {
                        return Err(BloomCraftError::invalid_parameters(format!(
                            "Bounded filter index {} would cause capacity overflow (max safe: {:.1})",
                            filter_index, max_safe_exp
                        )));
                    }

                    let computed =
                        self.inner.config.initial_capacity as f64 * scale.powi(filter_index as i32);

                    if computed > MAX_CAPACITY || !computed.is_finite() {
                        return Err(BloomCraftError::invalid_parameters(
                            "Bounded capacity calculation overflow",
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
                    let computed =
                        self.inner.config.initial_capacity as f64 * SCALE.powi(filter_index as i32);
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

    /// FPR that the `filter_index`-th sub-filter should target.
    ///
    /// The tightening formula is:
    ///
    /// ```text
    /// FPR_i = target_fpr × error_ratio^i
    /// ```
    ///
    /// where `i = min(filter_index, 1000)`. The exponent is clamped at 1000 to
    /// guard against extreme inputs that might cause the `powi` loop to
    /// overflow the exponent range or dominate the growth hot path. The clamp
    /// has no observable effect on the result — by index 1000 the FPR has
    /// long since been
    /// clamped to [`MIN_FPR`] for any realistic `error_ratio`.
    ///
    /// The result is clamped to `[MIN_FPR, 1.0]` to prevent degenerate values
    /// from reaching `ShardedFilter::new`. The upper clamp is a safety net:
    /// `error_ratio^i` can approach 1.0 from below but should never exceed it.
    ///
    /// Uses `Relaxed` ordering to read `error_ratio` — the ratio is immutable
    /// after construction, so all threads see the same value regardless of
    /// ordering. The `Relaxed` load compiles to a plain load on all
    /// architectures the crate targets, avoiding an unnecessary acquire
    /// barrier on the growth hot path.
    fn calculate_next_fpr(&self, filter_index: usize) -> f64 {
        let ratio = self.inner.config.error_ratio();

        const MAX_SAFE_EXP: i32 = 1000;
        let safe_index = (filter_index as i32).min(MAX_SAFE_EXP);

        let raw_fpr = self.inner.config.target_fpr * ratio.powi(safe_index);

        raw_fpr.clamp(MIN_FPR, 1.0)
    }

    // --- Accessors ---

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
        filters.iter().map(|f| f.expected_items()).sum()
    }

    /// Whether the filter has reached [`MAX_FILTERS`] sub-filters.
    ///
    /// The check is performed on the current filter count without locking — a
    /// concurrent growth event can make the returned value stale by the time
    /// the caller acts on it. Suitable for monitoring dashboards and
    /// capacity-alert thresholds, **not** for controlling insert logic.
    #[must_use]
    pub fn is_at_max_capacity(&self) -> bool {
        self.filter_count() >= MAX_FILTERS
    }

    /// Whether the filter is within `CAPACITY_WARNING_THRESHOLD` filters of
    /// [`MAX_FILTERS`].
    ///
    /// Same staleness caveat as [`is_at_max_capacity`](Self::is_at_max_capacity):
    /// the returned value is a snapshot and should be used for monitoring only,
    /// not for controlling insert logic.
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

        let product: f64 = filters.iter().map(|f| 1.0 - f.estimate_fpr()).product();

        1.0 - product
    }

    /// Total memory footprint in bytes.
    ///
    /// Includes the struct overhead and all shard bit arrays. Does not
    /// account for `Arc`/`RwLock` metadata or OS-level page table overhead.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let filters = self.inner.filters.read().unwrap();
        filters.iter().map(|f| f.memory_usage()).sum::<usize>() + std::mem::size_of::<Self>()
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

    /// Returns the number of shards per sub-filter.
    ///
    /// Determined at construction via `optimal_shard_count` and immutable
    /// thereafter. Changing the shard count mid-life would invalidate the
    /// routing invariant between `insert` and `contains`.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.inner.config.shard_count
    }

    /// Returns the total number of set bits across all sub-filters and shards.
    #[must_use]
    pub fn count_set_bits(&self) -> usize {
        let filters = self.inner.filters.read().unwrap();
        filters.iter().map(|f| f.count_set_bits()).sum()
    }

    /// Returns the total bit capacity across all sub-filters and shards.
    #[must_use]
    pub fn bit_count(&self) -> usize {
        let filters = self.inner.filters.read().unwrap();
        filters.iter().map(|f| f.bit_count()).sum()
    }

    /// Returns the number of hash functions per shard (uniform across all sub-filters).
    #[must_use]
    pub fn hash_count(&self) -> usize {
        let filters = self.inner.filters.read().unwrap();
        filters.first().map(|f| f.hash_count()).unwrap_or(0)
    }

    /// Returns the total expected item capacity across all sub-filters.
    #[must_use]
    pub fn expected_items(&self) -> usize {
        let filters = self.inner.filters.read().unwrap();
        filters.iter().map(|f| f.expected_items()).sum()
    }

    /// Estimated item count via the inverse Bloom filter formula.
    ///
    /// Uses `hash_count()` from the first sub-filter (`k₀`) applied across
    /// the aggregate bits of all sub-filters. Because later sub-filters have
    /// progressively higher `k` values (due to FPR tightening), the
    /// single-`k` formula becomes increasingly inaccurate as the filter
    /// grows — it may over-count items that land predominantly in later
    /// tiers. For monitoring-grade estimates this is typically acceptable;
    /// production alerting should use generous margins (~2×) around the
    /// reported value.
    #[must_use]
    pub fn estimate_count(&self) -> usize {
        let total_bits = self.bit_count();
        let set_bits = self.count_set_bits();
        if set_bits == 0 || total_bits == 0 {
            return 0;
        }
        let m = total_bits as f64;
        let k = self.hash_count() as f64;
        if k == 0.0 {
            return 0;
        }
        let fill_ratio = set_bits as f64 / m;
        if fill_ratio >= 1.0 {
            return total_bits;
        }
        (-(m / k) * (1.0 - fill_ratio).ln()).round() as usize
    }

    /// Alias for [`estimate_fpr`](Self::estimate_fpr).
    #[must_use]
    pub fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }
}

// --- SharedBloomFilter ---

impl<T, H> SharedBloomFilter<T> for AtomicScalableBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn insert(&self, item: &T) {
        AtomicScalableBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        AtomicScalableBloomFilter::contains(self, item)
    }

    fn clear(&self) {
        AtomicScalableBloomFilter::clear(self);
    }

    /// Returns the number of set bits across all sub-filters.
    ///
    /// **Note:** This is *not* the same as the inherent [`len`](crate::filters::AtomicScalableBloomFilter::len),
    /// which returns `total_items` (the number of insert calls). The two
    /// metrics diverge because a single insert sets multiple bits. Use this
    /// trait method when you need the bit-level fill metric for cross-filter
    /// comparison, and the inherent `len` for item-count semantics.
    fn len(&self) -> usize {
        self.count_set_bits()
    }

    /// Returns `true` when the count of set bits is zero.
    ///
    /// Unlike the inherent [`is_empty`](crate::filters::AtomicScalableBloomFilter::is_empty),
    /// which checks `total_items == 0`, this checks bit-level emptiness.
    fn is_empty(&self) -> bool {
        self.count_set_bits() == 0
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn estimate_count(&self) -> usize {
        AtomicScalableBloomFilter::estimate_count(self)
    }

    fn expected_items(&self) -> usize {
        self.expected_items()
    }

    fn bit_count(&self) -> usize {
        AtomicScalableBloomFilter::bit_count(self)
    }

    fn hash_count(&self) -> usize {
        self.hash_count()
    }

    fn count_set_bits(&self) -> usize {
        AtomicScalableBloomFilter::count_set_bits(self)
    }
}

// --- Clone ---

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    // --- basic operations ---------------------------------------------------------

    #[test]
    fn test_new() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
        assert_eq!(f.filter_count(), 1);
    }

    #[test]
    fn test_insert_and_contains() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        f.insert(&42);
        assert!(f.contains(&42));
        // A fresh filter won't falsely report 42 for an unrelated large key,
        // but this is not a statistical FPR assertion.
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_no_false_negatives() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        for i in 0..500u64 {
            f.insert(&i);
        }
        for i in 0..500u64 {
            assert!(f.contains(&i), "false negative for {}", i);
        }
    }

    #[test]
    fn test_len_is_empty() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
        f.insert(&1);
        assert!(!f.is_empty());
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn test_clear() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        for i in 0..100u64 {
            f.insert(&i);
        }
        assert_eq!(f.len(), 100);
        f.clear();
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());
        // clear preserves single sub-filter.
        assert_eq!(f.filter_count(), 1);
    }

    // --- concurrency --------------------------------------------------------------

    #[test]
    fn test_concurrent_insert() {
        let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
        let threads: Vec<_> = (0..4)
            .map(|tid| {
                let f = Arc::clone(&f);
                thread::spawn(move || {
                    for i in 0..250 {
                        f.insert(&(tid * 250 + i));
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().unwrap();
        }
        assert_eq!(f.len(), 1000);
        for i in 0..1000u64 {
            assert!(f.contains(&i), "false negative for {}", i);
        }
    }

    #[test]
    fn test_concurrent_contains_during_insert() {
        let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
        // Prime with some data.
        for i in 0..200u64 {
            f.insert(&i);
        }
        let ready = Arc::new(Barrier::new(5));
        let stop = Arc::new(AtomicBool::new(false));

        // 4 reader threads hammering contains().
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let f = Arc::clone(&f);
                let ready = Arc::clone(&ready);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    ready.wait();
                    let mut i = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let _ = f.contains(&(i % 200));
                        i = i.wrapping_add(1);
                    }
                })
            })
            .collect();

        // 1 writer inserting.
        let writer = {
            let f = Arc::clone(&f);
            let ready = Arc::clone(&ready);
            thread::spawn(move || {
                ready.wait();
                for i in 200..1000u64 {
                    f.insert(&i);
                }
            })
        };

        // Let the writer finish, then signal readers to stop.
        writer.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for h in readers {
            h.join().unwrap();
        }

        // Final assertions: no false negatives, len is correct.
        assert_eq!(f.len(), 1000);
        for i in 0..1000u64 {
            assert!(f.contains(&i), "false negative for {}", i);
        }
    }

    #[test]
    fn test_clear_under_concurrent_readers() {
        let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
        for i in 0..500u64 {
            f.insert(&i);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(3)); // 2 readers + 1 clearer

        let readers: Vec<_> = (0..2)
            .map(|_| {
                let f = Arc::clone(&f);
                let stop = Arc::clone(&stop);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut i = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let _ = f.contains(&(i % 500));
                        i = i.wrapping_add(1);
                    }
                })
            })
            .collect();

        barrier.wait();
        f.clear();
        stop.store(true, Ordering::Relaxed);

        for h in readers {
            h.join().unwrap();
        }

        // After clear the filter is reusable.
        assert_eq!(f.len(), 0);
        f.insert(&99);
        assert!(f.contains(&99));
    }

    // --- growth -------------------------------------------------------------------

    #[test]
    fn test_automatic_growth() {
        let f = AtomicScalableBloomFilter::<u64>::new(100, 0.01).unwrap();
        // default fill_threshold = 0.9, but check_interval is
        // (100 * 0.9) = 90. Insert 120 items → at least 1 growth event.
        for i in 0..120u64 {
            f.insert(&i);
        }
        assert!(
            f.filter_count() >= 2,
            "expected growth, got {} filters",
            f.filter_count()
        );
        for i in 0..120u64 {
            assert!(f.contains(&i), "false negative after growth for {}", i);
        }
    }

    #[test]
    fn test_growth_preserves_items() {
        let f = AtomicScalableBloomFilter::<u64>::new(100, 0.01).unwrap();
        // Force several growth events.
        for i in 0..10_000u64 {
            f.insert(&i);
        }
        for i in 0..10_000u64 {
            assert!(f.contains(&i), "false negative after growth for {}", i);
        }
    }

    #[test]
    fn test_capacity_exhausted_error() {
        // Use a tiny initial capacity so MAX_FILTERS is reachable quickly.
        // Constant growth means each new filter has the same capacity (1 item),
        // but we need to fill above the threshold. We just verify the Error
        // behavior exists and returns Err rather than panicking.
        let strategies = [GrowthStrategy::Geometric(1.001), GrowthStrategy::Constant];
        for &strategy in &strategies {
            let f = AtomicScalableBloomFilter::<u64>::with_strategy(1, 0.5, 0.9, strategy).unwrap();
            // Insert enough items to hit MAX_FILTERS (64).
            for i in 0..200u64 {
                f.insert(&i);
            }
            // The filter should still be usable — items are present.
            assert!(f.contains(&0));
            assert!(f.contains(&199));
        }
    }

    // --- constructors -------------------------------------------------------------

    #[test]
    fn test_new_invalid_parameters() {
        assert!(AtomicScalableBloomFilter::<u64>::new(0, 0.01).is_err());
        assert!(AtomicScalableBloomFilter::<u64>::new(1_000, 0.0).is_err());
        assert!(AtomicScalableBloomFilter::<u64>::new(1_000, 1.0).is_err());
        assert!(AtomicScalableBloomFilter::<u64>::new(1_000, -0.01).is_err());
    }

    #[test]
    fn test_with_strategy() {
        let strategies = [
            (GrowthStrategy::Constant, 0.5),
            (GrowthStrategy::Geometric(2.0), 0.5),
            (GrowthStrategy::Geometric(1.5), 0.5),
            (
                GrowthStrategy::Bounded {
                    scale: 2.0,
                    max_filter_size: 10_000,
                },
                0.5,
            ),
            (
                GrowthStrategy::Adaptive {
                    initial_ratio: 0.5,
                    min_ratio: 0.3,
                    max_ratio: 0.9,
                },
                0.5,
            ),
        ];
        for &(strategy, error_ratio) in &strategies {
            let f =
                AtomicScalableBloomFilter::<u64>::with_strategy(1_000, 0.01, error_ratio, strategy)
                    .unwrap();
            f.insert(&42);
            assert!(f.contains(&42));
            assert!(f.filter_count() >= 1);
        }
    }

    #[test]
    fn test_with_preallocated() {
        let f = AtomicScalableBloomFilter::<u64>::with_preallocated(1_000, 0.01, 10_000).unwrap();
        // with_preallocated pre-builds sub-filters to cover estimated_total_items.
        assert!(f.filter_count() >= 1);
        // Insert past initial capacity — the pre-built filters are activated.
        for i in 0..2_000u64 {
            f.insert(&i);
        }
        for i in 0..2_000u64 {
            assert!(f.contains(&i), "false negative in preallocated for {}", i);
        }
    }

    // --- accessors ----------------------------------------------------------------

    #[test]
    fn test_accessors() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        assert_eq!(f.filter_count(), 1);
        assert!(f.total_capacity() >= 1_000);
        assert!(f.shard_count() > 0);
        assert!(f.hash_count() > 0);
        assert!(f.bit_count() > 0);
        assert_eq!(f.count_set_bits(), 0);
        assert!(f.expected_items() >= 1_000);
        assert!(!f.is_at_max_capacity());
        assert!(!f.is_near_capacity());

        f.insert(&1);
        assert!(f.count_set_bits() > 0);
    }

    #[test]
    fn test_current_fill_rate() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        assert_eq!(f.current_fill_rate(), 0.0);
        f.insert(&1);
        let rate = f.current_fill_rate();
        assert!(rate > 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_estimate_count() {
        let f = AtomicScalableBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        assert_eq!(f.estimate_count(), 0);
        for i in 0..500u64 {
            f.insert(&i);
        }
        let est = f.estimate_count();
        // estimate_count is approximate but should be in the right ballpark.
        assert!(est > 100, "estimate_count too low: {}", est);
        assert!(est < 5_000, "estimate_count too high: {}", est);
    }

    #[test]
    fn test_estimate_fpr() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let fpr = f.estimate_fpr();
        assert!((0.0..=1.0).contains(&fpr));
        // After inserting nothing, FPR should be very low.
        assert!(fpr < 0.1);
    }

    #[test]
    fn test_false_positive_rate_alias() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        assert_eq!(f.false_positive_rate(), f.estimate_fpr());
    }

    #[test]
    fn test_bit_statistics() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let (total, set, utilization) = f.bit_statistics();
        assert_eq!(total, f.bit_count());
        assert_eq!(set, 0);
        assert_eq!(utilization, 0.0);
        for i in 0..100u64 {
            f.insert(&i);
        }
        let (_, set2, utilization2) = f.bit_statistics();
        assert!(set2 > 0);
        assert!(utilization2 > 0.0);
    }

    // --- batch ops ----------------------------------------------------------------

    #[test]
    fn test_insert_batch() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let items: Vec<u64> = (0..100).collect();
        f.insert_batch(&items).unwrap();
        assert_eq!(f.len(), 100);
        for i in 0..100u64 {
            assert!(f.contains(&i));
        }
    }

    #[test]
    fn test_insert_batch_empty() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let empty: Vec<u64> = vec![];
        f.insert_batch(&empty).unwrap();
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_contains_batch() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        for i in 0..50u64 {
            f.insert(&i);
        }
        let queries: Vec<u64> = (0..100).collect();
        let results = f.contains_batch(&queries);
        assert_eq!(results.len(), 100);
        for (i, &present) in results.iter().enumerate() {
            assert_eq!(present, i < 50, "mismatch at index {}", i);
        }
    }

    // --- clone --------------------------------------------------------------------

    #[test]
    fn test_clone() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        f.insert(&42);
        let g = f.clone();
        // Both see the same state (Arc semantics).
        assert!(g.contains(&42));
        assert_eq!(g.len(), 1);

        // Mutation through one clone is visible to the other.
        f.insert(&99);
        assert!(g.contains(&99));
    }

    // --- SharedBloomFilter trait --------------------------------------------------

    #[test]
    fn test_shared_bloom_filter_trait() {
        use crate::core::filter::SharedBloomFilter;

        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        // Dispatch through the trait.
        SharedBloomFilter::insert(&f, &42);
        assert!(SharedBloomFilter::contains(&f, &42));
        assert_eq!(SharedBloomFilter::len(&f), f.count_set_bits());
        assert!(!SharedBloomFilter::is_empty(&f));
        assert!(SharedBloomFilter::false_positive_rate(&f) > 0.0);

        SharedBloomFilter::clear(&f);
        assert!(SharedBloomFilter::is_empty(&f));
    }

    // --- display ------------------------------------------------------------------

    #[test]
    fn test_display() {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let s = format!("{}", f);
        assert!(s.starts_with("AtomicScalableBloomFilter {"));
        assert!(s.contains("filters: 1"));
    }

    // --- growth strategy validation ------------------------------------------------

    #[test]
    fn test_growth_strategy_validate() {
        assert!(GrowthStrategy::Geometric(1.0).validate().is_err());
        assert!(GrowthStrategy::Geometric(0.5).validate().is_err());
        assert!(GrowthStrategy::Geometric(f64::NAN).validate().is_err());
        assert!(GrowthStrategy::Geometric(f64::INFINITY).validate().is_err());
        assert!(GrowthStrategy::Geometric(2.0).validate().is_ok());
        assert!(GrowthStrategy::Geometric(100.0).validate().is_ok());

        assert!(GrowthStrategy::Bounded {
            scale: 0.0,
            max_filter_size: 100
        }
        .validate()
        .is_err());
        assert!(GrowthStrategy::Bounded {
            scale: -1.0,
            max_filter_size: 100
        }
        .validate()
        .is_err());
        assert!(GrowthStrategy::Bounded {
            scale: f64::NAN,
            max_filter_size: 100
        }
        .validate()
        .is_err());
        assert!(GrowthStrategy::Bounded {
            scale: 2.0,
            max_filter_size: 100
        }
        .validate()
        .is_ok());

        assert!(GrowthStrategy::Constant.validate().is_ok());
        assert!(GrowthStrategy::Adaptive {
            initial_ratio: 0.5,
            min_ratio: 0.3,
            max_ratio: 0.9
        }
        .validate()
        .is_ok());
    }
}
