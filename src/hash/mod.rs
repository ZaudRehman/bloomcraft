//! Hash function implementations and strategies for Bloom filters.
//!
//! This module provides multiple hash functions and strategies optimised for
//! Bloom filter operations, with a focus on speed, quality, and correctness.
//!
//! # Module Structure
//!
//! ```text
//! hash/
//! ├── hasher.rs      - BloomHasher trait, StdHasher (FNV-1a), HashWriter bridge
//! ├── strategies.rs  - HashStrategy trait + DoubleHashing / EnhancedDoubleHashing / TripleHashing
//! ├── wyhash.rs      - WyHash (optional, feature = "wyhash")
//! ├── xxhash.rs      - XXHash3 (optional, feature = "xxhash")
//! ├── simd.rs        - SIMD batch hashing (optional, feature = "simd")
//! └── mod.rs         - Public API surface (this file)
//! ```
//!
//! # Quick Start
//!
//! ```
//! use bloomcraft::hash::{BloomHasher, StdHasher};
//!
//! let hasher = StdHasher::new();
//!
//! // Raw bytes
//! let hash = hasher.hash_bytes(b"hello");
//!
//! // Generic item (canonical bridge — zero allocation for keys ≤32 bytes)
//! let (h1, h2) = hasher.hash_item(&"hello".to_string());
//! ```
//!
//! # Choosing a Hash Function
//!
//! | Hash Function  | Speed     | Deterministic | Use Case                        |
//! |----------------|-----------|---------------|---------------------------------|
//! | [`StdHasher`]  | Fast      | Yes           | Default; deterministic FNV-1a   |
//! | [`WyHasher`]   | Very fast | Yes           | Trusted input, small keys       |
//! | [`XxHasher`]   | Very fast | Yes           | Trusted input, large keys       |
//! | [`SimdHasher`] | Fastest*  | Yes           | Batch operations (≥8 items)     |
//!
//! *SIMD throughput peaks at batches of ≥8 items. For smaller batches, scalar is faster.
//!
//! **None of the above are DoS-resistant.** For hash tables exposed to adversarial
//! input, use a different tool. For Bloom filters, adversaries cannot force false
//! negatives via hash collisions alone.
//!
//! # Hash Strategies
//!
//! Strategies produce k bit-array indices from a pair (or triple) of base hashes.
//! The [`HashStrategy`] trait is implemented by three zero-sized types:
//!
//! | Strategy                | Formula                                      | Use Case              |
//! |-------------------------|----------------------------------------------|-----------------------|
//! | [`DoubleHashing`]       | `h_i = (h1 + i·h2) mod m`                   | General purpose       |
//! | [`EnhancedDoubleHashing`] | `h_i = (h1 + i·h2 + (i²+i)/2) mod m`     | High-accuracy default |
//! | [`TripleHashing`]       | `h_i = (h1 + i·h2 + i²·h3) mod m`          | Research / validation |
//!
//! For runtime selection between strategies, use [`IndexingStrategy`].
//!
//! # Feature Flags
//!
//! | Feature     | Enables                                         |
//! |-------------|--------------------------------------------------|
//! | (default)   | [`StdHasher`] — deterministic FNV-1a            |
//! | `wyhash`    | [`WyHasher`] — inline custom WyHash             |
//! | `xxhash`    | [`XxHasher`] — backed by `xxhash-rust`          |
//! | `simd`      | [`SimdHasher`] — SIMD batch hashing via std::arch|
//! | `fast-hash` | Enables both `wyhash` and `xxhash`              |
//!
//! # References
//!
//! - Kirsch & Mitzenmacher (2006): "Less Hashing, Same Performance"
//! - Dillinger & Manolios (2004): "Fast and Accurate Bitstate Verification for SPIN"
//! - Wang Yi: wyhash — <https://github.com/wangyi-fudan/wyhash>
//! - Yann Collet: XXHash — <https://github.com/Cyan4973/xxHash>

// ── Submodules ────────────────────────────────────────────────────────────────

pub mod hasher;
pub mod strategies;

#[cfg(feature = "wyhash")]
pub mod wyhash;

#[cfg(feature = "xxhash")]
pub mod xxhash;

#[cfg(feature = "simd")]
pub mod simd;

// ── Core re-exports ───────────────────────────────────────────────────────────

pub use hasher::{BloomHasher, StdHasher};
pub use strategies::{DoubleHashing, EnhancedDoubleHashing, HashStrategy, TripleHashing};

#[cfg(feature = "wyhash")]
pub use wyhash::{WyHasher, WyHasherBuilder};

#[cfg(feature = "xxhash")]
pub use xxhash::{XxHasher, XxHasherBuilder};

#[cfg(feature = "simd")]
pub use simd::SimdHasher;

// ── Type aliases ──────────────────────────────────────────────────────────────

/// Stable alias for the default hash function used across all filter types.
///
/// Code that needs to name the default hasher without depending on the specific
/// type can use `DefaultHasher`. If the default is changed in a future version,
/// this alias will be updated and all call sites remain valid.
pub type HashStrategyKind = IndexingStrategy;

// ── IndexingStrategy enum ────────────────────────────────────────────────────

/// Runtime-selectable hash indexing strategy.
///
/// This enum mirrors the three [`HashStrategy`] implementations as variants,
/// enabling runtime strategy selection (e.g., from config or CLI flags) while
/// delegating to the same zero-sized-type implementations used on the hot path.
///
/// # Naming: `IndexingStrategy` vs `HashStrategy`
///
/// [`HashStrategy`] is the **trait** implemented by `DoubleHashing`,
/// `EnhancedDoubleHashing`, and `TripleHashing`. This **enum** provides a
/// serialisable, runtime-dispatchable handle to those same strategies.
/// The names are distinct to prevent import ambiguity.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::IndexingStrategy;
///
/// let strategy = IndexingStrategy::EnhancedDouble;
/// assert_eq!(strategy.base_hash_count(), 2);
/// assert_eq!(strategy.name(), "EnhancedDouble");
///
/// let indices = strategy.generate_indices(12345, 67890, 0, 7, 1000);
/// assert_eq!(indices.len(), 7);
/// assert!(indices.iter().all(|&i| i < 1000));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndexingStrategy {
    /// Standard double hashing: `h_i = (h1 + i·h2) mod m`
    ///
    /// Fastest strategy. Proven asymptotically optimal by Kirsch & Mitzenmacher (2006).
    Double,

    /// Enhanced double hashing: `h_i = (h1 + i·h2 + (i²+i)/2) mod m`
    ///
    /// Adds a quadratic anti-clustering term. Better distribution for large k (>10).
    /// Recommended default.
    #[default]
    EnhancedDouble,

    /// Triple hashing: `h_i = (h1 + i·h2 + i²·h3) mod m`
    ///
    /// Requires three base hashes. Near-perfect index independence.
    /// Use for research or empirical validation; overkill for production.
    Triple,
}

impl IndexingStrategy {
    /// Number of base hash values this strategy requires.
    ///
    /// Returns `2` for `Double` and `EnhancedDouble`; `3` for `Triple`.
    /// Filters use this to decide whether to call
    /// [`hash_bytes_pair`](BloomHasher::hash_bytes_pair) or
    /// [`hash_bytes_triple`](BloomHasher::hash_bytes_triple).
    #[must_use]
    pub const fn base_hash_count(&self) -> usize {
        match self {
            Self::Double | Self::EnhancedDouble => 2,
            Self::Triple => 3,
        }
    }

    /// Human-readable name, stable across serialisation roundtrips.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Double => "Double",
            Self::EnhancedDouble => "EnhancedDouble",
            Self::Triple => "Triple",
        }
    }

    /// Generate `k` hash indices in `[0, m)` using this strategy.
    ///
    /// Delegates to the corresponding zero-sized-type implementation.
    ///
    /// # Arguments
    ///
    /// * `h1` — First base hash
    /// * `h2` — Second base hash
    /// * `h3` — Third base hash (used only by `Triple`; pass `0` otherwise)
    /// * `k`  — Number of indices to generate
    /// * `m`  — Filter size in bits; all returned indices are in `[0, m)`
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::IndexingStrategy;
    ///
    /// let indices = IndexingStrategy::Double.generate_indices(111, 222, 0, 5, 1000);
    /// assert_eq!(indices.len(), 5);
    /// assert!(indices.iter().all(|&i| i < 1000));
    /// ```
    #[must_use]
    pub fn generate_indices(&self, h1: u64, h2: u64, h3: u64, k: usize, m: usize) -> Vec<usize> {
        match self {
            Self::Double => DoubleHashing.generate_indices(h1, h2, h3, k, m),
            Self::EnhancedDouble => EnhancedDoubleHashing.generate_indices(h1, h2, h3, k, m),
            Self::Triple => TripleHashing.generate_indices(h1, h2, h3, k, m),
        }
    }
}

// ── Factory functions ─────────────────────────────────────────────────────────

/// Return the recommended hasher for the current feature configuration.
///
/// Selection priority (first enabled feature wins):
/// 1. `wyhash` — fastest for most workloads
/// 2. `xxhash` — very fast, industry-standard
/// 3. `StdHasher` — deterministic FNV-1a (default)
///
/// Uses static dispatch: zero runtime cost.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{recommended_hasher, BloomHasher};
///
/// let hasher = recommended_hasher();
/// assert_ne!(hasher.hash_bytes(b"test"), 0);
/// ```
#[must_use]
pub fn recommended_hasher() -> impl BloomHasher {
    #[cfg(feature = "wyhash")]
    {
        WyHasher::new()
    }

    #[cfg(all(not(feature = "wyhash"), feature = "xxhash"))]
    {
        XxHasher::new()
    }

    #[cfg(not(any(feature = "wyhash", feature = "xxhash")))]
    {
        StdHasher::new()
    }
}

/// Return a SIMD-capable hasher if `feature = "simd"` is enabled, otherwise
/// falls back to [`recommended_hasher`].
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{simd_hasher, BloomHasher};
///
/// let hasher = simd_hasher();
/// assert_ne!(hasher.hash_bytes(b"test"), 0);
/// ```
#[must_use]
#[cfg(feature = "simd")]
pub fn simd_hasher() -> SimdHasher {
    SimdHasher::new()
}

/// Return a SIMD-capable hasher if `feature = "simd"` is enabled, otherwise
/// falls back to [`recommended_hasher`].
#[must_use]
#[cfg(not(feature = "simd"))]
pub fn simd_hasher() -> impl BloomHasher {
    recommended_hasher()
}

/// Return a hasher initialised with a specific seed.
///
/// Useful for creating multiple independent hash functions from a single
/// algorithm, or for reproducible hashing with a known seed.
/// Uses the same priority as [`recommended_hasher`].
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{hasher_with_seed, BloomHasher};
///
/// let h1 = hasher_with_seed(1).hash_bytes(b"test");
/// let h2 = hasher_with_seed(2).hash_bytes(b"test");
/// assert_ne!(h1, h2);
/// ```
#[must_use]
pub fn hasher_with_seed(seed: u64) -> impl BloomHasher {
    #[cfg(feature = "wyhash")]
    {
        WyHasher::with_seed(seed)
    }

    #[cfg(all(not(feature = "wyhash"), feature = "xxhash"))]
    {
        XxHasher::with_seed(seed)
    }

    #[cfg(not(any(feature = "wyhash", feature = "xxhash")))]
    {
        StdHasher::with_seed(seed)
    }
}

// ── Prelude ───────────────────────────────────────────────────────────────────

/// Convenience prelude — import everything needed to start hashing.
///
/// ```
/// use bloomcraft::hash::prelude::*;
///
/// let hasher = StdHasher::new();
/// let (h1, h2) = hasher.hash_item(&"hello".to_string());
/// assert_ne!(h1, h2);
/// ```
pub mod prelude {
    pub use super::hasher::{BloomHasher, StdHasher};
    pub use super::strategies::{DoubleHashing, EnhancedDoubleHashing, HashStrategy, TripleHashing};
    pub use super::IndexingStrategy;

    #[cfg(feature = "wyhash")]
    pub use super::wyhash::WyHasher;

    #[cfg(feature = "xxhash")]
    pub use super::xxhash::XxHasher;

    #[cfg(feature = "simd")]
    pub use super::simd::SimdHasher;
}

// ── Benchmarking utilities ────────────────────────────────────────────────────

/// Hasher comparison and micro-benchmark utilities.
///
/// These functions are intended for integration tests and internal profiling.
/// For production benchmarks use the `benches/` directory with Criterion.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::bench::compare_hashers;
///
/// let items: Vec<Vec<u8>> = (0..100)
///     .map(|i| format!("item{}", i).into_bytes())
///     .collect();
/// let refs: Vec<&[u8]> = items.iter().map(Vec::as_slice).collect();
///
/// for result in compare_hashers(&refs) {
///     println!("{}: {:.2} ns/hash", result.name, result.time_per_hash_ns);
/// }
/// ```
pub mod bench {
    use super::*;
    use std::time::{Duration, Instant};

    /// Timing result for a single hash function benchmark.
    #[derive(Debug, Clone)]
    pub struct HashBenchmark {
        /// Hash function name (matches [`BloomHasher::name`])
        pub name: String,
        /// Mean time to hash one item, in nanoseconds
        pub time_per_hash_ns: f64,
        /// Items hashed per second
        pub throughput: f64,
        /// Total items hashed during the run
        pub items_hashed: usize,
        /// Wall-clock duration of the entire run
        pub duration: Duration,
    }

    impl HashBenchmark {
        pub(crate) fn new(name: String, items_hashed: usize, duration: Duration) -> Self {
            let time_per_hash_ns = duration.as_nanos() as f64 / items_hashed as f64;
            let throughput = items_hashed as f64 / duration.as_secs_f64();
            Self { name, time_per_hash_ns, throughput, items_hashed, duration }
        }
    }

    /// Benchmark a single hasher against a set of byte slices.
    ///
    /// Returns timing results. Results are only meaningful in `--release` builds.
    pub fn benchmark_hasher<H: BloomHasher>(
        hasher: &H,
        items: &[&[u8]],
        name: &str,
    ) -> HashBenchmark {
        let start = Instant::now();
        for &item in items {
            // Prevent the compiler from optimising away the hash call.
            let _ = std::hint::black_box(hasher.hash_bytes(item));
        }
        HashBenchmark::new(name.to_string(), items.len(), start.elapsed())
    }

    /// Benchmark all enabled hash functions and return results sorted fastest-first.
    pub fn compare_hashers(items: &[&[u8]]) -> Vec<HashBenchmark> {
        let mut results = vec![
            benchmark_hasher(&StdHasher::new(), items, "StdHasher"),
        ];

        #[cfg(feature = "wyhash")]
        results.push(benchmark_hasher(&WyHasher::new(), items, "WyHasher"));

        #[cfg(feature = "xxhash")]
        results.push(benchmark_hasher(&XxHasher::new(), items, "XxHasher"));

        #[cfg(feature = "simd")]
        results.push(benchmark_hasher(&SimdHasher::new(), items, "SimdHasher"));

        results.sort_by(|a, b| {
            a.time_per_hash_ns
                .partial_cmp(&b.time_per_hash_ns)
                .unwrap()
        });
        results
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::StdHasher;

    // ── IndexingStrategy ─────────────────────────────────────────────────────

    #[test]
    fn test_indexing_strategy_base_hash_count() {
        assert_eq!(IndexingStrategy::Double.base_hash_count(), 2);
        assert_eq!(IndexingStrategy::EnhancedDouble.base_hash_count(), 2);
        assert_eq!(IndexingStrategy::Triple.base_hash_count(), 3);
    }

    #[test]
    fn test_indexing_strategy_names() {
        assert_eq!(IndexingStrategy::Double.name(), "Double");
        assert_eq!(IndexingStrategy::EnhancedDouble.name(), "EnhancedDouble");
        assert_eq!(IndexingStrategy::Triple.name(), "Triple");
    }

    #[test]
    fn test_indexing_strategy_default_is_enhanced_double() {
        assert_eq!(IndexingStrategy::default(), IndexingStrategy::EnhancedDouble);
    }

    #[test]
    fn test_indexing_strategy_generate_indices_bounds() {
        for variant in [
            IndexingStrategy::Double,
            IndexingStrategy::EnhancedDouble,
            IndexingStrategy::Triple,
        ] {
            let indices = variant.generate_indices(12345, 67890, 11111, 7, 1000);
            assert_eq!(indices.len(), 7, "{:?} should generate exactly k indices", variant);
            assert!(
                indices.iter().all(|&i| i < 1000),
                "{:?} produced out-of-bounds index",
                variant
            );
        }
    }

    #[test]
    fn test_indexing_strategy_is_deterministic() {
        for variant in [
            IndexingStrategy::Double,
            IndexingStrategy::EnhancedDouble,
            IndexingStrategy::Triple,
        ] {
            let a = variant.generate_indices(999, 888, 777, 5, 500);
            let b = variant.generate_indices(999, 888, 777, 5, 500);
            assert_eq!(a, b, "{:?} must be deterministic", variant);
        }
    }

    #[test]
    fn test_indexing_strategy_variants_differ() {
        // Using large k so the quadratic / cubic terms have room to diverge.
        let (h1, h2, h3) = (0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888u64, 0x9999_aaaabbbb_ccccu64);
        let double   = IndexingStrategy::Double.generate_indices(h1, h2, h3, 20, 10_000);
        let enhanced = IndexingStrategy::EnhancedDouble.generate_indices(h1, h2, h3, 20, 10_000);
        let triple   = IndexingStrategy::Triple.generate_indices(h1, h2, h3, 20, 10_000);
        assert_ne!(double, enhanced);
        assert_ne!(enhanced, triple);
        assert_ne!(double, triple);
    }

    #[test]
    fn test_indexing_strategy_wrapping_safety() {
        // u64::MAX inputs must not panic — all arithmetic is wrapping.
        for variant in [
            IndexingStrategy::Double,
            IndexingStrategy::EnhancedDouble,
            IndexingStrategy::Triple,
        ] {
            let indices = variant.generate_indices(u64::MAX, u64::MAX, u64::MAX, 10, 1000);
            assert_eq!(indices.len(), 10);
            assert!(indices.iter().all(|&i| i < 1000));
        }
    }

    // ── Factory functions ─────────────────────────────────────────────────────

    #[test]
    fn test_recommended_hasher_is_deterministic() {
        let h = recommended_hasher();
        assert_eq!(h.hash_bytes(b"test"), h.hash_bytes(b"test"));
    }

    #[test]
    fn test_recommended_hasher_nonzero() {
        assert_ne!(recommended_hasher().hash_bytes(b"test"), 0);
    }

    #[test]
    fn test_simd_hasher_nonzero() {
        assert_ne!(simd_hasher().hash_bytes(b"test"), 0);
    }

    #[test]
    fn test_hasher_with_seed_independence() {
        let h1 = hasher_with_seed(1).hash_bytes(b"test");
        let h2 = hasher_with_seed(2).hash_bytes(b"test");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hasher_with_seed_deterministic() {
        let h1 = hasher_with_seed(42).hash_bytes(b"test");
        let h2 = hasher_with_seed(42).hash_bytes(b"test");
        assert_eq!(h1, h2);
    }

    // ── Type alias ────────────────────────────────────────────────────────────

    #[test]
    fn test_default_hasher_alias_is_std_hasher() {
        // Verify the alias compiles and behaves identically to StdHasher.
        let a: StdHasher = StdHasher::new();
        let b: StdHasher = StdHasher::new();
        assert_eq!(
            a.hash_bytes(b"alias check"),
            b.hash_bytes(b"alias check")
        );
    }

    // ── Prelude ───────────────────────────────────────────────────────────────

    #[test]
    fn test_prelude_imports_compile() {
        use prelude::*;
        let h = StdHasher::new();
        let (h1, h2) = h.hash_item(&"prelude test".to_string());
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_prelude_strategy_accessible() {
        use prelude::*;
        let indices = DoubleHashing.generate_indices(1, 2, 0, 5, 100);
        assert_eq!(indices.len(), 5);
    }

    // ── Bench module ──────────────────────────────────────────────────────────

    #[test]
    fn test_bench_benchmark_hasher_fields() {
        let hasher = StdHasher::new();
        let items: Vec<Vec<u8>> = (0..100).map(|i| format!("item{}", i).into_bytes()).collect();
        let refs: Vec<&[u8]> = items.iter().map(Vec::as_slice).collect();
        let result = bench::benchmark_hasher(&hasher, &refs, "StdHasher");
        assert_eq!(result.name, "StdHasher");
        assert_eq!(result.items_hashed, 100);
        assert!(result.time_per_hash_ns > 0.0);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_bench_compare_hashers_nonempty_sorted() {
        let items: Vec<Vec<u8>> = (0..50).map(|i| format!("i{}", i).into_bytes()).collect();
        let refs: Vec<&[u8]> = items.iter().map(Vec::as_slice).collect();
        let results = bench::compare_hashers(&refs);
        assert!(!results.is_empty());
        for w in results.windows(2) {
            assert!(
                w[0].time_per_hash_ns <= w[1].time_per_hash_ns,
                "Results not sorted: {} ({:.2} ns) should be ≤ {} ({:.2} ns)",
                w[0].name, w[0].time_per_hash_ns,
                w[1].name, w[1].time_per_hash_ns,
            );
        }
    }

    // ── Feature flag coverage ────────────────────────────────────────────────

    #[test]
    #[cfg(feature = "wyhash")]
    fn test_wyhash_reexport() {
        let h = WyHasher::new();
        assert_ne!(h.hash_bytes(b"test"), 0);
    }

    #[test]
    #[cfg(feature = "xxhash")]
    fn test_xxhash_reexport() {
        let h = XxHasher::new();
        assert_ne!(h.hash_bytes(b"test"), 0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_reexport() {
        let h = SimdHasher::new();
        assert_ne!(h.hash_bytes(b"test"), 0);
    }

    // ── Integration: hasher + IndexingStrategy ────────────────────────────────

    #[test]
    fn test_hash_item_with_indexing_strategy() {
        let hasher = StdHasher::new();
        let (h1, h2) = hasher.hash_item(&"end to end".to_string());
        let indices = IndexingStrategy::EnhancedDouble.generate_indices(h1, h2, 0, 7, 10_000);
        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&i| i < 10_000));
    }

    #[test]
    fn test_reexports_all_compile() {
        let _ = StdHasher::new();
        let _ = DoubleHashing;
        let _ = EnhancedDoubleHashing;
        let _ = TripleHashing;
        let _ = IndexingStrategy::default();
        let _ = StdHasher::new();

        #[cfg(feature = "wyhash")]
        { let _ = WyHasher::new(); let _ = WyHasherBuilder::new(); }

        #[cfg(feature = "xxhash")]
        { let _ = XxHasher::new(); let _ = XxHasherBuilder::new(); }

        #[cfg(feature = "simd")]
        { let _ = SimdHasher::new(); }
    }
}
