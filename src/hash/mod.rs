//! Hash function implementations and strategies for Bloom filters.
//!
//! This module provides multiple hash functions and strategies optimized for
//! Bloom filter operations, with a focus on speed, quality, and correctness.
//!
//! # Module Structure
//!
//! ```text
//! hash/
//! ├── hasher.rs      - Core BloomHasher trait and StdHasher (SipHash wrapper)
//! ├── strategies.rs  - Hash index generation strategies
//! ├── wyhash.rs      - WyHash implementation (optional, feature = "wyhash")
//! ├── xxhash.rs      - XXHash3 implementation (optional, feature = "xxhash")
//! ├── simd.rs        - SIMD batch hashing (optional, feature = "simd")
//! └── mod.rs         - This file (public API)
//! ```
//!
//! # Quick Start
//!
//! ```
//! use bloomcraft::hash::{StdHasher, BloomHasher};
//!
//! let hasher = StdHasher::new();
//! let hash = hasher.hash_bytes(b"hello");
//! ```
//!
//! # Choosing a Hash Function
//!
//! | Hash Function | Speed        | Quality   | Use Case                       |
//! |---------------|--------------|-----------|--------------------------------|
//! | [`StdHasher`] | Medium       | Excellent | Default, DoS-resistant (SipHash) |
//! | [`WyHasher`]  | Very Fast    | Excellent | Small keys (<100 bytes)        |
//! | [`XxHasher`]  | Very Fast    | Excellent | Large keys (>100 bytes)        |
//! | [`SimdHasher`]| Fastest*     | Excellent | Batch operations (≥8 items)    |
//!
//! *SIMD is fastest when processing batches ≥8 items. For smaller batches, scalar is faster.
//!
//! # Hash Strategies
//!
//! Strategies generate multiple hash indices from base hashes:
//!
//! - **Double Hashing**: `h_i = (h1 + i*h2) mod m` - Standard, proven optimal
//! - **Enhanced Double Hashing**: `h_i = (h1 + i*h2 + i²) mod m` - Better distribution
//! - **Triple Hashing**: `h_i = (h1 + i*h2 + i²*h3) mod m` - Best distribution
//!
//! # Feature Flags
//!
//! | Feature      | Enables                                    |
//! |--------------|--------------------------------------------|
//! | (default)    | [`StdHasher`] (Rust's SipHash)             |
//! | `wyhash`     | [`WyHasher`]                               |
//! | `xxhash`     | [`XxHasher`]                               |
//! | `simd`       | [`SimdHasher`] with SIMD optimizations     |
//! | `fast-hash`  | Enables both `wyhash` and `xxhash`         |
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::hash::{StdHasher, BloomHasher};
//! use bloomcraft::hash::strategies::{DoubleHashing, HashStrategy};
//!
//! let hasher = StdHasher::new();
//! let strategy = DoubleHashing;
//!
//! // Get two base hashes
//! let (h1, h2) = hasher.hash_bytes_pair(b"test");
//!
//! // Generate 7 indices for a 1000-bit filter
//! let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);
//! assert_eq!(indices.len(), 7);
//! ```
//!
//! ## Using Fast Hash Functions
//!
//! ```
//! # #[cfg(feature = "wyhash")]
//! # {
//! use bloomcraft::hash::{WyHasher, BloomHasher};
//!
//! let hasher = WyHasher::new();
//! let hash = hasher.hash_bytes(b"fast");
//! # }
//! ```
//!
//! ## Batch Operations with SIMD
//!
//! ```
//! # #[cfg(feature = "simd")]
//! # {
//! use bloomcraft::hash::simd::SimdHasher;
//!
//! let hasher = SimdHasher::new();
//! let values = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
//!
//! // Hash all values in parallel (SIMD-accelerated)
//! let hashes = hasher.hash_batch_u64(&values);
//! assert_eq!(hashes.len(), 8);
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! ## Hash Function Selection
//!
//! - **DoS protection needed?** → Use [`StdHasher`] (SipHash, cryptographically strong)
//! - **Maximum speed, small keys (<100 bytes)?** → Use [`WyHasher`]
//! - **Maximum speed, large keys (>100 bytes)?** → Use [`XxHasher`]
//! - **Batch insertions (≥8 items)?** → Use [`SimdHasher`]
//!
//! ## Strategy Selection
//!
//! - **Default/recommended** → Enhanced Double Hashing (best balance)
//! - **Maximum speed** → Double Hashing (marginal speed gain)
//! - **Research/validation** → Triple Hashing (overkill for most cases)
//!
//! # References
//!
//! - Kirsch & Mitzenmacher (2006): "Less Hashing, Same Performance: Building a Better Bloom Filter"
//! - Dillinger & Manolios (2004): "Fast and Accurate Bitstate Verification for SPIN"
//! - Wang Yi: "wyhash and wyrand"
//! - Yann Collet: "XXHash - Extremely fast non-cryptographic hash algorithm"

// Core hash abstractions
pub mod hasher;
pub mod strategies;

// Optional fast hash implementations
#[cfg(feature = "wyhash")]
pub mod wyhash;

#[cfg(feature = "xxhash")]
pub mod xxhash;

// SIMD-optimized batch hashing
#[cfg(feature = "simd")]
pub mod simd;

// Re-export main types for convenience
pub use hasher::{BloomHasher, StdHasher};
pub use strategies::{DoubleHashing, EnhancedDoubleHashing, HashStrategy as HashStrategyTrait, TripleHashing};

/// Type alias for the default hasher used by Bloom filters.
///
/// This provides a stable name for the default hash function, allowing
/// code to reference `DefaultHasher` without depending on the specific
/// implementation (currently `StdHasher`).
pub type DefaultHasher = StdHasher;

/// Hash strategy configuration enum.
///
/// This enum allows selecting between different hash index generation strategies
/// at runtime. Each variant corresponds to a strategy implementation.
///
/// # Variants
///
/// - `Double`: Standard double hashing (fastest)
/// - `EnhancedDouble`: Enhanced double hashing with quadratic probing (recommended)
/// - `Triple`: Triple hashing (best distribution, slowest)
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::HashStrategy;
///
/// let strategy = HashStrategy::EnhancedDouble;
/// assert_eq!(strategy.base_hash_count(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HashStrategy {
    /// Standard double hashing: `h_i = (h1 + i*h2) mod m`
    Double,
    /// Enhanced double hashing with quadratic term: `h_i = (h1 + i*h2 + (i²+i)/2) mod m`
    #[default]
    EnhancedDouble,
    /// Triple hashing: `h_i = (h1 + i*h2 + i²*h3) mod m`
    Triple,
}

impl HashStrategy {
    /// Get the number of base hash values required by this strategy.
    ///
    /// # Returns
    ///
    /// - `2` for Double and EnhancedDouble
    /// - `3` for Triple
    #[must_use]
    pub const fn base_hash_count(&self) -> usize {
        match self {
            Self::Double | Self::EnhancedDouble => 2,
            Self::Triple => 3,
        }
    }

    /// Get the human-readable name of this strategy.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Double => "Double",
            Self::EnhancedDouble => "EnhancedDouble",
            Self::Triple => "Triple",
        }
    }

    /// Generate k hash indices using this strategy.
    ///
    /// # Arguments
    ///
    /// * `h1` - First base hash value
    /// * `h2` - Second base hash value
    /// * `h3` - Third base hash value (only used by Triple strategy)
    /// * `k` - Number of indices to generate
    /// * `m` - Filter size (indices will be in range `[0, m)`)
    ///
    /// # Returns
    ///
    /// Vector of k hash indices
    #[must_use]
    pub fn generate_indices(&self, h1: u64, h2: u64, h3: u64, k: usize, m: usize) -> Vec<usize> {
        match self {
            Self::Double => DoubleHashing.generate_indices(h1, h2, h3, k, m),
            Self::EnhancedDouble => EnhancedDoubleHashing.generate_indices(h1, h2, h3, k, m),
            Self::Triple => TripleHashing.generate_indices(h1, h2, h3, k, m),
        }
    }
}

#[cfg(feature = "wyhash")]
pub use wyhash::{WyHasher, WyHasherBuilder};

#[cfg(feature = "xxhash")]
pub use xxhash::{XxHasher, XxHasherBuilder};

#[cfg(feature = "simd")]
pub use simd::SimdHasher;

/// Prelude module for convenient imports.
///
/// Import everything from the prelude to get started quickly:
///
/// ```
/// use bloomcraft::hash::prelude::*;
///
/// let hasher = StdHasher::new();
/// let hash = hasher.hash_bytes(b"test");
/// ```
pub mod prelude {
    pub use super::hasher::{BloomHasher, StdHasher};
    pub use super::strategies::{DoubleHashing, EnhancedDoubleHashing, HashStrategy, TripleHashing};

    #[cfg(feature = "wyhash")]
    pub use super::wyhash::WyHasher;

    #[cfg(feature = "xxhash")]
    pub use super::xxhash::XxHasher;

    #[cfg(feature = "simd")]
    pub use super::simd::SimdHasher;
}

/// Get the recommended hash function for the current platform.
///
/// Selection priority:
/// 1. **WyHash** (if available) - Fastest for most workloads
/// 2. **XXHash** (if available) - Very fast, industry-standard
/// 3. **StdHasher** (fallback) - DoS-resistant SipHash
///
/// Uses static dispatch for zero-cost abstraction.
///
/// # Returns
///
/// A hasher instance optimized for the current platform.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{recommended_hasher, BloomHasher};
///
/// let hasher = recommended_hasher();
/// let hash = hasher.hash_bytes(b"test");
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

/// Get a SIMD-capable hasher if available, otherwise falls back to recommended hasher.
///
/// If the `simd` feature is enabled, returns a SIMD hasher optimized for batch
/// operations. Otherwise, returns the platform's recommended scalar hasher.
///
/// Uses static dispatch for zero-cost abstraction.
///
/// # Returns
///
/// A hasher optimized for batch operations when SIMD is available.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{simd_hasher, BloomHasher};
///
/// let hasher = simd_hasher();
/// let hash = hasher.hash_bytes(b"test");
/// ```
#[must_use]
#[cfg(feature = "simd")]
pub fn simd_hasher() -> SimdHasher {
    SimdHasher::new()
}

/// Get a SIMD-capable hasher if available, otherwise falls back to recommended hasher.
///
/// If the `simd` feature is enabled, returns a SIMD hasher optimized for batch
/// operations. Otherwise, returns the platform's recommended scalar hasher.
///
/// Uses static dispatch for zero-cost abstraction.
///
/// # Returns
///
/// A hasher optimized for batch operations when SIMD is available.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{simd_hasher, BloomHasher};
///
/// let hasher = simd_hasher();
/// let hash = hasher.hash_bytes(b"test");
/// ```
#[must_use]
#[cfg(not(feature = "simd"))]
pub fn simd_hasher() -> impl BloomHasher {
    recommended_hasher()
}

/// Create a hasher with a specific seed.
///
/// Useful for creating multiple independent hash functions or for
/// reproducible hashing across runs.
///
/// Uses the same priority as [`recommended_hasher`]: WyHash > XXHash > StdHasher.
///
/// # Arguments
///
/// * `seed` - Seed value for the hash function
///
/// # Returns
///
/// A hasher configured with the specified seed.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::{hasher_with_seed, BloomHasher};
///
/// let hasher1 = hasher_with_seed(1);
/// let hasher2 = hasher_with_seed(2);
///
/// let h1 = hasher1.hash_bytes(b"test");
/// let h2 = hasher2.hash_bytes(b"test");
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

/// Hash function comparison and benchmarking utilities.
///
/// This module provides utilities for comparing different hash functions
/// and measuring their performance characteristics.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::bench::compare_hashers;
///
/// let items: Vec<&[u8]> = vec![b"item1", b"item2", b"item3"];
/// let results = compare_hashers(&items);
///
/// for result in results {
///     println!("{}: {:.2} ns/hash", result.name, result.time_per_hash_ns);
/// }
/// ```
pub mod bench {
    use super::*;
    use std::time::{Duration, Instant};

    /// Benchmark results for a hash function.
    ///
    /// Contains timing and throughput information for a hash function benchmark.
    #[derive(Debug, Clone)]
    pub struct HashBenchmark {
        /// Hash function name
        pub name: String,
        /// Time to hash a single item (nanoseconds)
        pub time_per_hash_ns: f64,
        /// Throughput in items per second
        pub throughput: f64,
        /// Number of items hashed
        pub items_hashed: usize,
        /// Total duration
        pub duration: Duration,
    }

    impl HashBenchmark {
        /// Create a new benchmark result.
        pub fn new(name: String, items_hashed: usize, duration: Duration) -> Self {
            let time_per_hash_ns = duration.as_nanos() as f64 / items_hashed as f64;
            let throughput = (items_hashed as f64) / duration.as_secs_f64();

            Self {
                name,
                time_per_hash_ns,
                throughput,
                items_hashed,
                duration,
            }
        }
    }

    /// Benchmark a hash function on arbitrary byte slices.
    ///
    /// # Arguments
    ///
    /// * `hasher` - Hash function to benchmark
    /// * `items` - Byte slices to hash
    /// * `name` - Name for the benchmark
    ///
    /// # Returns
    ///
    /// Benchmark results
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::{StdHasher, bench::benchmark_hasher};
    ///
    /// let hasher = StdHasher::new();
    /// let items: Vec<Vec<u8>> = (0..1000)
    ///     .map(|i| format!("item{}", i).into_bytes())
    ///     .collect();
    /// let item_refs: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();
    /// let results = benchmark_hasher(&hasher, &item_refs, "StdHasher");
    ///
    /// println!("{:.2} ns/hash", results.time_per_hash_ns);
    /// ```
    pub fn benchmark_hasher<H: BloomHasher>(
        hasher: &H,
        items: &[&[u8]],
        name: &str,
    ) -> HashBenchmark {
        let start = Instant::now();

        // Hash all items
        for &item in items {
            let _ = hasher.hash_bytes(item);
        }

        let duration = start.elapsed();
        HashBenchmark::new(name.to_string(), items.len(), duration)
    }

    /// Compare all available hash functions.
    ///
    /// Benchmarks all hash functions enabled via feature flags and returns
    /// results sorted by speed (fastest first).
    ///
    /// # Arguments
    ///
    /// * `items` - Byte slices to hash for comparison
    ///
    /// # Returns
    ///
    /// Vector of benchmark results for all available hash functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::bench::compare_hashers;
    ///
    /// let items: Vec<Vec<u8>> = (0..1000)
    ///     .map(|i| format!("item{}", i).into_bytes())
    ///     .collect();
    /// let item_refs: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();
    /// let results = compare_hashers(&item_refs);
    ///
    /// for result in &results {
    ///     println!("{}: {:.2} ns/hash, {:.2} M hashes/sec",
    ///         result.name,
    ///         result.time_per_hash_ns,
    ///         result.throughput / 1_000_000.0
    ///     );
    /// }
    /// ```
    pub fn compare_hashers(items: &[&[u8]]) -> Vec<HashBenchmark> {
        let mut results = Vec::new();

        // Always benchmark StdHasher
        let std_hasher = StdHasher::new();
        results.push(benchmark_hasher(&std_hasher, items, "StdHasher"));

        #[cfg(feature = "wyhash")]
        {
            let wy_hasher = WyHasher::new();
            results.push(benchmark_hasher(&wy_hasher, items, "WyHasher"));
        }

        #[cfg(feature = "xxhash")]
        {
            let xx_hasher = XxHasher::new();
            results.push(benchmark_hasher(&xx_hasher, items, "XxHasher"));
        }

        #[cfg(feature = "simd")]
        {
            let simd_hasher = SimdHasher::new();
            results.push(benchmark_hasher(&simd_hasher, items, "SimdHasher"));
        }

        // Sort by speed (fastest first)
        results.sort_by(|a, b| a.time_per_hash_ns.partial_cmp(&b.time_per_hash_ns).unwrap());

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

         // Hasher Factory Tests
     
    #[test]
    fn test_recommended_hasher() {
        let hasher = recommended_hasher();
        let hash = hasher.hash_bytes(b"test");
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_recommended_hasher_deterministic() {
        let hasher = recommended_hasher();
        let h1 = hasher.hash_bytes(b"test");
        let h2 = hasher.hash_bytes(b"test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_simd_hasher() {
        let hasher = simd_hasher();
        let hash = hasher.hash_bytes(b"test");
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_hasher_with_seed() {
        let hasher1 = hasher_with_seed(1);
        let hasher2 = hasher_with_seed(2);

        let h1 = hasher1.hash_bytes(b"test");
        let h2 = hasher2.hash_bytes(b"test");

        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

    #[test]
    fn test_hasher_with_seed_deterministic() {
        let hasher = hasher_with_seed(42);
        let h1 = hasher.hash_bytes(b"test");
        let h2 = hasher.hash_bytes(b"test");
        assert_eq!(h1, h2);
    }

         // Prelude Tests
     
    #[test]
    fn test_prelude_imports() {
        use prelude::*;

        let hasher = StdHasher::new();
        let hash = hasher.hash_bytes(b"test");
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_prelude_strategy_imports() {
        use prelude::*;

        let strategy = DoubleHashing;
        let (h1, h2) = (12345u64, 67890u64);
        let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);
        assert_eq!(indices.len(), 7);
    }

         // Strategy Tests
     
    #[test]
    fn test_hash_strategy_enum() {
        let strategy = HashStrategy::EnhancedDouble;
        assert_eq!(strategy.base_hash_count(), 2);
    }

    #[test]
    fn test_double_hashing_strategy() {
        let strategy = DoubleHashing;
        let (h1, h2) = (100u64, 200u64);
        let indices = strategy.generate_indices(h1, h2, 0, 5, 1000);

        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_enhanced_double_hashing_strategy() {
        let strategy = EnhancedDoubleHashing;
        let (h1, h2) = (100u64, 200u64);
        let indices = strategy.generate_indices(h1, h2, 0, 5, 1000);

        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_triple_hashing_strategy() {
        let strategy = TripleHashing;
        let (h1, h2, h3) = (100u64, 200u64, 300u64);
        let indices = strategy.generate_indices(h1, h2, h3, 5, 1000);

        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

         // Integration Tests (All Hashers)
     
    #[test]
    fn test_all_hashers_consistent() {
        let data = b"test";

        // Test StdHasher
        let hasher = StdHasher::new();
        let h1 = hasher.hash_bytes(data);
        let h2 = hasher.hash_bytes(data);
        assert_eq!(h1, h2);

        // Test WyHasher if available
        #[cfg(feature = "wyhash")]
        {
            let hasher = WyHasher::new();
            let h1 = hasher.hash_bytes(data);
            let h2 = hasher.hash_bytes(data);
            assert_eq!(h1, h2);
        }

        // Test XxHasher if available
        #[cfg(feature = "xxhash")]
        {
            let hasher = XxHasher::new();
            let h1 = hasher.hash_bytes(data);
            let h2 = hasher.hash_bytes(data);
            assert_eq!(h1, h2);
        }

        // Test SimdHasher if available
        #[cfg(feature = "simd")]
        {
            let hasher = SimdHasher::new();
            let h1 = hasher.hash_bytes(data);
            let h2 = hasher.hash_bytes(data);
            assert_eq!(h1, h2);
        }
    }

    #[test]
    fn test_all_hashers_different_seeds() {
        let data = b"test";

        let h1 = hasher_with_seed(1).hash_bytes(data);
        let h2 = hasher_with_seed(2).hash_bytes(data);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_all_hashers_with_strategies() {
        let hasher = StdHasher::new();
        let strategy = EnhancedDoubleHashing;
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);
        let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

         // Benchmark Module Tests
     
    #[test]
    fn test_bench_benchmark_hasher() {
        let hasher = StdHasher::new();
        let items: Vec<Vec<u8>> = (0..100).map(|i| format!("item{}", i).into_bytes()).collect();
        let item_refs: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();

        let results = bench::benchmark_hasher(&hasher, &item_refs, "test");

        assert_eq!(results.name, "test");
        assert_eq!(results.items_hashed, 100);
        assert!(results.time_per_hash_ns > 0.0);
        assert!(results.throughput > 0.0);
    }

    #[test]
    fn test_bench_compare_hashers() {
        let items: Vec<Vec<u8>> = (0..100).map(|i| format!("item{}", i).into_bytes()).collect();
        let item_refs: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();

        let results = bench::compare_hashers(&item_refs);

        // Should at least have StdHasher
        assert!(!results.is_empty());

        for result in &results {
            assert!(result.time_per_hash_ns > 0.0);
            assert!(result.throughput > 0.0);
            assert_eq!(result.items_hashed, 100);
        }
    }

    #[test]
    fn test_bench_results_sorted() {
        let items: Vec<Vec<u8>> = (0..100).map(|i| format!("item{}", i).into_bytes()).collect();
        let item_refs: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();

        let results = bench::compare_hashers(&item_refs);

        // Verify results are sorted by speed (fastest first)
        for i in 1..results.len() {
            assert!(
                results[i - 1].time_per_hash_ns <= results[i].time_per_hash_ns,
                "Results not sorted: {} ({:.2}) should be <= {} ({:.2})",
                results[i - 1].name,
                results[i - 1].time_per_hash_ns,
                results[i].name,
                results[i].time_per_hash_ns
            );
        }
    }

         // Feature Flag Tests
     
    #[test]
    #[cfg(feature = "wyhash")]
    fn test_wyhash_available() {
        let hasher = WyHasher::new();
        let hash = hasher.hash_bytes(b"test");
        assert_ne!(hash, 0);
    }

    #[test]
    #[cfg(feature = "xxhash")]
    fn test_xxhash_available() {
        let hasher = XxHasher::new();
        let hash = hasher.hash_bytes(b"test");
        assert_ne!(hash, 0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_available() {
        let hasher = SimdHasher::new();
        let hash = hasher.hash_bytes(b"test");
        assert_ne!(hash, 0);
    }

         // Re-export Tests
     
    #[test]
    fn test_reexports_compile() {
        // Just verify all re-exports are accessible
        let _ = StdHasher::new();
        let _ = DoubleHashing;
        let _ = EnhancedDoubleHashing;
        let _ = TripleHashing;

        #[cfg(feature = "wyhash")]
        {
            let _ = WyHasher::new();
            let _ = WyHasherBuilder::new();
        }

        #[cfg(feature = "xxhash")]
        {
            let _ = XxHasher::new();
            let _ = XxHasherBuilder::new();
        }

        #[cfg(feature = "simd")]
        {
            let _ = SimdHasher::new();
        }
    }
}
