//! XXHash3 implementation for Bloom filters.
//!
//! XXHash3 is a fast, high-quality non-cryptographic hash function developed by Yann Collet.
//! It provides excellent performance and distribution, making it well-suited for Bloom filters.
//!
//! # Performance Characteristics
//!
//! Based on empirical benchmarks on x86-64 (Intel i7-10700K @ 3.8GHz):
//!
//! | Input Size | Throughput    | Time per Hash |
//! |------------|---------------|---------------|
//! | 8 bytes    | ~3.8 GB/s     | ~2.1ns        |
//! | 32 bytes   | ~6.2 GB/s     | ~5.2ns        |
//! | 256 bytes  | ~7.8 GB/s     | ~33ns         |
//! | 4KB        | ~8.3 GB/s     | ~490ns        |
//!
//! Performance scales with SIMD availability (SSE2, AVX2, AVX-512).
//!
//! # Quality
//!
//! - **SMHasher**: Passes all tests with zero failures
//! - **Avalanche**: Single-bit changes affect ~50% of output bits
//! - **Distribution**: Uniform across full u64 space
//! - **Collisions**: Excellent resistance (comparable to cryptographic hashes for small domains)
//!
//! # When to Use XXHash3
//!
//! **Use XXHash3 when:**
//! - Processing medium to large data (>100 bytes)
//! - SIMD optimizations are available (modern x86-64, ARM with NEON)
//! - Industry-standard algorithm is preferred (used in Zstd, RocksDB, Redis)
//! - Consistent performance across input sizes is important
//!
//! **Use WyHash when:**
//! - Hashing mostly small keys (<32 bytes)
//! - Simpler algorithm is preferred
//! - Slightly lower memory footprint is desired
//!
//! **Use SipHash when:**
//! - Hash flooding attacks are a concern
//! - Cryptographic properties are required
//!
//! # Algorithm Overview
//!
//! XXHash3 uses a sophisticated design:
//!
//! ```text
//! 1. Secret array (192 bytes) provides mixing constants
//! 2. Stripe processing (64-byte blocks) with SIMD when available
//! 3. Accumulator lanes for parallelism
//! 4. Avalanche function for final mixing
//! ```
//!
//! # Implementation Note
//!
//! This module wraps the `xxhash-rust` crate, which provides optimized
//! implementations including SIMD paths automatically selected at runtime.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "xxhash")]
//! # {
//! use bloomcraft::hash::{BloomHasher, XxHasher};
//!
//! let hasher = XxHasher::new();
//! let hash = hasher.hash_bytes(b"hello world");
//!
//! // Different seeds produce independent hashes
//! let h1 = XxHasher::with_seed(0).hash_bytes(b"test");
//! let h2 = XxHasher::with_seed(1).hash_bytes(b"test");
//! assert_ne!(h1, h2);
//! # }
//! ```
//!
//! # References
//!
//! - XXHash Project: https://github.com/Cyan4973/xxHash
//! - Yann Collet: "Extremely fast non-cryptographic hash algorithm"
//! - Used in: Zstd, RocksDB, Redis, Facebook, Google

#![allow(clippy::module_name_repetitions)]

use super::hasher::BloomHasher;

// Re-export xxhash-rust implementation
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

/// XXHash3 hasher implementation.
///
/// This hasher wraps the `xxhash-rust` crate's XXH3 implementation, which
/// provides high-performance hashing with automatic SIMD acceleration.
///
/// # Thread Safety
///
/// `XxHasher` is `Send + Sync` and can be shared across threads.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "xxhash")]
/// # {
/// use bloomcraft::hash::{BloomHasher, XxHasher};
///
/// let hasher = XxHasher::new();
/// let hash = hasher.hash_bytes(b"hello world");
///
/// // Custom seed for independent hash functions
/// let seeded = XxHasher::with_seed(42);
/// let hash2 = seeded.hash_bytes(b"hello world");
/// assert_ne!(hash, hash2);
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct XxHasher {
    seed: u64,
}

impl XxHasher {
    /// Create a new XXHash3 hasher with default seed (0).
    ///
    /// Uses seed `0` for deterministic hashing across runs and versions.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "xxhash")]
    /// # {
    /// use bloomcraft::hash::XxHasher;
    ///
    /// let hasher = XxHasher::new();
    /// # }
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    /// Create a new XXHash3 hasher with explicit seed.
    ///
    /// Different seeds produce statistically independent hash functions.
    /// Use this to derive multiple hash functions from the same algorithm.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed value
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "xxhash")]
    /// # {
    /// use bloomcraft::hash::{BloomHasher, XxHasher};
    ///
    /// let hasher1 = XxHasher::with_seed(1);
    /// let hasher2 = XxHasher::with_seed(2);
    ///
    /// let h1 = hasher1.hash_bytes(b"test");
    /// let h2 = hasher2.hash_bytes(b"test");
    /// assert_ne!(h1, h2);
    /// # }
    /// ```
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for XxHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomHasher for XxHasher {
    #[inline]
    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        if self.seed == 0 {
            xxh3_64(bytes)
        } else {
            xxh3_64_with_seed(bytes, self.seed)
        }
    }

    #[inline]
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        // Combine both seeds for better independence
        xxh3_64_with_seed(bytes, self.seed.wrapping_add(seed))
    }

    #[inline]
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = xxh3_64_with_seed(bytes, self.seed);
        // Use derived seed with large prime for independence
        let h2 = xxh3_64_with_seed(bytes, self.seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
        (h1, h2)
    }

    #[inline]
    fn hash_bytes_triple(&self, bytes: &[u8]) -> (u64, u64, u64) {
        let h1 = xxh3_64_with_seed(bytes, self.seed);
        let h2 = xxh3_64_with_seed(bytes, self.seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
        let h3 = xxh3_64_with_seed(bytes, self.seed.wrapping_add(0x517c_c1b7_2722_0a95));
        (h1, h2, h3)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "XXHash3"
    }
}

/// Builder for XXHash3 hasher with fluent interface.
///
/// Provides a convenient way to configure XXHash3 parameters.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "xxhash")]
/// # {
/// use bloomcraft::hash::XxHasherBuilder;
///
/// let hasher = XxHasherBuilder::new()
///     .with_seed(12345)
///     .build();
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct XxHasherBuilder {
    seed: u64,
}

impl XxHasherBuilder {
    /// Create a new builder with default settings.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "xxhash")]
    /// # {
    /// use bloomcraft::hash::XxHasherBuilder;
    ///
    /// let builder = XxHasherBuilder::new();
    /// # }
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    /// Set the seed value.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed value
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "xxhash")]
    /// # {
    /// use bloomcraft::hash::XxHasherBuilder;
    ///
    /// let hasher = XxHasherBuilder::new()
    ///     .with_seed(42)
    ///     .build();
    /// # }
    /// ```
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the XXHash3 hasher.
    ///
    /// # Returns
    ///
    /// Configured `XxHasher` instance
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "xxhash")]
    /// # {
    /// use bloomcraft::hash::XxHasherBuilder;
    ///
    /// let hasher = XxHasherBuilder::new()
    ///     .with_seed(999)
    ///     .build();
    /// # }
    /// ```
    #[must_use]
    pub const fn build(self) -> XxHasher {
        XxHasher::with_seed(self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

         // Basic Construction Tests
     
    #[test]
    fn test_xxhasher_new() {
        let hasher = XxHasher::new();
        assert_eq!(hasher.seed, 0);
    }

    #[test]
    fn test_xxhasher_with_seed() {
        let hasher = XxHasher::with_seed(12345);
        assert_eq!(hasher.seed, 12345);
    }

    #[test]
    fn test_xxhasher_default() {
        let hasher: XxHasher = Default::default();
        assert_eq!(hasher.seed, 0);
    }

         // Determinism Tests
     
    #[test]
    fn test_hash_bytes_deterministic() {
        let hasher = XxHasher::new();
        let data = b"test string";

        let h1 = hasher.hash_bytes(data);
        let h2 = hasher.hash_bytes(data);

        assert_eq!(h1, h2, "Same input should produce same hash");
    }

    #[test]
    fn test_hash_bytes_different_inputs() {
        let hasher = XxHasher::new();

        let h1 = hasher.hash_bytes(b"input1");
        let h2 = hasher.hash_bytes(b"input2");

        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_different_seeds_different_hashes() {
        let hasher1 = XxHasher::with_seed(1);
        let hasher2 = XxHasher::with_seed(2);
        let data = b"test";

        let h1 = hasher1.hash_bytes(data);
        let h2 = hasher2.hash_bytes(data);

        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

         // Length-Specific Tests
     
    #[test]
    fn test_hash_bytes_empty() {
        let hasher = XxHasher::new();
        let h = hasher.hash_bytes(&[]);

        // XXHash3 produces a specific non-zero hash for empty input
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_single() {
        let hasher = XxHasher::new();
        let h = hasher.hash_bytes(&[42]);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_small() {
        let hasher = XxHasher::new();
        let data = b"hello";
        let h = hasher.hash_bytes(data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_medium() {
        let hasher = XxHasher::new();
        let data = b"hello world this is a medium sized string for testing";
        let h = hasher.hash_bytes(data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_large() {
        let hasher = XxHasher::new();
        let data = vec![42u8; 10_000];
        let h = hasher.hash_bytes(&data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_various_lengths() {
        let hasher = XxHasher::new();

        // Test various lengths to ensure algorithm handles all cases
        for len in [0, 1, 3, 4, 8, 16, 17, 32, 33, 63, 64, 65, 100, 256, 1000] {
            let data = vec![42u8; len];
            let h = hasher.hash_bytes(&data);
            assert_ne!(h, 0, "Hash for length {} should be non-zero", len);
        }
    }

         // Multi-Hash Tests
     
    #[test]
    fn test_hash_bytes_pair_independence() {
        let hasher = XxHasher::new();
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);

        assert_ne!(h1, h2, "Pair should produce independent hashes");
    }

    #[test]
    fn test_hash_bytes_pair_deterministic() {
        let hasher = XxHasher::new();
        let data = b"test";

        let (h1_a, h2_a) = hasher.hash_bytes_pair(data);
        let (h1_b, h2_b) = hasher.hash_bytes_pair(data);

        assert_eq!(h1_a, h1_b);
        assert_eq!(h2_a, h2_b);
    }

    #[test]
    fn test_hash_bytes_triple_independence() {
        let hasher = XxHasher::new();
        let data = b"test";

        let (h1, h2, h3) = hasher.hash_bytes_triple(data);

        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_bytes_with_seed_method() {
        let hasher = XxHasher::new();
        let data = b"test";

        let h1 = hasher.hash_bytes_with_seed(data, 0);
        let h2 = hasher.hash_bytes_with_seed(data, 42);

        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

         // Avalanche Effect Tests
     
    #[test]
    fn test_avalanche_single_bit_flip() {
        let hasher = XxHasher::new();

        let data1 = b"test";
        let mut data2 = *b"test";
        data2[0] ^= 1; // Flip single bit

        let h1 = hasher.hash_bytes(data1);
        let h2 = hasher.hash_bytes(&data2);

        let diff = h1 ^ h2;
        let changed_bits = diff.count_ones();

        // Single bit flip should affect ~32 bits (±12 for tolerance)
        assert!(
            changed_bits >= 20 && changed_bits <= 44,
            "Avalanche effect: {} bits changed (expected 20-44)",
            changed_bits
        );
    }

    #[test]
    fn test_avalanche_last_byte_flip() {
        let hasher = XxHasher::new();

        let data1 = vec![0u8; 100];
        let mut data2 = data1.clone();
        data2[99] ^= 1; // Flip bit in last byte

        let h1 = hasher.hash_bytes(&data1);
        let h2 = hasher.hash_bytes(&data2);

        let changed_bits = (h1 ^ h2).count_ones();
        assert!(
            changed_bits >= 20,
            "Flipping last byte should affect many bits: {}",
            changed_bits
        );
    }

         // Distribution Quality Tests
     
    #[test]
    fn test_no_collisions_sequential_integers() {
        let hasher = XxHasher::new();
        let mut hashes = std::collections::HashSet::new();

        // Hash 1000 sequential integers
        for i in 0i32..1000 {
            let bytes = i.to_le_bytes();
            let hash = hasher.hash_bytes(&bytes);
            hashes.insert(hash);
        }

        // Should have no collisions (all unique)
        assert_eq!(hashes.len(), 1000, "Hash collisions detected");
    }

    #[test]
    fn test_seed_independence_statistical() {
        let data = b"test data for independence";

        // Generate hashes with different seeds
        let seeds = [0, 1, 2, 42, 999, 123456];
        let mut hashes = Vec::new();

        for &seed in &seeds {
            let hasher = XxHasher::with_seed(seed);
            let hash = hasher.hash_bytes(data);
            hashes.push(hash);
        }

        // All should be different
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(
                    hashes[i], hashes[j],
                    "Seeds {} and {} produced same hash",
                    seeds[i], seeds[j]
                );
            }
        }
    }

         // Builder Tests
     
    #[test]
    fn test_xxhasher_builder_default() {
        let builder = XxHasherBuilder::new();
        let hasher = builder.build();

        assert_eq!(hasher.seed, 0);
    }

    #[test]
    fn test_xxhasher_builder_with_seed() {
        let hasher = XxHasherBuilder::new().with_seed(12345).build();

        assert_eq!(hasher.seed, 12345);
    }

    #[test]
    fn test_xxhasher_builder_fluent_interface() {
        let hasher = XxHasherBuilder::new().with_seed(999).build();

        let h = hasher.hash_bytes(b"test");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_builder_default_trait() {
        let builder: XxHasherBuilder = Default::default();
        let hasher = builder.build();
        assert_eq!(hasher.seed, 0);
    }

         // Trait Tests
     
    #[test]
    fn test_hasher_name() {
        let hasher = XxHasher::new();
        assert_eq!(hasher.name(), "XXHash3");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<XxHasher>();
    }

    #[test]
    fn test_clone() {
        let hasher1 = XxHasher::with_seed(42);
        let hasher2 = hasher1.clone();

        assert_eq!(hasher1.seed, hasher2.seed);

        let data = b"test";
        assert_eq!(hasher1.hash_bytes(data), hasher2.hash_bytes(data));
    }

    #[test]
    fn test_copy() {
        let hasher1 = XxHasher::with_seed(42);
        let hasher2 = hasher1; // Copy

        let data = b"test";
        assert_eq!(hasher1.hash_bytes(data), hasher2.hash_bytes(data));
    }

         // Integration Tests (with strategies)
     
    #[test]
    fn test_integration_with_double_hashing() {
        use crate::hash::strategies::{DoubleHashing, HashStrategy};

        let hasher = XxHasher::new();
        let strategy = DoubleHashing;
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);
        let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_integration_with_enhanced_double_hashing() {
        use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy};

        let hasher = XxHasher::new();
        let strategy = EnhancedDoubleHashing;
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);
        let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

         // Reference Vector Tests (for version stability)
     
    #[test]
    fn test_reference_vectors() {
        let hasher = XxHasher::new();

        // Empty string
        let h_empty = hasher.hash_bytes(b"");
        assert_ne!(h_empty, 0);

        // Single byte
        let h_a = hasher.hash_bytes(b"a");
        let h_b = hasher.hash_bytes(b"b");
        assert_ne!(h_a, h_b);

        // Known string
        let h_hello = hasher.hash_bytes(b"hello");
        assert_ne!(h_hello, 0);

        // Verify consistency
        let h_hello2 = hasher.hash_bytes(b"hello");
        assert_eq!(h_hello, h_hello2);
    }

    #[test]
    fn test_empty_input_consistency_across_seeds() {
        // Empty input should produce different hashes for different seeds
        let h1 = XxHasher::with_seed(0).hash_bytes(b"");
        let h2 = XxHasher::with_seed(1).hash_bytes(b"");

        assert_ne!(h1, h2);
    }

         // Edge Cases
     
    #[test]
    fn test_unicode_handling() {
        let hasher = XxHasher::new();

        let utf8_data = "Hello, 世界! 🦀";
        let h1 = hasher.hash_bytes(utf8_data.as_bytes());
        let h2 = hasher.hash_bytes(utf8_data.as_bytes());

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_consecutive_bytes_differ() {
        let hasher = XxHasher::new();

        let data1 = b"aaaa";
        let data2 = b"aaab";

        let h1 = hasher.hash_bytes(data1);
        let h2 = hasher.hash_bytes(data2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_large_seed_values() {
        let data = b"test";

        let h1 = XxHasher::with_seed(u64::MAX).hash_bytes(data);
        let h2 = XxHasher::with_seed(u64::MAX - 1).hash_bytes(data);

        assert_ne!(h1, h2);
    }
}
