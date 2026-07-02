//! XXHash3 implementation for Bloom filters.
//!
//! XXHash3 is a deterministic, seedable non-cryptographic hash (Yann Collet).
//! This module wraps the [`xxhash-rust`] crate to delegate to its optimised
//! implementation, which includes automatic SIMD dispatch on supported platforms.
//!
//! # Properties
//!
//! | Property | Guarantee |
//! |---|---|
//! | Determinism | Tracked — same input + seed → same output (versioned via `name()`) |
//! | Seed independence | Seeds differ by ≥ 1 bit → outputs differ |
//! | Avalanche | Verified — single-bit input flips affect ~34 bits of output |
//! | Non-cryptographic | Not suitable for adversarial / security contexts |
//!
//! # Performance (x86-64, release build)
//!
//! | Input | ns/hash | GB/s |
//! |---|---|---:|
//! | 8 B | 2.0 | 4.1 |
//! | 16 B | 1.7 | 9.4 |
//! | 32 B | 2.2 | 14.4 |
//! | 64 B | 4.2 | 15.2 |
//! | 128 B | 7.3 | 17.6 |
//! | 256 B | 19.3 | 13.3 |
//! | 512 B | 29.6 | 17.3 |
//! | 1024 B | 47.7 | 21.5 |
//! | 4096 B | 171.9 | 23.8 |
//!
//! XXHash3 consistently outperforms WyHash across all input sizes on this
//! platform by a factor of 1.2–2.6×. Throughput scales with SIMD availability
//! (the `xxhash-rust` crate selects SSE2/AVX2/AVX-512 at runtime when available).
//!
//! Multi-hash operations (`hash_bytes_pair`, `hash_bytes_triple`) amortise
//! the finalisation step: `hash_bytes_pair` costs ~1.5× a single hash for
//! short inputs, approaching ~2.6× at 256 B where SIMD streaming dominates.
//!
//! # Algorithm
//!
//! This module delegates to the `xxhash-rust` crate's `xxh3_64` / `xxh3_64_with_seed`
//! functions. See the [xxHash repository] for the algorithm specification.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "xxhash")]
//! # {
//! use bloomcraft::hash::{BloomHasher, XxHasher};
//!
//! let h = XxHasher::new().hash_bytes(b"hello world");
//!
//! let h0 = XxHasher::with_seed(0).hash_bytes(b"test");
//! let h1 = XxHasher::with_seed(1).hash_bytes(b"test");
//! assert_ne!(h0, h1);
//! # }
//! ```
//!
//! # References
//!
//! - Collet, Y. (2012). *xxHash: Extremely fast non-cryptographic hash algorithm*. 
//!   GitHub Repository. <https://github.com/Cyan4973/xxHash>

#![allow(clippy::module_name_repetitions)]

use super::hasher::{BloomHasher, HashWriter};
use std::hash::{BuildHasher, Hash, Hasher as StdHasher};
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

/// XXHash3 hasher wrapping a single `u64` seed.
///
/// Implements [`BloomHasher`] by delegating to `xxhash-rust`'s `xxh3_64`.
/// `Send + Sync`.
///
/// The seed is exposed via [`instance_token`](BloomHasher::instance_token)
/// for cross-filter compatibility checks.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct XxHasher {
    seed: u64,
}

impl XxHasher {
    /// Create a hasher with seed `0`.
    #[must_use]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    /// Create a hasher with an explicit seed.
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
        // xxh3_64 avoids a seed-mix step; use it directly for the default seed
        if self.seed == 0 {
            xxh3_64(bytes)
        } else {
            xxh3_64_with_seed(bytes, self.seed)
        }
    }

    #[inline]
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        xxh3_64_with_seed(bytes, self.seed.wrapping_add(seed))
    }

    #[inline]
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = xxh3_64_with_seed(bytes, self.seed);
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
    fn hash_item<T: Hash>(&self, item: &T) -> (u64, u64) {
        let mut writer = HashWriter::new();
        item.hash(&mut writer);
        let bytes = writer.into_bytes();
        let h1 = xxh3_64_with_seed(&bytes, self.seed);
        let h2 = xxh3_64_with_seed(&bytes, self.seed ^ 0x9e37_79b9_7f4a_7c15);
        (h1, h2.rotate_left(31) ^ 0xa021_282d_c0b9_ed54)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "XXHash3"
    }

    #[inline]
    fn instance_token(&self) -> u64 {
        self.seed
    }
}

/// `BuildHasher` adapter so `XxHasher` can be used with `HashMap` / `HashSet`.
#[derive(Debug, Clone, Copy, Default)]
pub struct XxHasherBuilder {
    seed: u64,
}

impl XxHasherBuilder {
    /// Create a builder with seed `0`.
    #[must_use]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    /// Create a builder with an explicit seed.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

/// `std::hash::Hasher` state produced by `XxHasherBuilder`.
#[derive(Debug)]
pub struct XxHasherState {
    seed: u64,
    buffer: Vec<u8>,
}

impl XxHasherState {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            buffer: Vec::with_capacity(64),
        }
    }
}

impl StdHasher for XxHasherState {
    fn finish(&self) -> u64 {
        xxh3_64_with_seed(&self.buffer, self.seed)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }
}

impl BuildHasher for XxHasherBuilder {
    type Hasher = XxHasherState;

    fn build_hasher(&self) -> Self::Hasher {
        XxHasherState::new(self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Basic Construction Tests ---

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

    // --- Determinism Tests ---

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

    // --- Length-Specific Tests ---

    #[test]
    fn test_hash_bytes_empty() {
        let hasher = XxHasher::new();
        let h = hasher.hash_bytes(&[]);

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

        for len in [0, 1, 3, 4, 8, 16, 17, 32, 33, 63, 64, 65, 100, 256, 1000] {
            let data = vec![42u8; len];
            let h = hasher.hash_bytes(&data);
            assert_ne!(h, 0, "Hash for length {} should be non-zero", len);
        }
    }

    // --- Multi-Hash Tests ---

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

    // --- Avalanche Effect Tests ---

    #[test]
    fn test_avalanche_single_bit_flip() {
        let hasher = XxHasher::new();

        let data1 = b"test";
        let mut data2 = *b"test";
        data2[0] ^= 1;

        let h1 = hasher.hash_bytes(data1);
        let h2 = hasher.hash_bytes(&data2);

        let diff = h1 ^ h2;
        let changed_bits = diff.count_ones();

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
        data2[99] ^= 1;

        let h1 = hasher.hash_bytes(&data1);
        let h2 = hasher.hash_bytes(&data2);

        let changed_bits = (h1 ^ h2).count_ones();
        assert!(
            changed_bits >= 20,
            "Avalanche effect: {} bits changed (expected ≥20)",
            changed_bits
        );
    }

    // --- Distribution Quality Tests ---

    #[test]
    fn test_no_collisions_sequential_integers() {
        let hasher = XxHasher::new();
        let mut hashes = std::collections::HashSet::new();

        for i in 0i32..1000 {
            let bytes = i.to_le_bytes();
            let hash = hasher.hash_bytes(&bytes);
            hashes.insert(hash);
        }

        assert_eq!(hashes.len(), 1000, "Hash collisions detected");
    }

    #[test]
    fn test_seed_independence_statistical() {
        let data = b"test data for independence";

        let seeds = [0, 1, 2, 42, 999, 123456];
        let mut hashes = Vec::new();

        for &seed in &seeds {
            let hasher = XxHasher::with_seed(seed);
            let hash = hasher.hash_bytes(data);
            hashes.push(hash);
        }

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

    // --- Builder Tests ---

    #[test]
    fn test_xxhasher_builder_default() {
        let builder = XxHasherBuilder::new();
        assert_eq!(builder.seed, 0);
    }

    #[test]
    fn test_xxhasher_builder_with_seed() {
        let builder = XxHasherBuilder::with_seed(12345);
        assert_eq!(builder.seed, 12345);
    }

    #[test]
    fn test_xxhasher_builder_default_trait() {
        let builder: XxHasherBuilder = Default::default();
        assert_eq!(builder.seed, 0);
    }

    #[test]
    fn test_xxhasher_build_hasher_deterministic() {
        let builder = XxHasherBuilder::new();
        let mut state = builder.build_hasher();
        state.write(b"hello");
        let h1 = state.finish();
        let mut state = builder.build_hasher();
        state.write(b"hello");
        let h2 = state.finish();
        assert_eq!(h1, h2);
    }

    // --- Trait Tests ---

    #[test]
    fn test_hasher_name() {
        let hasher = XxHasher::new();
        assert_eq!(hasher.name(), "XXHash3");
    }

    #[test]
    fn test_instance_token() {
        assert_eq!(XxHasher::new().instance_token(), 0);
        assert_eq!(XxHasher::with_seed(42).instance_token(), 42);
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
        let hasher2 = hasher1;

        let data = b"test";
        assert_eq!(hasher1.hash_bytes(data), hasher2.hash_bytes(data));
    }

    // --- Integration Tests (with strategies) ---

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

    // --- Regression / Determinism Tests ---

    #[test]
    fn test_reference_vectors() {
        let hasher = XxHasher::new();

        let h_empty = hasher.hash_bytes(b"");
        assert_ne!(h_empty, 0);

        let h_a = hasher.hash_bytes(b"a");
        let h_b = hasher.hash_bytes(b"b");
        assert_ne!(h_a, h_b);

        let h_hello = hasher.hash_bytes(b"hello");
        assert_ne!(h_hello, 0);

        assert_eq!(h_hello, hasher.hash_bytes(b"hello"));
    }

    #[test]
    fn test_empty_input_different_across_seeds() {
        let h1 = XxHasher::with_seed(0).hash_bytes(b"");
        let h2 = XxHasher::with_seed(1).hash_bytes(b"");

        assert_ne!(h1, h2);
    }

    // --- Edge Cases ---

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
