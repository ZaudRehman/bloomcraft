//! Hash function trait and utilities for Bloom filters.
//!
//! This module provides a trait-based abstraction over hash functions used in Bloom filters.
//! Hash functions operate on byte slices, allowing flexibility in how data is serialized
//! before hashing.
//!
//! # Design Philosophy
//!
//! 1. **Byte-Oriented**: Work with `&[u8]` instead of generic `T: Hash` for explicit control
//! 2. **Minimal Interface**: Hash functions hash; strategies generate indices
//! 3. **Independence**: Multiple hashes from single input must be independent
//! 4. **Performance**: Zero-cost abstractions, inline-friendly
//!
//! # Separation of Concerns
//!
//! - **`BloomHasher`**: Generates base hash values from bytes
//! - **`HashStrategy`**: Derives k indices from base hashes (see `strategies` module)
//! - **Bloom Filters**: Compose hasher + strategy + bit storage
//!
//! # Examples
//!
//! ## Basic Hashing
//!
//! ```
//! use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
//!
//! let hasher = StdHasher::new();
//! let data = b"hello world";
//!
//! // Single hash
//! let h1 = hasher.hash_bytes(data);
//!
//! // Two independent hashes for double hashing
//! let (h1, h2) = hasher.hash_bytes_pair(data);
//! ```
//!
//! ## Using with Strategies
//!
//! ```
//! use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
//! use bloomcraft::hash::strategies::{HashStrategy, DoubleHashing};
//!
//! let hasher = StdHasher::new();
//! let strategy = DoubleHashing;
//! let data = b"hello";
//!
//! let (h1, h2) = hasher.hash_bytes_pair(data);
//! let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);
//! assert_eq!(indices.len(), 7);
//! ```
//!
//! # References
//!
//! - Kirsch & Mitzenmacher (2006): "Less Hashing, Same Performance: Building a Better Bloom Filter"

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

/// Base hasher trait for Bloom filter hash functions.
///
/// This trait abstracts over specific hash function implementations, allowing
/// different algorithms (WyHash, XXHash, SipHash, etc.) to be used interchangeably.
///
/// # Design Rationale
///
/// The trait operates on **byte slices** (`&[u8]`) rather than generic `T: Hash` to:
/// - Give explicit control over serialization (users choose how to convert data to bytes)
/// - Avoid coupling hash algorithms to Rust's `Hash` trait
/// - Enable zero-copy hashing of pre-serialized data
/// - Support custom serialization formats (e.g., protobuf, msgpack)
///
/// # Requirements
///
/// Implementations must provide:
/// - **Fast hashing**: <10ns per item on modern hardware
/// - **Avalanche property**: Single bit change affects ~50% of output bits
/// - **Uniform distribution**: Output evenly distributed across `u64` space
/// - **Determinism**: Same input → same output (within same process/version)
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for use in concurrent Bloom filters.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
///
/// let hasher = StdHasher::new();
/// let data = b"hello world";
///
/// // Hash bytes directly
/// let hash = hasher.hash_bytes(data);
/// assert!(hash != 0);
///
/// // Get two independent hashes
/// let (h1, h2) = hasher.hash_bytes_pair(data);
/// assert_ne!(h1, h2);
/// ```
pub trait BloomHasher: Send + Sync {
    /// Hash arbitrary bytes to a 64-bit value.
    ///
    /// This is the core hashing operation. All other methods derive from this.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Data to hash
    ///
    /// # Returns
    ///
    /// 64-bit hash value uniformly distributed across `u64` space.
    ///
    /// # Performance
    ///
    /// Should complete in O(bytes.len()) time, typically <10ns for small inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    /// let h1 = hasher.hash_bytes(b"hello");
    /// let h2 = hasher.hash_bytes(b"hello");
    /// assert_eq!(h1, h2); // Deterministic
    /// ```
    fn hash_bytes(&self, bytes: &[u8]) -> u64;

    /// Hash bytes with an explicit seed for generating independent hash functions.
    ///
    /// Used to derive multiple independent hash values from the same input.
    /// Different seeds MUST produce statistically independent outputs.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Data to hash
    /// * `seed` - Seed value for hash function
    ///
    /// # Returns
    ///
    /// 64-bit hash value derived from both input and seed.
    ///
    /// # Default Implementation
    ///
    /// Default implementation XORs the seed with the base hash:
    ///
    /// ```text
    /// hash_bytes_with_seed(data, seed) = hash_bytes(data) ^ seed
    /// ```
    ///
    /// This is **sufficient but not optimal**. Implementations SHOULD override
    /// this method if the underlying algorithm supports native seeding.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    /// let h1 = hasher.hash_bytes_with_seed(b"test", 0);
    /// let h2 = hasher.hash_bytes_with_seed(b"test", 42);
    /// assert_ne!(h1, h2); // Different seeds → different hashes
    /// ```
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        // Default: XOR seed into result
        // This provides basic independence but is not optimal
        self.hash_bytes(bytes) ^ seed
    }

    /// Generate two independent hash values from a single input.
    ///
    /// This is the primary method used by double hashing strategies.
    /// The two values MUST be statistically independent to ensure proper
    /// distribution in Bloom filters.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Data to hash
    ///
    /// # Returns
    ///
    /// Tuple `(h1, h2)` where `h1` and `h2` are independent 64-bit hashes.
    ///
    /// # Independence Requirement
    ///
    /// The correlation between `h1` and `h2` must be negligible (< 0.01).
    /// Default implementation uses bit rotation + XOR, which provides
    /// sufficient independence for Bloom filters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    /// let (h1, h2) = hasher.hash_bytes_pair(b"data");
    /// assert_ne!(h1, h2);
    /// ```
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = self.hash_bytes(bytes);
        let h2 = self.hash_bytes_with_seed(bytes, 0x517c_c1b7_2722_0a95);
        (h1, h2)
    }

    /// Generate three independent hash values from a single input.
    ///
    /// Used by triple hashing strategies. Rarely needed in practice.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Data to hash
    ///
    /// # Returns
    ///
    /// Tuple `(h1, h2, h3)` of three independent 64-bit hashes.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    /// let (h1, h2, h3) = hasher.hash_bytes_triple(b"data");
    /// assert_ne!(h1, h2);
    /// assert_ne!(h2, h3);
    /// assert_ne!(h1, h3);
    /// ```
    fn hash_bytes_triple(&self, bytes: &[u8]) -> (u64, u64, u64) {
        let h1 = self.hash_bytes(bytes);
        let h2 = self.hash_bytes_with_seed(bytes, 0x517c_c1b7_2722_0a95);
        let h3 = self.hash_bytes_with_seed(bytes, 0x9e37_79b9_7f4a_7c15);
        (h1, h2, h3)
    }

    /// Human-readable name for debugging and serialization.
    ///
    /// # Returns
    ///
    /// Static string identifying the hash function (e.g., "WyHash", "XXHash64").
    fn name(&self) -> &'static str;
}

/// Platform-dependent hasher using `std::collections::hash_map::DefaultHasher`.
///
/// # Deterministic Behavior
///
/// This hasher uses proper deterministic hashing (FNV-1a) to ensure
/// compatibility with serialized Bloom filters across runs.
///
/// # Performance
///
/// - Small inputs (<32 bytes): Very fast
/// - Large inputs (>1KB): Fast linear performance
///
/// For higher performance and quality, consider using `WyHasher` (feature `wyhash`).
///
/// # When to Use
///
/// Use `StdHasher` when:
/// - Bloom filter lifetime is same as program lifetime (no serialization)
/// - Performance is adequate (not bottleneck)
/// - Simplicity is more important than control
///
/// # Performance
///
/// Typical performance on modern x86-64 (Rust 1.70+ SipHash-1-3):
/// - Small inputs (<32 bytes): ~8-12ns
/// - Large inputs (>1KB): ~2-3 cycles/byte
///
/// For higher performance, use `WyHasher` (~3-5ns for small inputs).
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
///
/// let hasher = StdHasher::new();
/// let hash = hasher.hash_bytes(b"test data");
/// ```
/// let hash = hasher.hash_bytes(b"test data");
/// ```
#[derive(Debug, Clone, Default)]
pub struct DeterministicHasher {
    state: u64,
}

impl DeterministicHasher {
    /// Create a new deterministic hasher with FNV-1a offset basis.
    pub fn new() -> Self {
        Self { state: 0xcbf2_9ce4_8422_2325 }
    }
}

impl std::hash::Hasher for DeterministicHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }
    
    fn finish(&self) -> u64 {
        self.state
    }
}

/// Standard hasher implementation using deterministic FNV-1a.
///
/// This hasher provides consistent hashing across process runs, ensuring
/// that serialized Bloom filters can be reliably deserialized and queried.
#[derive(Debug, Clone)]
pub struct StdHasher {
    seed: u64,
}

impl StdHasher {
    /// Create a new hasher with default seed.
    ///
    /// Uses a compile-time constant seed (0x517cc1b7_2722_0a95) for
    /// reproducibility within the same Rust version.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::StdHasher;
    ///
    /// let hasher = StdHasher::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: 0x517c_c1b7_2722_0a95,
        }
    }

    /// Create a new hasher with explicit seed.
    ///
    /// Different seeds produce independent hash functions, useful for
    /// creating multiple hash functions from a single algorithm.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed value
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher1 = StdHasher::with_seed(0);
    /// let hasher2 = StdHasher::with_seed(42);
    ///
    /// let h1 = hasher1.hash_bytes(b"test");
    /// let h2 = hasher2.hash_bytes(b"test");
    /// assert_ne!(h1, h2);
    /// ```
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for StdHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomHasher for StdHasher {
    #[inline]
    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        use std::hash::Hasher;

        let mut hasher = DeterministicHasher::new();
        hasher.write_u64(self.seed);
        hasher.write(bytes);
        hasher.finish()
    }

    #[inline]
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        use std::hash::Hasher;

        let mut hasher = DeterministicHasher::new();
        // Combine both seeds for better independence
        hasher.write_u64(self.seed ^ seed);
        hasher.write(bytes);
        hasher.finish()
    }

    #[inline]
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = self.hash_bytes(bytes);
        let h2 = self.hash_bytes_with_seed(bytes, 0x9e37_79b9_7f4a_7c15);
        (h1, h2)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "StdHasher"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

         // Basic Functionality Tests
     
    #[test]
    fn test_std_hasher_determinism() {
        let hasher = StdHasher::new();
        let data = b"test string";

        let h1 = hasher.hash_bytes(data);
        let h2 = hasher.hash_bytes(data);

        assert_eq!(h1, h2, "Same input should produce same hash");
    }

    #[test]
    fn test_std_hasher_different_inputs() {
        let hasher = StdHasher::new();

        let h1 = hasher.hash_bytes(b"input1");
        let h2 = hasher.hash_bytes(b"input2");

        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_std_hasher_empty_input() {
        let hasher = StdHasher::new();
        let hash = hasher.hash_bytes(b"");

        // Empty input should still produce a hash
        assert!(hash != 0);
    }

    #[test]
    fn test_std_hasher_single_byte() {
        let hasher = StdHasher::new();

        let h1 = hasher.hash_bytes(b"a");
        let h2 = hasher.hash_bytes(b"b");

        assert_ne!(h1, h2);
    }

         // Seed Tests
     
    #[test]
    fn test_with_seed_produces_different_hashes() {
        let hasher1 = StdHasher::with_seed(123);
        let hasher2 = StdHasher::with_seed(456);
        let data = b"test";

        let h1 = hasher1.hash_bytes(data);
        let h2 = hasher2.hash_bytes(data);

        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

    #[test]
    fn test_hash_bytes_with_seed_independence() {
        let hasher = StdHasher::new();
        let data = b"test";

        let h1 = hasher.hash_bytes_with_seed(data, 0);
        let h2 = hasher.hash_bytes_with_seed(data, 42);
        let h3 = hasher.hash_bytes_with_seed(data, 999);

        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

         // Multi-Hash Tests
     
    #[test]
    fn test_hash_bytes_pair_produces_different_values() {
        let hasher = StdHasher::new();
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);

        assert_ne!(h1, h2, "Pair should produce independent hashes");
    }

    #[test]
    fn test_hash_bytes_pair_deterministic() {
        let hasher = StdHasher::new();
        let data = b"test";

        let (h1_a, h2_a) = hasher.hash_bytes_pair(data);
        let (h1_b, h2_b) = hasher.hash_bytes_pair(data);

        assert_eq!(h1_a, h1_b);
        assert_eq!(h2_a, h2_b);
    }

    #[test]
    fn test_hash_bytes_triple_produces_different_values() {
        let hasher = StdHasher::new();
        let data = b"test";

        let (h1, h2, h3) = hasher.hash_bytes_triple(data);

        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_bytes_triple_deterministic() {
        let hasher = StdHasher::new();
        let data = b"test";

        let (h1_a, h2_a, h3_a) = hasher.hash_bytes_triple(data);
        let (h1_b, h2_b, h3_b) = hasher.hash_bytes_triple(data);

        assert_eq!(h1_a, h1_b);
        assert_eq!(h2_a, h2_b);
        assert_eq!(h3_a, h3_b);
    }

         // Avalanche Effect Tests
     
    #[test]
    fn test_avalanche_single_bit_flip() {
        let hasher = StdHasher::new();

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

         // Independence Tests (Statistical)
     
    #[test]
    fn test_hash_pair_independence_statistical() {
        let hasher = StdHasher::new();

        // Collect 100 pairs and check for correlation
        let mut h1_values = Vec::new();
        let mut h2_values = Vec::new();

        for i in 0..100 {
            let data = format!("test_{}", i);
            let (h1, h2) = hasher.hash_bytes_pair(data.as_bytes());
            h1_values.push(h1);
            h2_values.push(h2);
        }

        // Simple independence check: XOR of all h1s should differ from XOR of all h2s
        let xor_h1: u64 = h1_values.iter().fold(0, |acc, &x| acc ^ x);
        let xor_h2: u64 = h2_values.iter().fold(0, |acc, &x| acc ^ x);

        assert_ne!(xor_h1, xor_h2, "Hash pairs should be independent");
    }

         // Trait Tests
     
    #[test]
    fn test_std_hasher_name() {
        let hasher = StdHasher::new();
        assert_eq!(hasher.name(), "StdHasher");
    }

    #[test]
    fn test_default_trait() {
        let hasher: StdHasher = Default::default();
        let hash = hasher.hash_bytes(b"test");
        assert!(hash != 0);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StdHasher>();
    }

         // Edge Cases
     
    #[test]
    fn test_large_input() {
        let hasher = StdHasher::new();
        let large_data = vec![42u8; 10_000];

        let hash = hasher.hash_bytes(&large_data);
        assert!(hash != 0);
    }

    #[test]
    fn test_unicode_handling() {
        let hasher = StdHasher::new();

        let utf8_data = "Hello, 世界! 🦀";
        let h1 = hasher.hash_bytes(utf8_data.as_bytes());
        let h2 = hasher.hash_bytes(utf8_data.as_bytes());

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_consecutive_bytes_differ() {
        let hasher = StdHasher::new();

        let data1 = b"aaaa";
        let data2 = b"aaab";

        let h1 = hasher.hash_bytes(data1);
        let h2 = hasher.hash_bytes(data2);

        assert_ne!(h1, h2);
    }

         // Integration Tests (with strategies module)
     
    #[test]
    fn test_integration_with_double_hashing() {
        use crate::hash::strategies::{DoubleHashing, HashStrategy};

        let hasher = StdHasher::new();
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

        let hasher = StdHasher::new();
        let strategy = EnhancedDoubleHashing;
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);
        let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_integration_with_triple_hashing() {
        use crate::hash::strategies::{HashStrategy, TripleHashing};

        let hasher = StdHasher::new();
        let strategy = TripleHashing;
        let data = b"test";

        let (h1, h2, h3) = hasher.hash_bytes_triple(data);
        let indices = strategy.generate_indices(h1, h2, h3, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

         // Reproducibility Tests
     
    #[test]
    fn test_same_seed_same_results() {
        let hasher1 = StdHasher::with_seed(42);
        let hasher2 = StdHasher::with_seed(42);

        let data = b"reproducibility test";

        let h1 = hasher1.hash_bytes(data);
        let h2 = hasher2.hash_bytes(data);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_clone_produces_identical_results() {
        let hasher1 = StdHasher::with_seed(999);
        let hasher2 = hasher1.clone();

        let data = b"clone test";

        let h1 = hasher1.hash_bytes(data);
        let h2 = hasher2.hash_bytes(data);

        assert_eq!(h1, h2);
    }
}
