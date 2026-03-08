//! Hash function trait and default implementation for Bloom filters.
//!
//! This module defines [`BloomHasher`] — the core hashing abstraction — and
//! [`StdHasher`], a deterministic FNV-1a implementation suitable for production use.
//!
//! # Design
//!
//! `BloomHasher` is **byte-oriented**: all primitive methods take `&[u8]`. This
//! decouples hash algorithms from Rust's `Hash` trait and gives callers explicit
//! control over serialization. For the canonical `T: Hash → (u64, u64)` bridge,
//! use the provided method [`BloomHasher::hash_item`].
//!
//! Concerns are strictly separated:
//! - [`BloomHasher`] produces raw `u64` values from bytes.
//! - [`crate::hash::strategies::HashStrategy`] derives k bit-array indices from
//!   those values.
//! - Bloom filters compose the two.
//!
//! # Hash Algorithm: FNV-1a
//!
//! [`StdHasher`] uses **FNV-1a** (Fowler–Noll–Vo, 64-bit, variant 1a). This is
//! **not** SipHash and is **not** cryptographically secure. It is:
//! - Deterministic across processes and Rust versions (fixed constants)
//! - Suitable for Bloom filters where input sources are trusted
//! - NOT DoS-resistant (predictable output for known inputs)
//!
//! # Security
//!
//! `StdHasher` MUST NOT be used for hash tables exposed to adversarial input.
//! For Bloom filters specifically, adversaries cannot force false negatives without
//! corrupting the bit array, so FNV-1a is acceptable as a default.
//!
//! # Examples
//!
//! ## Byte-Level Hashing
//!
//! ```
//! use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
//!
//! let hasher = StdHasher::new();
//! let (h1, h2) = hasher.hash_bytes_pair(b"hello world");
//! assert_ne!(h1, h2);
//! ```
//!
//! ## Generic Item Hashing
//!
//! ```
//! use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
//!
//! let hasher = StdHasher::new();
//!
//! // Canonical T: Hash → (u64, u64) bridge — zero extra allocation for keys ≤32 bytes
//! let (h1, h2) = hasher.hash_item(&"hello world".to_string());
//! assert_ne!(h1, h2);
//!
//! // Deterministic
//! let (a1, a2) = hasher.hash_item(&42u64);
//! let (b1, b2) = hasher.hash_item(&42u64);
//! assert_eq!(a1, b1);
//! assert_eq!(a2, b2);
//! ```
//!
//! # References
//!
//! - Kirsch & Mitzenmacher (2006): "Less Hashing, Same Performance: Building a Better Bloom Filter"
//! - Fowler, Noll, Vo: FNV Hash — <http://www.isthe.com/chongo/tech/comp/fnv/>

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use smallvec::SmallVec;
use std::hash::Hash;

// ── HashWriter ────────────────────────────────────────────────────────────────

/// Byte-collecting shim bridging `std::hash::Hasher` to `BloomHasher`.
///
/// Used exclusively by [`BloomHasher::hash_item`]. It accumulates every byte
/// written by `T::hash` into a stack-allocated `SmallVec<[u8; 32]>`, spilling
/// to the heap only for keys longer than 32 bytes.
///
/// Stack capacity covers without allocation:
/// - All integer primitives (`u8`–`u128`, 1–16 bytes)
/// - Short strings and byte slices up to 32 bytes
/// - `[u8; N]` for N ≤ 32
///
/// # Why Not Pre-Hash to `u64`?
///
/// Pre-hashing `T` to a `u64` via `DefaultHasher` and then passing the 8-byte
/// result to `hash_bytes` is a double-hash anti-pattern: it collapses full input
/// entropy to 64 bits before the Bloom hasher sees any data. `HashWriter` avoids
/// this by feeding the raw bytes directly to `hash_bytes_pair`.
///
/// # `finish()` Contract
///
/// The `finish() -> u64` method required by `std::hash::Hasher` returns `0` and
/// must never be called on `HashWriter`. The accumulated bytes are retrieved via
/// [`HashWriter::into_bytes`].
pub(crate) struct HashWriter {
    buf: SmallVec<[u8; 32]>,
}

impl HashWriter {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            buf: SmallVec::new(),
        }
    }

    /// Consume the writer and return the accumulated byte buffer.
    #[inline]
    pub(crate) fn into_bytes(self) -> SmallVec<[u8; 32]> {
        self.buf
    }
}

impl std::hash::Hasher for HashWriter {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        // Never called in practice. Returns 0 to make any accidental call
        // obviously wrong rather than silently producing a bad hash.
        0
    }
}

// ── BloomHasher trait ─────────────────────────────────────────────────────────

/// Core hashing abstraction for all Bloom filter variants.
///
/// All filter types in this crate are generic over `H: BloomHasher`. This trait
/// defines the minimal required interface: hash bytes to `u64`, generate
/// independent hash pairs for double hashing, and provide a canonical
/// `T: Hash → (u64, u64)` bridge via [`hash_item`](BloomHasher::hash_item).
///
/// # Byte-Oriented Design
///
/// The primitive methods (`hash_bytes`, `hash_bytes_pair`) operate on `&[u8]`
/// rather than `T: Hash`. This:
/// - Decouples hash algorithms from Rust's `Hash` trait
/// - Allows zero-copy hashing of pre-serialized data
/// - Supports custom serialization formats (protobuf, messagepack, etc.)
///
/// For generic `T: Hash`, use [`hash_item`](BloomHasher::hash_item).
///
/// # Implementing a Custom `BloomHasher`
///
/// You MUST implement:
/// - [`hash_bytes`](BloomHasher::hash_bytes) — the core primitive
/// - [`name`](BloomHasher::name) — stable identifier for compatibility checks
///
/// You SHOULD override:
/// - [`hash_bytes_with_seed`](BloomHasher::hash_bytes_with_seed) — the default
///   XOR-only implementation is insufficient for custom hashers (see Warning below)
/// - [`hash_bytes_pair`](BloomHasher::hash_bytes_pair) — override if your algorithm
///   supports a more efficient two-value derivation
///
/// # Requirements
///
/// Implementations MUST guarantee:
/// - **Determinism**: identical input → identical output, always
/// - **Uniformity**: output is uniformly distributed across `u64` space
/// - **Avalanche**: single input bit flip changes ~50% of output bits
/// - **Independence**: `hash_bytes_pair` returns two statistically uncorrelated values
///
/// # Warning: `hash_bytes_with_seed` Default
///
/// The default implementation is:
/// ```text
/// hash_bytes_with_seed(data, seed) = hash_bytes(data) ^ seed
/// ```
///
/// This provides only XOR-level independence and is **insufficient** for custom
/// implementations. `StdHasher`, `WyHasher`, and `XxHasher` all override this
/// with proper seed mixing. If you implement a custom `BloomHasher`, you MUST
/// override `hash_bytes_with_seed` or `hash_bytes_pair` to avoid correlated
/// hash values in double-hashing strategies.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync`. Hasher state must be immutable
/// after construction (seeded) or protected by atomics.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
///
/// let hasher = StdHasher::new();
///
/// // Raw bytes
/// assert_ne!(hasher.hash_bytes(b"hello"), 0);
///
/// // Independent pair (double hashing input)
/// let (h1, h2) = hasher.hash_bytes_pair(b"hello");
/// assert_ne!(h1, h2);
///
/// // Generic item (canonical bridge)
/// let (h1, h2) = hasher.hash_item(&"hello".to_string());
/// assert_ne!(h1, h2);
/// ```
pub trait BloomHasher: Send + Sync {
    /// Hash a byte slice to a 64-bit value.
    ///
    /// This is the primitive operation from which all other methods derive.
    ///
    /// # Performance
    ///
    /// O(bytes.len()). Typical: <10 ns for inputs ≤64 bytes on modern x86-64.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    /// assert_eq!(hasher.hash_bytes(b"test"), hasher.hash_bytes(b"test")); // deterministic
    /// assert_ne!(hasher.hash_bytes(b"a"),    hasher.hash_bytes(b"b"));    // sensitive
    /// ```
    fn hash_bytes(&self, bytes: &[u8]) -> u64;

    /// Hash bytes with an explicit seed to produce an independent hash value.
    ///
    /// Different seeds MUST produce statistically independent outputs — the
    /// correlation between `hash_bytes_with_seed(data, s1)` and
    /// `hash_bytes_with_seed(data, s2)` for `s1 ≠ s2` must be negligible.
    ///
    /// # Warning: Default Implementation
    ///
    /// The default `hash_bytes(data) ^ seed` provides only XOR-level independence.
    /// Custom implementors MUST override this. See trait-level warning.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    /// let h1 = hasher.hash_bytes_with_seed(b"test", 0);
    /// let h2 = hasher.hash_bytes_with_seed(b"test", 42);
    /// assert_ne!(h1, h2);
    /// ```
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        self.hash_bytes(bytes) ^ seed
    }

    /// Generate two independent 64-bit hashes from a single input.
    ///
    /// The returned `(h1, h2)` are the direct inputs to all double-hashing
    /// strategies. They MUST be statistically independent (correlation < 0.01).
    ///
    /// The default implementation uses a distinct seed constant and bit rotation.
    /// Implementations backed by natively seeded algorithms (WyHash, XXHash3)
    /// SHOULD override this for tighter independence guarantees.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let (h1, h2) = StdHasher::new().hash_bytes_pair(b"data");
    /// assert_ne!(h1, h2);
    /// ```
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = self.hash_bytes(bytes);
        let h2 = self.hash_bytes_with_seed(bytes, 0x517c_c1b7_2722_0a95);
        (h1, h2.rotate_left(31) ^ 0xa021_282d_c0b9_ed54)
    }

    /// Generate three independent 64-bit hashes from a single input.
    ///
    /// Used by [`crate::hash::strategies::TripleHashing`]. Rarely needed outside
    /// research contexts.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let (h1, h2, h3) = StdHasher::new().hash_bytes_triple(b"data");
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

    /// Hash any `T: Hash` item to a pair of independent 64-bit values.
    ///
    /// This is the **canonical bridge** from Rust's `Hash` trait to the
    /// byte-oriented `BloomHasher` API. Every `insert` and `contains` hot path
    /// in this crate calls this method — no ad-hoc `T → bytes` conversions exist
    /// elsewhere.
    ///
    /// # Implementation
    ///
    /// 1. `item.hash(&mut writer)` accumulates all bytes into a `SmallVec<[u8; 32]>`
    ///    (stack-only for keys ≤32 bytes; heap-spill for larger keys).
    /// 2. `hash_bytes_pair` is called on the accumulated bytes.
    ///
    /// This preserves the full entropy of `T`'s `Hash` implementation and avoids
    /// the double-hash anti-pattern (pre-hashing `T` to a `u64` then feeding 8
    /// bytes into a second hash collapses the input space).
    ///
    /// # Performance
    ///
    /// - Keys ≤32 bytes: zero heap allocation (stack SmallVec)
    /// - Keys >32 bytes: one heap allocation for the overflow buffer
    ///
    /// # Determinism
    ///
    /// Output is deterministic across processes and Rust versions when using
    /// `StdHasher`. It depends on the stability of `T`'s `Hash` implementation
    /// across Rust versions — standard library types (`String`, integers, etc.)
    /// are stable.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let hasher = StdHasher::new();
    ///
    /// // String
    /// let (h1, h2) = hasher.hash_item(&"hello".to_string());
    /// assert_ne!(h1, h2);
    ///
    /// // Integer
    /// let (h3, h4) = hasher.hash_item(&42u64);
    /// assert_ne!(h3, h4);
    ///
    /// // Deterministic: same item always produces same pair
    /// let (a1, a2) = hasher.hash_item(&"test".to_string());
    /// let (b1, b2) = hasher.hash_item(&"test".to_string());
    /// assert_eq!(a1, b1);
    /// assert_eq!(a2, b2);
    /// ```
    #[inline]
    fn hash_item<T: Hash>(&self, item: &T) -> (u64, u64) {
        let mut writer = HashWriter::new();
        item.hash(&mut writer);
        let bytes = writer.into_bytes();
        self.hash_bytes_pair(&bytes)
    }

    /// Human-readable identifier for this hash function.
    ///
    /// This string is used by [`crate::core::filter::MergeableBloomFilter::is_compatible`]
    /// to verify that two filters use the same hash function before performing
    /// bitwise union or intersection. Two hasher instances that produce identical
    /// outputs for all inputs MUST return the same name.
    ///
    /// The name MUST be stable across process runs for any hasher used with
    /// serialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// assert_eq!(StdHasher::new().name(), "StdHasher");
    /// ```
    fn name(&self) -> &'static str;
}

// ── DeterministicHasher (crate-internal) ─────────────────────────────────────

/// Internal `std::hash::Hasher` implementation using FNV-1a (64-bit).
///
/// Used by [`StdHasher`] to implement [`std::hash::BuildHasher`], enabling
/// `StdHasher` to be used as a hasher for `HashMap` and `HashSet`.
///
/// # Algorithm
///
/// ```text
/// state = FNV_OFFSET_BASIS  (0xcbf29ce484222325)
/// for each byte b:
///     state ^= b as u64
///     state *= FNV_PRIME     (0x100000001b3)
/// ```
///
/// Deterministic across processes and Rust versions. NOT cryptographically secure.
#[derive(Debug, Clone, Default)]
pub struct DeterministicHasher {
    state: u64,
}

impl DeterministicHasher {
    /// FNV-1a 64-bit offset basis.
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    /// FNV-1a 64-bit prime.
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }
}

impl std::hash::Hasher for DeterministicHasher {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(Self::FNV_PRIME);
        }
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        self.state
    }
}

// ── StdHasher ─────────────────────────────────────────────────────────────────

/// Default Bloom filter hasher using deterministic FNV-1a.
///
/// `StdHasher` is the default `H` in `StandardBloomFilter<T, H = StdHasher>`.
/// It uses FNV-1a with a configurable seed, ensuring consistent output across
/// processes and Rust versions.
///
/// # Algorithm
///
/// FNV-1a (64-bit). The instance seed is mixed into the initial state before
/// processing input bytes. Two `StdHasher` instances with different seeds
/// produce independent hash functions.
///
/// # Performance
///
/// | Input size  | Typical latency (x86-64, release) |
/// |-------------|-----------------------------------|
/// | ≤32 bytes   | ~5–8 ns                           |
/// | 1 KB        | ~100–150 ns                       |
/// | 64 KB       | ~6–8 µs                           |
///
/// For higher throughput, enable the `wyhash` or `xxhash` features.
///
/// # Determinism
///
/// | Scope                                         | Deterministic? |
/// |-----------------------------------------------|----------------|
/// | Within a single run                           | Yes            |
/// | Across multiple processes (same binary)       | Yes            |
/// | Across machines (same crate version)          | Yes            |
/// | Across Rust compiler versions                 | Yes (FNV, not SipHash) |
/// | Across crate versions                         | Not guaranteed |
///
/// # Security
///
/// FNV-1a is NOT DoS-resistant. Do not use `StdHasher` as the backing hasher
/// for hash tables exposed to adversarial input. For Bloom filters, where the
/// adversary cannot force false negatives, this is acceptable.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
///
/// // Default construction
/// let hasher = StdHasher::new();
/// assert_ne!(hasher.hash_bytes(b"test"), 0);
///
/// // Seeded construction
/// let h1 = StdHasher::with_seed(1).hash_bytes(b"x");
/// let h2 = StdHasher::with_seed(2).hash_bytes(b"x");
/// assert_ne!(h1, h2);
/// ```
#[derive(Debug, Clone)]
pub struct StdHasher {
    seed: u64,
}

impl StdHasher {
    /// Default seed — a well-distributed 64-bit constant.
    const DEFAULT_SEED: u64 = 0x517c_c1b7_2722_0a95;

    /// Create a `StdHasher` with the default seed.
    ///
    /// The default seed is a compile-time constant, ensuring identical output
    /// across all runs and processes with the same crate version.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            seed: Self::DEFAULT_SEED,
        }
    }

    /// Create a `StdHasher` with an explicit seed.
    ///
    /// Different seeds produce independent hash outputs. Useful for constructing
    /// multiple independent hash functions from a single algorithm, or for
    /// reproducible hashing with a known seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::hash::hasher::{BloomHasher, StdHasher};
    ///
    /// let h1 = StdHasher::with_seed(0).hash_bytes(b"test");
    /// let h2 = StdHasher::with_seed(1).hash_bytes(b"test");
    /// assert_ne!(h1, h2);
    /// ```
    #[must_use]
    #[inline]
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
        let mut h = DeterministicHasher::new();
        h.write_u64(self.seed);
        h.write(bytes);
        h.finish()
    }

    /// Override to mix the call-site seed with the instance seed before hashing,
    /// rather than XOR-ing after the fact. This ensures the seed affects the
    /// entire FNV accumulation, providing true independence between h1 and h2.
    #[inline]
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut h = DeterministicHasher::new();
        // XOR combines the two seeds before feeding to FNV.
        // Both seeds affect every byte processed, not just the final output.
        h.write_u64(self.seed ^ seed);
        h.write(bytes);
        h.finish()
    }

    /// Override for tighter independence: uses a well-chosen seed constant and
    /// bit rotation to maximally decorrelate h1 and h2 in their low bits
    /// (where modulo reduction operates).
    #[inline]
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = self.hash_bytes(bytes);
        let h2 = self.hash_bytes_with_seed(bytes, 0x9e37_79b9_7f4a_7c15);
        (h1, h2.rotate_left(31) ^ 0xa021_282d_c0b9_ed54)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "StdHasher"
    }
}

/// Implement `BuildHasher` so `StdHasher` can back `HashMap` and `HashSet`.
impl std::hash::BuildHasher for StdHasher {
    type Hasher = DeterministicHasher;

    fn build_hasher(&self) -> Self::Hasher {
        use std::hash::Hasher;
        let mut h = DeterministicHasher::new();
        h.write_u64(self.seed);
        h
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic functionality ──────────────────────────────────────────────────

    #[test]
    fn test_determinism() {
        let hasher = StdHasher::new();
        let data = b"test string";
        assert_eq!(hasher.hash_bytes(data), hasher.hash_bytes(data));
    }

    #[test]
    fn test_different_inputs_differ() {
        let hasher = StdHasher::new();
        assert_ne!(hasher.hash_bytes(b"input1"), hasher.hash_bytes(b"input2"));
    }

    #[test]
    fn test_empty_input_nonzero() {
        // FNV offset basis is non-zero, so empty input must produce a non-zero hash.
        assert_ne!(StdHasher::new().hash_bytes(b""), 0);
    }

    #[test]
    fn test_single_byte_sensitivity() {
        let hasher = StdHasher::new();
        assert_ne!(hasher.hash_bytes(b"a"), hasher.hash_bytes(b"b"));
    }

    // ── Seed behaviour ───────────────────────────────────────────────────────

    #[test]
    fn test_different_seeds_produce_different_hashes() {
        let data = b"test";
        assert_ne!(
            StdHasher::with_seed(123).hash_bytes(data),
            StdHasher::with_seed(456).hash_bytes(data),
        );
    }

    #[test]
    fn test_same_seed_same_results_cross_instance() {
        let h1 = StdHasher::with_seed(42).hash_bytes(b"reproducibility");
        let h2 = StdHasher::with_seed(42).hash_bytes(b"reproducibility");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_bytes_with_seed_produces_independent_values() {
        let hasher = StdHasher::new();
        let data = b"test";
        let h0 = hasher.hash_bytes_with_seed(data, 0);
        let h1 = hasher.hash_bytes_with_seed(data, 42);
        let h2 = hasher.hash_bytes_with_seed(data, 999);
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    // ── Pair and triple ──────────────────────────────────────────────────────

    #[test]
    fn test_pair_values_are_distinct() {
        let (h1, h2) = StdHasher::new().hash_bytes_pair(b"test");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_pair_is_deterministic() {
        let hasher = StdHasher::new();
        let (a1, a2) = hasher.hash_bytes_pair(b"test");
        let (b1, b2) = hasher.hash_bytes_pair(b"test");
        assert_eq!(a1, b1);
        assert_eq!(a2, b2);
    }

    #[test]
    fn test_triple_values_are_mutually_distinct() {
        let (h1, h2, h3) = StdHasher::new().hash_bytes_triple(b"test");
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_triple_is_deterministic() {
        let hasher = StdHasher::new();
        let (a1, a2, a3) = hasher.hash_bytes_triple(b"test");
        let (b1, b2, b3) = hasher.hash_bytes_triple(b"test");
        assert_eq!(a1, b1);
        assert_eq!(a2, b2);
        assert_eq!(a3, b3);
    }

    // ── hash_item bridge ─────────────────────────────────────────────────────

    #[test]
    fn test_hash_item_string_is_deterministic() {
        let hasher = StdHasher::new();
        let (a1, a2) = hasher.hash_item(&"hello world".to_string());
        let (b1, b2) = hasher.hash_item(&"hello world".to_string());
        assert_eq!(a1, b1);
        assert_eq!(a2, b2);
    }

    #[test]
    fn test_hash_item_integer_is_deterministic() {
        let hasher = StdHasher::new();
        let (a1, a2) = hasher.hash_item(&42u64);
        let (b1, b2) = hasher.hash_item(&42u64);
        assert_eq!(a1, b1);
        assert_eq!(a2, b2);
    }

    #[test]
    fn test_hash_item_pair_values_are_distinct() {
        let hasher = StdHasher::new();
        let (h1, h2) = hasher.hash_item(&"test".to_string());
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_item_different_items_differ() {
        let hasher = StdHasher::new();
        let (a1, _) = hasher.hash_item(&"foo".to_string());
        let (b1, _) = hasher.hash_item(&"bar".to_string());
        assert_ne!(a1, b1);
    }

    #[test]
    fn test_hash_item_integer_differs_by_value() {
        let hasher = StdHasher::new();
        let (h1, _) = hasher.hash_item(&1u64);
        let (h2, _) = hasher.hash_item(&2u64);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_item_large_key_no_panic() {
        // Forces a heap spill in HashWriter (>32 bytes)
        let hasher = StdHasher::new();
        let large_key = "x".repeat(256);
        let (h1, h2) = hasher.hash_item(&large_key);
        assert_ne!(h1, h2);
        // Deterministic even on the heap path
        let (h3, h4) = hasher.hash_item(&large_key);
        assert_eq!(h1, h3);
        assert_eq!(h2, h4);
    }

    #[test]
    fn test_hash_item_consistent_with_hash_bytes_pair() {
        // hash_item must produce the same result as manually collecting bytes
        // and calling hash_bytes_pair — this pins the implementation contract.
        let hasher = StdHasher::new();
        let item = "determinism check".to_string();

        let (item_h1, item_h2) = hasher.hash_item(&item);

        let mut writer = HashWriter::new();
        item.hash(&mut writer);
        let bytes = writer.into_bytes();
        let (bytes_h1, bytes_h2) = hasher.hash_bytes_pair(&bytes);

        assert_eq!(item_h1, bytes_h1);
        assert_eq!(item_h2, bytes_h2);
    }

    // ── Statistical independence ─────────────────────────────────────────────

    #[test]
    fn test_pair_statistical_independence() {
        // XOR of all h1s must differ from XOR of all h2s across 200 distinct inputs.
        let hasher = StdHasher::new();
        let (xor_h1, xor_h2) = (0..200u64).fold((0u64, 0u64), |(x1, x2), i| {
            let data = format!("item_{}", i);
            let (h1, h2) = hasher.hash_bytes_pair(data.as_bytes());
            (x1 ^ h1, x2 ^ h2)
        });
        assert_ne!(xor_h1, xor_h2, "h1 and h2 streams must be independent");
    }

    // ── Avalanche effect ─────────────────────────────────────────────────────

    #[test]
    fn test_avalanche_single_bit_flip() {
        let hasher = StdHasher::new();
        let data1 = b"test data!";
        let mut data2 = *b"test data!";
        data2[0] ^= 1;

        let h1 = hasher.hash_bytes(data1);
        let h2 = hasher.hash_bytes(&data2);
        let changed_bits = (h1 ^ h2).count_ones();

        // FNV-1a achieves reasonable avalanche; allow wide tolerance.
        // A single input bit change should statistically affect 20–44 of 64 output bits.
        assert!(
            changed_bits >= 16 && changed_bits <= 48,
            "Avalanche: {} bits changed (expected 16–48)",
            changed_bits
        );
    }

    // ── Name ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(StdHasher::new().name(), "StdHasher");
    }

    #[test]
    fn test_clone_produces_identical_results() {
        let h1 = StdHasher::with_seed(999);
        let h2 = h1.clone();
        assert_eq!(h1.hash_bytes(b"clone test"), h2.hash_bytes(b"clone test"));
    }

    // ── Trait bounds ─────────────────────────────────────────────────────────

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StdHasher>();
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_large_byte_slice() {
        let hasher = StdHasher::new();
        let large = vec![0xabu8; 65_536];
        let hash = hasher.hash_bytes(&large);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_unicode_bytes() {
        let hasher = StdHasher::new();
        let s = "こんにちは 🦀";
        let h1 = hasher.hash_bytes(s.as_bytes());
        let h2 = hasher.hash_bytes(s.as_bytes());
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_consecutive_inputs_differ() {
        let hasher = StdHasher::new();
        assert_ne!(hasher.hash_bytes(b"aaaa"), hasher.hash_bytes(b"aaab"));
    }

    // ── BuildHasher integration ───────────────────────────────────────────────

    #[test]
    fn test_build_hasher_produces_consistent_result() {
        use std::hash::{BuildHasher, Hasher};
        let builder = StdHasher::new();
        let mut h1 = builder.build_hasher();
        let mut h2 = builder.build_hasher();
        h1.write(b"test");
        h2.write(b"test");
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── Strategy integration ─────────────────────────────────────────────────

    #[test]
    fn test_integration_with_double_hashing() {
        use crate::hash::strategies::{DoubleHashing, HashStrategy};
        let hasher = StdHasher::new();
        let (h1, h2) = hasher.hash_bytes_pair(b"test");
        let indices = DoubleHashing.generate_indices(h1, h2, 0, 7, 1000);
        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&i| i < 1000));
    }

    #[test]
    fn test_integration_with_enhanced_double_hashing() {
        use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy};
        let hasher = StdHasher::new();
        let (h1, h2) = hasher.hash_bytes_pair(b"test");
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, 7, 1000);
        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&i| i < 1000));
    }

    #[test]
    fn test_integration_with_triple_hashing() {
        use crate::hash::strategies::{HashStrategy, TripleHashing};
        let hasher = StdHasher::new();
        let (h1, h2, h3) = hasher.hash_bytes_triple(b"test");
        let indices = TripleHashing.generate_indices(h1, h2, h3, 7, 1000);
        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&i| i < 1000));
    }

    #[test]
    fn test_hash_item_with_strategy() {
        use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy};
        let hasher = StdHasher::new();
        let (h1, h2) = hasher.hash_item(&"integration test".to_string());
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, 7, 10_000);
        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&i| i < 10_000));
    }
}
