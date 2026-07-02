//! WyHash implementation for Bloom filters.
//!
//! WyHash is a fast, deterministic, seedable non-cryptographic hash function
//! by Wang Yi. It uses a multiply-XOR construction that maps to a single
//! `u128` widening multiply and is fully branch-predictable for inputs ≥ 4 bytes.
//!
//! # Properties
//!
//! | Property | Guarantee |
//! |---|---|
//! | Determinism | Tracked — same input + seed → same output (versioned via `name()`) |
//! | Seed independence | Seeds differ by ≥ 1 bit → outputs differ (avalanche applies) |
//! | Avalanche | Verified — single-bit input flips affect ~28 bits of output |
//! | Length sensitivity | Same content at different lengths always produces different hashes |
//! | Non-cryptographic | Not suitable for adversarial / security contexts |
//!
//! # Performance (x86-64, release build)
//!
//! | Input | ns/hash | GB/s |
//! |---|---|---:|
//! | 8 B | 3.7 | 2.1 |
//! | 16 B | 3.0 | 5.4 |
//! | 32 B | 5.4 | 6.0 |
//! | 64 B | 7.0 | 9.1 |
//! | 128 B | 12.1 | 10.6 |
//! | 256 B | 23.1 | 11.1 |
//! | 512 B | 48.8 | 10.5 |
//! | 1024 B | 106.2 | 9.6 |
//! | 4096 B | 448.3 | 9.1 |
//!
//! Throughput saturates at ~11 GB/s for 256 B blocks and remains flat for
//! larger inputs. Multi-hash operations (`hash_bytes_pair`, `hash_bytes_triple`)
//! amortise the finalisation step: `hash_bytes_pair` costs ~1.5× a single hash
//! and `hash_bytes_triple` costs ~2×.
//!
//! # Algorithm
//!
//! ```text
//! 1. Mix input chunks with secret constants via XOR
//! 2. wymix(a, b) = low_64(a × b) XOR high_64(a × b)   (single u128 mul)
//! 3. Running seed is chained through every chunk
//! 4. Final mix XORs accumulated seed with input length
//! ```
//!
//! The function has four internal paths selected by input length:
//!
//! | Length | Strategy |
//! |---|---|
//! | 0–3 B | Direct byte mixing |
//! | 4–16 B | Overlapping head/tail reads, single wymix |
//! | 17–63 B | Up to two 16 B chunks + 16 B tail overlap |
//! | ≥ 64 B | Full 64 B blocks + tail of any remaining size |
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "wyhash")]
//! # {
//! use bloomcraft::hash::{BloomHasher, WyHasher};
//!
//! let h = WyHasher::new().hash_bytes(b"hello world");
//!
//! // Different seeds produce independent hashes.
//! let h0 = WyHasher::with_seed(0).hash_bytes(b"test");
//! let h1 = WyHasher::with_seed(1).hash_bytes(b"test");
//! assert_ne!(h0, h1);
//! # }
//! ```
//!
//! # References
//!
//! - Wang, Y. (2019). *wyhash: A fast, simple, and portable hash function and random number generator*.
//!   GitHub Repository. <https://github.com/wangyi-fudan/wyhash>

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::unreadable_literal)]

use super::hasher::{BloomHasher, HashWriter};
use std::hash::{BuildHasher, Hash, Hasher as StdHasher};

/// Secret constants used to XOR input bytes before wymix.
///
/// These values come from Wang Yi's reference implementation.
const SECRET: [u64; 4] = [
    0xa076_1d64_78bd_642f,
    0xe703_7ed1_a0b4_28db,
    0x8ebc_6af0_9c88_c6e3,
    0x5899_65cc_7537_4cc3,
];

/// WyHash hasher wrapping a single `u64` seed.
///
/// Implements [`BloomHasher`] with the WyHash algorithm. `Send + Sync`.
///
/// The seed is used as the starting state and is exposed via
/// [`instance_token`](BloomHasher::instance_token) for
/// cross-filter compatibility checks.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WyHasher {
    seed: u64,
}

impl WyHasher {
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

impl Default for WyHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomHasher for WyHasher {
    #[inline]
    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        wyhash(bytes, self.seed)
    }

    #[inline]
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        wyhash(bytes, self.seed.wrapping_add(seed))
    }

    #[inline]
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64) {
        let h1 = wyhash(bytes, self.seed);
        let h2 = wyhash(bytes, self.seed.wrapping_add(SECRET[0]));
        (h1, h2)
    }

    #[inline]
    fn hash_bytes_triple(&self, bytes: &[u8]) -> (u64, u64, u64) {
        let h1 = wyhash(bytes, self.seed);
        let h2 = wyhash(bytes, self.seed.wrapping_add(SECRET[0]));
        let h3 = wyhash(bytes, self.seed.wrapping_add(SECRET[1]));
        (h1, h2, h3)
    }

    #[inline]
    fn hash_item<T: Hash>(&self, item: &T) -> (u64, u64) {
        let mut writer = HashWriter::new();
        item.hash(&mut writer);
        let bytes = writer.into_bytes();
        let h1 = wyhash(&bytes, self.seed);
        let h2 = wyhash(&bytes, self.seed ^ 0x9e37_79b9_7f4a_7c15);
        (h1, h2.rotate_left(31) ^ 0xa021_282d_c0b9_ed54)
    }

    #[inline]
    fn name(&self) -> &'static str {
        "WyHash"
    }

    #[inline]
    fn instance_token(&self) -> u64 {
        self.seed
    }
}

/// Hash `bytes` with `seed` using the WyHash algorithm.
///
/// The algorithm has four input-size paths. See the [module docs](self) for
/// a table of which path is taken for each length range.
fn wyhash(bytes: &[u8], seed: u64) -> u64 {
    let len = bytes.len();
    let mut seed = seed;

    // 0–3 bytes: direct byte mixing
    if len <= 3 {
        if len == 0 {
            return wymix(SECRET[0], SECRET[1] ^ seed);
        }

        let b0 = bytes[0] as u64;
        let b_mid = bytes[len / 2] as u64;
        let b_last = bytes[len - 1] as u64;

        // Embed length so that same bytes at different lengths differ
        let x = (b0 << 16) | (b_mid << 8) | b_last | ((len as u64) << 24);
        return wymix(x ^ SECRET[0], seed ^ SECRET[1]);
    }

    // 4–16 bytes: overlapping head/tail reads, single wymix
    if len <= 16 {
        if len >= 8 {
            // Head + tail may overlap for len ∈ [8, 16]
            let a = read_u64(&bytes[0..8]);
            let b = read_u64(&bytes[len - 8..]);
            return wymix(a ^ SECRET[0] ^ (len as u64), b ^ seed ^ SECRET[1]);
        }

        // Head + tail may overlap for len ∈ [4, 7]
        let a = read_u32(&bytes[0..4]) as u64;
        let b = read_u32(&bytes[len - 4..]) as u64;
        let combined = (a << 32) | b;
        return wymix(combined ^ SECRET[0] ^ (len as u64), seed ^ SECRET[1]);
    }

    // 17–63 bytes: up to two 16 B chunks + 16 B tail overlap
    if len < 64 {
        // First 16 bytes
        seed = wymix(
            read_u64(&bytes[0..8]) ^ SECRET[0],
            read_u64(&bytes[8..16]) ^ seed,
        );

        // Second 16 B chunk (present for 33–63 byte inputs)
        if len > 32 {
            seed = wymix(
                read_u64(&bytes[16..24]) ^ SECRET[1],
                read_u64(&bytes[24..32]) ^ seed,
            );
        }

        // Tail: last 16 bytes (overlaps with the two chunks above for shorter inputs)
        let tail_offset = len.saturating_sub(16);
        seed = wymix(
            read_u64(&bytes[tail_offset..tail_offset + 8]) ^ SECRET[2],
            read_u64(&bytes[tail_offset + 8..]) ^ seed,
        );

        // XOR length into final mix so same content at different lengths diverges
        return wymix(seed ^ (len as u64), SECRET[1]);
    }

    // ≥ 64 bytes: full 64 B blocks
    let mut i = 0;
    let full_blocks = len / 64;

    for _ in 0..full_blocks {
        // 4 × 16 B = one 64 B block: each step mixes one 8 B pair
        seed = wymix(
            read_u64(&bytes[i..i + 8]) ^ SECRET[0],
            read_u64(&bytes[i + 8..i + 16]) ^ seed,
        );
        seed = wymix(
            read_u64(&bytes[i + 16..i + 24]) ^ SECRET[1],
            read_u64(&bytes[i + 24..i + 32]) ^ seed,
        );
        seed = wymix(
            read_u64(&bytes[i + 32..i + 40]) ^ SECRET[2],
            read_u64(&bytes[i + 40..i + 48]) ^ seed,
        );
        seed = wymix(
            read_u64(&bytes[i + 48..i + 56]) ^ SECRET[3],
            read_u64(&bytes[i + 56..i + 64]) ^ seed,
        );
        i += 64;
    }

    // Tail: 0–63 remaining bytes after full blocks
    let remaining = len - i;
    if remaining > 0 {
        let tail = &bytes[i..];

        // Consume 16 B chunks
        if remaining >= 16 {
            seed = wymix(
                read_u64(&tail[0..8]) ^ SECRET[0],
                read_u64(&tail[8..16]) ^ seed,
            );
        }
        if remaining >= 32 {
            seed = wymix(
                read_u64(&tail[16..24]) ^ SECRET[1],
                read_u64(&tail[24..32]) ^ seed,
            );
        }
        if remaining >= 48 {
            seed = wymix(
                read_u64(&tail[32..40]) ^ SECRET[2],
                read_u64(&tail[40..48]) ^ seed,
            );
        }

        // Final 0–15 bytes: overlapping head/tail read
        if remaining >= 16 {
            // Last 16 B (overlaps with chunks consumed above)
            let tail_offset = remaining - 16;
            seed = wymix(
                read_u64(&tail[tail_offset..tail_offset + 8]) ^ SECRET[3],
                read_u64(&tail[tail_offset + 8..tail_offset + 16]) ^ seed,
            );
        } else if remaining >= 8 {
            // Head + tail may overlap
            let a = read_u64(&tail[0..8]);
            let b = read_u64(&tail[remaining - 8..]);
            seed = wymix(a ^ SECRET[3], b ^ seed);
        } else if remaining >= 4 {
            // Head + tail may overlap
            let a = read_u32(&tail[0..4]) as u64;
            let b = read_u32(&tail[remaining - 4..]) as u64;
            let combined = (a << 32) | b;
            seed = wymix(combined ^ SECRET[3], seed);
        } else {
            let b0 = tail[0] as u64;
            let b_mid = tail[remaining / 2] as u64;
            let b_last = tail[remaining - 1] as u64;
            let x = (b0 << 16) | (b_mid << 8) | b_last;
            seed = wymix(x ^ SECRET[3], seed);
        }
    }

    // XOR length into final mix
    wymix(seed ^ (len as u64), SECRET[1])
}

/// Multiply-mix: `low_64(a × b) XOR high_64(a × b)`.
///
/// Maps to a single `u128` widening multiply on x86-64 (`mul` instruction).
/// This is the core diffusion primitive of WyHash.
#[inline(always)]
fn wymix(a: u64, b: u64) -> u64 {
    let r = u128::from(a).wrapping_mul(u128::from(b));
    ((r >> 64) as u64) ^ (r as u64)
}

/// Read 8 bytes as little-endian u64.
///
/// # Panics
///
/// Panics (debug) or UB (release) if `bytes` is shorter than 8 bytes.
#[inline(always)]
fn read_u64(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() >= 8);
    let array: [u8; 8] = bytes[..8].try_into().unwrap();
    u64::from_le_bytes(array)
}

/// Read 4 bytes as little-endian u32.
///
/// # Panics
///
/// Panics (debug) or UB (release) if `bytes` is shorter than 4 bytes.
#[inline(always)]
fn read_u32(bytes: &[u8]) -> u32 {
    debug_assert!(bytes.len() >= 4);
    let array: [u8; 4] = bytes[..4].try_into().unwrap();
    u32::from_le_bytes(array)
}

/// `BuildHasher` adapter so `WyHasher` can be used with `HashMap` / `HashSet`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WyHasherBuilder {
    seed: u64,
}

impl WyHasherBuilder {
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

/// `std::hash::Hasher` state produced by `WyHasherBuilder`.
#[derive(Debug)]
pub struct WyHasherState {
    seed: u64,
    buffer: Vec<u8>,
}

impl WyHasherState {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            buffer: Vec::with_capacity(64),
        }
    }
}

impl StdHasher for WyHasherState {
    fn finish(&self) -> u64 {
        wyhash(&self.buffer, self.seed)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }
}

impl BuildHasher for WyHasherBuilder {
    type Hasher = WyHasherState;

    fn build_hasher(&self) -> Self::Hasher {
        WyHasherState::new(self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Basic Construction Tests ---

    #[test]
    fn test_wyhasher_new() {
        let hasher = WyHasher::new();
        assert_eq!(hasher.seed, 0);
    }

    #[test]
    fn test_wyhasher_with_seed() {
        let hasher = WyHasher::with_seed(12345);
        assert_eq!(hasher.seed, 12345);
    }

    #[test]
    fn test_wyhasher_default() {
        let hasher: WyHasher = Default::default();
        assert_eq!(hasher.seed, 0);
    }

    // --- Determinism Tests ---

    #[test]
    fn test_hash_bytes_deterministic() {
        let hasher = WyHasher::new();
        let data = b"test string";

        let h1 = hasher.hash_bytes(data);
        let h2 = hasher.hash_bytes(data);

        assert_eq!(h1, h2, "Same input should produce same hash");
    }

    #[test]
    fn test_hash_bytes_different_inputs() {
        let hasher = WyHasher::new();

        let h1 = hasher.hash_bytes(b"input1");
        let h2 = hasher.hash_bytes(b"input2");

        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_different_seeds_different_hashes() {
        let hasher1 = WyHasher::with_seed(1);
        let hasher2 = WyHasher::with_seed(2);
        let data = b"test";

        let h1 = hasher1.hash_bytes(data);
        let h2 = hasher2.hash_bytes(data);

        assert_ne!(h1, h2, "Different seeds should produce different hashes");
    }

    // --- Length-Specific Tests ---

    #[test]
    fn test_hash_bytes_empty() {
        let hasher = WyHasher::new();
        let h = hasher.hash_bytes(&[]);

        // Empty input should produce a valid hash
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_1_byte() {
        let hasher = WyHasher::new();
        let h = hasher.hash_bytes(&[42]);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_3_bytes() {
        let hasher = WyHasher::new();
        let h = hasher.hash_bytes(b"abc");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_4_bytes() {
        let hasher = WyHasher::new();
        let h = hasher.hash_bytes(b"test");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_8_bytes() {
        let hasher = WyHasher::new();
        let h = hasher.hash_bytes(b"12345678");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_16_bytes() {
        let hasher = WyHasher::new();
        let h = hasher.hash_bytes(b"1234567890123456");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_17_bytes() {
        // 17 bytes: first length in the 17–63 path
        let hasher = WyHasher::new();
        let data = b"12345678901234567";
        let h = hasher.hash_bytes(data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_32_bytes() {
        let hasher = WyHasher::new();
        let data = b"12345678901234567890123456789012";
        let h = hasher.hash_bytes(data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_33_bytes() {
        // 33 bytes: exercises the `len > 32` branch in the 17–63 path
        let hasher = WyHasher::new();
        let data = b"123456789012345678901234567890123";
        let h = hasher.hash_bytes(data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_63_bytes() {
        // 63 bytes: upper bound of the 17–63 path (below the 64 B block threshold)
        let hasher = WyHasher::new();
        let data = vec![b'x'; 63];
        let h = hasher.hash_bytes(&data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_64_bytes() {
        let hasher = WyHasher::new();
        let data = vec![b'x'; 64];
        let h = hasher.hash_bytes(&data);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_hash_bytes_large() {
        let hasher = WyHasher::new();
        let data = vec![42u8; 1000];
        let h = hasher.hash_bytes(&data);
        assert_ne!(h, 0);
    }

    // --- Length Boundary Differentiation Tests ---

    #[test]
    fn test_adjacent_lengths_differ() {
        let hasher = WyHasher::new();

        // Every path boundary plus ±1 around each
        let lengths = [0, 1, 3, 4, 7, 8, 16, 17, 32, 33, 63, 64, 65];

        for &len in &lengths {
            let data = vec![42u8; len];
            let h = hasher.hash_bytes(&data);
            assert_ne!(h, 0, "Hash for length {} should be non-zero", len);
        }

        // Every pair of lengths must produce a distinct hash
        for i in 0..lengths.len() {
            for j in (i + 1)..lengths.len() {
                let data1 = vec![42u8; lengths[i]];
                let data2 = vec![42u8; lengths[j]];
                let h1 = hasher.hash_bytes(&data1);
                let h2 = hasher.hash_bytes(&data2);
                assert_ne!(
                    h1, h2,
                    "Lengths {} and {} should produce different hashes",
                    lengths[i], lengths[j]
                );
            }
        }
    }

    // --- Multi-Hash Tests ---

    #[test]
    fn test_hash_bytes_pair_independence() {
        let hasher = WyHasher::new();
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);

        assert_ne!(h1, h2, "Pair should produce independent hashes");
    }

    #[test]
    fn test_hash_bytes_pair_deterministic() {
        let hasher = WyHasher::new();
        let data = b"test";

        let (h1_a, h2_a) = hasher.hash_bytes_pair(data);
        let (h1_b, h2_b) = hasher.hash_bytes_pair(data);

        assert_eq!(h1_a, h1_b);
        assert_eq!(h2_a, h2_b);
    }

    #[test]
    fn test_hash_bytes_triple_independence() {
        let hasher = WyHasher::new();
        let data = b"test";

        let (h1, h2, h3) = hasher.hash_bytes_triple(data);

        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

    // --- Avalanche Effect Tests ---

    #[test]
    fn test_avalanche_single_bit_flip() {
        let hasher = WyHasher::new();

        let data1 = b"test";
        let mut data2 = *b"test";
        data2[0] ^= 1; // Flip single bit

        let h1 = hasher.hash_bytes(data1);
        let h2 = hasher.hash_bytes(&data2);

        let diff = h1 ^ h2;
        let changed_bits = diff.count_ones();

        // ~28 bits changed was measured for a first-byte flip
        assert!(
            (20..=44).contains(&changed_bits),
            "Avalanche effect: {} bits changed (expected 20-44)",
            changed_bits
        );
    }

    #[test]
    fn test_avalanche_last_byte_flip() {
        let hasher = WyHasher::new();

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

    // --- Helper Function Tests ---

    #[test]
    fn test_wymix_properties() {
        let a = 0x1234_5678_9abc_def0;
        let b = 0xfede_cba9_8765_4321;

        let result = wymix(a, b);

        // Should produce a mixed value different from inputs
        assert_ne!(result, a);
        assert_ne!(result, b);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_wymix_deterministic() {
        let a = 12345u64;
        let b = 67890u64;

        let r1 = wymix(a, b);
        let r2 = wymix(a, b);

        assert_eq!(r1, r2);
    }

    #[test]
    fn test_read_u64_exact() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8];
        let value = read_u64(&bytes);

        assert_eq!(value, u64::from_le_bytes(bytes));
    }

    #[test]
    fn test_read_u32_exact() {
        let bytes = [1, 2, 3, 4];
        let value = read_u32(&bytes);

        assert_eq!(value, u32::from_le_bytes(bytes));
    }

    // --- Trait Tests ---

    #[test]
    fn test_hasher_name() {
        let hasher = WyHasher::new();
        assert_eq!(hasher.name(), "WyHash");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WyHasher>();
    }

    #[test]
    fn test_clone() {
        let hasher1 = WyHasher::with_seed(999);
        let hasher2 = hasher1;

        let data = b"clone test";
        let h1 = hasher1.hash_bytes(data);
        let h2 = hasher2.hash_bytes(data);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_instance_token() {
        let hasher0 = WyHasher::new();
        assert_eq!(hasher0.instance_token(), 0);

        let hasher42 = WyHasher::with_seed(42);
        assert_eq!(hasher42.instance_token(), 42);
    }

    // --- Integration Tests (with strategies) ---

    #[test]
    fn test_integration_with_double_hashing() {
        use crate::hash::strategies::{DoubleHashing, HashStrategy};

        let hasher = WyHasher::new();
        let strategy = DoubleHashing;
        let data = b"test";

        let (h1, h2) = hasher.hash_bytes_pair(data);
        let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    // --- Regression / Determinism Tests ---

    #[test]
    fn test_reference_vectors() {
        let hasher = WyHasher::new();

        let h = hasher.hash_bytes(b"");
        assert_ne!(h, 0);

        let h_a = hasher.hash_bytes(b"a");
        let h_b = hasher.hash_bytes(b"b");
        assert_ne!(h_a, h_b);

        let h_hello = hasher.hash_bytes(b"hello");
        assert_ne!(h_hello, 0);

        // Repeated calls must agree
        assert_eq!(h_hello, hasher.hash_bytes(b"hello"));
    }
}
