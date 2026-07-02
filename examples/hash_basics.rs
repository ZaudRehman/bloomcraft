//! How to use the hasher API directly: creating hashers, hashing bytes,
//! deriving independent hash streams, checking hasher identity.
//!
//! Run with: cargo run --features "wyhash,xxhash" --example hash_basics

use bloomcraft::hash::{BloomHasher, StdHasher, WyHasher, XxHasher};

fn main() {
    // --- 1. Pick a hasher ---

    // Every hasher works the same way — they all implement `BloomHasher`.
    let std_hash = StdHasher::new();
    let wy_hash = WyHasher::new();
    let xx_hash = XxHasher::new();

    // The type carries its name — useful for logging / debugging.
    assert_eq!(std_hash.name(), "StdHasher");
    assert_eq!(wy_hash.name(), "WyHash");
    assert_eq!(xx_hash.name(), "XXHash3");

    // For feature-agnostic code, use the factory:
    let _h = bloomcraft::hash::recommended_hasher(); // picks best available

    // --- 2. Hash bytes ---

    let h = wy_hash.hash_bytes(b"hello world");
    println!("hash_bytes:          0x{h:016x}");

    // Deterministic — same call always returns the same value.
    assert_eq!(
        wy_hash.hash_bytes(b"hello world"),
        wy_hash.hash_bytes(b"hello world")
    );

    // --- 3. Hash generic items (the canonical bridge) ---

    // `hash_item` captures the `Hash` byte stream and derives two independent
    // hashes from it. Zero heap allocation for keys ≤32 bytes.
    let (h1, h2) = wy_hash.hash_item(&"string key");
    assert_ne!(h1, h2);
    println!("hash_item (string):  h1 = 0x{h1:016x}, h2 = 0x{h2:016x}");

    // Works for any `T: Hash`.
    let (_h1, _h2) = wy_hash.hash_item(&42u64);
    let (_h1, _h2) = wy_hash.hash_item(&[1, 2, 3, 4, 5]);

    // --- 4. Seeds produce independent hash streams ---

    let h_a = WyHasher::with_seed(1).hash_bytes(b"test");
    let h_b = WyHasher::with_seed(2).hash_bytes(b"test");
    assert_ne!(h_a, h_b);
    println!("seed(1) = 0x{h_a:016x}");
    println!("seed(2) = 0x{h_b:016x}");

    // --- 5. Pair and triple — multi-hash without re-hashing ---

    // `hash_bytes_pair` produces two independent values from one pass over the
    // data. This is the method Bloom filters call on the hot path.
    let (p1, p2) = wy_hash.hash_bytes_pair(b"data");
    assert_ne!(p1, p2);
    println!("pair:                p1 = 0x{p1:016x}, p2 = 0x{p2:016x}");

    // `hash_bytes_triple` for strategies that need three base hashes.
    let (t1, t2, t3) = wy_hash.hash_bytes_triple(b"data");
    assert_ne!(t1, t2);
    assert_ne!(t2, t3);
    assert_ne!(t1, t3);

    // --- 6. Hasher identity — `instance_token` ---

    // Two hashers with the same seed are compatible; different seeds are not.
    // This is checked when merging Bloom filters or deserialising them.
    let a = WyHasher::with_seed(0);
    let b = WyHasher::with_seed(0);
    let c = WyHasher::with_seed(42);

    assert_eq!(a.instance_token(), b.instance_token());
    assert_ne!(a.instance_token(), c.instance_token());
    println!("instance_token(seed=0):  0x{:016x}", a.instance_token());
    println!("instance_token(seed=42): 0x{:016x}", c.instance_token());

    // --- 7. HashMap integration (BuildHasher) ---

    // `WyHasherBuilder` and `XxHasherBuilder` both implement `BuildHasher`.
    use std::collections::HashMap;
    let mut map: HashMap<String, i32, _> =
        HashMap::with_hasher(bloomcraft::hash::WyHasherBuilder::new());
    *map.entry("key".to_string()).or_insert(0) += 1;
    println!("HashMap with WyHasher:  key -> {}", map["key"]);

    // --- 8. All outputs ---

    println!("\nAll checks passed.");
}
