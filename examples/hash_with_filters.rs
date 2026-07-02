//! How to use different hashers with real Bloom filters: type parameters,
//! merge compatibility, feature-gated hashing.
//!
//! Run with: cargo run --features "wyhash,xxhash" --example hash_with_filters

use bloomcraft::filters::standard::StandardBloomFilter;
use bloomcraft::hash::{BloomHasher, WyHasher};

fn main() {
    println!("=== Default hasher (StdHasher — FNV-1a) ===\n");

    // Every filter defaults to StdHasher. No type annotation needed.
    let f1: StandardBloomFilter<String> =
        StandardBloomFilter::new(100_000, 0.01).expect("valid params");
    f1.insert(&"alice@example.com".to_string());
    assert!(f1.contains(&"alice@example.com".to_string()));
    assert!(!f1.contains(&"nobody@example.com".to_string()));
    println!("StdHasher filter:  insert + contains OK");

    // --- Pick a faster hasher ---

    println!("\n=== WyHasher (fast, small keys) ===\n");

    let f2: StandardBloomFilter<String, WyHasher> =
        StandardBloomFilter::with_hasher(100_000, 0.01, WyHasher::new()).expect("valid params");
    f2.insert(&"alice@example.com".to_string());
    assert!(f2.contains(&"alice@example.com".to_string()));
    println!("WyHasher filter:  insert + contains OK");

    // With a custom seed.
    let f3: StandardBloomFilter<String, WyHasher> =
        StandardBloomFilter::with_hasher(100_000, 0.01, WyHasher::with_seed(42))
            .expect("valid params");
    f3.insert(&"data".to_string());
    assert!(f3.contains(&"data".to_string()));
    println!("WyHasher (seed=42) filter:  insert + contains OK");

    // --- XXHash3 — strongest on large inputs ---

    println!("\n=== XXHash3 (fast, large keys) ===\n");

    let f4: StandardBloomFilter<String, bloomcraft::hash::XxHasher> =
        StandardBloomFilter::with_hasher(100_000, 0.01, bloomcraft::hash::XxHasher::new())
            .expect("valid params");
    f4.insert(&"alice@example.com".to_string());
    assert!(f4.contains(&"alice@example.com".to_string()));
    println!("XXHash3 filter:  insert + contains OK");

    // --- Merge requires matching seeds ---

    println!("\n=== Merge compatibility check ===\n");

    let mut a: StandardBloomFilter<String, WyHasher> =
        StandardBloomFilter::with_hasher(100, 0.01, WyHasher::with_seed(0)).expect("valid params");
    let b: StandardBloomFilter<String, WyHasher> =
        StandardBloomFilter::with_hasher(100, 0.01, WyHasher::with_seed(0)).expect("valid params");
    a.insert(&"item_a".to_string());
    b.insert(&"item_b".to_string());
    a = a.union(&b).expect("same hasher + seed → merge OK");
    assert!(a.contains(&"item_b".to_string()));
    println!("Same hasher + seed:  merge succeeded");

    // Two filters with different seeds → merge is rejected.
    use bloomcraft::core::MergeableBloomFilter;
    let mut c: StandardBloomFilter<String, WyHasher> =
        StandardBloomFilter::with_hasher(100, 0.01, WyHasher::with_seed(0)).expect("valid params");
    let d: StandardBloomFilter<String, WyHasher> =
        StandardBloomFilter::with_hasher(100, 0.01, WyHasher::with_seed(1)).expect("valid params");
    c.insert(&"data".to_string());
    let result = MergeableBloomFilter::union(&mut c, &d);
    assert!(result.is_err());
    println!("Different seeds:     merge rejected (expected error)");

    // --- Feature-gated hashing ---

    println!("\n=== Feature-gated hashing ===\n");

    let h = bloomcraft::hash::recommended_hasher();
    let hash = h.hash_bytes(b"hello");
    assert_ne!(hash, 0);
    println!("recommended_hasher: {} — 0x{hash:016x}", h.name());

    let h = bloomcraft::hash::hasher_with_seed(42);
    println!(
        "hasher_with_seed(42): {} — 0x{:016x}",
        h.name(),
        h.hash_bytes(b"hello")
    );

    println!("\nAll checks passed.");
}
