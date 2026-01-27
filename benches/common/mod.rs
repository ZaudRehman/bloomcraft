//! Shared utilities and data generators for all benchmarks
//!
//! This module provides:
//! - Realistic data generators (strings, numbers, UUIDs)
//! - Common benchmark constants (sizes, FPRs, thread counts)
//! - Helper functions for filter setup
//! - Statistical utilities for analysis
//!
//! All benchmarks use these utilities to ensure consistency
//! and eliminate code duplication.
#![allow(dead_code)]
#![allow(unused_imports)]
use rand::distributions::{Alphanumeric, Distribution, Standard};
use rand::{thread_rng, Rng};
use std::collections::HashSet;

// DATA GENERATORS

/// Generate random alphanumeric string of specified length
///
/// # Examples
/// ```
/// let s = random_string(32);
/// assert_eq!(s.len(), 32);
/// ```
#[inline]
pub fn random_string(len: usize) -> String {
    thread_rng()
        .sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

/// Generate batch of random strings with specified length
///
/// Each string is independently random. For sequential strings,
/// use `generate_sequential_strings()`.
///
/// # Arguments
/// * `count` - Number of strings to generate
/// * `len` - Length of each string in bytes
///
/// # Examples
/// ```
/// let items = generate_strings(1000, 32);
/// assert_eq!(items.len(), 1000);
/// assert_eq!(items.len(), 32);
/// ```
pub fn generate_strings(count: usize, len: usize) -> Vec<String> {
    (0..count).map(|_| random_string(len)).collect()
}

/// Generate sequential strings with predictable format
///
/// Format: "item_00000001", "item_00000002", ...
/// Useful for testing scenarios where you need to verify exact items.
///
/// # Examples
/// ```
/// let items = generate_sequential_strings(100);
/// assert_eq!(items, "item_00000000");
/// assert_eq!(items, "item_00000099");
/// ```
pub fn generate_sequential_strings(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("item_{:08}", i)).collect()
}

/// Generate URL-like strings (realistic workload for web crawlers)
///
/// Format: "https://example.com/path/XXXXXXXX"
/// Where X is random alphanumeric
pub fn generate_urls(count: usize) -> Vec<String> {
    (0..count)
        .map(|_| format!("https://example.com/path/{}", random_string(16)))
        .collect()
}

/// Generate email-like strings (realistic for user deduplication)
///
/// Format: "userXXXXXXXX@example.com"
pub fn generate_emails(count: usize) -> Vec<String> {
    (0..count)
        .map(|_| format!("user{}@example.com", random_string(12)))
        .collect()
}

/// Generate random u64 values using cryptographically strong RNG
pub fn generate_u64s(count: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

/// Generate sequential u64 values (0, 1, 2, ...)
///
/// Useful for testing worst-case hash distribution scenarios
pub fn generate_sequential_u64s(count: usize) -> Vec<u64> {
    (0..count).map(|i| i as u64).collect()
}

/// Generate random u32 values
pub fn generate_u32s(count: usize) -> Vec<u32> {
    let mut rng = thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

/// Generate random UUIDs represented as [u8; 16]
///
/// Simulates realistic UUID storage (128-bit identifiers)
pub fn generate_uuids(count: usize) -> Vec<[u8; 16]> {
    let mut rng = thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

/// Generate IPv4 addresses as u32 values
///
/// Useful for testing network packet filtering scenarios
pub fn generate_ipv4s(count: usize) -> Vec<u32> {
    generate_u32s(count)
}

/// Generate realistic timestamp values (Unix epoch milliseconds)
///
/// Range: 2020-01-01 to 2030-01-01
pub fn generate_timestamps(count: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    let start = 1577836800000u64; // 2020-01-01
    let end = 1893456000000u64;   // 2030-01-01
    (0..count)
        .map(|_| rng.gen_range(start..end))
        .collect()
}

/// Generate mixed-case strings (tests hash function quality)
pub fn generate_mixed_case_strings(count: usize, len: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            (0..len)
                .map(|_| {
                    let c = rng.gen_range(b'a'..=b'z');
                    if rng.gen_bool(0.5) {
                        (c - 32) as char // Uppercase
                    } else {
                        c as char
                    }
                })
                .collect()
        })
        .collect()
}

/// Generate strings with common prefixes (worst case for some hash functions)
///
/// All strings share first 16 bytes, differ only in last 16 bytes
pub fn generate_prefixed_strings(count: usize) -> Vec<String> {
    let prefix = "common_prefix___";
    (0..count)
        .map(|i| format!("{}{:016}", prefix, i))
        .collect()
}

// BENCHMARK CONSTANTS

/// Standard filter sizes for benchmarking
///
/// Covers range from small caches (1K) to large databases (1M)
pub const SIZES: &[usize] = &[
    1_000,      // Small: L1 cache fits (~10KB)
    10_000,     // Medium: L2 cache (~100KB)
    100_000,    // Large: L3 cache (~1MB)
    1_000_000,  // XLarge: RAM (~10MB)
];

/// Extended size range for scalability testing
pub const SIZES_EXTENDED: &[usize] = &[
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
];

/// Common false positive rates
///
/// - 0.1 (10%): Fast, minimal hash functions
/// - 0.01 (1%): Common default, balanced
/// - 0.001 (0.1%): Tight, more memory
/// - 0.0001 (0.01%): Very tight, significant memory
pub const FP_RATES: &[f64] = &[0.1, 0.01, 0.001, 0.0001];

/// Minimal FPR set for quick benchmarks
pub const FP_RATES_QUICK: &[f64] = &[0.1, 0.01, 0.001];

/// Batch sizes for bulk operations
///
/// Covers single-item (1) to large batches (10K)
pub const BATCH_SIZES: &[usize] = &[1, 10, 100, 1_000, 10_000];

/// Small batch sizes for quick benchmarks
pub const BATCH_SIZES_SMALL: &[usize] = &[10, 100, 1_000];

/// Thread counts for concurrency benchmarks
///
/// Covers single-threaded to highly parallel (32 threads)
pub const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32];

/// Realistic thread counts for most systems
pub const THREAD_COUNTS_REALISTIC: &[usize] = &[1, 2, 4, 8];

/// Load factors as percentages (10% = 0.1 fill rate)
///
/// Tests performance at different filter saturation levels
pub const LOAD_FACTORS: &[usize] = &[10, 25, 50, 75, 90, 99];

/// String lengths for type benchmarks (bytes)
pub const STRING_LENGTHS: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024];

/// Typical string lengths
pub const STRING_LENGTHS_TYPICAL: &[usize] = &[32, 64, 128, 256];

// FILTER SETUP HELPERS

/// Fill filter to specified percentage
///
/// # Arguments
/// * `filter` - The filter to fill
/// * `items` - Items to insert
/// * `target_pct` - Target fill percentage (0-100)
///
/// # Returns
/// Number of items actually inserted
pub fn fill_filter<F, T>(filter: &mut F, items: &[T], target_pct: usize) -> usize
where
    F: bloomcraft::core::BloomFilter<T>,
    T: std::hash::Hash,
{
    let capacity = filter.expected_items();
    let target_count = (capacity * target_pct) / 100;
    let actual_count = target_count.min(items.len());
    
    for i in 0..actual_count {
        filter.insert(&items[i]);
    }
    
    actual_count
}

/// Create two disjoint sets of items (for testing hits vs misses)
///
/// Returns (present_items, absent_items)
pub fn create_disjoint_sets(
    present_count: usize,
    absent_count: usize,
    item_size: usize,
) -> (Vec<String>, Vec<String>) {
    let mut seen = HashSet::new();
    let mut present = Vec::with_capacity(present_count);
    let mut absent = Vec::with_capacity(absent_count);
    
    // Generate present items
    while present.len() < present_count {
        let item = random_string(item_size);
        if seen.insert(item.clone()) {
            present.push(item);
        }
    }
    
    // Generate absent items (guaranteed different)
    while absent.len() < absent_count {
        let item = random_string(item_size);
        if seen.insert(item.clone()) {
            absent.push(item);
        }
    }
    
    (present, absent)
}

// STATISTICAL UTILITIES

/// Calculate actual false positive rate from queries
///
/// # Arguments
/// * `true_negatives` - Items definitely not in set
/// * `filter_results` - Filter's answers for those items
///
/// # Returns
/// Measured FPR as a fraction (0.0 to 1.0)
pub fn measure_fpr(true_negatives: &[String], filter_results: &[bool]) -> f64 {
    assert_eq!(true_negatives.len(), filter_results.len());
    
    let false_positives = filter_results.iter().filter(|&&r| r).count();
    false_positives as f64 / true_negatives.len() as f64
}

/// Calculate throughput in operations per second
pub fn calculate_throughput(operations: usize, duration_ns: u128) -> f64 {
    (operations as f64 / duration_ns as f64) * 1_000_000_000.0
}

/// Format throughput in human-readable form
pub fn format_throughput(ops_per_sec: f64) -> String {
    if ops_per_sec >= 1_000_000.0 {
        format!("{:.2}M ops/s", ops_per_sec / 1_000_000.0)
    } else if ops_per_sec >= 1_000.0 {
        format!("{:.2}K ops/s", ops_per_sec / 1_000.0)
    } else {
        format!("{:.2} ops/s", ops_per_sec)
    }
}

/// Format memory size in human-readable form
pub fn format_memory(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format latency in human-readable form
pub fn format_latency(nanos: u128) -> String {
    if nanos >= 1_000_000_000 {
        format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
    } else if nanos >= 1_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.2} µs", nanos as f64 / 1_000.0)
    } else {
        format!("{} ns", nanos)
    }
}

// WORKLOAD PATTERNS

/// Generate Zipfian-distributed access pattern (realistic for caches)
///
/// Models real-world access patterns where some items are accessed
/// much more frequently than others (80/20 rule).
///
/// # Arguments
/// * `item_count` - Total number of unique items
/// * `access_count` - Number of accesses to generate
/// * `skew` - Zipfian skew parameter (1.0 = moderate, 2.0 = heavy)
pub fn generate_zipfian_pattern(
    item_count: usize,
    access_count: usize,
    skew: f64,
) -> Vec<usize> {
    // Simplified Zipfian using power law
    let mut rng = thread_rng();
    let mut accesses = Vec::with_capacity(access_count);
    
    for _ in 0..access_count {
        let u: f64 = rng.gen();
        let rank = ((item_count as f64).powf(u.powf(skew))) as usize;
        accesses.push(rank.min(item_count - 1));
    }
    
    accesses
}

/// Generate uniform random access pattern
pub fn generate_uniform_pattern(item_count: usize, access_count: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    (0..access_count)
        .map(|_| rng.gen_range(0..item_count))
        .collect()
}

/// Generate sequential access pattern (best case for cache)
pub fn generate_sequential_pattern(item_count: usize, access_count: usize) -> Vec<usize> {
    (0..access_count).map(|i| i % item_count).collect()
}

/// Generate worst-case pattern (alternating between far-apart items)
pub fn generate_worst_case_pattern(item_count: usize, access_count: usize) -> Vec<usize> {
    let stride = item_count / 2;
    (0..access_count)
        .map(|i| (i * stride) % item_count)
        .collect()
}

// CONFIGURATION

/// Default benchmark configuration
pub struct BenchConfig {
    pub size: usize,
    pub fpr: f64,
    pub item_size: usize,
    pub load_factor: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            size: 100_000,
            fpr: 0.01,
            item_size: 32,
            load_factor: 50,
        }
    }
}

impl BenchConfig {
    /// Quick benchmark (small size, fast)
    pub fn quick() -> Self {
        Self {
            size: 10_000,
            fpr: 0.01,
            item_size: 32,
            load_factor: 50,
        }
    }
    
    /// Standard benchmark (medium size, balanced)
    pub fn standard() -> Self {
        Self::default()
    }
    
    /// Intensive benchmark (large size, thorough)
    pub fn intensive() -> Self {
        Self {
            size: 1_000_000,
            fpr: 0.001,
            item_size: 64,
            load_factor: 75,
        }
    }
}


// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_string() {
        let s = random_string(32);
        assert_eq!(s.len(), 32);
        assert!(s.chars().all(|c| c.is_alphanumeric()));
    }
    
    #[test]
    fn test_generate_strings() {
        let items = generate_strings(100, 16);
        assert_eq!(items.len(), 100);
        assert!(items.iter().all(|s| s.len() == 16));
    }
    
    #[test]
    fn test_sequential_strings() {
        let items = generate_sequential_strings(5);
        assert_eq!(items[0], "item_00000000");
        assert_eq!(items[4], "item_00000004");
    }
    
    #[test]
    fn test_disjoint_sets() {
        let (present, absent) = create_disjoint_sets(100, 100, 16);
        assert_eq!(present.len(), 100);
        assert_eq!(absent.len(), 100);
        
        let present_set: HashSet<_> = present.iter().collect();
        let absent_set: HashSet<_> = absent.iter().collect();
        assert_eq!(present_set.intersection(&absent_set).count(), 0);
    }
    
    #[test]
    fn test_format_memory() {
        assert_eq!(format_memory(512), "512 B");
        assert_eq!(format_memory(2048), "2.00 KB");
        assert_eq!(format_memory(5 * 1024 * 1024), "5.00 MB");
    }
    
    #[test]
    fn test_format_throughput() {
        assert_eq!(format_throughput(500.0), "500.00 ops/s");
        assert_eq!(format_throughput(15_000.0), "15.00K ops/s");
        assert_eq!(format_throughput(12_500_000.0), "12.50M ops/s");
    }
}
