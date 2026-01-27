//! Core types, traits, and utilities for BloomCraft.
//!
//! This module contains the fundamental building blocks used by all Bloom filter variants:
//!
//! - **Traits**: `BloomFilter`, `ConcurrentBloomFilter`, `SharedBloomFilter`, `DeletableBloomFilter`, `MergeableBloomFilter`, `ScalableBloomFilter`
//! - **Data Structures**: `BitVec` (lock-free bit vector)
//! - **Utilities**: Parameter calculation functions
//!
//! # Module Organization
//!
//! ```text
//! core/
//! ├── filter.rs    - Trait definitions
//! ├── bitvec.rs    - Bit vector implementation
//! ├── params.rs    - Parameter calculations
//! └── mod.rs       - This file (public API)
//! ```
//!
//! # Design Principles
//!
//! 1. **Separation of Concerns**: Traits, data structures, and utilities are independent
//! 2. **Zero-Cost Abstractions**: No runtime overhead for type safety
//! 3. **Compile-Time Safety**: Invalid configurations rejected at compile time where possible
//! 4. **Thread Safety**: All types are `Send + Sync` for concurrent use
//! 5. **Three Concurrency Models**: Single-threaded, lock-free atomic, and interior mutability patterns
//!
//! # Concurrency Architecture
//!
//! BloomCraft provides three distinct concurrency patterns:
//!
//! ## Single-Threaded (`BloomFilter`)
//! - Methods require `&mut self`
//! - Zero synchronization overhead
//! - Examples: CountingBloomFilter, ScalableBloomFilter
//! - Usage: Direct access or wrap in `Mutex`/`RwLock`
//!
//! ## Lock-Free Atomic (`ConcurrentBloomFilter`)
//! - Extension trait for filters using atomic operations
//! - Methods: `insert_concurrent(&self)`
//! - Example: StandardBloomFilter (BitVec uses AtomicU64)
//! - Usage: `Arc<StandardBloomFilter>` - no Mutex needed!
//!
//! ## Interior Mutability (`SharedBloomFilter`)
//! - Separate trait for sharded/striped filters
//! - Methods take `&self` (standard names)
//! - Examples: ShardedBloomFilter, StripedBloomFilter
//! - Usage: `Arc<ShardedBloomFilter>` - no Mutex needed!
//!
//! # Examples
//!
//! ## Single-Threaded Filter
//!
//! ```ignore
//! use bloomcraft::core::BloomFilter;
//! use bloomcraft::StandardBloomFilter;
//!
//! let mut filter = StandardBloomFilter::<String>::new(10000, 0.01);
//! filter.insert(&"alice".to_string());
//! assert!(filter.contains(&"alice".to_string()));
//! ```
//!
//! ## Lock-Free Atomic Filter (StandardBloomFilter)
//!
//! ```ignore
//! use bloomcraft::core::ConcurrentBloomFilter;
//! use bloomcraft::StandardBloomFilter;
//! use std::sync::Arc;
//! use std::thread;
//!
//! // No Mutex needed - atomic operations!
//! let filter = Arc::new(StandardBloomFilter::<String>::new(10000, 0.01));
//!
//! let handles: Vec<_> = (0..4).map(|i| {
//!     let f = Arc::clone(&filter);
//!     thread::spawn(move || {
//!         f.insert_concurrent(&format!("item-{}", i));
//!     })
//! }).collect();
//!
//! for h in handles { h.join().unwrap(); }
//! ```
//!
//! ## Sharded Interior Mutability Filter
//!
//! ```ignore
//! use bloomcraft::core::SharedBloomFilter;
//! use bloomcraft::sync::ShardedBloomFilter;
//! use std::sync::Arc;
//! use std::thread;
//!
//! // No Mutex needed - sharding provides concurrency!
//! let filter = Arc::new(ShardedBloomFilter::<String>::new(10000, 0.01));
//!
//! let handles: Vec<_> = (0..4).map(|i| {
//!     let f = Arc::clone(&filter);
//!     thread::spawn(move || {
//!         f.insert(&format!("item-{}", i));  // &self method!
//!     })
//! }).collect();
//!
//! for h in handles { h.join().unwrap(); }
//! ```
//!
//! ## Using MergeableBloomFilter
//!
//! ```ignore
//! use bloomcraft::core::{BloomFilter, MergeableBloomFilter};
//! use bloomcraft::StandardBloomFilter;
//!
//! let mut filter1 = StandardBloomFilter::<String>::new(10000, 0.01);
//! let mut filter2 = StandardBloomFilter::<String>::new(10000, 0.01);
//!
//! filter1.insert(&"alice".to_string());
//! filter2.insert(&"bob".to_string());
//!
//! filter1.union(&filter2).unwrap();
//! assert!(filter1.contains(&"alice".to_string()));
//! assert!(filter1.contains(&"bob".to_string()));
//! ```
//!
//! ## Using Parameter Calculations
//!
//! ```
//! use bloomcraft::core::params::{optimal_bit_count, optimal_hash_count};
//!
//! // Calculate optimal parameters for 10K items with 1% FP rate
//! let m = optimal_bit_count(10_000, 0.01).unwrap();
//! let k = optimal_hash_count(m, 10_000).unwrap();
//!
//! println!("Need {} bits and {} hash functions", m, k);
//! // Output: Need 95851 bits and 7 hash functions
//! ```
//!
//! ## Using BitVec Directly
//!
//! ```
//! use bloomcraft::core::BitVec;
//!
//! let bv = BitVec::new(1000).expect("BitVec creation should succeed");
//! bv.set(42);
//! bv.set(100);
//! bv.set(999);
//!
//! assert!(bv.get(42));
//! assert!(!bv.get(43));
//! assert_eq!(bv.count_ones(), 3);
//! ```

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Public modules
pub mod filter;
pub mod params;
pub mod bitvec;

// Re-export main traits for convenience
pub use filter::{
    BloomFilter,
    MutableBloomFilter,
    ConcurrentBloomFilter,      // Extension trait for StandardBloomFilter
    SharedBloomFilter,           // For sharded/striped filters
    DeletableBloomFilter,
    MergeableBloomFilter,
    ScalableBloomFilter,
};

// Re-export BitVec as it's needed by variant implementations
pub use bitvec::BitVec;

// Re-export commonly used parameter functions
pub use params::{
    optimal_bit_count,
    optimal_hash_count,
    expected_fp_rate,
    calculate_filter_params,
    validate_params,
    bits_per_element,
};

/// Prelude module for convenient imports.
///
/// Import everything from the prelude to get started quickly:
///
/// ```ignore
/// use bloomcraft::core::prelude::*;
///
/// let mut filter = StandardBloomFilter::<String>::new(10000, 0.01);
/// filter.insert(&"hello".to_string());
/// ```
pub mod prelude {
    pub use super::filter::{
        BloomFilter,
        ConcurrentBloomFilter,
        SharedBloomFilter,
        DeletableBloomFilter,
        MergeableBloomFilter,
        ScalableBloomFilter,
    };
    pub use super::params::{
        optimal_bit_count,
        optimal_hash_count,
        calculate_filter_params,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports_compile() {
        // Verify prelude contains expected items
        use prelude::*;
        
        // These should compile if prelude is correct
        let _ = optimal_bit_count(1000, 0.01);
        let _ = optimal_hash_count(9585, 1000);
    }

    #[test]
    fn test_module_reexports() {
        // Verify re-exports work
        let bv = BitVec::new(100).expect("BitVec creation should succeed");
        assert_eq!(bv.len(), 100);
        
        let m = optimal_bit_count(1000, 0.01).unwrap();
        assert!(m > 9500 && m < 9600);
    }

    #[test]
    fn test_integration_params_and_bitvec() {
        // Test that params and BitVec work together
        let n = 1000;
        let fp_rate = 0.01;
        
        let m = optimal_bit_count(n, fp_rate).unwrap();
        let k = optimal_hash_count(m, n).unwrap();
        
        // Create a BitVec with calculated size
        let bv = BitVec::new(m).expect("BitVec creation should succeed");
        assert_eq!(bv.len(), m);
        
        // Verify parameters are reasonable
        assert!(k >= 5 && k <= 10);
    }

    #[test]
    fn test_bitvec_basic_operations() {
        let bv = BitVec::new(1000).expect("BitVec creation should succeed");
        bv.set(42);
        bv.set(100);
        bv.set(999);
        
        assert!(bv.get(42));
        assert!(bv.get(100));
        assert!(bv.get(999));
        assert!(!bv.get(43));
        
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_parameter_validation() {
        // Valid parameters
        assert!(validate_params(1000, 100, 7).is_ok());
        
        // Invalid: zero bits
        assert!(validate_params(0, 100, 7).is_err());
        
        // Invalid: zero items
        assert!(validate_params(1000, 0, 7).is_err());
        
        // Invalid: too many hash functions
        assert!(validate_params(1000, 100, 100).is_err());
    }

    #[test]
    fn test_calculate_filter_params_consistency() {
        // Calculate params and verify they're internally consistent
        let n = 5000;
        let target_fp = 0.01;
        
        let (m, k) = calculate_filter_params(n, target_fp).unwrap();
        
        // Verify parameters are valid
        assert!(validate_params(m, n, k).is_ok());
        
        // Verify FP rate is close to target
        let actual_fp = expected_fp_rate(m, n, k).unwrap();
        assert!((actual_fp - target_fp).abs() / target_fp < 0.15);
    }

    #[test]
    fn test_bits_per_element_calculation() {
        // 1% FP rate should require ~9.6 bits per element
        let bpe = bits_per_element(0.01).unwrap();
        assert!((bpe - 9.6).abs() < 0.1);
        
        // Verify it matches optimal_bit_count
        let n = 1000;
        let m = optimal_bit_count(n, 0.01).unwrap();
        let calculated_bpe = m as f64 / n as f64;
        assert!((calculated_bpe - bpe).abs() < 0.1);
    }

    #[test]
    fn test_bitvec_concurrent_safety() {
        use std::sync::Arc;
        use std::thread;
        
        let bv = Arc::new(BitVec::new(10000).expect("BitVec creation should succeed"));
        let mut handles = vec![];
        
        // Spawn 8 threads that each set 100 bits
        for t in 0..8 {
            let bv = Arc::clone(&bv);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    bv.set(t * 100 + i);
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // All 800 bits should be set
        assert_eq!(bv.count_ones(), 800);
    }

    #[test]
    fn test_expected_fp_rate_empty_filter() {
        // Empty filter should have 0% FP rate
        let fp = expected_fp_rate(1000, 0, 7).unwrap();
        assert_eq!(fp, 0.0);
    }

    #[test]
    fn test_expected_fp_rate_full_filter() {
        // Filter with n=m (one item per bit) should have very high FP rate
        let fp = expected_fp_rate(1000, 1000, 7).unwrap();
        assert!(fp > 0.5);
    }

    #[test]
    fn test_bitvec_set_multiple_and_test_all() {
        let bv = BitVec::new(1000).expect("BitVec creation should succeed");
        let indices = [10, 20, 30, 40, 50];
        
        for &idx in &indices {
            bv.set(idx);
        }
        
        // Test all indices are set
        assert!(indices.iter().all(|&idx| bv.get(idx)));
        
        // Test that 99 is not set
        assert!(!bv.get(99));
    }

    #[test]
    fn test_bitvec_union_operation() {
        let bv1 = BitVec::new(1000).expect("BitVec creation should succeed");
        let bv2 = BitVec::new(1000).expect("BitVec creation should succeed");
        
        bv1.set(10);
        bv1.set(20);
        
        bv2.set(20);
        bv2.set(30);
        
        let result = bv1.union(&bv2).unwrap();
        
        assert!(result.get(10));
        assert!(result.get(20));
        assert!(result.get(30));
        assert_eq!(result.count_ones(), 3);
    }

    #[test]
    fn test_bitvec_intersect_operation() {
        let bv1 = BitVec::new(1000).expect("BitVec creation should succeed");
        let bv2 = BitVec::new(1000).expect("BitVec creation should succeed");
        
        bv1.set(10);
        bv1.set(20);
        bv1.set(30);
        
        bv2.set(20);
        bv2.set(30);
        bv2.set(40);
        
        let result = bv1.intersect(&bv2).unwrap();
        
        assert!(!result.get(10)); // Not in intersection
        assert!(result.get(20));  // In both
        assert!(result.get(30));  // In both
        assert!(!result.get(40)); // Not in bv1 originally
        
        assert_eq!(result.count_ones(), 2);
    }

    #[test]
    fn test_bitvec_fill_fraction() {
        let bv = BitVec::new(1000).expect("BitVec creation should succeed");
        
        // Initially empty
        assert_eq!(bv.count_ones(), 0);
        
        // Set 250 bits (25%)
        for i in 0..250 {
            bv.set(i);
        }
        
        let one_fraction = bv.count_ones() as f64 / bv.len() as f64;
        assert!((one_fraction - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_bitvec_clear() {
        let mut bv = BitVec::new(1000).expect("BitVec creation should succeed");
        bv.set(10);
        bv.set(20);
        bv.set(30);
        
        assert_eq!(bv.count_ones(), 3);
        
        bv.clear();
        
        assert_eq!(bv.count_ones(), 0);
    }

    #[test]
    fn test_bitvec_is_full() {
        let bv = BitVec::new(64).expect("BitVec creation should succeed");
        
        // Initially not full
        assert!(bv.count_ones() < bv.len());
        
        // Set all bits
        for i in 0..64 {
            bv.set(i);
        }
        
        // Now full
        assert_eq!(bv.count_ones(), bv.len());
    }

    #[test]
    fn test_bitvec_memory_usage() {
        let bv = BitVec::new(1000).expect("BitVec creation should succeed");
        
        // ⌈1000/64⌉ × 8 = 16 × 8 = 128 bytes (plus struct overhead)
        assert!(bv.memory_usage() >= 128);
        
        let bv2 = BitVec::new(64).expect("BitVec creation should succeed");
        assert!(bv2.memory_usage() >= 8);
    }

    #[test]
    fn test_optimal_bit_count_various_fp_rates() {
        let n = 1000;
        
        // Test different FP rates with expected theoretical values
        let test_cases = vec![
            (0.1, 4792),
            (0.01, 9585),
            (0.001, 14377),
            (0.0001, 19170),
        ];
        
        for (fp_rate, expected_m) in test_cases {
            let m = optimal_bit_count(n, fp_rate).unwrap();
            assert!(
                (m as i32 - expected_m).abs() <= 1,
                "For fp_rate={}, expected ~{}, got {}",
                fp_rate,
                expected_m,
                m
            );
        }
    }

    #[test]
    fn test_optimal_hash_count_various_ratios() {
        // Test different m/n ratios
        let test_cases = vec![
            (1000, 100, 7),   // m/n = 10
            (2000, 100, 14),  // m/n = 20
            (500, 100, 3),    // m/n = 5
        ];
        
        for (m, n, expected_k) in test_cases {
            let k = optimal_hash_count(m, n).unwrap();
            assert_eq!(
                k, expected_k,
                "For m={}, n={}, expected k={}, got {}",
                m, n, expected_k, k
            );
        }
    }

    #[test]
    fn test_parameter_error_conditions() {
        // Zero items
        assert!(optimal_bit_count(0, 0.01).is_err());
        
        // Invalid FP rates
        assert!(optimal_bit_count(1000, 0.0).is_err());
        assert!(optimal_bit_count(1000, 1.0).is_err());
        assert!(optimal_bit_count(1000, -0.1).is_err());
        assert!(optimal_bit_count(1000, 1.5).is_err());
        
        // Zero bits
        assert!(optimal_hash_count(0, 1000).is_err());
        
        // Invalid hash count in expected_fp_rate
        assert!(expected_fp_rate(1000, 100, 0).is_err());
        assert!(expected_fp_rate(1000, 100, 100).is_err());
    }

    #[test]
    fn test_roundtrip_parameter_calculation() {
        // Calculate params, then verify FP rate matches target
        let n = 5000;
        let target_fp = 0.005;
        
        let (m, k) = calculate_filter_params(n, target_fp).unwrap();
        let actual_fp = expected_fp_rate(m, n, k).unwrap();
        
        // Should be within 15% of target
        let error = (actual_fp - target_fp).abs() / target_fp;
        assert!(
            error < 0.15,
            "FP rate error {:.1}% exceeds 15%. Target: {}, Actual: {}",
            error * 100.0,
            target_fp,
            actual_fp
        );
    }

    #[test]
    fn test_bitvec_alignment() {
        let bv = BitVec::new(1000).expect("BitVec creation should succeed");
        let ptr = &bv as *const _ as usize;
        
        // Check alignment (may not be 64-byte aligned without explicit annotation)
        // This test documents current behavior
        assert!(ptr % 8 == 0, "BitVec should be at least 8-byte aligned");
    }

    #[test]
    fn test_bitvec_basic() {
        let bv = BitVec::new(1000).expect("BitVec creation should succeed");
        assert_eq!(bv.len(), 1000);
        assert_eq!(bv.count_ones(), 0);
        assert!(!bv.is_empty());
    }

    #[test]
    fn test_mathematical_consistency() {
        // Verify optimal_bit_count and bits_per_element are consistent
        let n = 10000;
        let fp_rate = 0.001;
        
        let m = optimal_bit_count(n, fp_rate).unwrap();
        let bpe = bits_per_element(fp_rate).unwrap();
        let expected_m = (n as f64 * bpe).ceil() as usize;
        
        assert_eq!(m, expected_m);
    }

    #[test]
    fn test_validate_params_edge_cases() {
        // Valid edge cases
        assert!(validate_params(1, 1, 1).is_ok());
        assert!(validate_params(100, 1, 1).is_ok());
        
        // Invalid: m < k
        assert!(validate_params(5, 100, 10).is_err());
        
        // Invalid: load factor > 2.0
        assert!(validate_params(100, 250, 7).is_err());
    }

    #[test]
    fn test_validate_params_allows_moderate_saturation() {
        // Load factor between 1.0 and 2.0 should be allowed
        assert!(validate_params(100, 150, 7).is_ok());
        assert!(validate_params(100, 200, 7).is_ok());
    }
}
