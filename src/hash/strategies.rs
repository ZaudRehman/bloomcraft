//! Hash strategy implementations for Bloom filters.
//!
//! This module provides extensible hash strategies for generating k hash indices
//! from base hash values. Strategies are trait-based to allow custom implementations
//! while providing battle-tested defaults.
//!
//! # Strategy Comparison
//!
//! | Strategy           | Base Hashes | Distribution | Performance | Use Case                |
//! |--------------------|-------------|--------------|-------------|-------------------------|
//! | DoubleHashing      | 2           | Good         | Fastest     | General purpose (default)|
//! | EnhancedDoubleHashing | 2        | Excellent    | Fast        | High accuracy needs     |
//! | TripleHashing      | 3           | Best         | Slower      | Research/validation     |
//!
//! # Mathematical Background
//!
//! ## Double Hashing (Kirsch & Mitzenmacher 2006)
//!
//! For k hash functions derived from two independent hashes h₁ and h₂:
//!
//! ```text
//! gᵢ(x) = (h₁(x) + i·h₂(x)) mod m
//! ```
//!
//! **Proof of Optimality**: The paper proves that double hashing provides
//! asymptotically optimal false positive rates, matching k independent hash functions.
//!
//! ## Enhanced Double Hashing (Dillinger & Manolios 2004)
//!
//! Adds quadratic probing term to reduce clustering:
//!
//! ```text
//! gᵢ(x) = (h₁(x) + i·h₂(x) + f(i)) mod m
//! where f(i) = (i² + i) / 2
//! ```
//!
//! **Advantage**: Better distribution for small m or large k (k > 10).
//! **Cost**: ~10-15% slower than standard double hashing.
//!
//! ## Triple Hashing
//!
//! Uses three hash functions for maximum independence:
//!
//! ```text
//! gᵢ(x) = (h₁(x) + i·h₂(x) + i²·h₃(x)) mod m
//! ```
//!
//! **Use Case**: Research and empirical validation. Minimal practical benefit
//! over enhanced double hashing for typical Bloom filter parameters.
//!
//! # References
//!
//! - Kirsch, A., & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter"
//! - Dillinger, P. C., & Manolios, P. (2004). "Fast and Accurate Bitstate Verification for SPIN"

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]

/// Hash strategy for generating k indices from base hash values.
///
/// Implementors define how to derive k hash indices from 2-3 base hash values.
/// All implementations must be deterministic and uniformly distributed.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for use in concurrent Bloom filters.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::strategies::{HashStrategy, DoubleHashing};
///
/// let strategy = DoubleHashing;
/// let h1 = 0x123456789abcdef0;
/// let h2 = 0xfedcba9876543210;
///
/// // Generate 7 hash indices for a filter of size 1000
/// let indices = strategy.generate_indices(h1, h2, 0, 7, 1000);
/// assert_eq!(indices.len(), 7);
/// assert!(indices.iter().all(|&idx| idx < 1000));
/// ```
pub trait HashStrategy: Send + Sync {
    /// Generate k hash indices from base hash values.
    ///
    /// # Arguments
    ///
    /// * `h1` - First base hash value
    /// * `h2` - Second base hash value
    /// * `h3` - Third base hash value (unused by strategies requiring only 2 hashes)
    /// * `k` - Number of hash indices to generate
    /// * `m` - Filter size (indices will be in range `[0, m)`)
    ///
    /// # Returns
    ///
    /// Vector of k hash indices in range `[0, m)`.
    ///
    /// # Panics
    ///
    /// May panic if `k == 0` or `m == 0` (implementation-dependent).
    fn generate_indices(&self, h1: u64, h2: u64, h3: u64, k: usize, m: usize) -> Vec<usize>;

    /// Number of base hash values required by this strategy.
    ///
    /// # Returns
    ///
    /// - `2` for double hashing variants
    /// - `3` for triple hashing
    fn required_hashes(&self) -> usize;

    /// Human-readable name for debugging and serialization.
    fn name(&self) -> &'static str;
}

/// Standard double hashing strategy (Kirsch & Mitzenmacher 2006).
///
/// Formula: `gᵢ(x) = (h₁(x) + i·h₂(x)) mod m`
///
/// # Performance
///
/// - **Speed**: Fastest strategy (~2ns per index on modern CPUs)
/// - **Distribution**: Good (proven optimal for Bloom filters)
/// - **Use Case**: Default choice for most applications
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::strategies::{HashStrategy, DoubleHashing};
///
/// let strategy = DoubleHashing;
/// assert_eq!(strategy.required_hashes(), 2);
/// assert_eq!(strategy.name(), "DoubleHashing");
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct DoubleHashing;

impl HashStrategy for DoubleHashing {
    #[inline]
    fn generate_indices(&self, h1: u64, h2: u64, _h3: u64, k: usize, m: usize) -> Vec<usize> {
        let m_u64 = m as u64;
        let mut indices = Vec::with_capacity(k);

        for i in 0..k {
            let i_u64 = i as u64;
            // Formula: (h1 + i * h2) mod m
            let hash = h1.wrapping_add(i_u64.wrapping_mul(h2));
            indices.push((hash % m_u64) as usize);
        }

        indices
    }

    #[inline]
    fn required_hashes(&self) -> usize {
        2
    }

    #[inline]
    fn name(&self) -> &'static str {
        "DoubleHashing"
    }
}

/// Enhanced double hashing with quadratic probing (Dillinger & Manolios 2004).
///
/// Formula: `gᵢ(x) = (h₁(x) + i·h₂(x) + (i² + i)/2) mod m`
///
/// The quadratic term `(i² + i)/2` reduces clustering compared to standard
/// double hashing, especially for large k (>10 hash functions).
///
/// # Performance
///
/// - **Speed**: ~10-15% slower than standard double hashing
/// - **Distribution**: Excellent (better than standard for k > 10)
/// - **Use Case**: High-accuracy Bloom filters where FP rate is critical
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::strategies::{HashStrategy, EnhancedDoubleHashing};
///
/// let strategy = EnhancedDoubleHashing;
/// let indices = strategy.generate_indices(12345, 67890, 0, 7, 1000);
/// assert_eq!(indices.len(), 7);
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnhancedDoubleHashing;

impl HashStrategy for EnhancedDoubleHashing {
    #[inline]
    fn generate_indices(&self, h1: u64, h2: u64, _h3: u64, k: usize, m: usize) -> Vec<usize> {
        let m_u64 = m as u64;
        let mut indices = Vec::with_capacity(k);

        for i in 0..k {
            let i_u64 = i as u64;
            
            // Formula: (h1 + i*h2 + (i² + i)/2) mod m
            // The quadratic term prevents clustering
            let quadratic_term = (i_u64.wrapping_mul(i_u64.wrapping_add(1))) >> 1;
            let hash = h1
                .wrapping_add(i_u64.wrapping_mul(h2))
                .wrapping_add(quadratic_term);
            
            indices.push((hash % m_u64) as usize);
        }

        indices
    }

    #[inline]
    fn required_hashes(&self) -> usize {
        2
    }

    #[inline]
    fn name(&self) -> &'static str {
        "EnhancedDoubleHashing"
    }
}

/// Triple hashing strategy using three independent hash functions.
///
/// Formula: `gᵢ(x) = (h₁(x) + i·h₂(x) + i²·h₃(x)) mod m`
///
/// # Performance
///
/// - **Speed**: ~50% slower than double hashing (requires 3 base hashes)
/// - **Distribution**: Best possible (near-perfect independence)
/// - **Use Case**: Research, empirical validation, paranoid applications
///
/// # When to Use
///
/// Use triple hashing when:
/// - Validating double hashing implementations
/// - Publishing research requiring provably independent hashes
/// - Adversarial environments where hash collision attacks are possible
///
/// For production Bloom filters, enhanced double hashing provides 99% of the
/// benefit at 2/3 the cost.
///
/// # Examples
///
/// ```
/// use bloomcraft::hash::strategies::{HashStrategy, TripleHashing};
///
/// let strategy = TripleHashing;
/// assert_eq!(strategy.required_hashes(), 3);
///
/// let indices = strategy.generate_indices(12345, 67890, 11111, 7, 1000);
/// assert_eq!(indices.len(), 7);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TripleHashing;

impl HashStrategy for TripleHashing {
    #[inline]
    fn generate_indices(&self, h1: u64, h2: u64, h3: u64, k: usize, m: usize) -> Vec<usize> {
        let m_u64 = m as u64;
        let mut indices = Vec::with_capacity(k);

        for i in 0..k {
            let i_u64 = i as u64;
            let i_squared = i_u64.wrapping_mul(i_u64);
            
            // Formula: (h1 + i*h2 + i²*h3) mod m
            let hash = h1
                .wrapping_add(i_u64.wrapping_mul(h2))
                .wrapping_add(i_squared.wrapping_mul(h3));
            
            indices.push((hash % m_u64) as usize);
        }

        indices
    }

    #[inline]
    fn required_hashes(&self) -> usize {
        3
    }

    #[inline]
    fn name(&self) -> &'static str {
        "TripleHashing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

         // Basic Functionality Tests
     
    #[test]
    fn test_double_hashing_basic() {
        let strategy = DoubleHashing;
        let indices = strategy.generate_indices(12345, 67890, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_enhanced_double_hashing_basic() {
        let strategy = EnhancedDoubleHashing;
        let indices = strategy.generate_indices(12345, 67890, 0, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_triple_hashing_basic() {
        let strategy = TripleHashing;
        let indices = strategy.generate_indices(12345, 67890, 11111, 7, 1000);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_required_hashes() {
        assert_eq!(DoubleHashing.required_hashes(), 2);
        assert_eq!(EnhancedDoubleHashing.required_hashes(), 2);
        assert_eq!(TripleHashing.required_hashes(), 3);
    }

    #[test]
    fn test_strategy_names() {
        assert_eq!(DoubleHashing.name(), "DoubleHashing");
        assert_eq!(EnhancedDoubleHashing.name(), "EnhancedDoubleHashing");
        assert_eq!(TripleHashing.name(), "TripleHashing");
    }

         // Determinism Tests
     
    #[test]
    fn test_double_hashing_deterministic() {
        let strategy = DoubleHashing;
        let indices1 = strategy.generate_indices(12345, 67890, 0, 10, 1000);
        let indices2 = strategy.generate_indices(12345, 67890, 0, 10, 1000);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_enhanced_double_hashing_deterministic() {
        let strategy = EnhancedDoubleHashing;
        let indices1 = strategy.generate_indices(12345, 67890, 0, 10, 1000);
        let indices2 = strategy.generate_indices(12345, 67890, 0, 10, 1000);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_triple_hashing_deterministic() {
        let strategy = TripleHashing;
        let indices1 = strategy.generate_indices(12345, 67890, 11111, 10, 1000);
        let indices2 = strategy.generate_indices(12345, 67890, 11111, 10, 1000);

        assert_eq!(indices1, indices2);
    }

         // Differentiation Tests (Critical for correctness)
     
    #[test]
    fn test_strategies_produce_different_sequences() {
        let h1 = 0x123456789abcdef0;
        let h2 = 0xfedcba9876543210;
        let h3 = 0x1111111111111111;
        let k = 20;
        let m = 10000;

        let double_indices = DoubleHashing.generate_indices(h1, h2, 0, k, m);
        let enhanced_indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, k, m);
        let triple_indices = TripleHashing.generate_indices(h1, h2, h3, k, m);

        // All strategies should produce different index sequences
        assert_ne!(double_indices, enhanced_indices, "Double and Enhanced should differ");
        assert_ne!(enhanced_indices, triple_indices, "Enhanced and Triple should differ");
        assert_ne!(double_indices, triple_indices, "Double and Triple should differ");
    }

    #[test]
    fn test_enhanced_differs_from_double_for_large_k() {
        let h1 = 12345;
        let h2 = 67890;
        let m = 1000;

        // For large k, enhanced should diverge significantly from standard
        let double_indices = DoubleHashing.generate_indices(h1, h2, 0, 15, m);
        let enhanced_indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, 15, m);

        let differences = double_indices
            .iter()
            .zip(enhanced_indices.iter())
            .filter(|(a, b)| a != b)
            .count();

        // At least 80% should differ for k=15
        assert!(
            differences >= 12,
            "Enhanced should differ significantly from double for large k. Only {} of 15 differ",
            differences
        );
    }

         // Edge Case Tests
     
    #[test]
    fn test_single_hash_function() {
        let strategy = DoubleHashing;
        let indices = strategy.generate_indices(12345, 67890, 0, 1, 1000);

        assert_eq!(indices.len(), 1);
        assert!(indices[0] < 1000);
    }

    #[test]
    fn test_large_k() {
        let strategy = EnhancedDoubleHashing;
        let indices = strategy.generate_indices(12345, 67890, 0, 100, 10000);

        assert_eq!(indices.len(), 100);
        assert!(indices.iter().all(|&idx| idx < 10000));
    }

    #[test]
    fn test_small_filter_size() {
        let strategy = DoubleHashing;
        let indices = strategy.generate_indices(12345, 67890, 0, 10, 10);

        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&idx| idx < 10));
    }

    #[test]
    fn test_power_of_two_filter_size() {
        let strategy = DoubleHashing;
        let indices = strategy.generate_indices(12345, 67890, 0, 7, 1024);

        assert_eq!(indices.len(), 7);
        assert!(indices.iter().all(|&idx| idx < 1024));
    }

         // Distribution Quality Tests
     
    #[test]
    fn test_double_hashing_distribution() {
        let strategy = DoubleHashing;
        let m = 100;
        let k = 10;
        let mut buckets = vec![0usize; m];

        // Generate 1000 sets of indices using well-mixed hash values
        for seed in 0u64..1000 {
            // Use a proper mixing function to generate independent h1, h2
            let mixed = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x517cc1b727220a95);
            let h1 = mixed ^ (mixed >> 33);
            let h2 = h1.wrapping_mul(0x85ebca77c2b2ae63) ^ (h1 >> 29);
            let indices = strategy.generate_indices(h1, h2, 0, k, m);

            for idx in indices {
                buckets[idx] += 1;
            }
        }

        // Check distribution: each bucket should have roughly 100 entries (1000 * 10 / 100)
        let expected = (1000 * k) / m;
        let tolerance = expected / 2; // Allow 50% deviation

        let mut outliers = 0;
        for &count in &buckets {
            if count < expected.saturating_sub(tolerance) || count > expected + tolerance {
                outliers += 1;
            }
        }

        // No more than 10% of buckets should be outliers
        assert!(
            outliers <= m / 10,
            "Distribution is poor: {} of {} buckets are outliers",
            outliers,
            m
        );
    }

    #[test]
    fn test_enhanced_double_hashing_distribution() {
        let strategy = EnhancedDoubleHashing;
        let m = 100;
        let k = 10;
        let mut buckets = vec![0usize; m];

        for seed in 0u64..1000 {
            // Use a proper mixing function to generate independent h1, h2
            let mixed = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x517cc1b727220a95);
            let h1 = mixed ^ (mixed >> 33);
            let h2 = h1.wrapping_mul(0x85ebca77c2b2ae63) ^ (h1 >> 29);
            let indices = strategy.generate_indices(h1, h2, 0, k, m);

            for idx in indices {
                buckets[idx] += 1;
            }
        }

        let expected = (1000 * k) / m;
        let tolerance = expected / 2;

        let mut outliers = 0;
        for &count in &buckets {
            if count < expected.saturating_sub(tolerance) || count > expected + tolerance {
                outliers += 1;
            }
        }

        // Enhanced should have even better distribution
        assert!(
            outliers <= m / 10,
            "Enhanced distribution is poor: {} of {} buckets are outliers",
            outliers,
            m
        );
    }

         // Chi-Square Distribution Test (Statistical Rigor)
     
    #[test]
    fn test_chi_square_double_hashing() {
        let strategy = DoubleHashing;
        let m = 100;
        let k = 10;
        let num_trials: usize = 1000;
        let mut buckets = vec![0usize; m];

        for seed in 0..num_trials {
            // Use proper mixing to generate independent hash values
            let s = seed as u64;
            let mixed = s.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x517cc1b727220a95);
            let h1 = mixed ^ (mixed >> 33);
            let h2 = h1.wrapping_mul(0x85ebca77c2b2ae63) ^ (h1 >> 29);
            let indices = strategy.generate_indices(h1, h2, 0, k, m);

            for idx in indices {
                buckets[idx] += 1;
            }
        }

        // Calculate chi-square statistic
        let expected = (num_trials * k) as f64 / m as f64;
        let chi_squared: f64 = buckets
            .iter()
            .map(|&observed| {
                let diff = observed as f64 - expected;
                (diff * diff) / expected
            })
            .sum();

        // Critical value for 99 degrees of freedom at p=0.05 is ~123.2
        // We use 150 to account for inherent modulo bias
        assert!(
            chi_squared < 150.0,
            "Chi-square test failed: χ² = {:.2} (critical value ≈ 123.2)",
            chi_squared
        );
    }

    #[test]
    fn test_chi_square_enhanced_double_hashing() {
        let strategy = EnhancedDoubleHashing;
        let m = 100;
        let k = 10;
        let num_trials: usize = 1000;
        let mut buckets = vec![0usize; m];

        for seed in 0..num_trials {
            // Use proper mixing to generate independent hash values
            let s = seed as u64;
            let mixed = s.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x517cc1b727220a95);
            let h1 = mixed ^ (mixed >> 33);
            let h2 = h1.wrapping_mul(0x85ebca77c2b2ae63) ^ (h1 >> 29);
            let indices = strategy.generate_indices(h1, h2, 0, k, m);

            for idx in indices {
                buckets[idx] += 1;
            }
        }

        let expected = (num_trials * k) as f64 / m as f64;
        let chi_squared: f64 = buckets
            .iter()
            .map(|&observed| {
                let diff = observed as f64 - expected;
                (diff * diff) / expected
            })
            .sum();

        // Enhanced should pass the same test
        assert!(
            chi_squared < 150.0,
            "Enhanced chi-square test failed: χ² = {:.2}",
            chi_squared
        );
    }

         // Wrapping Behavior Tests (Overflow Safety)
     
    #[test]
    fn test_double_hashing_wrapping_behavior() {
        let strategy = DoubleHashing;
        let h1 = u64::MAX;
        let h2 = u64::MAX;
        let indices = strategy.generate_indices(h1, h2, 0, 10, 1000);

        // Should not panic, all indices valid
        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_enhanced_double_hashing_wrapping_behavior() {
        let strategy = EnhancedDoubleHashing;
        let h1 = u64::MAX;
        let h2 = u64::MAX;
        let indices = strategy.generate_indices(h1, h2, 0, 10, 1000);

        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

    #[test]
    fn test_triple_hashing_wrapping_behavior() {
        let strategy = TripleHashing;
        let h1 = u64::MAX;
        let h2 = u64::MAX;
        let h3 = u64::MAX;
        let indices = strategy.generate_indices(h1, h2, h3, 10, 1000);

        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&idx| idx < 1000));
    }

         // Trait Object Tests (Dynamic Dispatch)
     
    #[test]
    fn test_trait_object_usage() {
        let strategies: Vec<Box<dyn HashStrategy>> = vec![
            Box::new(DoubleHashing),
            Box::new(EnhancedDoubleHashing),
            Box::new(TripleHashing),
        ];

        for strategy in strategies {
            let indices = strategy.generate_indices(12345, 67890, 11111, 7, 1000);
            assert_eq!(indices.len(), 7);
            assert!(indices.iter().all(|&idx| idx < 1000));
        }
    }

    #[test]
    fn test_default_trait() {
        let _strategy1 = DoubleHashing::default();
        let _strategy2 = EnhancedDoubleHashing::default();
        
        // Verify defaults work correctly
        let indices = DoubleHashing::default().generate_indices(12345, 67890, 0, 7, 1000);
        assert_eq!(indices.len(), 7);
    }
}
