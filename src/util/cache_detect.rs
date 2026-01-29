//! CPU cache size detection for optimal partition sizing
//!
//! This module provides runtime detection of L1/L2/L3 cache sizes
//! to automatically tune PartitionedBloomFilter parameters.

#![allow(clippy::pedantic)]

use std::sync::OnceLock;

/// Detected CPU cache sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSizes {
    /// L1 data cache size in bytes (per core).
    pub l1_data_bytes: usize,
    /// L1 cache line size in bytes (typically 64).
    pub l1_line_bytes: usize,
    /// L2 cache size in bytes (per core).
    pub l2_bytes: usize,
    /// L3 cache size in bytes (shared).
    pub l3_bytes: usize,
}

impl CacheSizes {
    /// Conservative defaults for unknown architectures.
    pub const fn default_conservative() -> Self {
        Self {
            l1_data_bytes: 32 * 1024,      // 32 KB
            l1_line_bytes: 64,              // 64 bytes
            l2_bytes: 256 * 1024,           // 256 KB
            l3_bytes: 8 * 1024 * 1024,      // 8 MB
        }
    }
}

impl Default for CacheSizes {
    fn default() -> Self {
        Self::default_conservative()
    }
}

/// Detect CPU cache sizes at runtime (cached result).
pub fn detect_cache_sizes() -> CacheSizes {
    static CACHE_SIZES: OnceLock<CacheSizes> = OnceLock::new();

    *CACHE_SIZES.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            detect_x86_64_cache_sizes()
        }

        #[cfg(target_arch = "aarch64")]
        {
            detect_aarch64_cache_sizes()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CacheSizes::default_conservative()
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_64_cache_sizes() -> CacheSizes {
    use std::arch::x86_64::__cpuid;
    
    unsafe {
        // Trying modern Intel method first (CPUID leaf 0x04)
        let cpuid_max = __cpuid(0);
        if cpuid_max.eax >= 0x04 {
            if let Some(sizes) = detect_intel_deterministic_cache() {
                return sizes;
            }
        }
        
        // Fallback to AMD extended method (0x80000006)
        let cpuid_ext_max = __cpuid(0x80000000);
        if cpuid_ext_max.eax >= 0x80000006 {
            let cpuid = __cpuid(0x80000006);
            
            // ECX[31:24] = L1 data cache size in KB
            let l1_data_kb = ((cpuid.ecx >> 24) & 0xFF) as usize;
            // ECX[7:0] = L1 cache line size
            let l1_line = (cpuid.ecx & 0xFF) as usize;
            // ECX[31:16] = L2 size in KB
            let l2_kb = ((cpuid.ecx >> 16) & 0xFFFF) as usize;
            // EDX[31:18] = L3 size in 512KB units
            let l3_units = ((cpuid.edx >> 18) & 0x3FFF) as usize;
            let l3_kb = l3_units * 512;
            
            // Validate results
            if l1_data_kb >= 16 && l1_data_kb <= 128 {
                return CacheSizes {
                    l1_data_bytes: l1_data_kb * 1024,
                    l1_line_bytes: if l1_line == 64 || l1_line == 128 { l1_line } else { 64 },
                    l2_bytes: if l2_kb > 0 && l2_kb < 2048 { l2_kb * 1024 } else { 256 * 1024 },
                    l3_bytes: if l3_kb > 0 && l3_kb < 64 * 1024 { l3_kb * 1024 } else { 8 * 1024 * 1024 },
                };
            }
        }
        
        // Fallback to conservative defaults
        CacheSizes::default_conservative()
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn detect_intel_deterministic_cache() -> Option<CacheSizes> {
    use std::arch::x86_64::__cpuid_count;
    
    let mut l1_data = 0;
    let mut l1_line = 64;
    let mut l2 = 0;
    let mut l3 = 0;
    
    // Iterate through cache levels
    for i in 0..32 {
        let cpuid = __cpuid_count(0x04, i);
        let cache_type = cpuid.eax & 0x1F;
        
        if cache_type == 0 {
            break; // No more caches
        }
        
        let level = (cpuid.eax >> 5) & 0x07;
        let line_size = (cpuid.ebx & 0xFFF) + 1;
        let partitions = ((cpuid.ebx >> 12) & 0x3FF) + 1;
        let ways = ((cpuid.ebx >> 22) & 0x3FF) + 1;
        let sets = cpuid.ecx + 1;
        
        let size = (ways * partitions * line_size * sets) as usize;
        
        match (level, cache_type) {
            (1, 1) => { // L1 data
                l1_data = size;
                l1_line = line_size as usize;
            },
            (2, 3) => l2 = size, // L2 unified
            (3, 3) => l3 = size, // L3 unified
            _ => {}
        }
    }
    
    // Validate
    if l1_data >= 16 * 1024 {
        Some(CacheSizes {
            l1_data_bytes: l1_data,
            l1_line_bytes: l1_line,
            l2_bytes: if l2 > 0 { l2 } else { 256 * 1024 },
            l3_bytes: if l3 > 0 { l3 } else { 8 * 1024 * 1024 },
        })
    } else {
        None
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64_cache_sizes() -> CacheSizes {
    // On Linux, read from sysfs
    #[cfg(target_os = "linux")]
    {
        if let Some(sizes) = read_linux_sysfs_cache() {
            return sizes;
        }
    }

    // Fallback to architecture-specific defaults
    // Modern ARM (Apple Silicon, AWS Graviton) typically have:
    CacheSizes {
        l1_data_bytes: 64 * 1024,       // 64 KB (larger than x86)
        l1_line_bytes: 128,              // 128 bytes (larger than x86)
        l2_bytes: 4 * 1024 * 1024,      // 4 MB
        l3_bytes: 32 * 1024 * 1024,     // 32 MB (Apple M1/M2)
    }
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn read_linux_sysfs_cache() -> Option<CacheSizes> {
    use std::fs;

    let base_path = "/sys/devices/system/cpu/cpu0/cache";

    let l1_data = fs::read_to_string(format!("{}/index0/size", base_path))
        .ok()
        .and_then(|s| parse_cache_size(&s))?;

    let l1_line = fs::read_to_string(format!("{}/index0/coherency_line_size", base_path))
        .ok()
        .and_then(|s| s.trim().parse().ok())?;

    let l2 = fs::read_to_string(format!("{}/index2/size", base_path))
        .ok()
        .and_then(|s| parse_cache_size(&s))?;

    let l3 = fs::read_to_string(format!("{}/index3/size", base_path))
        .ok()
        .and_then(|s| parse_cache_size(&s))
        .unwrap_or(8 * 1024 * 1024); // Default if L3 not present

    Some(CacheSizes {
        l1_data_bytes: l1_data,
        l1_line_bytes: l1_line,
        l2_bytes: l2,
        l3_bytes: l3,
    })
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn parse_cache_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if s.ends_with('K') {
        s[..s.len() - 1].parse::<usize>().ok().map(|v| v * 1024)
    } else if s.ends_with('M') {
        s[..s.len() - 1].parse::<usize>().ok().map(|v| v * 1024 * 1024)
    } else {
        s.parse().ok()
    }
}

/// Calculate optimal partition size for detected cache.
///
/// Strategy:
/// - Target: Fit 2-4 partitions in L1 cache for hot accesses
/// - Fallback: Single partition in L2 cache
///
/// Returns partition size in bits.
pub fn optimal_partition_size_for_cache(_k: usize) -> usize {
    let cache = detect_cache_sizes();

    // Try to fit 2-4 partitions in L1 cache
    let l1_per_partition = cache.l1_data_bytes / 4;
    let optimal_bits = l1_per_partition * 8;

    // Clamp to reasonable range
    const MIN_PARTITION_BITS: usize = 512; // 64 bytes
    const MAX_PARTITION_BITS: usize = 256 * 1024 * 8; // 256 KB

    optimal_bits.clamp(MIN_PARTITION_BITS, MAX_PARTITION_BITS)
}

/// Recommend partition size based on filter size and cache.
///
/// # Arguments
///
/// * `total_bits` - Total filter size in bits (m)
/// * `k` - Number of hash functions (partitions)
///
/// # Returns
///
/// Recommended partition size in bits, optimized for cache locality.
pub fn recommend_partition_size(total_bits: usize, k: usize) -> usize {
    let cache = detect_cache_sizes();

    // Base partition size from total bits
    let base_partition = (total_bits + k - 1) / k;

    // Calculate cache-optimal size
    let cache_optimal = optimal_partition_size_for_cache(k);

    // If base partition fits in L1, use it
    if base_partition <= cache_optimal {
        return base_partition;
    }

    // If base partition is huge, warn and use L2-optimized size
    if base_partition > cache.l2_bytes * 8 {
        eprintln!(
            "Warning: Partition size {} KB exceeds L2 cache {} KB.              Consider using standard Bloom filter.",
            base_partition / 8192,
            cache.l2_bytes / 1024
        );
    }

    // Return L2-optimized size if partition doesn't fit in L1
    (cache.l2_bytes * 8).min(base_partition)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cache_sizes() {
        let sizes = detect_cache_sizes();

        println!("Detected cache sizes:");
        println!("  L1 data: {} KB", sizes.l1_data_bytes / 1024);
        println!("  L1 line: {} bytes", sizes.l1_line_bytes);
        println!("  L2: {} KB", sizes.l2_bytes / 1024);
        println!("  L3: {} MB", sizes.l3_bytes / (1024 * 1024));

        // Sanity checks
        assert!(sizes.l1_data_bytes >= 16 * 1024); // At least 16KB
        assert!(sizes.l1_data_bytes <= 128 * 1024); // At most 128KB
        assert!(sizes.l1_line_bytes == 64 || sizes.l1_line_bytes == 128);
        assert!(sizes.l2_bytes >= 128 * 1024); // At least 128KB
        assert!(sizes.l3_bytes >= 1 * 1024 * 1024); // At least 1MB
    }

    #[test]
    fn test_optimal_partition_size() {
        let k = 7;
        let size = optimal_partition_size_for_cache(k);

        println!("Optimal partition size for k={}: {} KB", 
            k, size / 8192);

        // Should be reasonable
        assert!(size >= 512); // At least 64 bytes
        assert!(size <= 256 * 1024 * 8); // At most 256KB
    }

    #[test]
    fn test_recommend_partition_size() {
        let total_bits = 1_000_000;
        let k = 7;

        let recommended = recommend_partition_size(total_bits, k);

        println!("Recommended partition size: {} KB", recommended / 8192);

        // Should divide filter reasonably
        assert!(recommended > 0);
        assert!(recommended <= total_bits);
    }

    #[test]
    fn test_cache_detection_cached() {
        // Multiple calls should return same result (cached)
        let sizes1 = detect_cache_sizes();
        let sizes2 = detect_cache_sizes();

        assert_eq!(sizes1, sizes2);
    }

    #[test]
    fn test_large_filter_warning() {
        // Very large filter should trigger warning
        let total_bits = 1_000_000_000; // 125 MB
        let k = 7;

        let _ = recommend_partition_size(total_bits, k);
        // Should print warning (check stderr manually)
    }
}
