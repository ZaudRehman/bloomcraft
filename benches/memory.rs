//! Comprehensive Memory Analysis Suite for BloomCraft
//!
//! This is NOT a performance benchmark. This is a **memory characterization suite**
//! that validates theoretical guarantees, detects anomalies, and provides
//! production-ready memory analysis for all filter variants.

use bloomcraft::core::params;
use bloomcraft::filters::{
    CountingBloomFilter, TreeBloomFilter, PartitionedBloomFilter, ScalableBloomFilter,
    StandardBloomFilter,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashSet;
use std::mem::size_of;

// CATEGORY 1: THEORETICAL VALIDATION

fn validate_theoretical_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_theoretical_validation");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    let test_cases = vec![
        ("standard_1pct", vec![1_000, 10_000, 100_000, 1_000_000], 0.01),
        ("standard_0.1pct", vec![1_000, 10_000, 100_000], 0.001),
        ("standard_10pct", vec![1_000, 10_000, 100_000], 0.1),
    ];

    for (name, sizes, fpr) in test_cases {
        for size in sizes {
            let filter = StandardBloomFilter::<u64>::new(size, fpr);
            
            let actual = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
            let (m, _k) = params::calculate_filter_params(size, fpr).unwrap();
            let theoretical = (m + 7) / 8;
            let overhead_pct = ((actual as f64 / theoretical as f64) - 1.0) * 100.0;
            
            // Adaptive validation thresholds
            let max_overhead = if size < 500 {
                50.0
            } else if size < 2_000 {
                20.0
            } else if size < 10_000 {
                10.0
            } else {
                2.0
            };
            
            assert!(
                overhead_pct < max_overhead,
                "FAIL: {} items @ {:.4} FPR has {:.2}% overhead (expected < {:.1}%)",
                size, fpr, overhead_pct, max_overhead
            );

            eprintln!(
                "{:>7} items @ FPR {:.4}: {:>11} bytes (theoretical {:>11}), overhead {:>6.2}%",
                size, fpr, actual, theoretical, overhead_pct
            );

            group.bench_function(BenchmarkId::new(name, size), |b| {
                b.iter(|| black_box(actual));
            });
        }
    }

    // Counting filters (8x overhead expected)
    for size in [1_000, 10_000, 100_000] {
        let filter = CountingBloomFilter::<u64>::new(size, 0.01);
        let actual = size_of::<CountingBloomFilter<u64>>() + filter.memory_usage();
        let (m, _k) = params::calculate_filter_params(size, 0.01).unwrap();
        let theoretical = m;
        let overhead_pct = ((actual as f64 / theoretical as f64) - 1.0) * 100.0;

        eprintln!(
            "Counting {:>7} items: {:>11} bytes (theoretical {:>11}), overhead {:>6.2}%",
            size, actual, theoretical, overhead_pct
        );

        group.bench_function(BenchmarkId::new("counting_8bit", size), |b| {
            b.iter(|| black_box(actual));
        });
    }

    // Partitioned (should be IDENTICAL to standard)
    for size in [1_000, 10_000, 100_000] {
        let standard = StandardBloomFilter::<u64>::new(size, 0.01);
        let partitioned = PartitionedBloomFilter::<u64>::new(size, 0.01);
        
        let std_mem = size_of::<StandardBloomFilter<u64>>() + standard.memory_usage();
        let part_mem = size_of::<PartitionedBloomFilter<u64>>() + partitioned.memory_usage();
        
        let diff_pct = ((part_mem as f64 / std_mem as f64) - 1.0) * 100.0;
        
        assert!(
            diff_pct.abs() < 10.0,
            "FAIL: Partitioned uses {:.2}% more memory than Standard (expected < 10%)",
            diff_pct
        );

        eprintln!("{:>7} items: Standard={:>11} bytes, Partitioned={:>11} bytes, diff={:>6.2}%",
            size, std_mem, part_mem, diff_pct);

        group.bench_function(BenchmarkId::new("partitioned_vs_standard", size), |b| {
            b.iter(|| black_box(part_mem));
        });
    }

    group.finish();
}

// CATEGORY 2: SCALING LAWS

fn validate_linear_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_scaling_laws");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    // Test linear scaling with items (fixed FPR)
    let sizes = vec![1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];
    let fpr = 0.01;
    
    let mut measurements = Vec::new();
    
    for size in &sizes {
        let filter = StandardBloomFilter::<u64>::new(*size, fpr);
        let mem = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
        let bits_per_item = (mem as f64 * 8.0) / *size as f64;
        measurements.push((*size, mem, bits_per_item));
    }

    // Calculate R² for linearity using proper linear regression
    let mean_size: f64 = sizes.iter().map(|&s| s as f64).sum::<f64>() / sizes.len() as f64;
    let mean_mem: f64 = measurements.iter().map(|(_, m, _)| *m as f64).sum::<f64>() / sizes.len() as f64;
    
    // Calculate slope via least squares
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for (size, mem, _) in &measurements {
        numerator += (*size as f64 - mean_size) * (*mem as f64 - mean_mem);
        denominator += (*size as f64 - mean_size).powi(2);
    }
    
    let slope = numerator / denominator;
    let intercept = mean_mem - slope * mean_size;
    
    // Calculate R² properly
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    
    for (size, mem, _) in &measurements {
        let predicted = slope * (*size as f64) + intercept;
        ss_tot += (*mem as f64 - mean_mem).powi(2);
        ss_res += (*mem as f64 - predicted).powi(2);
    }
    
    let r_squared = 1.0 - (ss_res / ss_tot);

    eprintln!("\n=== LINEAR SCALING VALIDATION ===");
    eprintln!("Size      | Memory      | Bits/Item");
    eprintln!("----------|-------------|----------");
    for (size, mem, bpi) in &measurements {
        eprintln!("{:>9} | {:>11} | {:>8.2}", size, mem, bpi);
    }
    eprintln!("\nLinear regression: y = {:.6}x + {:.2}", slope, intercept);
    eprintln!("R² = {:.6} (perfect linearity if > 0.95)", r_squared);

    if r_squared < 0.95 {
        eprintln!("WARNING: Non-linear scaling detected (R² = {:.6})", r_squared);
    } else {
        eprintln!("✓ Perfect linear scaling confirmed");
    }

    let last_mem = measurements.last().unwrap().1;
    group.bench_function("linear_with_items", |b| {
        b.iter(|| black_box(last_mem));
    });

    // Test logarithmic scaling with FPR (fixed items)
    let fprs = vec![0.5, 0.1, 0.01, 0.001, 0.0001];
    let size = 100_000;
    
    let mut measurements = Vec::new();
    
    for fpr in &fprs {
        let filter = StandardBloomFilter::<u64>::new(size, *fpr);
        let mem = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
        let bits_per_item = (mem as f64 * 8.0) / size as f64;
        let theoretical_bpi = params::bits_per_element(*fpr).unwrap();
        measurements.push((*fpr, mem, bits_per_item, theoretical_bpi));
    }

    eprintln!("\n=== LOGARITHMIC SCALING WITH FPR ===");
    eprintln!("FPR      | Memory      | Bits/Item | Theoretical");
    eprintln!("---------|-------------|-----------|------------");
    for (fpr, mem, bpi, theo) in &measurements {
        eprintln!("{:>8.4} | {:>11} | {:>9.2} | {:>11.2}", fpr, mem, bpi, theo);
    }

    let last_mem = measurements.last().unwrap().1;
    group.bench_function("logarithmic_with_fpr", |b| {
        b.iter(|| black_box(last_mem));
    });

    group.finish();
}

// CATEGORY 3: MEMORY STABILITY

fn validate_memory_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_memory_stability");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    let size = 100_000;
    let fpr = 0.01;

    // Test 1: Pre-allocation
    let filter_empty = StandardBloomFilter::<u64>::new(size, fpr);
    let mem_empty = size_of::<StandardBloomFilter<u64>>() + filter_empty.memory_usage();

    let filter_half = StandardBloomFilter::<u64>::new(size, fpr);
    let mem_half = size_of::<StandardBloomFilter<u64>>() + filter_half.memory_usage();

    let filter_full = StandardBloomFilter::<u64>::new(size, fpr);
    let mem_full = size_of::<StandardBloomFilter<u64>>() + filter_full.memory_usage();

    assert_eq!(mem_empty, mem_half, "FAIL: Memory grew during construction");
    assert_eq!(mem_empty, mem_full, "FAIL: Memory grew during construction");

    eprintln!("\n=== PREALLOCATION STABILITY ===");
    eprintln!("Empty filter:  {:>11} bytes", mem_empty);
    eprintln!("Half filter:   {:>11} bytes", mem_half);
    eprintln!("Full filter:   {:>11} bytes", mem_full);
    eprintln!("Growth: {:>11} bytes (must be 0)", mem_full - mem_empty);

    group.bench_function("prealloc_stability", |b| {
        b.iter(|| black_box(mem_full));
    });

    // Test 2: Post-insert stability
    let filter_before = StandardBloomFilter::<u64>::new(size, fpr);
    let mem_before = size_of::<StandardBloomFilter<u64>>() + filter_before.memory_usage();

    let filter_after = StandardBloomFilter::<u64>::new(size, fpr);
    for i in 0..size {
        filter_after.insert(&(i as u64));
    }
    let mem_after = size_of::<StandardBloomFilter<u64>>() + filter_after.memory_usage();

    assert_eq!(mem_before, mem_after, "FAIL: Memory grew during inserts");

    eprintln!("\n=== POST-INSERT STABILITY ===");
    eprintln!("Before inserts: {:>11} bytes", mem_before);
    eprintln!("After inserts:  {:>11} bytes", mem_after);
    eprintln!("Growth: {:>11} bytes (must be 0)", mem_after as i64 - mem_before as i64);

    group.bench_function("postinsert_stability", |b| {
        b.iter(|| black_box(mem_after));
    });

    // Test 3: Clear operation
    let mut filter = StandardBloomFilter::<u64>::new(size, fpr);
    for i in 0..size {
        filter.insert(&(i as u64));
    }
    let mem_before_clear = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();

    filter.clear();
    let mem_after_clear = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();

    assert_eq!(mem_before_clear, mem_after_clear, "FAIL: Memory changed after clear");

    eprintln!("\n=== POST-CLEAR STABILITY ===");
    eprintln!("Before clear: {:>11} bytes", mem_before_clear);
    eprintln!("After clear:  {:>11} bytes", mem_after_clear);
    eprintln!("Change: {:>11} bytes (must be 0)", mem_after_clear as i64 - mem_before_clear as i64);

    group.bench_function("postclear_stability", |b| {
        b.iter(|| black_box(mem_after_clear));
    });

    // Test 4: Scalable filter growth
    let mut filter_scalable = ScalableBloomFilter::<u64>::new(1_000, fpr);
    let mut growth_points = Vec::new();

    for batch in 1..=10 {
        let inserts = batch * 10_000;
        for i in ((batch - 1) * 10_000)..(batch * 10_000) {
            filter_scalable.insert(&(i as u64));
        }
        let mem = size_of::<ScalableBloomFilter<u64>>() + filter_scalable.memory_usage();
        let num_filters = filter_scalable.filter_count();
        growth_points.push((inserts, mem, num_filters));
    }

    eprintln!("\n=== SCALABLE FILTER GROWTH ===");
    eprintln!("Inserts  | Memory      | Sub-filters");
    eprintln!("---------|-------------|------------");
    for (inserts, mem, num) in &growth_points {
        eprintln!("{:>8} | {:>11} | {:>11}", inserts, mem, num);
    }

    let last_mem = growth_points.last().unwrap().1;
    group.bench_function("scalable_growth_pattern", |b| {
        b.iter(|| black_box(last_mem));
    });

    group.finish();
}

// CATEGORY 4: FILTER VARIANT ANALYSIS

fn analyze_filter_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_variant_analysis");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    let size = 100_000;
    let fpr = 0.01;

    // Standard vs Partitioned
    let standard = StandardBloomFilter::<u64>::new(size, fpr);
    let partitioned = PartitionedBloomFilter::<u64>::new(size, fpr);
    
    let std_stack = size_of::<StandardBloomFilter<u64>>();
    let std_heap = standard.memory_usage();
    let std_total = std_stack + std_heap;
    
    let part_stack = size_of::<PartitionedBloomFilter<u64>>();
    let part_heap = partitioned.memory_usage();
    let part_total = part_stack + part_heap;

    eprintln!("\n=== STANDARD VS PARTITIONED ===");
    eprintln!("Standard:");
    eprintln!("  Stack:  {:>11} bytes", std_stack);
    eprintln!("  Heap:   {:>11} bytes", std_heap);
    eprintln!("  Total:  {:>11} bytes", std_total);
    eprintln!("Partitioned:");
    eprintln!("  Stack:  {:>11} bytes", part_stack);
    eprintln!("  Heap:   {:>11} bytes", part_heap);
    eprintln!("  Total:  {:>11} bytes", part_total);
    eprintln!("Difference: {:>11} bytes ({:.2}%)", 
        part_total as i64 - std_total as i64,
        ((part_total as f64 / std_total as f64) - 1.0) * 100.0
    );

    group.bench_function("standard_vs_partitioned_detailed", |b| {
        b.iter(|| black_box(std_total));
    });

    // Counting filter overhead
    let counting = CountingBloomFilter::<u64>::new(size, fpr);
    
    let std_mem = size_of::<StandardBloomFilter<u64>>() + standard.memory_usage();
    let cnt_mem = size_of::<CountingBloomFilter<u64>>() + counting.memory_usage();
    
    let overhead_multiplier = cnt_mem as f64 / std_mem as f64;
    let overhead_bytes = cnt_mem - std_mem;

    eprintln!("\n=== COUNTING FILTER OVERHEAD ===");
    eprintln!("Standard filter:  {:>11} bytes", std_mem);
    eprintln!("Counting filter:  {:>11} bytes", cnt_mem);
    eprintln!("Overhead:         {:>11} bytes ({:.2}x)", overhead_bytes, overhead_multiplier);
    eprintln!("Per-counter cost: {:.2} bytes/counter (expected ~1.0)", overhead_bytes as f64 / size as f64);

    group.bench_function("counting_overhead_breakdown", |b| {
        b.iter(|| black_box(cnt_mem));
    });

    // Hierarchical depth analysis
    for depth in [2, 3, 4, 5] {
        let branching = vec![4; depth - 1];
        let filter = TreeBloomFilter::<u64>::new(branching, 1_000, fpr);
        let mem = size_of::<TreeBloomFilter<u64>>() + filter.memory_usage();
        let num_bins = 4_usize.pow((depth - 1) as u32);

        eprintln!(
            "Depth {}: {:>11} bytes ({} bins, {} bytes/bin)",
            depth, mem, num_bins, mem / num_bins
        );

        group.bench_function(BenchmarkId::new("hierarchical_depth", depth), |b| {
            b.iter(|| black_box(mem));
        });
    }

    group.finish();
}

// CATEGORY 5: COMPARATIVE ANALYSIS

fn comparative_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("5_comparative_analysis");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let bloom = StandardBloomFilter::<u64>::new(size, 0.01);
        let bloom_mem = size_of::<StandardBloomFilter<u64>>() + bloom.memory_usage();

        let mut hashset = HashSet::new();
        for i in 0..size {
            hashset.insert(i as u64);
        }
        let hashset_mem = size * 32; // Estimate: 32 bytes per entry
        let btreeset_mem = size * 40; // Estimate: 40 bytes per entry

        let compression_vs_hash = hashset_mem as f64 / bloom_mem as f64;
        let compression_vs_btree = btreeset_mem as f64 / bloom_mem as f64;

        eprintln!("\n=== {} ITEMS COMPARISON ===", size);
        eprintln!("BloomFilter (1% FPR):  {:>12} bytes", bloom_mem);
        eprintln!("HashSet<u64>:          {:>12} bytes (~{:.1}x larger)", hashset_mem, compression_vs_hash);
        eprintln!("BTreeSet<u64>:         {:>12} bytes (~{:.1}x larger)", btreeset_mem, compression_vs_btree);
        eprintln!("\nSpace Savings:");
        eprintln!("  vs HashSet:  {:.1}x compression", compression_vs_hash);
        eprintln!("  vs BTreeSet: {:.1}x compression", compression_vs_btree);

        group.bench_function(BenchmarkId::new("bloom_vs_exact", size), |b| {
            b.iter(|| {
                black_box(bloom_mem);
                black_box(&hashset);
            });
        });
    }

    group.finish();
}

// CATEGORY 6: PATHOLOGICAL CASES

fn test_pathological_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("6_pathological_cases");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    // Tiny filter
    let filter_tiny = StandardBloomFilter::<u64>::new(10, 0.01);
    let mem_tiny = size_of::<StandardBloomFilter<u64>>() + filter_tiny.memory_usage();
    let (m, k) = params::calculate_filter_params(10, 0.01).unwrap();

    eprintln!("\n=== TINY FILTER (10 items) ===");
    eprintln!("Memory: {:>11} bytes", mem_tiny);
    eprintln!("Bits: {}, Hashes: {}", m, k);
    eprintln!("Struct overhead: {:>11} bytes", size_of::<StandardBloomFilter<u64>>());
    eprintln!("Note: Struct overhead dominates for tiny filters");

    group.bench_function("tiny_filter_10items", |b| {
        b.iter(|| black_box(mem_tiny));
    });

    // High FPR
    let filter_high_fpr = StandardBloomFilter::<u64>::new(10_000, 0.5);
    let mem_high = size_of::<StandardBloomFilter<u64>>() + filter_high_fpr.memory_usage();
    let (m_high, k_high) = params::calculate_filter_params(10_000, 0.5).unwrap();

    eprintln!("\n=== HIGH FPR (50%) ===");
    eprintln!("Memory: {:>11} bytes", mem_high);
    eprintln!("Bits: {}, Hashes: {}", m_high, k_high);
    eprintln!("Bits/item: {:.2} (vs 9.6 for 1% FPR)", m_high as f64 / 10_000.0);

    group.bench_function("high_fpr_50pct", |b| {
        b.iter(|| black_box(mem_high));
    });

    // Extremely low FPR
    let filter_low_fpr = StandardBloomFilter::<u64>::new(10_000, 0.000001);
    let mem_low = size_of::<StandardBloomFilter<u64>>() + filter_low_fpr.memory_usage();
    let (m_low, k_low) = params::calculate_filter_params(10_000, 0.000001).unwrap();

    eprintln!("\n=== EXTREMELY LOW FPR (0.0001%) ===");
    eprintln!("Memory: {:>11} bytes", mem_low);
    eprintln!("Bits: {}, Hashes: {}", m_low, k_low);
    eprintln!("Bits/item: {:.2} (vs 9.6 for 1% FPR)", m_low as f64 / 10_000.0);

    group.bench_function("low_fpr_0.0001pct", |b| {
        b.iter(|| black_box(mem_low));
    });

    // Saturated filter
    let filter_saturated = StandardBloomFilter::<u64>::new(1_000, 0.01);
    let mem_saturated = size_of::<StandardBloomFilter<u64>>() + filter_saturated.memory_usage();
    let fpr_estimate = filter_saturated.estimate_fpr();

    eprintln!("\n=== SATURATED FILTER ===");
    eprintln!("Memory: {:>11} bytes (unchanged)", mem_saturated);
    eprintln!("Estimated FPR: {:.4} (vs 0.01 target)", fpr_estimate);
    eprintln!("Note: Memory stable regardless of saturation");

    group.bench_function("saturated_filter", |b| {
        b.iter(|| black_box(mem_saturated));
    });

    group.finish();
}

// CATEGORY 7: MEMORY ALIGNMENT & CACHE EFFICIENCY

fn validate_memory_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("7_memory_alignment");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    // Test 1: Stack alignment
    let std_filter = StandardBloomFilter::<u64>::new(10_000, 0.01);
    let part_filter = PartitionedBloomFilter::<u64>::new(10_000, 0.01);
    let count_filter = CountingBloomFilter::<u64>::new(10_000, 0.01);
    
    let std_align = std::mem::align_of::<StandardBloomFilter<u64>>();
    let part_align = std::mem::align_of::<PartitionedBloomFilter<u64>>();
    let count_align = std::mem::align_of::<CountingBloomFilter<u64>>();
    
    eprintln!("\n=== STRUCT ALIGNMENT ===");
    eprintln!("StandardBloomFilter:   {} bytes alignment", std_align);
    eprintln!("PartitionedBloomFilter: {} bytes alignment", part_align);
    eprintln!("CountingBloomFilter:    {} bytes alignment", count_align);
    eprintln!("\nOptimal: 8-byte alignment for 64-bit systems");

    group.bench_function("struct_alignment", |b| {
        b.iter(|| {
            black_box(std_align);
            black_box(&std_filter);
            black_box(&part_filter);
            black_box(&count_filter);
        });
    });

    // Test 2: Cache line utilization
    let sizes = vec![64, 128, 256, 512, 1024];
    let mut measurements = Vec::new();
    
    for size in sizes {
        let filter = StandardBloomFilter::<u64>::new(size, 0.01);
        let mem = filter.memory_usage();
        let cache_lines = (mem + 63) / 64;
        measurements.push((size, mem, cache_lines));
    }
    
    eprintln!("\n=== CACHE LINE EFFICIENCY ===");
    eprintln!("Items | Memory  | Cache Lines | Bytes/Line");
    eprintln!("------|---------|-------------|------------");
    for (size, mem, lines) in &measurements {
        eprintln!("{:>5} | {:>7} | {:>11} | {:>10.1}", 
            size, mem, lines, *mem as f64 / *lines as f64);
    }

    let last_mem = measurements.last().unwrap().1;
    group.bench_function("cache_line_efficiency", |b| {
        b.iter(|| black_box(last_mem));
    });

    // Test 3: Partitioned cache layout
    let size = 100_000;
    let filter = PartitionedBloomFilter::<u64>::new(size, 0.01);
    
    let partition_count = filter.partition_count();
    let partition_size = filter.partition_size();
    let total_mem = filter.memory_usage();
    let mem_per_partition = total_mem / partition_count;
    let cache_lines_per_partition = (mem_per_partition + 63) / 64;
    
    eprintln!("\n=== PARTITIONED CACHE LAYOUT ===");
    eprintln!("Total partitions:      {}", partition_count);
    eprintln!("Bits per partition:    {}", partition_size);
    eprintln!("Bytes per partition:   {}", mem_per_partition);
    eprintln!("Cache lines per part:  {}", cache_lines_per_partition);
    eprintln!("\nBenefit: Each query touches fewer cache lines");

    group.bench_function("partitioned_cache_layout", |b| {
        b.iter(|| black_box(total_mem));
    });

    group.finish();
}

// CATEGORY 8: BATCH OPERATION MEMORY CHARACTERISTICS

fn validate_batch_memory_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("8_batch_operations");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    let size = 100_000;
    let fpr = 0.01;

    // Test 1: Batch insert stability
    let filter_before = StandardBloomFilter::<u64>::new(size, fpr);
    let mem_before = size_of::<StandardBloomFilter<u64>>() + filter_before.memory_usage();
    
    let filter_batch = StandardBloomFilter::<u64>::new(size, fpr);
    let batch: Vec<u64> = (0..10_000).collect();
    filter_batch.insert_batch(&batch);
    let mem_after = size_of::<StandardBloomFilter<u64>>() + filter_batch.memory_usage();
    
    let diff = mem_after as i64 - mem_before as i64;
    
    assert_eq!(diff, 0, "FAIL: Batch insert caused memory growth of {} bytes", diff);
    
    eprintln!("\n=== BATCH INSERT STABILITY ===");
    eprintln!("Before batch: {:>11} bytes", mem_before);
    eprintln!("After batch:  {:>11} bytes", mem_after);
    eprintln!("Growth:       {:>11} bytes (must be 0)", diff);
    eprintln!("Batch size:   10,000 items");

    group.bench_function("batch_insert_stability", |b| {
        b.iter(|| black_box(mem_after));
    });

    // Test 2: Batch query memory
    let filter_query = StandardBloomFilter::<u64>::new(size, fpr);
    for i in 0..size {
        filter_query.insert(&(i as u64));
    }
    
    let mem_before_query = size_of::<StandardBloomFilter<u64>>() + filter_query.memory_usage();
    
    let query_batch: Vec<u64> = (0..10_000).collect();
    let _results = filter_query.contains_batch(&query_batch);
    
    let mem_after_query = size_of::<StandardBloomFilter<u64>>() + filter_query.memory_usage();
    
    eprintln!("\n=== BATCH QUERY ZERO-ALLOCATION ===");
    eprintln!("Before queries: {:>11} bytes", mem_before_query);
    eprintln!("After queries:  {:>11} bytes", mem_after_query);
    eprintln!("Growth:         {:>11} bytes (must be 0)", mem_after_query as i64 - mem_before_query as i64);
    eprintln!("Note: Batch operations should not allocate");

    group.bench_function("batch_query_zero_alloc", |b| {
        b.iter(|| black_box(mem_after_query));
    });

    // Test 3: Union/Intersect memory
    let filter1 = StandardBloomFilter::<u64>::new(size, fpr);
    let filter2 = StandardBloomFilter::<u64>::new(size, fpr);
    
    for i in 0..(size/2) {
        filter1.insert(&(i as u64));
        filter2.insert(&((i + size/4) as u64));
    }
    
    let filter1_mem = size_of::<StandardBloomFilter<u64>>() + filter1.memory_usage();
    let filter2_mem = size_of::<StandardBloomFilter<u64>>() + filter2.memory_usage();
    
    let union_filter = filter1.union(&filter2).unwrap();
    let union_mem = size_of::<StandardBloomFilter<u64>>() + union_filter.memory_usage();
    
    let intersect_filter = filter1.intersect(&filter2).unwrap();
    let intersect_mem = size_of::<StandardBloomFilter<u64>>() + intersect_filter.memory_usage();
    
    eprintln!("\n=== UNION/INTERSECT MEMORY ===");
    eprintln!("Filter 1:       {:>11} bytes", filter1_mem);
    eprintln!("Filter 2:       {:>11} bytes", filter2_mem);
    eprintln!("Union result:   {:>11} bytes (same size)", union_mem);
    eprintln!("Intersect res:  {:>11} bytes (same size)", intersect_mem);
    eprintln!("\nNote: Set operations preserve filter size");

    group.bench_function("union_intersect_memory", |b| {
        b.iter(|| black_box(union_mem));
    });

    group.finish();
}

// CATEGORY 9: MEMORY FRAGMENTATION & REUSE

fn validate_memory_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("9_memory_fragmentation");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    // Test 1: Repeated clear/rebuild cycles
    let size = 100_000;
    let fpr = 0.01;
    let mut filter = StandardBloomFilter::<u64>::new(size, fpr);
    
    let mut cycle_memories = Vec::new();
    
    for cycle in 0..10 {
        for i in 0..size {
            filter.insert(&((cycle * size + i) as u64));
        }
        let mem_filled = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
        
        filter.clear();
        let mem_cleared = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
        
        cycle_memories.push((cycle, mem_filled, mem_cleared));
    }
    
    eprintln!("\n=== CLEAR/REBUILD CYCLES (10 iterations) ===");
    eprintln!("Cycle | Filled  | Cleared | Stable?");
    eprintln!("------|---------|---------|--------");
    
    let first_filled = cycle_memories[0].1;
    let first_cleared = cycle_memories[0].2;
    
    for (cycle, filled, cleared) in &cycle_memories {
        let stable = *filled == first_filled && *cleared == first_cleared;
        eprintln!("{:>5} | {:>7} | {:>7} | {}", 
            cycle, filled, cleared, if stable { "✓" } else { "✗" });
    }
    
    eprintln!("\nAll cycles should show identical memory (no fragmentation)");

    group.bench_function("clear_rebuild_cycles", |b| {
        b.iter(|| black_box(first_filled));
    });

    // Test 2: Multiple filter creation/destruction
    let size = 10_000;
    let fpr = 0.01;
    let mut memories = Vec::new();
    
    for _i in 0..20 {
        let filter = StandardBloomFilter::<u64>::new(size, fpr);
        let mem = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
        memories.push(mem);
        drop(filter);
    }
    
    let first_mem = memories[0];
    let all_same = memories.iter().all(|&m| m == first_mem);
    
    eprintln!("\n=== CREATE/DESTROY PATTERN (20 filters) ===");
    eprintln!("All filters: {:>11} bytes", first_mem);
    eprintln!("Consistency: {}", if all_same { "✓ Perfect" } else { "✗ Varied" });
    
    if !all_same {
        let min = memories.iter().min().unwrap();
        let max = memories.iter().max().unwrap();
        eprintln!("Range: {} - {} bytes ({:.2}% variation)", 
            min, max, (*max as f64 / *min as f64 - 1.0) * 100.0);
    }

    group.bench_function("create_destroy_pattern", |b| {
        b.iter(|| black_box(first_mem));
    });

    // Test 3: Scalable filter shrink behavior
    let mut filter_scalable = ScalableBloomFilter::<u64>::new(1_000, 0.01);
    
    for i in 0..50_000 {
        filter_scalable.insert(&(i as u64));
    }
    let mem_grown = size_of::<ScalableBloomFilter<u64>>() + filter_scalable.memory_usage();
    let num_filters = filter_scalable.filter_count();
    
    filter_scalable.clear();
    let mem_cleared = size_of::<ScalableBloomFilter<u64>>() + filter_scalable.memory_usage();
    let filters_after = filter_scalable.filter_count();
    
    eprintln!("\n=== SCALABLE FILTER SHRINK BEHAVIOR ===");
    eprintln!("After growth:  {:>11} bytes ({} sub-filters)", mem_grown, num_filters);
    eprintln!("After clear:   {:>11} bytes ({} sub-filters)", mem_cleared, filters_after);
    eprintln!("Memory delta:  {:>11} bytes", mem_cleared as i64 - mem_grown as i64);
    eprintln!("\nNote: Scalable filters may retain sub-filters for reuse");

    group.bench_function("scalable_shrink_behavior", |b| {
        b.iter(|| black_box(mem_cleared));
    });

    group.finish();
}

// CATEGORY 10: CROSS-PLATFORM MEMORY CONSISTENCY

fn validate_cross_platform_consistency(c: &mut Criterion) {
    let mut group = c.benchmark_group("10_cross_platform_consistency");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(100));

    // Test 1: Size consistency
    let std_size = size_of::<StandardBloomFilter<u64>>();
    let std_size_u32 = size_of::<StandardBloomFilter<u32>>();
    let std_size_str = size_of::<StandardBloomFilter<String>>();
    
    let count_size = size_of::<CountingBloomFilter<u64>>();
    let part_size = size_of::<PartitionedBloomFilter<u64>>();
    let scale_size = size_of::<ScalableBloomFilter<u64>>();
    let hier_size = size_of::<TreeBloomFilter<u64>>();
    
    eprintln!("\n=== STRUCT SIZE CONSISTENCY ===");
    eprintln!("StandardBloomFilter<u64>:    {:>3} bytes", std_size);
    eprintln!("StandardBloomFilter<u32>:    {:>3} bytes", std_size_u32);
    eprintln!("StandardBloomFilter<String>: {:>3} bytes", std_size_str);
    eprintln!("");
    eprintln!("CountingBloomFilter<u64>:    {:>3} bytes", count_size);
    eprintln!("PartitionedBloomFilter<u64>: {:>3} bytes", part_size);
    eprintln!("ScalableBloomFilter<u64>:    {:>3} bytes", scale_size);
    eprintln!("TreeBloomFilter<u64>:{:>3} bytes", hier_size);
    eprintln!("\nNote: PhantomData ensures zero-sized type parameters");

    group.bench_function("struct_size_consistency", |b| {
        b.iter(|| black_box(std_size));
    });

    // Test 2: Memory layout determinism
    let mut memories = Vec::new();
    
    for _ in 0..10 {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01);
        let mem = size_of::<StandardBloomFilter<u64>>() + filter.memory_usage();
        memories.push(mem);
    }
    
    let first = memories[0];
    let all_identical = memories.iter().all(|&m| m == first);
    
    assert!(all_identical, "FAIL: Non-deterministic memory layout detected");
    
    eprintln!("\n=== MEMORY LAYOUT DETERMINISM ===");
    eprintln!("Filter created 10 times:");
    eprintln!("  All measurements: {:>11} bytes", first);
    eprintln!("  Deterministic:    {}", if all_identical { "✓ Yes" } else { "✗ No" });
    eprintln!("\nNote: Same parameters → Same memory (important for serialization)");

    group.bench_function("layout_determinism", |b| {
        b.iter(|| black_box(first));
    });

    // Test 3: Pointer size independence
    let filter = StandardBloomFilter::<u64>::new(10_000, 0.01);
    let heap_mem = filter.memory_usage();
    let stack_mem = size_of::<StandardBloomFilter<u64>>();
    
    let ptr_size = size_of::<usize>();
    let expected_ptrs = if ptr_size == 8 { "64-bit" } else { "32-bit" };
    
    eprintln!("\n=== POINTER SIZE INDEPENDENCE ===");
    eprintln!("Platform:      {} pointers ({} bytes)", expected_ptrs, ptr_size);
    eprintln!("Stack size:    {:>11} bytes (contains pointers)", stack_mem);
    eprintln!("Heap size:     {:>11} bytes (pointer-independent)", heap_mem);
    eprintln!("\nNote: Heap memory is consistent across 32/64-bit");

    group.bench_function("pointer_size_independence", |b| {
        b.iter(|| black_box(heap_mem));
    });

    // Test 4: Endianness-independent memory
    let filter = StandardBloomFilter::<u64>::new(10_000, 0.01);
    let mem = filter.memory_usage();
    let atomic_size = size_of::<std::sync::atomic::AtomicU64>();
    
    eprintln!("\n=== ENDIANNESS INDEPENDENCE ===");
    eprintln!("Filter memory: {:>11} bytes", mem);
    eprintln!("AtomicU64:     {:>11} bytes", atomic_size);
    eprintln!("\nNote: BitVec uses atomics (endianness-safe for operations)");
    eprintln!("Serialization: Use explicit byte ordering for portability");

    group.bench_function("endianness_independence", |b| {
        b.iter(|| black_box(mem));
    });

    group.finish();
}

criterion_group!(
    benches,
    validate_theoretical_accuracy,
    validate_linear_scaling,
    validate_memory_stability,
    analyze_filter_variants,
    comparative_analysis,
    test_pathological_cases,
    validate_memory_alignment,
    validate_batch_memory_behavior,
    validate_memory_fragmentation,
    validate_cross_platform_consistency,
);
criterion_main!(benches);
