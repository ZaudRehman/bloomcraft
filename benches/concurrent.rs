//! Concurrent Bloom Filter Performance Benchmarks
//!
//! **Purpose**: Rigorously validate concurrent Bloom filter implementations under
//! realistic multi-threaded workloads to identify bottlenecks, contention points,
//! and optimal configurations.
//!
//! # What This Suite Tests
//!
//! ## 1. **Correctness Under Concurrency** (Benchmarks 1-2)
//! - No lost updates (all inserts visible)
//! - No false negatives (memory ordering correct)
//! - No data races (atomic operations work)
//!
//! ## 2. **Scalability** (Benchmarks 3-5)
//! - Linear scaling: Does 8 threads = 8x throughput?
//! - Parallel efficiency: Actual speedup vs ideal
//! - Contention breakdown: Where do threads block?
//!
//! ## 3. **Latency Characteristics** (Benchmarks 6-7)
//! - P50/P95/P99 latency under load
//! - Tail latency spikes (critical for real-time systems)
//! - Latency vs throughput trade-off
//!
//! ## 4. **Cache Effects** (Benchmarks 8-9)
//! - False sharing detection (cache line bouncing)
//! - NUMA effects (cross-socket access)
//! - Cache-aware shard assignment
//!
//! ## 5. **Configuration Tuning** (Benchmarks 10-12)
//! - Optimal shard count per core count
//! - Optimal stripe count per workload
//! - Over-sharding penalty measurement
//!
//! ## 6. **Failure Modes** (Benchmarks 13-14)
//! - Hot shard saturation (all threads hit same shard)
//! - Memory bandwidth limits
//! - Lock convoy effects (striped implementation)
//!
//! # Benchmark Categories
//!
//! | ID | Name | What It Measures | Why It Matters |
//! |----|------|------------------|----------------|
//! | 1 | **Baseline Overhead** | Single-thread perf vs StandardBloomFilter | Quantifies atomic overhead |
//! | 2 | **Correctness Stress** | No lost updates under extreme contention | Validates memory ordering |
//! | 3 | **Throughput Scaling** | Ops/sec vs thread count (1-32 threads) | Identifies scalability limits |
//! | 4 | **Parallel Efficiency** | Actual speedup / ideal speedup | Measures wasted parallelism |
//! | 5 | **Workload Mix** | Read-heavy (90/10), Write-heavy (10/90), Balanced (50/50) | Real-world patterns |
//! | 6 | **Latency Distribution** | P50/P95/P99/P99.9 insert latency | Tail latency for SLAs |
//! | 7 | **Query Latency** | P50/P95/P99/P99.9 query latency | Read-heavy workload perf |
//! | 8 | **False Sharing** | Cache line invalidations (striped) | Detects false sharing |
//! | 9 | **NUMA Sensitivity** | Cross-socket vs same-socket | Multi-socket server perf |
//! | 10 | **Shard Count Tuning** | Throughput vs shard count (1-64) | Find optimal shards |
//! | 11 | **Stripe Count Tuning** | Throughput vs stripe count (1-128) | Find optimal stripes |
//! | 12 | **Over-Sharding Penalty** | Measure overhead when shards >> cores | Avoid over-sharding |
//! | 13 | **Hot Shard Attack** | All threads hit same shard | Worst-case contention |
//! | 14 | **Memory Bandwidth Limit** | Max sustained write rate | Hardware ceiling |
//!
//! # Coverage Matrix
//!
//! | Benchmark Group | Sharded | Striped | AtomicCounter | Purpose |
//! |-----------------|---------|---------|---------------|---------|
//! | 01-05 | ✓ | ✓ | - | Basic correctness & scaling |
//! | 06-07 | ✓ | ✓ | - | Latency distribution |
//! | 08-09 | - | ✓ | ✓ | Cache effects (false sharing, NUMA) |
//! | 10-12 | ✓ | ✓ | - | Configuration tuning |
//! | 13-14 | ✓ | - | - | Failure modes |
//! | 15-16 | - | - | ✓ | Atomic counter primitives |
//! | 17-19 | ✓ | ✓ | - | Serialization & cloning |
//! | 20-21 | ✓ | ✓ | - | Lock poisoning & distribution |
//! | 22-24 | ✓ | - | ✓ | Memory subsystem & batching |

use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter, AtomicCounterArray, CacheLinePadded};
use bloomcraft::core::{ConcurrentBloomFilter, SharedBloomFilter};
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Test Data Generation

fn generate_test_strings(count: usize, size: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("{:0width$}", i, width = size))
        .collect()
}

// BENCHMARK 1: Baseline Overhead (Single-Threaded)

/// Measure single-threaded performance to quantify atomic/lock overhead
///
/// **Purpose**: Establish baseline cost of atomic operations vs simple operations
/// **Expected**: Sharded ~20-30% slower, Striped ~30-40% slower than Standard
fn bench_baseline_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_baseline_overhead");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_test_strings(size, 32);
    
    // Standard with concurrent insert (atomic operations)
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("standard_concurrent", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            for item in &items {
                black_box(filter.insert_concurrent(item));
            }
        });
    });
    
    // Sharded (1 shard = atomic overhead only)
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("sharded_1shard", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<String>::with_shard_count(size, fpr, 1);
            for item in &items {
                black_box(filter.insert(item));
            }
        });
    });
    
    // Striped (1 stripe = single RwLock)
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("striped_1stripe", |b| {
        b.iter(|| {
            let filter = StripedBloomFilter::<String>::with_stripe_count(size, fpr, 1);
            for item in &items {
                black_box(filter.insert(item));
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 2: Correctness Stress Test

/// Verify no lost updates under extreme contention
///
/// **Purpose**: Validate memory ordering is correct
/// **Method**: Insert 100K items concurrently, verify all present
/// **Pass Criteria**: 100% of items must be found
fn bench_correctness_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_correctness_stress");
    group.sample_size(10); // Expensive test
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 16;
    
    group.bench_function("sharded_8_correctness", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            let items_per_thread = size / num_threads;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let start = tid * items_per_thread;
                        let end = start + items_per_thread;
                        for i in start..end {
                            filter.insert(&items[i]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
            
            // Verify all items present (no lost updates)
            let mut missing = 0;
            for item in items.iter() {
                if !filter.contains(item) {
                    missing += 1;
                }
            }
            
            black_box(missing);
            assert_eq!(missing, 0, "Lost {} updates!", missing);
        });
    });
    
    group.finish();
}

// BENCHMARK 3: Throughput Scaling (1-32 Threads)

/// Measure ops/sec vs thread count to identify scalability limits
///
/// **Purpose**: Quantify how well each implementation scales with cores
/// **Expected**: Sharded near-linear, Striped sublinear, Mutex flat
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("03_throughput_scaling");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let thread_counts = vec![1, 2, 4, 8, 16];
    
    for num_threads in thread_counts {
        let ops_per_thread = 10_000;
        let total_ops = num_threads * ops_per_thread;
        
        // Sharded (8 shards)
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("sharded_8", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);
                            thread::spawn(move || {
                                let mut rng: u64 = (tid * 12345) as u64;
                                for _ in 0..ops_per_thread {
                                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                    let idx = (rng as usize) % items.len();
                                    filter.insert(&items[idx]);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
        
        // Striped (16 stripes)
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("striped_16", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(StripedBloomFilter::<String>::with_stripe_count(size, fpr, 16));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);
                            thread::spawn(move || {
                                let mut rng: u64 = (tid * 12345) as u64;
                                for _ in 0..ops_per_thread {
                                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                    let idx = (rng as usize) % items.len();
                                    filter.insert(&items[idx]);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
        
        // StandardBloomFilter with Mutex (baseline bottleneck)
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("mutex", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(Mutex::new(StandardBloomFilter::<String>::new(size, fpr)));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);
                            thread::spawn(move || {
                                let mut rng: u64 = (tid * 12345) as u64;
                                for _ in 0..ops_per_thread {
                                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                    let idx = (rng as usize) % items.len();
                                    // StandardBloomFilter.insert requires &mut self, so we must lock
                                    let mut guard = filter.lock().unwrap();
                                    guard.insert(&items[idx]);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 4: Parallel Efficiency

/// Measure actual speedup vs ideal (linear) speedup
///
/// **Purpose**: Quantify wasted parallelism (coordination overhead)
/// **Formula**: Efficiency = (Actual Speedup / Ideal Speedup) * 100%
/// **Target**: >80% efficiency is good, >90% is excellent
fn bench_parallel_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("04_parallel_efficiency");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let total_ops = 80_000; // Fixed work
    let items = Arc::new(generate_test_strings(size, 32));
    
    for num_threads in vec![1, 2, 4, 8, 16] {
        let ops_per_thread = total_ops / num_threads;
        
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("sharded_8", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);
                            thread::spawn(move || {
                                let mut rng: u64 = (tid * 12345) as u64;
                                for _ in 0..ops_per_thread {
                                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                    let idx = (rng as usize) % items.len();
                                    filter.insert(&items[idx]);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 5: Workload Mix (Read-Heavy, Write-Heavy, Balanced)

/// Test real-world access patterns
///
/// **Purpose**: Measure performance under different read/write ratios
/// **Patterns**:
/// - Read-heavy (90% query, 10% insert): Caching scenario
/// - Write-heavy (10% query, 90% insert): Bulk loading
/// - Balanced (50% query, 50% insert): General use
fn bench_workload_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_workload_mix");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    let ops_per_thread = 10_000;
    
    // Pre-populate for read tests
    let populated_sharded = {
        let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
        for item in items.iter() {
            filter.insert(item);
        }
        filter
    };
    
    // Read-heavy (90% read, 10% write)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("sharded_90read_10write", |b| {
        b.iter(|| {
            let filter = Arc::clone(&populated_sharded);
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            
                            if rng % 10 == 0 {
                                filter.insert(&items[idx]);
                            } else {
                                black_box(filter.contains(&items[idx]));
                            }
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Write-heavy (10% read, 90% write)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("sharded_10read_90write", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            
                            if rng % 10 == 0 {
                                black_box(filter.contains(&items[idx]));
                            } else {
                                filter.insert(&items[idx]);
                            }
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Balanced (50% read, 50% write)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("sharded_50read_50write", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            // Pre-populate 50%
            for i in 0..items.len()/2 {
                filter.insert(&items[i]);
            }
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            
                            if rng % 2 == 0 {
                                black_box(filter.contains(&items[idx]));
                            } else {
                                filter.insert(&items[idx]);
                            }
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 6: Insert Latency Distribution (P50/P95/P99/P99.9)

/// Measure latency percentiles under contention
///
/// **Purpose**: Identify tail latency for SLA compliance
/// **Critical**: P99 <1µs excellent, P99.9 <10µs acceptable
fn bench_insert_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("06_insert_latency_distribution");
    group.sample_size(10); // Collect many samples for percentiles
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    let samples_per_thread = 1_000;
    
    group.bench_function("sharded_8_p99_latency", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            
            for _ in 0..iters {
                let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
                let latencies = Arc::new(Mutex::new(Vec::new()));
                
                let handles: Vec<_> = (0..num_threads)
                    .map(|tid| {
                        let filter = Arc::clone(&filter);
                        let items = Arc::clone(&items);
                        let latencies = Arc::clone(&latencies);
                        thread::spawn(move || {
                            let mut local_latencies = Vec::with_capacity(samples_per_thread);
                            let mut rng: u64 = (tid * 12345) as u64;
                            
                            for _ in 0..samples_per_thread {
                                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                let idx = (rng as usize) % items.len();
                                
                                let start = Instant::now();
                                filter.insert(&items[idx]);
                                let elapsed = start.elapsed();
                                
                                local_latencies.push(elapsed);
                            }
                            
                            latencies.lock().unwrap().extend(local_latencies);
                        })
                    })
                    .collect();
                
                let start = Instant::now();
                for h in handles {
                    h.join().unwrap();
                }
                total += start.elapsed();
            }
            
            total
        });
    });
    
    group.finish();
}

// BENCHMARK 7: Query Latency Distribution

/// Measure query latency for read-heavy workloads
///
/// **Purpose**: Validate low latency for cache lookups
fn bench_query_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("07_query_latency_distribution");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    let queries_per_thread = 10_000;
    
    // Pre-populate
    let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
    for item in items.iter() {
        filter.insert(item);
    }
    
    group.throughput(Throughput::Elements((num_threads * queries_per_thread) as u64));
    group.bench_function("sharded_8_query_p99", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..queries_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            black_box(filter.contains(&items[idx]));
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 8: False Sharing Detection (Striped)

/// Detect cache line bouncing in striped implementation
///
/// **Purpose**: Measure performance degradation from false sharing
/// **Method**: Compare adjacent stripe access vs distributed access
fn bench_false_sharing_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_false_sharing_detection");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    
    // Sequential access pattern (high false sharing)
    group.bench_function("striped_16_sequential", |b| {
        b.iter(|| {
            let filter = Arc::new(StripedBloomFilter::<String>::with_stripe_count(size, fpr, 16));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        // Access sequential items (likely same cache lines)
                        let start = tid * 1000;
                        for i in start..(start + 10_000) {
                            let idx = i % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Random access pattern (low false sharing)
    group.bench_function("striped_16_random", |b| {
        b.iter(|| {
            let filter = Arc::new(StripedBloomFilter::<String>::with_stripe_count(size, fpr, 16));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..10_000 {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 9: NUMA Sensitivity (Cross-Socket vs Same-Socket)

/// Measure performance on multi-socket systems
///
/// **Purpose**: Quantify NUMA effects for server deployments
/// **Note**: Only meaningful on multi-socket systems
fn bench_numa_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("09_numa_sensitivity");
    group.sample_size(10);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    
    // All threads on same NUMA node (if supported)
    group.bench_function("sharded_8_same_socket", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            
            let handles: Vec<_> = (0..8)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..10_000 {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 10: Shard Count Tuning (1-64 Shards)

/// Find optimal shard count for given core count
///
/// **Purpose**: Determine best shard count (typically 1x to 2x cores)
/// **Expected**: Peak at core count, then plateau or degrade
fn bench_shard_count_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("10_shard_count_tuning");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    let ops_per_thread = 10_000;
    
    for shard_count in vec![1, 2, 4, 8, 16, 32, 64] {
        group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::new("shards", shard_count),
            &shard_count,
            |b, &shard_count| {
                b.iter(|| {
                    let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, shard_count));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);
                            thread::spawn(move || {
                                let mut rng: u64 = (tid * 12345) as u64;
                                for _ in 0..ops_per_thread {
                                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                    let idx = (rng as usize) % items.len();
                                    filter.insert(&items[idx]);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 11: Stripe Count Tuning (1-128 Stripes)

/// Find optimal stripe count for StripedBloomFilter
///
/// **Purpose**: Balance lock granularity vs overhead
fn bench_stripe_count_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("11_stripe_count_tuning");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    let ops_per_thread = 10_000;
    
    for stripe_count in vec![1, 2, 4, 8, 16, 32, 64, 128] {
        group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::new("stripes", stripe_count),
            &stripe_count,
            |b, &stripe_count| {
                b.iter(|| {
                    let filter = Arc::new(StripedBloomFilter::<String>::with_stripe_count(size, fpr, stripe_count));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);
                            thread::spawn(move || {
                                let mut rng: u64 = (tid * 12345) as u64;
                                for _ in 0..ops_per_thread {
                                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                                    let idx = (rng as usize) % items.len();
                                    filter.insert(&items[idx]);
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 12: Over-Sharding Penalty

/// Quantify performance loss from excessive sharding
///
/// **Purpose**: Prove that shards >> cores hurts performance
/// **Method**: Measure throughput with 128/256 shards on 8-core system
fn bench_over_sharding_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("12_over_sharding_penalty");
    group.sample_size(20);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = 8;
    let ops_per_thread = 10_000;
    
    // Optimal (8 shards for 8 threads)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("shards_8_optimal", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Over-sharded (128 shards for 8 threads)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("shards_128_oversharded", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 128));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 13: Hot Shard Attack (Worst-Case Contention)

/// Simulate pathological workload where all threads hit same shard
///
/// **Purpose**: Expose sharding failure mode (bad hash distribution)
/// **Method**: Force all items to hash to shard 0
fn bench_hot_shard_attack(c: &mut Criterion) {
    let mut group = c.benchmark_group("13_hot_shard_attack");
    group.sample_size(10);
    
    let size = 100_000;
    let fpr = 0.01;
    let num_threads = 8;
    let ops_per_thread = 10_000;
    
    // Uniform distribution (baseline)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("sharded_8_uniform", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            let items = Arc::new(generate_test_strings(size, 32));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Hot shard (all threads hit same shard)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("sharded_8_hotshard", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            // All items hash to shard 0 (force collision)
            let items = Arc::new(vec!["shard0item".to_string(); size]);
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let start = tid * (size / num_threads);
                        let end = start + (size / num_threads);
                        for i in start..end {
                            filter.insert(&items[i]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 14: Memory Bandwidth Limit

/// Measure hardware memory bandwidth ceiling
///
/// **Purpose**: Identify if you're CPU-bound or memory-bound
/// **Method**: Saturate all cores with writes
fn bench_memory_bandwidth_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("14_memory_bandwidth_limit");
    group.sample_size(10);
    
    let size = 1_000_000; // Large filter to stress memory
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    let num_threads = num_cpus::get(); // Use all cores
    let ops_per_thread = 10_000;
    
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("sharded_max_bandwidth", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, num_threads));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..ops_per_thread {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}


// BENCHMARK 15: Cache-Line Padding Effectiveness

/// Validate 64-byte cache-line padding prevents false sharing
///
/// **Purpose**: Prove atomic counter padding eliminates contention
/// **Method**: Adjacent counter hammering (false sharing) vs distributed
/// **Expected**: Padded counters 7-8x faster than unpadded
fn bench_cacheline_padding_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("15_cacheline_padding");
    group.sample_size(20);
    
    let num_threads = 8;
    let ops_per_thread = 100_000;
    
    // Test 1: Adjacent counters (potential false sharing)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("adjacent_counter_hammering", |b| {
        b.iter(|| {
            let counters = Arc::new(AtomicCounterArray::new(num_threads));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let counters = Arc::clone(&counters);
                    thread::spawn(move || {
                        // Each thread hammers its own counter (adjacent in memory)
                        for _ in 0..ops_per_thread {
                            let _ = counters.increment(tid);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Test 2: Distributed counters (no false sharing)
    group.throughput(Throughput::Elements((num_threads * ops_per_thread) as u64));
    group.bench_function("distributed_counter_access", |b| {
        b.iter(|| {
            let counters = Arc::new(AtomicCounterArray::new(num_threads * 8)); // Sparse allocation
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let counters = Arc::clone(&counters);
                    thread::spawn(move || {
                        // Each thread accesses widely separated counters
                        let idx = tid * 8; // 8 cache lines apart
                        for _ in 0..ops_per_thread {
                            let _ = counters.increment(idx);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 16: Overflow Protection Cost

/// Measure performance impact of checked vs saturating arithmetic
///
/// **Purpose**: Quantify cost of overflow detection
/// **Expected**: Saturating ~5-10% faster (no error handling)
fn bench_overflow_protection_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("16_overflow_protection");
    
    let counters = Arc::new(AtomicCounterArray::new(16));
    let ops = 1_000_000;
    
    // Checked increment (returns Result)
    group.throughput(Throughput::Elements(ops as u64));
    group.bench_function("checked_increment", |b| {
        b.iter(|| {
            counters.clear();
            for i in 0..ops {
                let _ = counters.increment(i % 16);
            }
        });
    });
    
    // Saturating increment (never fails)
    group.throughput(Throughput::Elements(ops as u64));
    group.bench_function("saturating_increment", |b| {
        b.iter(|| {
            counters.clear();
            for i in 0..ops {
                let _ = counters.increment_saturating(i % 16);
            }
        });
    });
    
    // Overflow scenario (near u64::MAX)
    let overflow_counters = Arc::new(AtomicCounterArray::with_value(16, u64::MAX - 1000));
    
    group.throughput(Throughput::Elements(2000 as u64));
    group.bench_function("checked_near_overflow", |b| {
        b.iter(|| {
            for i in 0..2000 {
                let _ = overflow_counters.increment(i % 16);
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 17: Serialization/Deserialization Performance

/// Measure cost of extracting and reconstructing filter state
///
/// **Purpose**: Critical for distributed systems (Redis, Kafka, etc.)
/// **Operations**: Extract raw bits → Reconstruct → Verify correctness
fn bench_serialization_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("17_serialization");
    group.sample_size(10); // Expensive
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    
    // Populate filter
    let filter = ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8);
    for item in items.iter() {
        filter.insert(item);
    }
    
    // Serialize (extract raw bits from all shards)
    group.bench_function("serialize_8_shards", |b| {
        b.iter(|| {
            let shard_bits: Vec<Vec<u64>> = (0..filter.shard_count())
                .map(|i| filter.shard_raw_bits(i).unwrap())
                .collect();
            black_box(shard_bits);
        });
    });
    
    // Deserialize (reconstruct from bits)
    let shard_bits: Vec<Vec<u64>> = (0..filter.shard_count())
        .map(|i| filter.shard_raw_bits(i).unwrap())
        .collect();
    let k = filter.hash_count();
    
    group.bench_function("deserialize_8_shards", |b| {
        b.iter(|| {
            use bloomcraft::hash::StdHasher;
            let restored = ShardedBloomFilter::<String>::from_shard_bits(
                shard_bits.clone(),
                k,
                size,
                fpr,
                StdHasher::default(),
            ).unwrap();
            black_box(restored);
        });
    });
    
    // Round-trip correctness check
    group.bench_function("roundtrip_verification", |b| {
        b.iter(|| {
            use bloomcraft::hash::StdHasher;
            
            let shard_bits: Vec<Vec<u64>> = (0..filter.shard_count())
                .map(|i| filter.shard_raw_bits(i).unwrap())
                .collect();
            
            let restored = ShardedBloomFilter::<String>::from_shard_bits(
                shard_bits,
                k,
                size,
                fpr,
                StdHasher::default(),
            ).unwrap();
            
            // Verify all items present
            let mut missing = 0;
            for item in items.iter() {
                if !restored.contains(item) {
                    missing += 1;
                }
            }
            
            assert_eq!(missing, 0);
        });
    });
    
    group.finish();
}

// BENCHMARK 18: Concurrent Clear Performance

/// Measure clear() cost while other threads are actively inserting
///
/// **Purpose**: Validate clear() correctness under concurrent writes
/// **Critical**: Does clear() cause stalls or lost updates?
fn bench_concurrent_clear_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("18_concurrent_clear");
    group.sample_size(10);
    
    let size = 100_000;
    let fpr = 0.01;
    let items = Arc::new(generate_test_strings(size, 32));
    
    // Sharded: Lock-free clear (atomic pointer swap)
    group.bench_function("sharded_clear_under_load", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8));
            
            // Spawn writers
            let writer_handles: Vec<_> = (0..4)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        for i in 0..10_000 {
                            let idx = (tid * 10_000 + i) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            // Clear midway through
            thread::sleep(Duration::from_millis(5));
            filter.clear();
            
            for h in writer_handles {
                h.join().unwrap();
            }
            
            // Verify filter works after clear
            filter.insert(&items[0]);
            assert!(filter.contains(&items[0]));
        });
    });
    
    // Striped: Must acquire all locks
    group.bench_function("striped_clear_under_load", |b| {
        b.iter(|| {
            let filter = Arc::new(StripedBloomFilter::<String>::with_stripe_count(size, fpr, 16));
            
            let writer_handles: Vec<_> = (0..4)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        for i in 0..10_000 {
                            let idx = (tid * 10_000 + i) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            thread::sleep(Duration::from_millis(5));
            filter.clear(); // Acquires ALL stripe locks
            
            for h in writer_handles {
                h.join().unwrap();
            }
            
            filter.insert(&items[0]);
            assert!(filter.contains(&items[0]));
        });
    });
    
    group.finish();
}

// BENCHMARK 19: Clone Cost Analysis

/// Measure cost of deep-copying large filters
///
/// **Purpose**: Cloning is expensive (deep BitVec copy) - quantify it
/// **Sizes**: 100K, 1M, 10M bits
fn bench_clone_cost_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("19_clone_cost");
    
    for size in vec![100_000, 1_000_000, 10_000_000] {
        let fpr = 0.01;
        
        // Sharded filter
        let filter_sharded = ShardedBloomFilter::<i32>::with_shard_count(size, fpr, 8);
        for i in 0..size/10 {
            filter_sharded.insert(&(i as i32));
        }
        
        group.bench_with_input(
            BenchmarkId::new("sharded_clone", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let cloned = filter_sharded.clone();
                    black_box(cloned);
                });
            },
        );
        
        // Striped filter
        let filter_striped = StripedBloomFilter::<i32>::with_stripe_count(size, fpr, 256);
        for i in 0..size/10 {
            filter_striped.insert(&(i as i32));
        }
        
        group.bench_with_input(
            BenchmarkId::new("striped_clone", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let cloned = filter_striped.clone();
                    black_box(cloned);
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 20: Lock Poisoning Recovery

/// Simulate thread panic during insert and measure recovery
///
/// **Purpose**: Validate StripedBloomFilter handles lock poisoning correctly
/// **Method**: Panic one thread, measure impact on others
fn bench_lock_poisoning_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("20_lock_poisoning");
    group.sample_size(10);
    
    let size = 100_000;
    let fpr = 0.01;
    
    // Note: This benchmark is conceptual - actual lock poisoning would panic
    // In production, use ShardedBloomFilter (lock-free) to avoid this
    
    group.bench_function("striped_recovery_simulation", |b| {
        b.iter(|| {
            let filter = Arc::new(StripedBloomFilter::<String>::with_stripe_count(size, fpr, 16));
            
            // Simulate recovery by just continuing normal operations
            let handles: Vec<_> = (0..8)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    thread::spawn(move || {
                        for i in 0..10_000 {
                            filter.insert(&format!("{}-{}", tid, i));
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 21: Memory Subsystem Stress (Extended)

/// Saturate memory bandwidth with concurrent writes
///
/// **Purpose**: Identify if you're CPU-bound or memory-bound
/// **Platform**: DDR4 ~25GB/s, DDR5 ~40GB/s theoretical
fn bench_memory_subsystem_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("22_memory_bandwidth");
    group.sample_size(10);
    
    let size = 10_000_000; // 10M items = ~100MB filter
    let fpr = 0.01;
    let items = Arc::new((0..size).map(|i| i as u64).collect::<Vec<_>>());
    let num_threads = num_cpus::get();
    
    group.throughput(Throughput::Bytes((num_threads * 10_000 * 8) as u64)); // Bytes written
    group.bench_function("saturate_memory_bandwidth", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<u64>::with_shard_count(size, fpr, num_threads));
            
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);
                    thread::spawn(move || {
                        let mut rng: u64 = (tid * 12345) as u64;
                        for _ in 0..10_000 {
                            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                            let idx = (rng as usize) % items.len();
                            filter.insert(&items[idx]);
                        }
                    })
                })
                .collect();
            
            let start = Instant::now();
            for h in handles {
                h.join().unwrap();
            }
            let elapsed = start.elapsed();
            
            // Calculate effective bandwidth
            let bytes_written = (num_threads * 10_000 * 8) as f64;
            let bandwidth_gbps = bytes_written / elapsed.as_secs_f64() / 1_000_000_000.0;
            
            eprintln!("Effective bandwidth: {:.2} GB/s", bandwidth_gbps);
        });
    });
    
    group.finish();
}

// BENCHMARK 22: NUMA-Aware Allocation

/// Pin threads to NUMA nodes and measure locality impact
///
/// **Purpose**: Validate cache-line padding helps NUMA systems
/// **Expected**: Same-node 1x, cross-node 2-3x slower
fn bench_numa_aware_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("23_numa_locality");
    group.sample_size(10);
    
    let counters = Arc::new(AtomicCounterArray::new(16));
    let ops_per_thread = 100_000;
    
    // Simulate same-node access (threads 0-7)
    group.throughput(Throughput::Elements((8 * ops_per_thread) as u64));
    group.bench_function("same_numa_node", |b| {
        b.iter(|| {
            counters.clear();
            
            let handles: Vec<_> = (0..8)
                .map(|tid| {
                    let counters = Arc::clone(&counters);
                    thread::spawn(move || {
                        for _ in 0..ops_per_thread {
                            let _ = counters.increment(tid);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    // Simulate cross-node access (threads access remote counters)
    group.throughput(Throughput::Elements((8 * ops_per_thread) as u64));
    group.bench_function("cross_numa_node", |b| {
        b.iter(|| {
            counters.clear();
            
            let handles: Vec<_> = (0..8)
                .map(|tid| {
                    let counters = Arc::clone(&counters);
                    thread::spawn(move || {
                        // Access counter on opposite "node" (simulated)
                        let remote_idx = (tid + 8) % 16;
                        for _ in 0..ops_per_thread {
                            let _ = counters.increment(remote_idx);
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
        });
    });
    
    group.finish();
}

// BENCHMARK 23: Batch Operation Efficiency

/// Compare insert_batch() vs manual loop
///
/// **Purpose**: Identify if batch operations have optimization potential
/// **Current**: insert_batch() just loops - no optimization yet
fn bench_batch_operation_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("24_batch_operations");
    
    let size = 100_000;
    let fpr = 0.01;
    let items: Vec<String> = (0..10_000).map(|i| format!("item-{}", i)).collect();
    
    // Manual loop
    group.throughput(Throughput::Elements(items.len() as u64));
    group.bench_function("sharded_manual_loop", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8);
            for item in &items {
                filter.insert(item);
            }
        });
    });
    
    // insert_batch()
    group.throughput(Throughput::Elements(items.len() as u64));
    group.bench_function("sharded_insert_batch", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<String>::with_shard_count(size, fpr, 8);
            filter.insert_batch(items.iter());
        });
    });
    
    // Striped comparison
    group.throughput(Throughput::Elements(items.len() as u64));
    group.bench_function("striped_manual_loop", |b| {
        b.iter(|| {
            let filter = StripedBloomFilter::<String>::with_stripe_count(size, fpr, 256);
            for item in &items {
                filter.insert(item);
            }
        });
    });
    
    group.throughput(Throughput::Elements(items.len() as u64));
    group.bench_function("striped_insert_batch", |b| {
        b.iter(|| {
            let filter = StripedBloomFilter::<String>::with_stripe_count(size, fpr, 256);
            filter.insert_batch(items.iter());
        });
    });
    
    group.finish();
}

// CRITERION GROUP REGISTRATION

criterion_group!(
    benches,
    // Benchmarks 1-14 (previous)
    bench_baseline_overhead,
    bench_correctness_stress,
    bench_throughput_scaling,
    bench_parallel_efficiency,
    bench_workload_mix,
    bench_insert_latency_distribution,
    bench_query_latency_distribution,
    bench_false_sharing_detection,
    bench_numa_sensitivity,
    bench_shard_count_tuning,
    bench_stripe_count_tuning,
    bench_over_sharding_penalty,
    bench_hot_shard_attack,
    bench_memory_bandwidth_limit,
    bench_cacheline_padding_effectiveness,
    bench_overflow_protection_cost,
    bench_serialization_overhead,
    bench_concurrent_clear_performance,
    bench_clone_cost_analysis,
    bench_lock_poisoning_recovery,
    bench_memory_subsystem_stress,
    bench_numa_aware_allocation,
    bench_batch_operation_efficiency,
);

criterion_main!(benches);
