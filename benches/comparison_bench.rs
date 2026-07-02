//! Cross-variant Bloom filter benchmark suite.
//!
//! Groups:
//!   - `comparison/` — baseline single-threaded throughput, FPR, metadata
//!   - `scaling/`    — throughput as inputs, threads, batch sizes, or R/W mix vary
//!   - `quirks/`     — edge-case behaviours (overfill, string types)
//!   - `latency/`    — tail-latency percentiles under concurrent load
//!   - `memory/`     — actual per-variant memory footprint

use bloomcraft::core::filter::{BloomFilter, ConcurrentBloomFilter, SharedBloomFilter};
use bloomcraft::filters::{
    ClassicBitsFilter, ClassicHashFilter, CountingBloomFilter, PartitionedBloomFilter,
    RegisterBlockedBloomFilter, ScalableBloomFilter, StandardBloomFilter, TreeBloomFilter,
};
use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[cfg(feature = "concurrent")]
use bloomcraft::filters::{AtomicPartitionedBloomFilter, AtomicScalableBloomFilter};

const N: usize = 100_000;
const TARGET_FPR: f64 = 0.01;
const QUERIES: usize = 100_000;
const TREE_BRANCHING: [usize; 2] = [10, 10];

const INPUT_SIZES: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];
const THREAD_COUNTS: [usize; 5] = [1, 2, 4, 8, 16];
const BATCH_SIZES: [usize; 5] = [1, 10, 100, 1_000, 10_000];
const MIX_RATIOS: [(f64, &str); 3] = [
    (0.1, "10pct_write"),
    (0.5, "50pct_write"),
    (0.9, "90pct_write"),
];
const DICT_WORDS: &[&str] = &[
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with",
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us", "great", "between",
    "need", "large", "often", "hand", "high", "place", "small", "under", "long", "right", "still",
    "house", "world", "last", "school", "never", "city", "tree", "cross", "farm", "hard", "start",
    "might", "story", "saw", "far", "sea", "draw", "left", "late", "run", "while", "press",
    "close", "night", "real", "life", "few", "north", "open", "seem", "together", "next", "white",
    "children", "begin", "got", "walk", "example", "ease", "paper", "group", "always", "music",
    "those", "both", "mark", "book", "letter", "until", "mile", "river", "car", "feet", "care",
    "second", "enough", "plain", "girl", "usual", "young", "ready", "above", "ever", "red", "list",
    "though", "feel", "talk", "bird", "soon", "body", "dog", "family", "direct", "pose", "leave",
    "song", "measure", "door", "product", "black", "short", "number", "class", "wind", "question",
    "happen", "complete", "ship", "area", "half", "rock", "order", "fire", "south", "problem",
    "piece", "told", "knew", "pass", "since", "top", "whole", "king", "space", "heard", "best",
    "hour", "better", "true", "during", "hundred", "remember", "step", "early", "hold", "west",
    "ground", "interest", "reach", "fast", "verb", "sing", "listen", "six", "table", "travel",
    "less", "morning", "ten", "simple", "several", "vowel", "toward", "war", "lay", "against",
    "pattern",
];

// ─── Utilities ───────────────────────────────────────────────────────────────

fn seeded_data(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.gen()).collect()
}

fn seeded_queries(hits: &[u64], misses: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(99);
    let mut q = hits.to_vec();
    q.extend((0..misses).map(|_| rng.gen::<u64>()));
    q
}

fn gen_absent(count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(123);
    (0..count).map(|_| rng.gen()).collect()
}

fn measure_fpr<F, I>(filter: &F, absent: I) -> f64
where
    F: SharedBloomFilter<u64>,
    I: IntoIterator<Item = u64>,
{
    let items: Vec<_> = absent.into_iter().collect();
    let fp = items.iter().filter(|x| filter.contains(x)).count();
    fp as f64 / items.len() as f64
}

fn measure_fpr_bf<F>(filter: &F, absent: &[u64]) -> f64
where
    F: BloomFilter<u64>,
{
    let fp = absent.iter().filter(|x| filter.contains(x)).count();
    fp as f64 / absent.len() as f64
}

fn dict_words(count: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| DICT_WORDS[rng.gen_range(0..DICT_WORDS.len())].to_string())
        .collect()
}

enum MixedOp {
    Insert(u64),
    Contains(u64),
}

fn gen_mixed_ops(
    write_ratio: f64,
    total: usize,
    existing_hits: &[u64],
    absent: &[u64],
    seed: u64,
) -> Vec<MixedOp> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ops = Vec::with_capacity(total);
    for _ in 0..total {
        if rng.gen_bool(write_ratio) {
            ops.push(MixedOp::Insert(rng.gen()));
        } else {
            let src = if rng.gen_bool(0.5) {
                existing_hits
            } else {
                absent
            };
            ops.push(MixedOp::Contains(src[rng.gen_range(0..src.len())]));
        }
    }
    ops
}

// ─── comparison/insert_throughput ────────────────────────────────────────────

fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/insert_throughput");
    group.throughput(Throughput::Elements(N as u64));
    let data = seeded_data(N);

    group.bench_with_input(
        BenchmarkId::new("StandardBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("CountingBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = CountingBloomFilter::<u64>::new(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ScalableBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = ScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                let _ = f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("PartitionedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("RegisterBlockedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = RegisterBlockedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("TreeBloomFilter", N), &data, |b, data| {
        b.iter(|| {
            let cap = N / TREE_BRANCHING.iter().product::<usize>();
            let mut f =
                TreeBloomFilter::<u64>::new(TREE_BRANCHING.to_vec(), cap, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            black_box(f.count_set_bits());
        });
    });

    group.bench_with_input(
        BenchmarkId::new("ClassicBitsFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = ClassicBitsFilter::<u64>::with_fpr(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ClassicHashFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = ClassicHashFilter::<u64>::with_fpr(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.len());
            });
        },
    );

    #[cfg(feature = "concurrent")]
    group.bench_with_input(
        BenchmarkId::new("AtomicPartitionedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ShardedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = ShardedBloomFilter::<u64>::new(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("StripedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    #[cfg(feature = "concurrent")]
    group.bench_with_input(
        BenchmarkId::new("AtomicScalableBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                let _ = f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.finish();
}

// ─── comparison/contains_throughput ──────────────────────────────────────────

fn bench_contains_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/contains_throughput");
    group.throughput(Throughput::Elements(QUERIES as u64));
    let data = seeded_data(N);
    let queries = seeded_queries(&data[..N / 2], QUERIES - N / 2);

    group.bench_with_input(
        BenchmarkId::new("StandardBloomFilter", N),
        &queries,
        |b, q| {
            let f = StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("CountingBloomFilter", N),
        &queries,
        |b, q| {
            let mut f = CountingBloomFilter::<u64>::new(N, TARGET_FPR);
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ScalableBloomFilter", N),
        &queries,
        |b, q| {
            let mut f = ScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            let _ = f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("PartitionedBloomFilter", N),
        &queries,
        |b, q| {
            let mut f = PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("RegisterBlockedBloomFilter", N),
        &queries,
        |b, q| {
            let mut f = RegisterBlockedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("TreeBloomFilter", N), &queries, |b, q| {
        let cap = N / TREE_BRANCHING.iter().product::<usize>();
        let mut f = TreeBloomFilter::<u64>::new(TREE_BRANCHING.to_vec(), cap, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            for x in q {
                black_box(f.contains(x));
            }
        });
    });

    group.bench_with_input(
        BenchmarkId::new("ClassicBitsFilter", N),
        &queries,
        |b, q| {
            let mut f = ClassicBitsFilter::<u64>::with_fpr(N, TARGET_FPR);
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ClassicHashFilter", N),
        &queries,
        |b, q| {
            let mut f = ClassicHashFilter::<u64>::with_fpr(N, TARGET_FPR);
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    #[cfg(feature = "concurrent")]
    group.bench_with_input(
        BenchmarkId::new("AtomicPartitionedBloomFilter", N),
        &queries,
        |b, q| {
            let mut f = AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ShardedBloomFilter", N),
        &queries,
        |b, q| {
            let f = ShardedBloomFilter::<u64>::new(N, TARGET_FPR);
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("StripedBloomFilter", N),
        &queries,
        |b, q| {
            let f = StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    #[cfg(feature = "concurrent")]
    group.bench_with_input(
        BenchmarkId::new("AtomicScalableBloomFilter", N),
        &queries,
        |b, q| {
            let f = AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
            let _ = f.insert_batch(data.as_slice());
            b.iter(|| {
                for x in q {
                    black_box(f.contains(x));
                }
            });
        },
    );

    group.finish();
}

// ─── comparison/fpr_accuracy ─────────────────────────────────────────────────

fn bench_fpr_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/fpr_accuracy");
    let data = seeded_data(N);
    let absent = gen_absent(QUERIES);

    group.bench_function(BenchmarkId::new("StandardBloomFilter", N), |b| {
        let f = StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("CountingBloomFilter", N), |b| {
        let mut f = CountingBloomFilter::<u64>::new(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("ScalableBloomFilter", N), |b| {
        let mut f = ScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        let _ = f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("PartitionedBloomFilter", N), |b| {
        let mut f = PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("RegisterBlockedBloomFilter", N), |b| {
        let mut f = RegisterBlockedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("TreeBloomFilter", N), |b| {
        let cap = N / TREE_BRANCHING.iter().product::<usize>();
        let mut f = TreeBloomFilter::<u64>::new(TREE_BRANCHING.to_vec(), cap, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("ClassicBitsFilter", N), |b| {
        let mut f = ClassicBitsFilter::<u64>::with_fpr(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("ClassicHashFilter", N), |b| {
        let mut f = ClassicHashFilter::<u64>::with_fpr(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    #[cfg(feature = "concurrent")]
    group.bench_function(BenchmarkId::new("AtomicPartitionedBloomFilter", N), |b| {
        let mut f = AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr_bf(&f, &absent)));
    });

    group.bench_function(BenchmarkId::new("ShardedBloomFilter", N), |b| {
        let f = ShardedBloomFilter::<u64>::new(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr(&f, absent.iter().copied())));
    });

    group.bench_function(BenchmarkId::new("StripedBloomFilter", N), |b| {
        let f = StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr(&f, absent.iter().copied())));
    });

    #[cfg(feature = "concurrent")]
    group.bench_function(BenchmarkId::new("AtomicScalableBloomFilter", N), |b| {
        let f = AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        let _ = f.insert_batch(data.as_slice());
        b.iter(|| black_box(measure_fpr(&f, absent.iter().copied())));
    });

    group.finish();
}

// ─── comparison/bit_count_read_latency ───────────────────────────────────────

fn bench_bit_count_read_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/bit_count_read_latency");
    let data = seeded_data(N);

    group.bench_function(BenchmarkId::new("StandardBloomFilter", N), |b| {
        let f = StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("CountingBloomFilter", N), |b| {
        let mut f = CountingBloomFilter::<u64>::new(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("ScalableBloomFilter", N), |b| {
        let mut f = ScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        let _ = f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("PartitionedBloomFilter", N), |b| {
        let mut f = PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("RegisterBlockedBloomFilter", N), |b| {
        let mut f = RegisterBlockedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("TreeBloomFilter", N), |b| {
        let cap = N / TREE_BRANCHING.iter().product::<usize>();
        let mut f = TreeBloomFilter::<u64>::new(TREE_BRANCHING.to_vec(), cap, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("ClassicBitsFilter", N), |b| {
        let mut f = ClassicBitsFilter::<u64>::with_fpr(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("ClassicHashFilter", N), |b| {
        let mut f = ClassicHashFilter::<u64>::with_fpr(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    #[cfg(feature = "concurrent")]
    group.bench_function(BenchmarkId::new("AtomicPartitionedBloomFilter", N), |b| {
        let mut f = AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("ShardedBloomFilter", N), |b| {
        let f = ShardedBloomFilter::<u64>::new(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.bench_function(BenchmarkId::new("StripedBloomFilter", N), |b| {
        let f = StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    #[cfg(feature = "concurrent")]
    group.bench_function(BenchmarkId::new("AtomicScalableBloomFilter", N), |b| {
        let f = AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        let _ = f.insert_batch(data.as_slice());
        b.iter(|| {
            let bits = f.bit_count();
            let fpr = f.false_positive_rate();
            black_box((bits, fpr));
        });
    });

    group.finish();
}

// ─── scaling/input_insert ────────────────────────────────────────────────────

fn bench_input_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/input_insert");

    for &size in &INPUT_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        let data = seeded_data(size);

        group.bench_with_input(
            BenchmarkId::new("StandardBloomFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let f = StandardBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                    f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PartitionedBloomFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut f = PartitionedBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                    f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ScalableBloomFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut f = ScalableBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                    let _ = f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ShardedBloomFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let f = ShardedBloomFilter::<u64>::new(size, TARGET_FPR);
                    f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ClassicBitsFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut f = ClassicBitsFilter::<u64>::with_fpr(size, TARGET_FPR);
                    f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ClassicHashFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut f = ClassicHashFilter::<u64>::with_fpr(size, TARGET_FPR);
                    f.insert_batch(data.as_slice());
                    black_box(f.len());
                });
            },
        );

        #[cfg(feature = "concurrent")]
        group.bench_with_input(
            BenchmarkId::new("AtomicPartitionedBloomFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut f = AtomicPartitionedBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                    f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StripedBloomFilter", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let f = StripedBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                    f.insert_batch(data.as_slice());
                    black_box(f.count_set_bits());
                });
            },
        );
    }

    group.finish();
}

// ─── scaling/input_contains ──────────────────────────────────────────────────

fn bench_input_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/input_contains");

    for &size in &INPUT_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        let data = seeded_data(size);
        let queries = seeded_queries(&data[..size / 2], size - size / 2);

        group.bench_with_input(
            BenchmarkId::new("StandardBloomFilter", size),
            &queries,
            |b, q| {
                let f = StandardBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PartitionedBloomFilter", size),
            &queries,
            |b, q| {
                let mut f = PartitionedBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ScalableBloomFilter", size),
            &queries,
            |b, q| {
                let mut f = ScalableBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                let _ = f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ShardedBloomFilter", size),
            &queries,
            |b, q| {
                let f = ShardedBloomFilter::<u64>::new(size, TARGET_FPR);
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ClassicBitsFilter", size),
            &queries,
            |b, q| {
                let mut f = ClassicBitsFilter::<u64>::with_fpr(size, TARGET_FPR);
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ClassicHashFilter", size),
            &queries,
            |b, q| {
                let mut f = ClassicHashFilter::<u64>::with_fpr(size, TARGET_FPR);
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        #[cfg(feature = "concurrent")]
        group.bench_with_input(
            BenchmarkId::new("AtomicPartitionedBloomFilter", size),
            &queries,
            |b, q| {
                let mut f = AtomicPartitionedBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StripedBloomFilter", size),
            &queries,
            |b, q| {
                let f = StripedBloomFilter::<u64>::new(size, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                b.iter(|| {
                    for x in q {
                        black_box(f.contains(x));
                    }
                });
            },
        );
    }

    group.finish();
}

// ─── scaling/concurrent_threads ──────────────────────────────────────────────

fn bench_concurrent_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/concurrent_threads");

    for &threads in &THREAD_COUNTS {
        group.throughput(Throughput::Elements(N as u64));

        group.bench_with_input(
            BenchmarkId::new("StandardBloomFilter", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap()),
                    |filter| {
                        let chunk = N / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 { N } else { start + chunk };
                                thread::spawn(move || {
                                    for i in start..end {
                                        f.insert(&(i as u64));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ShardedBloomFilter", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(ShardedBloomFilter::<u64>::new(N, TARGET_FPR)),
                    |filter| {
                        let chunk = N / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 { N } else { start + chunk };
                                thread::spawn(move || {
                                    for i in start..end {
                                        f.insert(&(i as u64));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StripedBloomFilter", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap()),
                    |filter| {
                        let chunk = N / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 { N } else { start + chunk };
                                thread::spawn(move || {
                                    for i in start..end {
                                        f.insert(&(i as u64));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        #[cfg(feature = "concurrent")]
        group.bench_with_input(
            BenchmarkId::new("AtomicScalableBloomFilter", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap()),
                    |filter| {
                        let chunk = N / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 { N } else { start + chunk };
                                thread::spawn(move || {
                                    for i in start..end {
                                        f.insert(&(i as u64));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        #[cfg(feature = "concurrent")]
        group.bench_with_input(
            BenchmarkId::new("AtomicPartitionedBloomFilter", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap()),
                    |filter| {
                        let chunk = N / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 { N } else { start + chunk };
                                thread::spawn(move || {
                                    for i in start..end {
                                        f.insert_concurrent(&(i as u64));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ─── scaling/read_write_mix ─────────────────────────────────────────────────

fn bench_read_write_mix(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/read_write_mix");
    let threads = 8usize;
    let total_ops = N * 4;

    let data = seeded_data(N);
    let baseline_absent = gen_absent(N);

    for &(ratio, label) in &MIX_RATIOS {
        group.throughput(Throughput::Elements(total_ops as u64));
        let ops = gen_mixed_ops(ratio, total_ops, &data, &baseline_absent, 42);
        let ops = Arc::new(ops);

        group.bench_with_input(
            BenchmarkId::new("StandardBloomFilter", label),
            &label,
            |b, _| {
                b.iter_batched(
                    || {
                        let f = Arc::new(StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap());
                        f.insert_batch(data.as_slice());
                        f
                    },
                    |filter| {
                        let chunk = total_ops / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let o = Arc::clone(&ops);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 {
                                    total_ops
                                } else {
                                    start + chunk
                                };
                                thread::spawn(move || {
                                    for i in start..end {
                                        match o[i] {
                                            MixedOp::Insert(v) => {
                                                f.insert(&v);
                                            }
                                            MixedOp::Contains(v) => {
                                                black_box(f.contains(&v));
                                            }
                                        }
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ShardedBloomFilter", label),
            &label,
            |b, _| {
                b.iter_batched(
                    || {
                        let f = Arc::new(ShardedBloomFilter::<u64>::new(N, TARGET_FPR));
                        f.insert_batch(data.iter());
                        f
                    },
                    |filter| {
                        let chunk = total_ops / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let o = Arc::clone(&ops);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 {
                                    total_ops
                                } else {
                                    start + chunk
                                };
                                thread::spawn(move || {
                                    for i in start..end {
                                        match o[i] {
                                            MixedOp::Insert(v) => {
                                                f.insert(&v);
                                            }
                                            MixedOp::Contains(v) => {
                                                black_box(f.contains(&v));
                                            }
                                        }
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StripedBloomFilter", label),
            &label,
            |b, _| {
                b.iter_batched(
                    || {
                        let f = Arc::new(StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap());
                        f.insert_batch(data.iter());
                        f
                    },
                    |filter| {
                        let chunk = total_ops / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let o = Arc::clone(&ops);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 {
                                    total_ops
                                } else {
                                    start + chunk
                                };
                                thread::spawn(move || {
                                    for i in start..end {
                                        match o[i] {
                                            MixedOp::Insert(v) => {
                                                f.insert(&v);
                                            }
                                            MixedOp::Contains(v) => {
                                                black_box(f.contains(&v));
                                            }
                                        }
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        #[cfg(feature = "concurrent")]
        group.bench_with_input(
            BenchmarkId::new("AtomicScalableBloomFilter", label),
            &label,
            |b, _| {
                b.iter_batched(
                    || {
                        let f =
                            Arc::new(AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap());
                        let _ = f.insert_batch(data.as_slice());
                        f
                    },
                    |filter| {
                        let chunk = total_ops / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let o = Arc::clone(&ops);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 {
                                    total_ops
                                } else {
                                    start + chunk
                                };
                                thread::spawn(move || {
                                    for i in start..end {
                                        match o[i] {
                                            MixedOp::Insert(v) => {
                                                f.insert(&v);
                                            }
                                            MixedOp::Contains(v) => {
                                                black_box(f.contains(&v));
                                            }
                                        }
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        #[cfg(feature = "concurrent")]
        group.bench_with_input(
            BenchmarkId::new("AtomicPartitionedBloomFilter", label),
            &label,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut f =
                            AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                        f.insert_batch(data.as_slice());
                        Arc::new(f)
                    },
                    |filter| {
                        let chunk = total_ops / threads;
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let o = Arc::clone(&ops);
                                let start = tid * chunk;
                                let end = if tid == threads - 1 {
                                    total_ops
                                } else {
                                    start + chunk
                                };
                                thread::spawn(move || {
                                    for i in start..end {
                                        match o[i] {
                                            MixedOp::Insert(v) => {
                                                f.insert_concurrent(&v);
                                            }
                                            MixedOp::Contains(v) => {
                                                black_box(f.contains(&v));
                                            }
                                        }
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ─── scaling/batch_insert ────────────────────────────────────────────────────

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/batch_insert");
    let full_data = seeded_data(N);

    for &batch in &BATCH_SIZES {
        group.throughput(Throughput::Elements(N as u64));

        group.bench_with_input(
            BenchmarkId::new("StandardBloomFilter", batch),
            &batch,
            |b, &batch| {
                b.iter_batched(
                    || {
                        (
                            StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap(),
                            full_data.clone(),
                        )
                    },
                    |(f, data)| {
                        let chunks: Vec<_> = data.chunks(batch).map(|c| c.to_vec()).collect();
                        for chunk in &chunks {
                            f.insert_batch(chunk.as_slice());
                        }
                        black_box(f.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("PartitionedBloomFilter", batch),
            &batch,
            |b, &batch| {
                b.iter_batched(
                    || {
                        (
                            PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap(),
                            full_data.clone(),
                        )
                    },
                    |(mut f, data)| {
                        let chunks: Vec<_> = data.chunks(batch).map(|c| c.to_vec()).collect();
                        for chunk in &chunks {
                            f.insert_batch(chunk.as_slice());
                        }
                        black_box(f.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ShardedBloomFilter", batch),
            &batch,
            |b, &batch| {
                b.iter_batched(
                    || {
                        (
                            ShardedBloomFilter::<u64>::new(N, TARGET_FPR),
                            full_data.clone(),
                        )
                    },
                    |(f, data)| {
                        let chunks: Vec<_> = data.chunks(batch).map(|c| c.to_vec()).collect();
                        for chunk in &chunks {
                            f.insert_batch(chunk.as_slice());
                        }
                        black_box(f.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StripedBloomFilter", batch),
            &batch,
            |b, &batch| {
                b.iter_batched(
                    || {
                        (
                            StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap(),
                            full_data.clone(),
                        )
                    },
                    |(f, data)| {
                        let chunks: Vec<_> = data.chunks(batch).map(|c| c.to_vec()).collect();
                        for chunk in &chunks {
                            f.insert_batch(chunk.as_slice());
                        }
                        black_box(f.count_set_bits());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ─── quirks/fpr_under_overfill ─────────────────────────────────────────────

fn bench_fpr_under_overfill(c: &mut Criterion) {
    let mut group = c.benchmark_group("quirks/fpr_under_overfill");
    let data = seeded_data(N * 2);
    let absent = gen_absent(QUERIES);

    for &fill in &[1.0f64, 1.5, 2.0] {
        let count = (N as f64 * fill) as usize;

        eprintln!(
            "--- StandardBloomFilter at {:.0}% fill ({} items) ---",
            fill * 100.0,
            count
        );
        group.bench_with_input(
            BenchmarkId::new("StandardBloomFilter", format!("{:.0}pct", fill * 100.0)),
            &fill,
            |b, _fill| {
                let f = StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data[..count].as_ref());
                let fp = absent.iter().filter(|x| f.contains(x)).count();
                let fpr = fp as f64 / absent.len() as f64;
                eprintln!("  empirical FPR = {:.4}%", fpr * 100.0);
                b.iter(|| black_box(f.count_set_bits()));
            },
        );

        eprintln!(
            "--- ScalableBloomFilter at {:.0}% fill ({} items) ---",
            fill * 100.0,
            count
        );
        group.bench_with_input(
            BenchmarkId::new("ScalableBloomFilter", format!("{:.0}pct", fill * 100.0)),
            &fill,
            |b, _fill| {
                let mut f = ScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                let _ = f.insert_batch(data[..count].as_ref());
                let fp = absent.iter().filter(|x| f.contains(x)).count();
                let fpr = fp as f64 / absent.len() as f64;
                eprintln!("  empirical FPR = {:.4}%", fpr * 100.0);
                b.iter(|| black_box(f.count_set_bits()));
            },
        );

        eprintln!(
            "--- PartitionedBloomFilter at {:.0}% fill ({} items) ---",
            fill * 100.0,
            count
        );
        group.bench_with_input(
            BenchmarkId::new("PartitionedBloomFilter", format!("{:.0}pct", fill * 100.0)),
            &fill,
            |b, _fill| {
                let mut f = PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data[..count].as_ref());
                let fp = absent.iter().filter(|x| f.contains(x)).count();
                let fpr = fp as f64 / absent.len() as f64;
                eprintln!("  empirical FPR = {:.4}%", fpr * 100.0);
                b.iter(|| black_box(f.count_set_bits()));
            },
        );

        eprintln!(
            "--- ShardedBloomFilter at {:.0}% fill ({} items) ---",
            fill * 100.0,
            count
        );
        group.bench_with_input(
            BenchmarkId::new("ShardedBloomFilter", format!("{:.0}pct", fill * 100.0)),
            &fill,
            |b, _fill| {
                let f = ShardedBloomFilter::<u64>::new(N, TARGET_FPR);
                f.insert_batch(data[..count].iter());
                let fp = absent.iter().filter(|x| f.contains(x)).count();
                let fpr = fp as f64 / absent.len() as f64;
                eprintln!("  empirical FPR = {:.4}%", fpr * 100.0);
                b.iter(|| black_box(f.count_set_bits()));
            },
        );

        eprintln!(
            "--- CountingBloomFilter at {:.0}% fill ({} items) ---",
            fill * 100.0,
            count
        );
        group.bench_with_input(
            BenchmarkId::new("CountingBloomFilter", format!("{:.0}pct", fill * 100.0)),
            &fill,
            |b, _fill| {
                let mut f = CountingBloomFilter::<u64>::new(N, TARGET_FPR);
                f.insert_batch(data[..count].as_ref());
                let fp = absent.iter().filter(|x| f.contains(x)).count();
                let fpr = fp as f64 / absent.len() as f64;
                eprintln!("  empirical FPR = {:.4}%", fpr * 100.0);
                b.iter(|| black_box(f.count_set_bits()));
            },
        );
    }

    group.finish();
}

// ─── quirks/string_dictionary ────────────────────────────────────────────────

fn bench_string_dictionary(c: &mut Criterion) {
    let data = dict_words(N);
    let mut group = c.benchmark_group("quirks/string_dictionary");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_with_input(
        BenchmarkId::new("StandardBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = StandardBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("CountingBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = CountingBloomFilter::<String>::new(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ScalableBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = ScalableBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                let _ = f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("PartitionedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = PartitionedBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("RegisterBlockedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = RegisterBlockedBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("TreeBloomFilter", N), &data, |b, data| {
        b.iter(|| {
            let cap = N / TREE_BRANCHING.iter().product::<usize>();
            let mut f =
                TreeBloomFilter::<String>::new(TREE_BRANCHING.to_vec(), cap, TARGET_FPR).unwrap();
            f.insert_batch(data.as_slice());
            black_box(f.count_set_bits());
        });
    });

    group.bench_with_input(
        BenchmarkId::new("ClassicBitsFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = ClassicBitsFilter::<String>::with_fpr(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ClassicHashFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = ClassicHashFilter::<String>::with_fpr(N, TARGET_FPR);
                f.insert_batch(data.as_slice());
                black_box(f.len());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("ShardedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = ShardedBloomFilter::<String>::new(N, TARGET_FPR);
                f.insert_batch(data.iter());
                black_box(f.count_set_bits());
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("StripedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = StripedBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.iter());
                black_box(f.count_set_bits());
            });
        },
    );

    #[cfg(feature = "concurrent")]
    group.bench_with_input(
        BenchmarkId::new("AtomicPartitionedBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let mut f = AtomicPartitionedBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    #[cfg(feature = "concurrent")]
    group.bench_with_input(
        BenchmarkId::new("AtomicScalableBloomFilter", N),
        &data,
        |b, data| {
            b.iter(|| {
                let f = AtomicScalableBloomFilter::<String>::new(N, TARGET_FPR).unwrap();
                let _ = f.insert_batch(data.as_slice());
                black_box(f.count_set_bits());
            });
        },
    );

    group.finish();
}

// ─── latency/concurrent_contains_p99 ─────────────────────────────────────────

fn bench_concurrent_contains_p99(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/concurrent_contains_p99");
    let data = seeded_data(N);
    let threads = 8usize;
    let queries_per_thread = 5_000usize;

    let run_p99 = |name: &str, filter: Arc<dyn Fn(&u64) -> bool + Send + Sync>| {
        eprintln!("\n--- {} p99 latency ---", name);
        let mut all_latencies = Vec::with_capacity(threads * queries_per_thread);
        let done = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let f = Arc::clone(&filter);
                let done = Arc::clone(&done);
                thread::spawn(move || {
                    let mut rng = StdRng::seed_from_u64(tid as u64);
                    let mut latencies = Vec::with_capacity(queries_per_thread);
                    for _ in 0..queries_per_thread {
                        let q = rng.gen::<u64>();
                        let start = Instant::now();
                        black_box(f(&q));
                        let elapsed = start.elapsed();
                        latencies.push(elapsed);
                    }
                    done.store(true, Ordering::Relaxed);
                    latencies
                })
            })
            .collect();

        for h in handles {
            all_latencies.extend(h.join().unwrap());
        }

        all_latencies.sort();
        let n = all_latencies.len();
        if n > 0 {
            let p50 = all_latencies[(n as f64 * 0.50) as usize].as_secs_f64() * 1e9;
            let p95 = all_latencies[(n as f64 * 0.95) as usize].as_secs_f64() * 1e9;
            let p99 = all_latencies[(n as f64 * 0.99) as usize].as_secs_f64() * 1e9;
            let p999 = all_latencies[(n as f64 * 0.999) as usize].as_secs_f64() * 1e9;
            eprintln!(
                "  p50={:.1}ns  p95={:.1}ns  p99={:.1}ns  p99.9={:.1}ns",
                p50, p95, p99, p999
            );
        }
    };

    group.bench_function(BenchmarkId::new("StandardBloomFilter", N), |b| {
        let f = Arc::new(StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap());
        f.insert_batch(data.as_slice());
        let filter = Arc::new(move |x: &u64| f.contains(x));
        run_p99("StandardBloomFilter", filter.clone());
        b.iter(|| {
            black_box(filter(&0));
        });
    });

    group.bench_function(BenchmarkId::new("ShardedBloomFilter", N), |b| {
        let f = Arc::new(ShardedBloomFilter::<u64>::new(N, TARGET_FPR));
        f.insert_batch(data.iter());
        let filter = Arc::new(move |x: &u64| f.contains(x));
        run_p99("ShardedBloomFilter", filter.clone());
        b.iter(|| {
            black_box(filter(&0));
        });
    });

    group.bench_function(BenchmarkId::new("StripedBloomFilter", N), |b| {
        let f = Arc::new(StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap());
        f.insert_batch(data.iter());
        let filter = Arc::new(move |x: &u64| f.contains(x));
        run_p99("StripedBloomFilter", filter.clone());
        b.iter(|| {
            black_box(filter(&0));
        });
    });

    group.finish();
}

// ─── memory/actual_footprint ─────────────────────────────────────────────────

fn bench_actual_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/actual_footprint");
    let data = seeded_data(N);

    let report = |name: &str, bits: usize, extra: &str| {
        let bytes = bits / 8;
        let mb = bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "  {:<30}  {:>8} bits  {:>8.2} MB  {}",
            name, bits, mb, extra
        );
    };

    eprintln!("\n--- Memory footprint at N={} ---", N);

    group.bench_function(BenchmarkId::new("StandardBloomFilter", N), |b| {
        let f = StandardBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        report("StandardBloomFilter", f.bit_count(), "");
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("CountingBloomFilter", N), |b| {
        let mut f = CountingBloomFilter::<u64>::new(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        report(
            "CountingBloomFilter",
            f.bit_count(),
            "4-bit counters → 4× bit cost",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("ScalableBloomFilter", N), |b| {
        let mut f = ScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        let _ = f.insert_batch(data.as_slice());
        report("ScalableBloomFilter", f.bit_count(), "grows on demand");
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("PartitionedBloomFilter", N), |b| {
        let mut f = PartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        report("PartitionedBloomFilter", f.bit_count(), "");
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("RegisterBlockedBloomFilter", N), |b| {
        let mut f = RegisterBlockedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        report(
            "RegisterBlockedBloomFilter",
            f.bit_count(),
            "4-bit block counters",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("TreeBloomFilter", N), |b| {
        let cap = N / TREE_BRANCHING.iter().product::<usize>();
        let mut f = TreeBloomFilter::<u64>::new(TREE_BRANCHING.to_vec(), cap, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        report(
            "TreeBloomFilter",
            f.bit_count(),
            &format!("branching {:?}", TREE_BRANCHING),
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("ClassicBitsFilter", N), |b| {
        let mut f = ClassicBitsFilter::<u64>::with_fpr(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        report(
            "ClassicBitsFilter",
            f.bit_count(),
            "k bit-vectors of m bits each",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("ClassicHashFilter", N), |b| {
        let mut f = ClassicHashFilter::<u64>::with_fpr(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        report(
            "ClassicHashFilter",
            f.bit_count(),
            "stores actual elements (bit_count = m*d*size_of(T)*8)",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    #[cfg(feature = "concurrent")]
    group.bench_function(BenchmarkId::new("AtomicPartitionedBloomFilter", N), |b| {
        let mut f = AtomicPartitionedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        report(
            "AtomicPartitionedBloomFilter",
            f.bit_count(),
            "same as Partitioned + atomic storage",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("ShardedBloomFilter", N), |b| {
        let f = ShardedBloomFilter::<u64>::new(N, TARGET_FPR);
        f.insert_batch(data.as_slice());
        report(
            "ShardedBloomFilter",
            f.bit_count(),
            "sum of shard bit counts",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.bench_function(BenchmarkId::new("StripedBloomFilter", N), |b| {
        let f = StripedBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        f.insert_batch(data.as_slice());
        report(
            "StripedBloomFilter",
            f.bit_count(),
            "same as Standard + RwLock stripes",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    #[cfg(feature = "concurrent")]
    group.bench_function(BenchmarkId::new("AtomicScalableBloomFilter", N), |b| {
        let f = AtomicScalableBloomFilter::<u64>::new(N, TARGET_FPR).unwrap();
        let _ = f.insert_batch(data.as_slice());
        report(
            "AtomicScalableBloomFilter",
            f.bit_count(),
            "grows on demand + atomic ops",
        );
        b.iter(|| black_box(f.bit_count()));
    });

    group.finish();
}

// ─── Register and run ────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_insert_throughput,
    bench_contains_throughput,
    bench_fpr_accuracy,
    bench_bit_count_read_latency,
    bench_input_insert,
    bench_input_contains,
    bench_concurrent_threads,
    bench_read_write_mix,
    bench_batch_insert,
    bench_fpr_under_overfill,
    bench_string_dictionary,
    bench_concurrent_contains_p99,
    bench_actual_footprint,
);
criterion_main!(benches);
