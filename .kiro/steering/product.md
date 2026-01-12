# BloomCraft Product Overview

BloomCraft is a production-grade Rust library implementing multiple variants of Bloom filters - probabilistic data structures for approximate set membership queries.

## What it does
- Provides memory-efficient, cache-optimized, and thread-safe Bloom filter implementations
- Supports multiple filter variants: Standard, Counting, Scalable, Partitioned, and Hierarchical
- Includes historical implementations (Burton Bloom's 1970 methods) for educational purposes
- Offers comprehensive error handling, serialization support, and observability hooks

## Key use cases
- Database query optimization (avoiding disk lookups)
- Distributed caching and LSM-tree skip lists
- Network packet deduplication and filtering
- Web crawling (URL seen-set tracking)
- Large-scale data processing and streaming systems

## Core value proposition
Production-ready Bloom filter library that is indistinguishable from infrastructure code shipped by Principal Engineers at FAANG-scale systems, with focus on correctness, performance, and flexibility.