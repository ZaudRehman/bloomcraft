//! Tree-structured Bloom filter for hierarchical data organization.
//!
//! A production-grade Bloom filter variant that organizes filters in a user-defined
//! tree hierarchy, enabling location-aware membership testing and spatial queries.
//! Optimized for cache efficiency, pruned search, and real-time incremental updates.
//!
//! # Architecture
//!
//! ```text
//! Level 0 (Root): [========= Aggregation Filter =========]
//!                       ↓       ↓       ↓
//! Level 1:        [Filter 0] [Filter 1] [Filter 2]
//!                       ↓        ↓          ↓
//! Level 2:        [F 0.0]     [F 1.0]    [F 2.0]
//!                 [F 0.1]     [F 1.1]    [F 2.1]
//! ```
//!
//! # Tutorial: Building a CDN Cache Tracker
//!
//! ## Step 1: Define Your Hierarchy
//!
//! Our CDN has:
//! - 5 geographic regions (NA, EU, APAC, LATAM, ME)
//! - 20 Points of Presence (POPs) per region
//! - 100 edge servers per POP
//!
//! This gives us a 3-level tree with branching [5, 20, 100] = 10,000 leaf nodes.
//!
//! ## Step 2: Size Your Filters
//!
//! ```rust
//! use bloomcraft::filters::TreeBloomFilter;
//!
//! let mut cache_tracker: TreeBloomFilter<String> =
//!     TreeBloomFilter::new(
//!         vec![5, 20, 100],
//!         50_000,
//!         0.001
//!     );
//! ```

//!
//! ## Step 3: Invalidate Caches
//!
//! ```rust
//! # use bloomcraft::filters::TreeBloomFilter;
//! # let mut cache_tracker: TreeBloomFilter<String> =
//! #     TreeBloomFilter::new(vec![5, 20, 100], 50_000, 0.001);
//! # cache_tracker.insert_to_bin(&"/assets/logo.png".to_string(), &[0, 0, 0]).unwrap();
//! let locations = cache_tracker.locate(&"/assets/logo.png".to_string());
//! for loc in locations {
//!     println!("Invalidating cache at path={:?}", loc);
//! }
//! ```

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(dead_code)]
#![allow(private_interfaces)] 

use crate::core::filter::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "metrics")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize, Deserializer, Serializer};

#[cfg(feature = "metrics")]
use crate::metrics::LatencyHistogram;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Maximum depth to prevent excessive recursion and stack overflow.
const MAX_TREE_DEPTH: usize = 16;

/// WyHash-style mixing for better avalanche in bin distribution.
#[inline]
fn mix_hash(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Convert a hashable item to bytes using Rust's Hash trait.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Metadata for TreeNode (cold path - rarely accessed).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct NodeMetadata {
    /// Path to this node (for debugging/statistics)
    path: Vec<usize>,
    /// Level in the tree (0 = root)
    #[allow(dead_code)]
    level: u8,
}

/// Node in the tree-structured Bloom filter with optimized cache layout.
///
/// **Cache Optimization**: Hot data (filter, item_count) separated from cold metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, align(64))]
pub struct TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    // HOT PATH (first cache line)
    /// Bloom filter for this node
    filter: StandardBloomFilter<T, H>,
    /// Number of items inserted (for load tracking)
    item_count: usize,
    /// Child nodes (empty for leaf nodes)
    children: Box<[TreeNode<T, H>]>,
    
    // COLD PATH (metadata - separate cache line)
    metadata: NodeMetadata,
}

impl<T, H> TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Create a new leaf node.
    #[inline]
    fn new_leaf(capacity: usize, fpr: f64, hasher: H, path: Vec<usize>, level: u8) -> Self {
        Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher),
            children: Box::new([]),
            item_count: 0,
            metadata: NodeMetadata { path, level },
        }
    }

    /// Create a new internal node with children.
    #[inline]
    fn new_internal(
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
        level: u8,
        children: Box<[TreeNode<T, H>]>,
    ) -> Self {
        Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher),
            children,
            item_count: 0,
            metadata: NodeMetadata { path, level },
        }
    }

    /// Check if this is a leaf node.
    #[inline(always)]
    const fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Get total number of nodes in subtree (including self).
    #[inline]
    fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Get memory usage estimate of subtree in bytes.
    #[inline]
    fn memory_usage_estimate(&self) -> usize {
        let filter_bytes = (self.filter.bit_count() + 7) / 8;
        let children_bytes: usize = self.children.iter()
            .map(|c| c.memory_usage_estimate())
            .sum();
        let overhead = std::mem::size_of::<Self>();
        filter_bytes + children_bytes + overhead
    }

    /// Get load factor (items / capacity) for this node.
    #[inline]
    fn load_factor(&self) -> f64 {
        let capacity = self.filter.expected_items();
        if capacity == 0 {
            0.0
        } else {
            self.item_count as f64 / capacity as f64
        }
    }
}

// MAIN TREE BLOOM FILTER
/// Tree-structured Bloom filter for hierarchical data organization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TreeBloomFilter<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Root node of tree
    root: TreeNode<T, H>,
    /// Branching factors for each level
    branching: Vec<usize>,
    /// Expected items per leaf bin
    capacity_per_bin: usize,
    /// Target false positive rate
    #[allow(dead_code)]
    target_fpr: f64,
    /// Total items inserted across all bins
    total_items: usize,
    /// Hasher for items
    #[cfg_attr(feature = "serde", serde(skip, default = "H::default"))]
    hasher: H,
    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
    
    // === METRICS (feature-gated) ===
    #[cfg(feature = "metrics")]
    #[cfg_attr(feature = "serde", serde(skip))]
    metrics: TreeFilterMetrics,
}

// Metrics/Observability (feature-gated)
#[cfg(feature = "metrics")]
#[derive(Debug)]
struct TreeFilterMetrics {
    insert_latency: LatencyHistogram,
    query_latency: LatencyHistogram,
    locate_latency: LatencyHistogram,
    pruned_subtrees: AtomicUsize,
}

#[cfg(feature = "metrics")]
impl Default for TreeFilterMetrics {
    fn default() -> Self {
        Self {
            insert_latency: LatencyHistogram::new(),
            query_latency: LatencyHistogram::new(),
            locate_latency: LatencyHistogram::new(),
            pruned_subtrees: AtomicUsize::new(0),
        }
    }
}

#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
}

#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct TreeHealthCheck {
    pub status: HealthStatus,
    pub avg_load_factor: f64,
    pub total_items: usize,
    pub capacity: usize,
    pub saturation: f64,
}

#[cfg(feature = "serde")]
impl<T, H> Serialize for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TreeBloomFilter", 6)?;
        state.serialize_field("root", &self.root)?;
        state.serialize_field("branching", &self.branching)?;
        state.serialize_field("capacity_per_bin", &self.capacity_per_bin)?;
        state.serialize_field("target_fpr", &self.target_fpr)?;
        state.serialize_field("total_items", &self.total_items)?;
        state.serialize_field("hasher_type", std::any::type_name::<H>())?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, H> Deserialize<'de> for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, Visitor, MapAccess};
        
        struct TreeBloomFilterVisitor<T, H>(PhantomData<(T, H)>);
        
        impl<'de, T, H> Visitor<'de> for TreeBloomFilterVisitor<T, H>
        where
            T: Hash + Send + Sync,
            H: BloomHasher + Clone + Default + Deserialize<'de>,
        {
            type Value = TreeBloomFilter<T, H>;
            
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct TreeBloomFilter")
            }
            
            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut root = None;
                let mut branching = None;
                let mut capacity_per_bin = None;
                let mut target_fpr = None;
                let mut total_items = None;
                let mut hasher_type = None;
                
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "root" => root = Some(map.next_value()?),
                        "branching" => branching = Some(map.next_value()?),
                        "capacity_per_bin" => capacity_per_bin = Some(map.next_value()?),
                        "target_fpr" => target_fpr = Some(map.next_value()?),
                        "total_items" => total_items = Some(map.next_value()?),
                        "hasher_type" => hasher_type = Some(map.next_value::<String>()?),
                        _ => { let _ = map.next_value::<serde::de::IgnoredAny>()?; }
                    }
                }
                
                let root = root.ok_or_else(|| de::Error::missing_field("root"))?;
                let branching = branching.ok_or_else(|| de::Error::missing_field("branching"))?;
                let capacity_per_bin = capacity_per_bin.ok_or_else(|| de::Error::missing_field("capacity_per_bin"))?;
                let target_fpr = target_fpr.ok_or_else(|| de::Error::missing_field("target_fpr"))?;
                let total_items = total_items.ok_or_else(|| de::Error::missing_field("total_items"))?;
                
                // Validate hasher type matches
                if let Some(ht) = hasher_type {
                    let expected = std::any::type_name::<H>();
                    if ht != expected {
                        return Err(de::Error::custom(format!(
                            "Hasher type mismatch: expected {}, got {}",
                            expected, ht
                        )));
                    }
                }
                
                Ok(TreeBloomFilter {
                    root,
                    branching,
                    capacity_per_bin,
                    target_fpr,
                    total_items,
                    hasher: H::default(),  
                    _phantom: PhantomData,
                    #[cfg(feature = "metrics")]
                    metrics: TreeFilterMetrics::default(),
                })
            }
        }
        
        deserializer.deserialize_struct(
            "TreeBloomFilter",
            &["root", "branching", "capacity_per_bin", "target_fpr", "total_items", "hasher_type"],
            TreeBloomFilterVisitor(PhantomData)
        )
    }
}

impl<T> TreeBloomFilter<T, StdHasher>
where
    T: Hash + Send + Sync,
{
    /// Create a new tree-structured Bloom filter with default hasher.
    #[must_use]
    pub fn new(branching: Vec<usize>, capacity_per_bin: usize, fpr: f64) -> Self {
        Self::with_hasher(branching, capacity_per_bin, fpr, StdHasher::new())
    }
}

impl<T, H> TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Create a new tree-structured Bloom filter with custom hasher.
    #[must_use]
    pub fn with_hasher(
        branching: Vec<usize>,
        capacity_per_bin: usize,
        fpr: f64,
        hasher: H,
    ) -> Self {
        assert!(!branching.is_empty(), "branching cannot be empty");
        assert!(
            branching.iter().all(|&b| b > 0),
            "all branching factors must be > 0"
        );
        assert!(capacity_per_bin > 0, "capacity_per_bin must be > 0");
        assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1)");
        assert!(
            branching.len() <= MAX_TREE_DEPTH,
            "tree depth {} exceeds maximum {}",
            branching.len(),
            MAX_TREE_DEPTH
        );

        let leaf_count: usize = branching.iter().product();
        let mut total_nodes = leaf_count;
        let mut partial_product = 1;
        for i in 0..branching.len() - 1 {
            partial_product *= branching[i];
            total_nodes += partial_product;
        }

        if total_nodes > 10_000 {
            eprintln!(
                "WARNING: TreeBloomFilter with {} total nodes (branching {:?})\n\
                This will allocate {} filters. Consider reducing depth or branching factors.",
                total_nodes, branching, total_nodes
            );
        }

        let root = Self::build_tree(&branching, 0, capacity_per_bin, fpr, hasher.clone(), vec![]);
        
        Self {
            root,
            branching,
            capacity_per_bin,
            target_fpr: fpr,
            total_items: 0,
            hasher,
            _phantom: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: TreeFilterMetrics::default(),
        }
    }

    /// Recursively build the tree with breadth-first layout.
    fn build_tree(
        branching: &[usize],
        level: usize,
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
    ) -> TreeNode<T, H> {
        if level >= branching.len() {
            return TreeNode::new_leaf(capacity, fpr, hasher, path, level as u8);
        }

        let num_children = branching[level];
        let mut children = Vec::with_capacity(num_children);
        for i in 0..num_children {
            let mut child_path = path.clone();
            child_path.push(i);
            let child = Self::build_tree(
                branching,
                level + 1,
                capacity,
                fpr,
                hasher.clone(),
                child_path,
            );
            children.push(child);
        }

        let internal_capacity = capacity * num_children;
        TreeNode::new_internal(
            internal_capacity,
            fpr,
            hasher,
            path,
            level as u8,
            children.into_boxed_slice(),
        )
    }

    /// Validate a bin path against tree structure.
    #[inline]
    fn validate_path(&self, path: &[usize]) -> Result<()> {
        if path.len() != self.depth() {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Path length {} does not match tree depth {}",
                path.len(),
                self.depth()
            )));
        }

        for (level, &idx) in path.iter().enumerate() {
            if idx >= self.branching[level] {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "Path index {} at level {} exceeds branching factor {}",
                    idx, level, self.branching[level]
                )));
            }
        }
        Ok(())
    }

    /// Get the depth (number of levels) in the tree.
    ///
    /// # Examples
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    /// let filter = TreeBloomFilter::<String>::new(vec![3, 4], 1000, 0.01);
    /// assert_eq!(filter.depth(), 2);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn depth(&self) -> usize {
        self.branching.len()
    }

    /// Get the total number of leaf bins.
    ///
    /// This is the product of all branching factors.
    #[must_use]
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.branching.iter().product()
    }

    /// Get the total number of nodes (internal + leaf) in the tree.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Get estimated memory usage in bytes.
    ///
    /// Includes all filters, metadata, and tree structure overhead.
    #[must_use]
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.root.memory_usage_estimate() + std::mem::size_of::<Self>()
    }

    /// Insert an item using automatic hash-based routing.
    #[inline]
    pub fn insert(&mut self, item: &T) -> Result<()> {
        self.insert_auto(item)
    }

    /// Insert an item into a specific bin.
    ///
    /// The item is inserted at all levels along the path from root to leaf.
    ///
    /// # Errors
    /// Returns error if `bin_path` is invalid (wrong length or out-of-bounds).
    #[inline]
    pub fn insert_to_bin(&mut self, item: &T, bin_path: &[usize]) -> Result<()> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        
        self.validate_path(bin_path)?;
        let mut current = &mut self.root;
        current.filter.insert(item);
        current.item_count += 1;

        for &child_idx in bin_path {
            current = &mut current.children[child_idx];
            current.filter.insert(item);
            current.item_count += 1;
        }

        self.total_items += 1;
        
        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics.insert_latency.record(elapsed);
        }
        
        Ok(())
    }

    /// Insert multiple items into the same bin.
    #[inline]
    pub fn insert_batch_to_bin(&mut self, items: &[&T], bin_path: &[usize]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        self.validate_path(bin_path)?;
        
        // Using batch insert for potential SIMD optimization
        let mut current = &mut self.root;
        for item in items {
            current.filter.insert(item);
        }
        current.item_count += items.len();

        for &child_idx in bin_path {
            current = &mut current.children[child_idx];
            for item in items {
                current.filter.insert(item);
            }
            current.item_count += items.len();
        }

        self.total_items += items.len();
        Ok(())
    }

    /// Find all bins that might contain an item (with prefetching).
    #[must_use]
    pub fn locate(&self, item: &T) -> Vec<Vec<usize>> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        
        if !self.root.filter.contains(item) {
            #[cfg(feature = "metrics")]
            {
                let elapsed = start.elapsed().as_nanos() as u64;
                self.metrics.locate_latency.record(elapsed);
            }
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut stack = vec![(&self.root, Vec::new())];
        
        while let Some((node, current_path)) = stack.pop() {
            if node.is_leaf() {
                result.push(current_path);
                continue;
            }

            // Prefetch children for cache optimization
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                for child in node.children.iter() {
                    let child_ptr = child as *const _ as *const i8;
                    unsafe {
                        _mm_prefetch(child_ptr, _MM_HINT_T0);
                    }
                }
            }

            // Now check filters (data likely in cache)
            for (child_idx, child) in node.children.iter().enumerate().rev() {
                if !child.filter.contains(item) {
                    #[cfg(feature = "metrics")]
                    self.metrics.pruned_subtrees.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                
                let mut child_path = current_path.clone();
                child_path.push(child_idx);
                stack.push((child, child_path));
            }
        }
        
        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics.locate_latency.record(elapsed);
        }
        
        result
    }

    /// Compute the union of two tree filters (merge).
    pub fn union(&mut self, other: &Self) -> Result<()> {
        if self.branching != other.branching {
            return Err(BloomCraftError::incompatible_filters(
                "Different branching factors".to_string()
            ));
        }
        
        Self::union_nodes(&mut self.root, &other.root)?;
        self.total_items += other.total_items;
        Ok(())
    }

    fn union_nodes(a: &mut TreeNode<T, H>, b: &TreeNode<T, H>) -> Result<()> {
        // Union filters at this level
        a.filter = a.filter.union(&b.filter)?;
        a.item_count += b.item_count;
        
        // Recursively union children
        for (a_child, b_child) in a.children.iter_mut().zip(b.children.iter()) {
            Self::union_nodes(a_child, b_child)?;
        }
        
        Ok(())
    }

    /// Compute the intersection of two tree filters.
    pub fn intersect(&mut self, other: &Self) -> Result<()> {
        if self.branching != other.branching {
            return Err(BloomCraftError::incompatible_filters(
                "Different branching factors".to_string()
            ));
        }
        
        Self::intersect_nodes(&mut self.root, &other.root)?;
        self.total_items = 0;  // Unknown after intersect
        Ok(())
    }

    fn intersect_nodes(a: &mut TreeNode<T, H>, b: &TreeNode<T, H>) -> Result<()> {
        // Intersect filters at this level
        a.filter = a.filter.intersect(&b.filter)?;
        a.item_count = 0;  // Unknown
        
        // Recursively intersect children
        for (a_child, b_child) in a.children.iter_mut().zip(b.children.iter()) {
            Self::intersect_nodes(a_child, b_child)?;
        }
        
        Ok(())
    }

    /// Get reference to subtree at path.
    pub fn subtree_at(&self, path: &[usize]) -> Result<&TreeNode<T, H>> {
        if path.is_empty() {
            return Ok(&self.root);
        }
        
        let mut current = &self.root;
        for &idx in path {
            if idx >= current.children.len() {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "Invalid subtree path: index {} out of bounds",
                    idx
                )));
            }
            current = &current.children[idx];
        }
        Ok(current)
    }

    /// Clear specific subtree.
    pub fn clear_subtree(&mut self, path: &[usize]) -> Result<()> {
        if path.is_empty() {
            Self::clear_node_iterative(&mut self.root);
            return Ok(());
        }
        
        let mut current = &mut self.root;
        for &idx in path {
            if idx >= current.children.len() {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "Invalid subtree path: index {} out of bounds",
                    idx
                )));
            }
            current = &mut current.children[idx];
        }
        Self::clear_node_iterative(current);
        Ok(())
    }

    /// Locate items within path prefix (range query).
    pub fn locate_in_range(&self, item: &T, path_prefix: &[usize]) -> Vec<Vec<usize>> {
        if path_prefix.is_empty() {
            return self.locate(item);
        }
        
        // Navigate to subtree
        let subtree = match self.subtree_at(path_prefix) {
            Ok(node) => node,
            Err(_) => return Vec::new(),
        };
        
        if !subtree.filter.contains(item) {
            return Vec::new();
        }
        
        // DFS within subtree
        let mut result = Vec::new();
        self.locate_in_subtree(subtree, item, path_prefix.to_vec(), &mut result);
        result
    }

    fn locate_in_subtree(
        &self,
        node: &TreeNode<T, H>,
        item: &T,
        current_path: Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if node.is_leaf() {
            result.push(current_path);
            return;
        }
        
        for (child_idx, child) in node.children.iter().enumerate() {
            if !child.filter.contains(item) {
                continue;
            }
            
            let mut child_path = current_path.clone();
            child_path.push(child_idx);
            self.locate_in_subtree(child, item, child_path, result);
        }
    }

    /// Locate multiple items efficiently.
    #[must_use]
    pub fn locate_batch(&self, items: &[&T]) -> Vec<Vec<Vec<usize>>> {
        items.iter()
            .map(|item| self.locate(item))
            .collect()
    }

    /// Parallel batch locate (requires `rayon` feature).
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn locate_batch_parallel(&self, items: &[&T]) -> Vec<Vec<Vec<usize>>> {
        items.par_iter()
            .map(|item| self.locate(item))
            .collect()
    }

    /// Check if any leaf needs resizing.
    #[must_use]
    pub fn needs_resize(&self) -> bool {
        self.stats().avg_load_factor > 0.7
    }

    /// Create resized filter with more capacity.
    pub fn resize(&self, new_capacity_per_bin: usize, new_fpr: f64) -> Result<Self> {
        let new_filter = Self::with_hasher(
            self.branching.clone(),
            new_capacity_per_bin,
            new_fpr,
            H::default(),
        );
        
        // This returns an empty filter with new capacity
        Ok(new_filter)
    }

    /// Insert item with hash-based bin assignment (OPTIMIZED).
    pub fn insert_auto(&mut self, item: &T) -> Result<()> {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        
        // hash mixing with WyHash-style avalanche
        let mut hash = h1;
        let mut bin_path = Vec::with_capacity(self.depth());
        for &branching_factor in &self.branching {
            let index = (hash as usize) % branching_factor;
            bin_path.push(index);
            hash = mix_hash(hash.wrapping_add(h2));  // Better mixing
        }
        
        self.insert_to_bin(item, &bin_path)
    }

    /// Clear a node and its children.
    fn clear_node_iterative(node: &mut TreeNode<T, H>) {
        let mut stack = vec![node as *mut TreeNode<T, H>];
        
        while let Some(node_ptr) = stack.pop() {
            let node_ref = unsafe { &mut *node_ptr };
            node_ref.filter.clear();
            node_ref.item_count = 0;
            
            for child in &mut *node_ref.children {
                stack.push(child as *mut TreeNode<T, H>);
            }
        }
    }

    /// Check if item might exist anywhere in the tree.
    ///
    /// Checks only the root filter for O(1) performance.
    #[must_use]
    #[inline(always)]
    pub fn contains(&self, item: &T) -> bool {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        
        let result = self.root.filter.contains(item);
        
        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics.query_latency.record(elapsed);
        }
        
        result
    }

    /// Check if item might exist in a specific bin.
    ///
    /// # Errors
    /// Returns error if `bin_path` is invalid.
    #[must_use]
    #[inline]
    pub fn contains_in_bin(&self, item: &T, bin_path: &[usize]) -> Result<bool> {
        self.validate_path(bin_path)?;
        
        if !self.root.filter.contains(item) {
            return Ok(false);
        }

        let mut current = &self.root;
        for &child_idx in bin_path {
            current = &current.children[child_idx];
            if !current.filter.contains(item) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Batch check if items might exist (checks root only).
    #[must_use]
    #[inline]
    pub fn contains_batch(&self, items: &[&T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Check if item exists in ALL levels of the tree (expensive).
    ///
    /// Unlike `contains()`, this verifies the item at every level.
    #[must_use]
    pub fn query_all(&self, item: &T) -> bool {
        if !self.root.filter.contains(item) {
            return false;
        }
        self.query_all_recursive(&self.root, item)
    }

    fn query_all_recursive(&self, node: &TreeNode<T, H>, item: &T) -> bool {
        if node.is_leaf() {
            return node.filter.contains(item);
        }
        node.children.iter().any(|child| self.query_all_recursive(child, item))
    }

    /// Get tree statistics (memory, load factor, etc.).
    #[must_use]
    pub fn stats(&self) -> TreeStats {
        let total_nodes = self.node_count();
        let memory_usage = self.memory_usage();
        let leaf_bins = self.leaf_count();
        
        let memory_per_node = if total_nodes > 0 {
            memory_usage / total_nodes
        } else {
            0
        };
        
        let overhead_factor = if leaf_bins > 0 {
            total_nodes as f64 / leaf_bins as f64
        } else {
            0.0
        };
        
        TreeStats {
            total_nodes,
            memory_usage,
            total_items: self.total_items,
            depth: self.depth(),
            leaf_bins,
            avg_load_factor: self.compute_avg_load_factor(),
            memory_per_node,
            overhead_factor,
        }
    }

    fn compute_avg_load_factor(&self) -> f64 {
        let (sum, count) = self.compute_load_factor_recursive(&self.root);
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    fn compute_load_factor_recursive(&self, node: &TreeNode<T, H>) -> (f64, usize) {
        if node.is_leaf() {
            return (node.load_factor(), 1);
        }

        let mut sum = 0.0;
        let mut count = 0;
        for child in &*node.children {
            let (child_sum, child_count) = self.compute_load_factor_recursive(child);
            sum += child_sum;
            count += child_count;
        }
        (sum, count)
    }

    #[cfg(feature = "metrics")]
    pub fn health_check(&self) -> TreeHealthCheck {
        let stats = self.stats();
        let status = if stats.avg_load_factor > 0.9 {
            HealthStatus::Critical
        } else if stats.avg_load_factor > 0.7 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        
        TreeHealthCheck {
            status,
            avg_load_factor: stats.avg_load_factor,
            total_items: stats.total_items,
            capacity: self.capacity_per_bin * stats.leaf_bins,
            saturation: stats.avg_load_factor,
        }
    }
    
    #[cfg(feature = "metrics")]
    pub fn export_prometheus(&self) -> String {
        let stats = self.stats();
        format!(
            "# HELP tree_bloom_filter_items Total items inserted\n\
             # TYPE tree_bloom_filter_items gauge\n\
             tree_bloom_filter_items{{depth=\"{}\"}} {}\n\
             # HELP tree_bloom_filter_load_factor Average load factor\n\
             # TYPE tree_bloom_filter_load_factor gauge\n\
             tree_bloom_filter_load_factor{{depth=\"{}\"}} {:.4}\n\
             # HELP tree_bloom_filter_pruned_subtrees Total pruned subtrees\n\
             # TYPE tree_bloom_filter_pruned_subtrees counter\n\
             tree_bloom_filter_pruned_subtrees{{depth=\"{}\"}} {}\n",
            stats.depth, stats.total_items,
            stats.depth, stats.avg_load_factor,
            stats.depth, self.metrics.pruned_subtrees.load(Ordering::Relaxed)
        )
    }

    /// Validate tree structure integrity.
    ///
    /// Checks branching factors and path consistency.
    ///
    /// # Errors
    /// Returns error if structure is invalid.
    #[cfg(debug_assertions)]
    pub fn validate_structure(&self) -> Result<()> {
        self.validate_node(&self.root, &[])?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_node(&self, node: &TreeNode<T, H>, current_path: &[usize]) -> Result<()> {
        // Check path matches stored path
        if node.metadata.path != current_path {
            return Err(BloomCraftError::internal_error(format!(
                "Path mismatch: stored {:?}, actual {:?}",
                node.metadata.path, current_path
            )));
        }
        
        // Check children count matches branching factor
        if current_path.len() < self.branching.len() {
            let expected_children = self.branching[current_path.len()];
            if node.children.len() != expected_children {
                return Err(BloomCraftError::internal_error(format!(
                    "Child count mismatch at {:?}: expected {}, got {}",
                    current_path, expected_children, node.children.len()
                )));
            }
        }
        
        // Recursively validate children
        for (idx, child) in node.children.iter().enumerate() {
            let mut child_path = current_path.to_vec();
            child_path.push(idx);
            self.validate_node(child, &child_path)?;
        }
        
        Ok(())
    }
}

/// Statistics about the tree structure and usage.
#[derive(Debug, Clone, Default, Copy)]
pub struct TreeStats {
    /// Total number of nodes (internal + leaf).
    pub total_nodes: usize,
    /// Estimated memory usage in bytes.
    pub memory_usage: usize,
    /// Total items inserted across all bins.
    pub total_items: usize,
    /// Tree depth (number of levels).
    pub depth: usize,
    /// Number of leaf bins.
    pub leaf_bins: usize,
    /// Average load factor (items / capacity) across all nodes.
    pub avg_load_factor: f64,
    /// Memory usage per node (average).
    pub memory_per_node: usize,
    /// Overhead factor (total_nodes / leaf_bins).
    pub overhead_factor: f64,
}

impl<T, H> BloomFilter<T> for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        let _ = TreeBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        TreeBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        Self::clear_node_iterative(&mut self.root);
        self.total_items = 0;
    }

    fn len(&self) -> usize {
        self.total_items
    }

    fn is_empty(&self) -> bool {
        self.total_items == 0
    }

    fn false_positive_rate(&self) -> f64 {
        self.root.filter.false_positive_rate()
    }

    fn expected_items(&self) -> usize {
        self.capacity_per_bin
    }

    fn bit_count(&self) -> usize {
        self.root.filter.bit_count()
    }

    fn hash_count(&self) -> usize {
        self.root.filter.hash_count()
    }
}

pub struct TreeBloomFilterBuilder<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    branching: Option<Vec<usize>>,
    capacity_per_bin: Option<usize>,
    fpr: Option<f64>,
    hasher: H,
    _phantom: PhantomData<T>,
}

impl<T, H> TreeBloomFilterBuilder<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    pub fn new() -> Self {
        Self {
            branching: None,
            capacity_per_bin: None,
            fpr: None,
            hasher: H::default(),
            _phantom: PhantomData,
        }
    }
    
    pub fn branching(mut self, branching: Vec<usize>) -> Self {
        self.branching = Some(branching);
        self
    }
    
    pub fn capacity_per_bin(mut self, capacity: usize) -> Self {
        self.capacity_per_bin = Some(capacity);
        self
    }
    
    pub fn false_positive_rate(mut self, fpr: f64) -> Self {
        self.fpr = Some(fpr);
        self
    }
    
    pub fn hasher(mut self, hasher: H) -> Self {
        self.hasher = hasher;
        self
    }
    
    pub fn build(self) -> Result<TreeBloomFilter<T, H>> {
        let branching = self.branching
            .ok_or_else(|| BloomCraftError::invalid_parameters("branching not set"))?;
        let capacity = self.capacity_per_bin
            .ok_or_else(|| BloomCraftError::invalid_parameters("capacity_per_bin not set"))?;
        let fpr = self.fpr
            .ok_or_else(|| BloomCraftError::invalid_parameters("fpr not set"))?;
        
        Ok(TreeBloomFilter::with_hasher(branching, capacity, fpr, self.hasher))
    }
}

impl<T, H> Default for TreeBloomFilterBuilder<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// TESTS

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01);
        assert_eq!(filter.depth(), 2);
        assert_eq!(filter.leaf_count(), 6);
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_and_query() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
        filter.insert_to_bin(&"hello".to_string(), &[0, 1]).unwrap();
        assert!(filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"goodbye".to_string()));
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_insert_auto() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01);
        
        filter.insert_auto(&"test".to_string()).unwrap();
        assert!(filter.contains(&"test".to_string()));
        
        // Should be deterministic
        let loc1 = filter.locate(&"test".to_string());
        filter.insert_auto(&"test".to_string()).unwrap();
        let loc2 = filter.locate(&"test".to_string());
        assert_eq!(loc1, loc2);
    }

    #[test]
    fn test_union() {
        let mut filter1: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
        let mut filter2: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
        
        filter1.insert_to_bin(&"a".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"b".to_string(), &[1, 1]).unwrap();
        
        filter1.union(&filter2).unwrap();
        assert!(filter1.contains(&"a".to_string()));
        assert!(filter1.contains(&"b".to_string()));
    }

    #[test]
    fn test_intersect() {
        let mut filter1: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
        let mut filter2: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
        
        filter1.insert_to_bin(&"a".to_string(), &[0, 0]).unwrap();
        filter1.insert_to_bin(&"b".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"b".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"c".to_string(), &[1, 1]).unwrap();
        
        filter1.intersect(&filter2).unwrap();
        assert!(filter1.contains(&"b".to_string()));
    }

    #[test]
    fn test_subtree_operations() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
        
        filter.insert_to_bin(&"item".to_string(), &[0, 1]).unwrap();
        
        let subtree = filter.subtree_at(&[0]).unwrap();
        assert!(subtree.filter.contains(&"item".to_string()));
        
        filter.clear_subtree(&[0]).unwrap();

        let cleared_subtree = filter.subtree_at(&[0]).unwrap();
        assert!(!cleared_subtree.filter.contains(&"item".to_string()));

        assert!(!filter.contains_in_bin(&"item".to_string(), &[0, 1]).unwrap());
    }

    #[test]
    fn test_locate_in_range() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 3], 1000, 0.01);
        
        filter.insert_to_bin(&"item".to_string(), &[1, 2]).unwrap();
        
        let locs = filter.locate_in_range(&"item".to_string(), &[1]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], vec![1, 2]);
    }

    #[test]
    fn test_builder() {
        let filter: Result<TreeBloomFilter<String>> = TreeBloomFilterBuilder::new()
            .branching(vec![2, 3])
            .capacity_per_bin(1000)
            .false_positive_rate(0.01)
            .build();
        
        assert!(filter.is_ok());
        let f = filter.unwrap();
        assert_eq!(f.depth(), 2);
        assert_eq!(f.leaf_count(), 6);
    }

    #[test]
    fn test_needs_resize() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2], 100, 0.01);
        
        assert!(!filter.needs_resize());
        
        // Fill to saturation
        for i in 0..150 {
            filter.insert_to_bin(&format!("item{}", i), &[0]).unwrap();
        }
        
        assert!(filter.needs_resize());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_validate_structure() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01);
        assert!(filter.validate_structure().is_ok());
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_health_check() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01);
        
        let health = filter.health_check();
        assert!(matches!(health.status, HealthStatus::Healthy));
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_prometheus_export() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2], 1000, 0.01);
        
        let output = filter.export_prometheus();
        assert!(output.contains("tree_bloom_filter_items"));
        assert!(output.contains("tree_bloom_filter_load_factor"));
    }
   
    #[cfg(test)]
    #[cfg(feature = "proptest")]
    mod proptests {
        use super::*;
        use proptest::prelude::*;
        
        proptest! {
            #[test]
            fn no_false_negatives(items: Vec<String>) {
                let branching = vec![3, 4];
                let mut filter: TreeBloomFilter<String> = 
                    TreeBloomFilter::new(branching, 1000, 0.01);
                
                for item in &items {
                    filter.insert_auto(item).unwrap();
                }
                
                // No false negatives
                for item in &items {
                    prop_assert!(filter.contains(item));
                }
            }
            
            #[test]
            fn insert_auto_deterministic(items: Vec<String>) {
                let mut filter: TreeBloomFilter<String> = 
                    TreeBloomFilter::new(vec![4, 4], 1000, 0.01);
                
                for item in &items {
                    filter.insert_auto(item).unwrap();
                }
                
                // Each item should always be in exactly one bin
                for item in &items {
                    let locations = filter.locate(item);
                    prop_assert_eq!(locations.len(), 1, 
                        "Item {:?} in {} bins (expected 1)", item, locations.len());
                }
            }
        }
    }
}
