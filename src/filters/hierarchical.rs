//! Hierarchical Bloom filter for multi-level indexing.
//!
//! A hierarchical Bloom filter organizes data into multiple levels, where each level
//! acts as an index for the next. This is particularly useful for large-scale data
//! partitioning and approximate membership queries with locality information.
//!
//! # Architecture
//!
//! ```text
//! Level 0 (Root):    [========= Single Bloom Filter =========]
//!                              ↓         ↓         ↓
//! Level 1:          [Filter 0] [Filter 1] [Filter 2]
//!                        ↓          ↓          ↓
//! Level 2:          [F 0.0]    [F 1.0]    [F 2.0]
//!                   [F 0.1]    [F 1.1]    [F 2.1]
//! ```
//!
//! # Key Concepts
//!
//! ## Hierarchical Structure
//!
//! - Root level: Single filter representing all data
//! - Intermediate levels: Filters for data partitions
//! - Leaf level: Filters for individual data bins
//!
//! ## Query Process
//!
//! 1. Check root filter: Does item exist anywhere?
//! 2. Check level 1: Which partition might contain it?
//! 3. Check level 2: Which specific bin contains it?
//!
//! ## Use Cases
//!
//! ### Distributed Databases
//! - Level 0: Does data exist in cluster?
//! - Level 1: Which shard contains it?
//! - Level 2: Which node in shard?
//!
//! ### File Systems
//! - Level 0: Does file exist?
//! - Level 1: Which directory?
//! - Level 2: Which subdirectory?
//!
//! ### Genomics (HIBF-inspired)
//! - Level 0: Does k-mer exist in database?
//! - Level 1: Which experiment group?
//! - Level 2: Which specific sample?
//!
//! # Performance Characteristics
//!
//! | Operation | Time Complexity | Space Complexity |
//! |-----------|----------------|------------------|
//! | Insert | O(k × depth) | O(total_bins × m) |
//! | Query | O(k × depth) | - |
//! | Locate | O(k × depth × branching) | - |
//!
//! # Trade-offs
//!
//! | Aspect | Flat Bloom | Hierarchical Bloom |
//! |--------|------------|-------------------|
//! | Locality info | No | Yes (which partition) |
//! | Memory | Lower | Higher (multiple filters) |
//! | Query flexibility | Basic | Advanced (location) |
//! | Maintenance | Simple | Complex |
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::filters::HierarchicalBloomFilter;
//!
//! // Create 2-level hierarchy with 4 bins
//! let mut filter: HierarchicalBloomFilter<&str> = HierarchicalBloomFilter::new(
//!     vec![2, 2],  // 2 branches at each level
//!     1000,        // Items per bin
//!     0.01         // FPR
//! );
//!
//! // Insert item into specific bin (path must match depth)
//! filter.insert_to_bin(&"hello", &[0, 1]).unwrap();
//!
//! // Check if item exists anywhere
//! assert!(filter.contains(&"hello"));
//!
//! // Find which bins might contain the item
//! let bins = filter.locate(&"hello");
//! assert!(bins.contains(&vec![0, 1]));
//! ```
//!
//! ## Multi-Level Indexing
//!
//! ```
//! use bloomcraft::filters::HierarchicalBloomFilter;
//!
//! // 3-level hierarchy: 2 datacenters, 4 racks each, 8 servers each
//! let mut filter: HierarchicalBloomFilter<&str> = HierarchicalBloomFilter::new(
//!     vec![2, 4, 8],
//!     10_000,
//!     0.01
//! );
//!
//! // Insert data for datacenter 0, rack 2, server 5
//! filter.insert_to_bin(&"user:12345", &[0, 2, 5]).unwrap();
//!
//! // Query which bins contain the data
//! let locations = filter.locate(&"user:12345");
//! for location in locations {
//!     println!("Found in: DC {}, Rack {}, Server {}",
//!              location[0], location[1], location[2]);
//! }
//! ```
//!
//! ## Batch Operations
//!
//! ```
//! use bloomcraft::filters::HierarchicalBloomFilter;
//!
//! let mut filter: HierarchicalBloomFilter<&str> = HierarchicalBloomFilter::new(vec![2], 1000, 0.01);
//!
//! // Insert multiple items to same bin
//! let items = ["a", "b", "c"];
//! filter.insert_batch_to_bin(&items, &[1]).unwrap();
//!
//! // Query all items
//! for item in &items {
//!     assert!(filter.contains(item));
//! }
//! ```
//!
//! # References
//!
//! - Mehringer, S., et al. (2021). "Hierarchical Interleaved Bloom Filter: enabling
//!   ultrafast approximate sequence locations". Algorithms for Molecular Biology.
//! - Fan, B., et al. (2014). "Cuckoo Filter: Practically Better Than Bloom". CoNEXT.

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::filter::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Node in the hierarchical Bloom filter tree.
///
/// Each node contains:
/// - A Bloom filter for items at this level
/// - Optional children (for non-leaf nodes)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct HierarchicalNode<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Bloom filter for this node
    filter: StandardBloomFilter<T, H>,

    /// Child nodes (empty for leaf nodes)
    children: Vec<HierarchicalNode<T, H>>,

    /// Path to this node (for debugging/statistics)
    path: Vec<usize>,
}

impl<T, H> HierarchicalNode<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Create a new leaf node.
    fn new_leaf(capacity: usize, fpr: f64, hasher: H, path: Vec<usize>) -> Self {
        Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher),
            children: Vec::new(),
            path,
        }
    }

    /// Create a new internal node with children.
    fn new_internal(
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
        children: Vec<HierarchicalNode<T, H>>,
    ) -> Self {
        Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher),
            children,
            path,
        }
    }

    /// Check if this is a leaf node.
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Get total number of nodes in subtree (including self).
    fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Get memory usage of subtree.
    fn memory_usage(&self) -> usize {
        self.filter.memory_usage()
            + self
                .children
                .iter()
                .map(|c| c.memory_usage())
                .sum::<usize>()
    }
}

/// Hierarchical Bloom filter for multi-level indexing.
///
/// Organizes Bloom filters in a tree structure where each level represents
/// a different granularity of data partitioning.
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Structure
///
/// ```text
/// HierarchicalBloomFilter {
///     root: HierarchicalNode,          // Root of tree
///     branching: Vec<usize>,        // Branching factor per level
///     capacity_per_bin: usize,      // Items per leaf
///     fpr: f64,                     // False positive rate
/// }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HierarchicalBloomFilter<T, H = StdHasher>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Root node of hierarchy
    root: HierarchicalNode<T, H>,

    /// Branching factors for each level
    branching: Vec<usize>,

    /// Expected items per bin
    capacity_per_bin: usize,

    /// Target false positive rate
    target_fpr: f64,

    /// Total items inserted
    total_items: usize,

    /// Hasher for items (stored for future serialization support)
    #[cfg_attr(feature = "serde", serde(skip))]
    #[allow(dead_code)]
    hasher: H,

    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

impl<T> HierarchicalBloomFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new hierarchical Bloom filter with default hasher.
    ///
    /// # Arguments
    ///
    /// * `branching` - Branching factors for each level (e.g., [2, 4] = 2 branches, then 4)
    /// * `capacity_per_bin` - Expected items per leaf bin
    /// * `fpr` - Target false positive rate
    ///
    /// # Panics
    ///
    /// Panics if branching is empty or contains zeros, or if capacity/fpr are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// // 2 levels: 4 bins at level 1, 3 sub-bins each at level 2
    /// let filter: HierarchicalBloomFilter<&str> =
    ///     HierarchicalBloomFilter::new(vec![4, 3], 1000, 0.01);
    /// ```
    #[must_use]
    pub fn new(branching: Vec<usize>, capacity_per_bin: usize, fpr: f64) -> Self {
        Self::with_hasher(branching, capacity_per_bin, fpr, StdHasher::new())
    }
}

impl<T, H> HierarchicalBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Create a new hierarchical Bloom filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `branching` - Branching factors for each level
    /// * `capacity_per_bin` - Expected items per leaf bin
    /// * `fpr` - Target false positive rate
    /// * `hasher` - Custom hash function
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

        // Build the tree recursively
        let root = Self::build_tree(&branching, 0, capacity_per_bin, fpr, hasher.clone(), vec![]);

        Self {
            root,
            branching,
            capacity_per_bin,
            target_fpr: fpr,
            total_items: 0,
            hasher,
            _phantom: PhantomData,
        }
    }

    /// Recursively build the hierarchy tree.
    fn build_tree(
        branching: &[usize],
        level: usize,
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
    ) -> HierarchicalNode<T, H> {
        if level >= branching.len() {
            // Leaf node
            return HierarchicalNode::new_leaf(capacity, fpr, hasher, path);
        }

        // Internal node: create children
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

        // Calculate capacity for internal node (sum of children)
        let internal_capacity = capacity * num_children;

        HierarchicalNode::new_internal(internal_capacity, fpr, hasher, path, children)
    }

    /// Validate a bin path.
    ///
    /// Checks that the path length matches depth and all indices are valid.
    fn validate_path(&self, path: &[usize]) -> Result<()> {
        if path.len() != self.depth() {
            return Err(BloomCraftError::invalid_parameters(
                format!(
                    "Path length {} does not match hierarchy depth {}",
                    path.len(),
                    self.depth()
                )
            ));
        }

        for (level, &idx) in path.iter().enumerate() {
            if idx >= self.branching[level] {
                return Err(BloomCraftError::invalid_parameters(
                    format!(
                        "Path index {} at level {} exceeds branching factor {}",
                        idx, level, self.branching[level]
                    )
                ));
            }
        }

        Ok(())
    }

    /// Get the depth of the hierarchy.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// let filter: HierarchicalBloomFilter<&str> =
    ///     HierarchicalBloomFilter::new(vec![2, 3], 1000, 0.01);
    /// assert_eq!(filter.depth(), 2);
    /// ```
    #[must_use]
    pub fn depth(&self) -> usize {
        self.branching.len()
    }

    /// Get the total number of leaf bins.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// let filter: HierarchicalBloomFilter<&str> =
    ///     HierarchicalBloomFilter::new(vec![2, 4], 1000, 0.01);
    /// assert_eq!(filter.bin_count(), 8); // 2 * 4 = 8 leaves
    /// ```
    #[must_use]
    pub fn bin_count(&self) -> usize {
        self.branching.iter().product()
    }

    /// Get the total number of nodes in the hierarchy.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Insert an item into a specific bin.
    ///
    /// The path specifies which bin to insert into (e.g., [0, 2] = first branch, third sub-branch).
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    /// * `bin_path` - Path to target bin (length must equal depth)
    ///
    /// # Errors
    ///
    /// Returns error if bin_path length doesn't match depth or contains invalid indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// let mut filter: HierarchicalBloomFilter<i32> = HierarchicalBloomFilter::new(vec![2], 1000, 0.01);
    /// filter.insert_to_bin(&42, &[1]).unwrap();
    /// assert!(filter.contains(&42));
    /// ```
    pub fn insert_to_bin(&mut self, item: &T, bin_path: &[usize]) -> Result<()> {
        self.validate_path(bin_path)?;
        Self::insert_recursive_impl(&mut self.root, item, bin_path, 0, &self.branching)?;
        self.total_items += 1;
        Ok(())
    }

    /// Recursively insert item into hierarchy.
    fn insert_recursive_impl(
        node: &mut HierarchicalNode<T, H>,
        item: &T,
        bin_path: &[usize],
        level: usize,
        _branching: &[usize],
    ) -> Result<()> {
        // Insert into current node's filter
        node.filter.insert(item);

        // If not at leaf, recurse to appropriate child
        if !node.is_leaf() {
            let child_index = bin_path[level];

            if child_index >= node.children.len() {
                return Err(BloomCraftError::invalid_parameters(
                    format!(
                        "Child index {} at level {} exceeds number of children {}",
                        child_index,
                        level,
                        node.children.len()
                    )
                ));
            }

            Self::insert_recursive_impl(
                &mut node.children[child_index],
                item,
                bin_path,
                level + 1,
                _branching,
            )?;
        }

        Ok(())
    }

    /// Check if an item exists anywhere in the hierarchy.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to check
    ///
    /// # Returns
    ///
    /// `true` if item might exist, `false` if definitely doesn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// let mut filter: HierarchicalBloomFilter<&str> = HierarchicalBloomFilter::new(vec![2], 1000, 0.01);
    /// filter.insert_to_bin(&"hello", &[1]).unwrap();
    ///
    /// assert!(filter.contains(&"hello"));
    /// assert!(!filter.contains(&"world"));
    /// ```
    #[must_use]
    pub fn contains(&self, item: &T) -> bool {
        self.root.filter.contains(item)
    }

    /// Locate which bins might contain the item.
    ///
    /// Returns a list of bin paths where the item might exist.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to locate
    ///
    /// # Returns
    ///
    /// Vector of bin paths (each path is a vector of indices)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// let mut filter: HierarchicalBloomFilter<i32> = HierarchicalBloomFilter::new(vec![2], 1000, 0.01);
    /// filter.insert_to_bin(&42, &[1]).unwrap();
    ///
    /// let bins = filter.locate(&42);
    /// assert!(bins.contains(&vec![1]));
    /// ```
    #[must_use]
    pub fn locate(&self, item: &T) -> Vec<Vec<usize>> {
        if !self.root.filter.contains(item) {
            return Vec::new();
        }

        let mut result = Vec::new();
        self.locate_recursive(&self.root, item, &mut result);
        result
    }

    /// Recursively locate item in hierarchy.
    fn locate_recursive(
        &self,
        node: &HierarchicalNode<T, H>,
        item: &T,
        result: &mut Vec<Vec<usize>>,
    ) {
        if node.is_leaf() {
            // Leaf node: if item in filter, add path to results
            if node.filter.contains(item) {
                result.push(node.path.clone());
            }
        } else {
            // Internal node: check each child
            for child in &node.children {
                if child.filter.contains(item) {
                    self.locate_recursive(child, item, result);
                }
            }
        }
    }

    /// Check if an item exists in a specific bin.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to check
    /// * `bin_path` - Path to target bin
    ///
    /// # Errors
    ///
    /// Returns error if bin_path is invalid.
    pub fn contains_in_bin(&self, item: &T, bin_path: &[usize]) -> Result<bool> {
        self.validate_path(bin_path)?;
        Ok(self.contains_in_bin_recursive(&self.root, item, bin_path, 0))
    }

    /// Recursively check if item is in specific bin.
    fn contains_in_bin_recursive(
        &self,
        node: &HierarchicalNode<T, H>,
        item: &T,
        bin_path: &[usize],
        level: usize,
    ) -> bool {
        if !node.filter.contains(item) {
            return false;
        }

        if node.is_leaf() {
            return true;
        }

        let child_index = bin_path[level];
        if child_index >= node.children.len() {
            return false;
        }

        self.contains_in_bin_recursive(&node.children[child_index], item, bin_path, level + 1)
    }

    /// Clear all filters in the hierarchy.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::HierarchicalBloomFilter;
    ///
    /// let mut filter: HierarchicalBloomFilter<i32> = HierarchicalBloomFilter::new(vec![2], 1000, 0.01);
    /// filter.insert_to_bin(&42, &[1]).unwrap();
    /// filter.clear();
    /// assert!(!filter.contains(&42));
    /// ```
    pub fn clear(&mut self) {
        Self::clear_node_impl(&mut self.root);
        self.total_items = 0;
    }

    /// Recursively clear node and children.
    fn clear_node_impl(node: &mut HierarchicalNode<T, H>) {
        node.filter.clear();
        for child in &mut node.children {
            Self::clear_node_impl(child);
        }
    }

    /// Get total number of items inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_items
    }

    /// Check if hierarchy is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_items == 0
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.root.memory_usage() + std::mem::size_of::<Self>()
    }

    /// Insert multiple items to the same bin.
    ///
    /// # Arguments
    ///
    /// * `items` - Items to insert
    /// * `bin_path` - Target bin path
    ///
    /// # Errors
    ///
    /// Returns error if bin_path is invalid.
    pub fn insert_batch_to_bin(&mut self, items: &[T], bin_path: &[usize]) -> Result<()> {
        self.validate_path(bin_path)?;
        for item in items {
            Self::insert_recursive_impl(&mut self.root, item, bin_path, 0, &self.branching)?;
            self.total_items += 1;
        }
        Ok(())
    }

    /// Check multiple items for existence.
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Locate multiple items.
    #[must_use]
    pub fn locate_batch(&self, items: &[T]) -> Vec<Vec<Vec<usize>>> {
        items.iter().map(|item| self.locate(item)).collect()
    }

    /// Get statistics for the hierarchy.
    ///
    /// Returns (total_nodes, leaf_nodes, internal_nodes, total_memory).
    #[must_use]
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let total = self.node_count();
        let leaves = self.bin_count();
        let internal = total - leaves;
        let memory = self.memory_usage();

        (total, leaves, internal, memory)
    }

    /// Get fill rate for a specific bin.
    ///
    /// # Arguments
    ///
    /// * `bin_path` - Path to target bin
    ///
    /// # Errors
    ///
    /// Returns error if bin_path is invalid.
    pub fn bin_fill_rate(&self, bin_path: &[usize]) -> Result<f64> {
        self.validate_path(bin_path)?;
        Ok(self.bin_fill_rate_recursive(&self.root, bin_path, 0))
    }

    /// Recursively get fill rate for bin.
    fn bin_fill_rate_recursive(
        &self,
        node: &HierarchicalNode<T, H>,
        bin_path: &[usize],
        level: usize,
    ) -> f64 {
        if node.is_leaf() {
            return node.filter.fill_rate();
        }

        let child_index = bin_path[level];
        if child_index >= node.children.len() {
            return 0.0;
        }

        self.bin_fill_rate_recursive(&node.children[child_index], bin_path, level + 1)
    }

    /// Get fill rates for all leaf bins.
    #[must_use]
    pub fn all_bin_fill_rates(&self) -> HashMap<Vec<usize>, f64> {
        let mut result = HashMap::new();
        self.collect_fill_rates(&self.root, &mut result);
        result
    }

    /// Recursively collect fill rates.
    fn collect_fill_rates(
        &self,
        node: &HierarchicalNode<T, H>,
        result: &mut HashMap<Vec<usize>, f64>,
    ) {
        if node.is_leaf() {
            result.insert(node.path.clone(), node.filter.fill_rate());
        } else {
            for child in &node.children {
                self.collect_fill_rates(child, result);
            }
        }
    }

    /// Get the branching factors.
    #[must_use]
    pub fn branching(&self) -> &[usize] {
        &self.branching
    }

    /// Get the capacity per bin.
    #[must_use]
    pub fn capacity_per_bin(&self) -> usize {
        self.capacity_per_bin
    }

    /// Get the target FPR.
    #[must_use]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }
}

impl<T, H> BloomFilter<T> for HierarchicalBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Insert an item into the filter using the generic BloomFilter trait.
    ///
    /// **Important**: This inserts to the first bin (path [0, 0, ...]).
    /// For hierarchical filters, you typically want to use `insert_to_bin()`
    /// to specify which bin the item belongs to.
    ///
    /// This trait implementation is provided for compatibility with generic
    /// code that works with any BloomFilter, but loses the hierarchical
    /// locality information that makes this filter type useful.
    fn insert(&mut self, item: &T) {
        // For trait implementation, insert to first bin
        // Note: This is a limitation of the generic trait - hierarchical filters
        // are designed to be used with insert_to_bin() for proper bin assignment
        let default_path: Vec<usize> = vec![0; self.depth()];
        let _ = self.insert_to_bin(item, &default_path);
    }

    fn contains(&self, item: &T) -> bool {
        HierarchicalBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        HierarchicalBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        HierarchicalBloomFilter::len(self)
    }

    fn is_empty(&self) -> bool {
        HierarchicalBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.target_fpr
    }

    fn expected_items(&self) -> usize {
        self.capacity_per_bin * self.bin_count()
    }

    fn bit_count(&self) -> usize {
        fn count_bits<T, H>(node: &HierarchicalNode<T, H>) -> usize
        where
            T: Hash + Send + Sync,
            H: BloomHasher + Clone + Default,
        {
            let mut total = node.filter.bit_count();
            for child in &node.children {
                total += count_bits(child);
            }
            total
        }

        count_bits(&self.root)
    }

    fn hash_count(&self) -> usize {
        self.root.filter.hash_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        assert_eq!(filter.depth(), 2);
        assert_eq!(filter.bin_count(), 4); // 2 * 2
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_to_bin() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"hello", &[0, 1]).unwrap();
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_locate() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"item1", &[0, 1]).unwrap();
        filter.insert_to_bin(&"item2", &[1, 0]).unwrap();

        let bins1 = filter.locate(&"item1");
        assert!(bins1.contains(&vec![0, 1]));

        let bins2 = filter.locate(&"item2");
        assert!(bins2.contains(&vec![1, 0]));

        let bins3 = filter.locate(&"nonexistent");
        assert!(bins3.is_empty());
    }

    #[test]
    fn test_contains_in_bin() {
        let mut filter: HierarchicalBloomFilter<i32> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&42, &[0, 1]).unwrap();

        assert!(filter.contains_in_bin(&42, &[0, 1]).unwrap());
        assert!(!filter.contains_in_bin(&42, &[1, 0]).unwrap());
    }

    #[test]
    fn test_clear() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"item1", &[0, 0]).unwrap();
        filter.insert_to_bin(&"item2", &[1, 1]).unwrap();

        filter.clear();

        assert!(filter.is_empty());
        assert!(!filter.contains(&"item1"));
        assert!(!filter.contains(&"item2"));
    }

    #[test]
    fn test_len() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);
        assert_eq!(filter.len(), 0);

        filter.insert_to_bin(&"a", &[0, 0]).unwrap();
        filter.insert_to_bin(&"b", &[0, 1]).unwrap();

        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn test_node_count() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        // Root + 2 level-1 nodes + 4 leaf nodes = 7 nodes
        assert_eq!(filter.node_count(), 7);
    }

    #[test]
    fn test_memory_usage() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_insert_batch_to_bin() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        let items = vec!["a", "b", "c"];
        filter.insert_batch_to_bin(&items, &[1, 0]).unwrap();

        for item in &items {
            assert!(filter.contains(item));
            assert!(filter.contains_in_bin(item, &[1, 0]).unwrap());
        }
    }

    #[test]
    fn test_contains_batch() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"a", &[0, 0]).unwrap();
        filter.insert_to_bin(&"b", &[0, 1]).unwrap();

        let queries = vec!["a", "b", "c", "d"];
        let results = filter.contains_batch(&queries);

        assert_eq!(results, vec![true, true, false, false]);
    }

    #[test]
    fn test_locate_batch() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"item1", &[0, 1]).unwrap();
        filter.insert_to_bin(&"item2", &[1, 0]).unwrap();

        let items = vec!["item1", "item2", "item3"];
        let locations = filter.locate_batch(&items);

        assert_eq!(locations.len(), 3);
        assert!(locations[0].contains(&vec![0, 1]));
        assert!(locations[1].contains(&vec![1, 0]));
        assert!(locations[2].is_empty());
    }

    #[test]
    fn test_stats() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 3], 1000, 0.01);

        let (total, leaves, internal, mem) = filter.stats();

        assert_eq!(leaves, 6); // 2 * 3
        assert_eq!(internal, 3); // Root + 2 level-1 nodes
        assert_eq!(total, 9);
        assert!(mem > 0);
    }

    #[test]
    fn test_bin_fill_rate() {
        let mut filter: HierarchicalBloomFilter<i32> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        for i in 0..100 {
            filter.insert_to_bin(&i, &[0, 1]).unwrap();
        }

        let fill = filter.bin_fill_rate(&[0, 1]).unwrap();
        assert!(fill > 0.0 && fill < 1.0);

        // Other bins should be less full
        let fill2 = filter.bin_fill_rate(&[1, 0]).unwrap();
        assert!(fill2 < fill);
    }

    #[test]
    fn test_all_bin_fill_rates() {
        let mut filter: HierarchicalBloomFilter<i32> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&42, &[0, 0]).unwrap();

        let rates = filter.all_bin_fill_rates();
        assert_eq!(rates.len(), 4); // 4 leaf bins

        for (path, rate) in &rates {
            assert_eq!(path.len(), 2);
            assert!(*rate >= 0.0 && *rate <= 1.0);
        }
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter: HierarchicalBloomFilter<i32> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        BloomFilter::insert(&mut filter, &42);
        assert!(BloomFilter::contains(&filter, &42));
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    #[test]
    fn test_deep_hierarchy() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 2, 2, 2], 100, 0.01);

        assert_eq!(filter.depth(), 4);
        assert_eq!(filter.bin_count(), 16); // 2^4
    }

    #[test]
    fn test_asymmetric_branching() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![3, 5, 2], 100, 0.01);

        assert_eq!(filter.depth(), 3);
        assert_eq!(filter.bin_count(), 30); // 3 * 5 * 2
    }

    #[test]
    fn test_no_false_negatives() {
        let mut filter: HierarchicalBloomFilter<&str> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        let items = vec!["apple", "banana", "cherry"];
        for (i, item) in items.iter().enumerate() {
            filter.insert_to_bin(item, &[i % 2, i / 2]).unwrap();
        }

        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_invalid_path_length() {
        let mut filter: HierarchicalBloomFilter<i32> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        let result = filter.insert_to_bin(&42, &[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_path_index() {
        let mut filter: HierarchicalBloomFilter<i32> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        let result = filter.insert_to_bin(&42, &[3, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_branching_getter() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 4, 8], 1000, 0.01);

        assert_eq!(filter.branching(), &[2, 4, 8]);
    }

    #[test]
    fn test_capacity_and_fpr_getters() {
        let filter: HierarchicalBloomFilter<String> =
            HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        assert_eq!(filter.capacity_per_bin(), 1000);
        assert_eq!(filter.target_fpr(), 0.01);
    }

    #[test]
    fn test_clone() {
        let mut filter1 = HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);
        filter1.insert_to_bin(&"test", &[0, 1]).unwrap();

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"test"));
        assert_eq!(filter1.depth(), filter2.depth());
        assert_eq!(filter1.bin_count(), filter2.bin_count());
    }

    #[test]
    fn test_error_propagation() {
        let mut filter = HierarchicalBloomFilter::new(vec![2, 2], 1000, 0.01);

        // Invalid path should return error
        let result = filter.insert_to_bin(&42, &[5, 5]);
        assert!(result.is_err());

        // Filter should still be usable
        assert!(filter.insert_to_bin(&42, &[0, 0]).is_ok());
        assert!(filter.contains(&42));
    }
}
