//! Basic test - just to prove the filter works

use bloomcraft::filters::StandardBloomFilter;

#[test]
fn test_basic_insert_and_find() {
    // Create a filter and unwrap it
    let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();

    // Add one item
    filter.insert(&"test-item".to_string());

    // Check we can find it
    assert!(
        filter.contains(&"test-item".to_string()),
        "Should find the item we just added"
    );
}

#[test]
fn test_batch_operations() {
    let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
    
    // Insert multiple items
    let items: Vec<String> = vec!["apple", "banana", "cherry"]
        .into_iter()
        .map(String::from)
        .collect();
    
    filter.insert_batch(&items);
    
    // Check all items are found
    for item in &items {
        assert!(filter.contains(item), "Should find {}", item);
    }
}

#[test]
fn test_no_false_negatives() {
    let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
    
    // Insert 100 numbers
    for i in 0..100 {
        filter.insert(&i);
    }
    
    // All inserted items MUST be found (no false negatives allowed)
    for i in 0..100 {
        assert!(filter.contains(&i), "False negative for {}", i);
    }
}
