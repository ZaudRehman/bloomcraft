//! TreeBloomFilter capacity planning example.
//!
//! Run with: cargo run --example capacity_planning

use bloomcraft::filters::tree::TreeConfig;

fn main() {
    println!("TreeBloomFilter Capacity Planning Examples\n");
    
    let configs = vec![
        ("Small CDN", vec![5, 10, 20], 1000, 0.01),
        ("Enterprise Org", vec![10, 15, 20, 10], 5000, 0.001),
        ("Filesystem", vec![1, 4, 10, 10, 10], 2000, 0.01),
        ("Deep Parse Tree", vec![3; 8], 500, 0.01),
    ];
    
    for (name, branching, capacity, fpr) in configs {
        let config = TreeConfig {
            branching,
            capacity_per_bin: capacity,
            target_fpr: fpr,
        };
        
        println!("{}", name);
        println!("{}\n", config.report());
    }
}
