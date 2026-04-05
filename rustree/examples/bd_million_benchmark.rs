use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustree::bd::simulate_bd_tree_bwd;

fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn main() {
    println!("Birth-Death Tree Benchmark - Scaling Test");
    println!("=========================================");
    println!("Parameters: λ=1.0, μ=0.9");
    println!();

    let lambda = 1.0;
    let mu = 0.9;
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let runs_per_size = 3;

    for &n in &sizes {
        print!("n={:>10}: ", fmt_num(n));
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut total_time = 0.0;
        let mut total_nodes = 0usize;

        for i in 0..runs_per_size {
            let mut rng = StdRng::seed_from_u64(42 + i as u64);

            let start = Instant::now();
            let (tree, _events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap();
            let elapsed = start.elapsed().as_secs_f64();

            total_time += elapsed;
            total_nodes += tree.nodes.len();
        }

        let avg_time = total_time / runs_per_size as f64;
        let avg_nodes = total_nodes / runs_per_size;
        let trees_per_min = 60.0 / avg_time;

        println!("{:.3}s avg, {:>12} nodes, {:.1} trees/min",
            avg_time,
            fmt_num(avg_nodes),
            trees_per_min
        );
    }

    println!();
    println!("Note: With μ/λ=0.9, trees have ~10x more total nodes than extant species");
}
