// Generate as many DTL gene trees as possible within 1 second and export to XML
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::{simulate_dtl, count_events, count_extant_genes, DTLEvent};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::{Instant, Duration};
use std::fs;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let time_limit = Duration::from_secs(1);

    println!("=== Maximum Throughput DTL Benchmark with XML Export ===");
    println!("Time limit: {} second\n", time_limit.as_secs());

    // Phase 1: Generate species tree
    println!("Phase 1: Generating species tree (1000 leaves, pure birth)...");
    let phase1_start = Instant::now();
    let n_species = 1000;
    let lambda = 1.0;
    let mu = 0.0; // Pure birth

    let (species_tree, _) = simulate_bd_tree_bwd(n_species, lambda, mu, &mut rng).unwrap();
    let mut species_tree = species_tree; // Make mutable for assign_depths
    species_tree.assign_depths();
    let phase1_time = phase1_start.elapsed();
    println!("  ✓ Completed in {:.3}s", phase1_time.as_secs_f64());
    println!();

    // Phase 2: DTL simulations
    let lambda_d = 0.2;
    let lambda_t = 0.2;
    let lambda_l = 0.1;

    println!("Phase 2: Simulating DTL gene trees...");
    println!("  Parameters: d={}, t={}, l={}", lambda_d, lambda_t, lambda_l);
    println!("  Starting simulation...\n");

    let mut rec_trees = Vec::new();
    let mut all_events = Vec::new();
    let mut count = 0;
    let mut total_events = 0;
    let mut total_extant = 0;
    let mut total_speciations = 0;
    let mut total_duplications = 0;
    let mut total_transfers = 0;
    let mut total_losses = 0;
    let mut min_events = usize::MAX;
    let mut max_events = 0;
    let mut min_extant = usize::MAX;
    let mut max_extant = 0;

    let phase2_start = Instant::now();

    while phase2_start.elapsed() < time_limit {
        let (rec_tree, events) = simulate_dtl(
            &species_tree,
            species_tree.root,
            lambda_d,
            lambda_t,
            lambda_l,
            None,
            None,
            false,
            &mut rng,
        ).unwrap();

        let (s, d, t, l, _) = count_events(&rec_tree);
        let extant = count_extant_genes(&rec_tree);

        count += 1;
        let num_events = events.len();
        total_events += num_events;
        total_extant += extant;
        total_speciations += s;
        total_duplications += d;
        total_transfers += t;
        total_losses += l;

        min_events = min_events.min(num_events);
        max_events = max_events.max(num_events);
        min_extant = min_extant.min(extant);
        max_extant = max_extant.max(extant);

        // Store the results
        rec_trees.push(rec_tree);
        all_events.push(events);

        // Print progress every 10 simulations
        if count % 10 == 0 {
            let elapsed = phase2_start.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed;
            print!("\r  Simulations: {} | Rate: {:.1} trees/sec | Elapsed: {:.3}s",
                   count, rate, elapsed);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    let phase2_time = phase2_start.elapsed();
    println!("\n");
    println!("  ✓ Generated {} gene trees in {:.3}s", count, phase2_time.as_secs_f64());
    println!("  ✓ Throughput: {:.1} trees/second", count as f64 / phase2_time.as_secs_f64());
    println!();

    // Phase 3: Export to XML
    println!("Phase 3: Exporting {} gene trees to XML...", count);
    let phase3_start = Instant::now();

    // Create output directory
    fs::create_dir_all("dtl_gene_trees").expect("Failed to create directory");

    for (i, rec_tree) in rec_trees.iter().enumerate() {
        let xml = rec_tree.to_xml();
        let filename = format!("dtl_gene_trees/gene_tree_{:04}.xml", i);

        // Use BufWriter for faster file writing
        use std::io::BufWriter;
        let file = fs::File::create(&filename).expect("Failed to create XML file");
        let mut writer = BufWriter::new(file);
        use std::io::Write;
        writer.write_all(xml.as_bytes()).expect("Failed to write XML");
        writer.flush().expect("Failed to flush XML");

        if (i + 1) % 20 == 0 {
            print!("\r  Exported: {}/{} trees", i + 1, count);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    let phase3_time = phase3_start.elapsed();
    println!("\r  ✓ Exported {} XML files in {:.3}s", count, phase3_time.as_secs_f64());
    println!("  ✓ XML export rate: {:.1} files/second", count as f64 / phase3_time.as_secs_f64());
    println!();

    // Phase 3b: Export to Newick format
    println!("Phase 3b: Exporting {} gene trees to Newick format...", count);
    let phase3b_start = Instant::now();

    // Create output directory for Newick files
    fs::create_dir_all("dtl_gene_trees_newick").expect("Failed to create newick directory");

    for (i, rec_tree) in rec_trees.iter().enumerate() {
        let newick = rec_tree.gene_tree.to_newick()
            .expect("Failed to convert gene tree to Newick");
        let filename = format!("dtl_gene_trees_newick/gene_tree_{:04}.newick", i);

        // Use BufWriter for faster file writing
        use std::io::BufWriter;
        let file = fs::File::create(&filename).expect("Failed to create Newick file");
        let mut writer = BufWriter::new(file);
        use std::io::Write;
        writer.write_all(newick.as_bytes()).expect("Failed to write Newick");
        writer.flush().expect("Failed to flush Newick");

        if (i + 1) % 20 == 0 {
            print!("\r  Exported: {}/{} trees", i + 1, count);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    let phase3b_time = phase3b_start.elapsed();
    println!("\r  ✓ Exported {} Newick files in {:.3}s", count, phase3b_time.as_secs_f64());
    println!("  ✓ Newick export rate: {:.1} files/second", count as f64 / phase3b_time.as_secs_f64());
    println!();

    // Phase 4: Export events to CSV
    println!("Phase 4: Exporting events to CSV...");
    let phase4_start = Instant::now();

    // Combine all events into a single CSV with tree_id column
    use std::fs::File;
    use std::io::{Write, BufWriter};

    let file = File::create("dtl_all_events.csv").expect("Failed to create CSV");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "tree_id,{}", DTLEvent::csv_header())
        .expect("Failed to write header");

    for (tree_id, (events, rec_tree)) in all_events.iter().zip(rec_trees.iter()).enumerate() {
        for event in events {
            writeln!(
                writer,
                "{},{}",
                tree_id,
                event.to_csv_row(&species_tree, &rec_tree.gene_tree)
            ).expect("Failed to write event");
        }
    }

    writer.flush().expect("Failed to flush CSV");

    let phase4_time = phase4_start.elapsed();
    println!("  ✓ Exported {} events to dtl_all_events.csv in {:.3}s",
             total_events, phase4_time.as_secs_f64());
    println!();

    // Calculate statistics
    let throughput = count as f64 / phase2_time.as_secs_f64();
    let avg_events = total_events / count;
    let avg_extant = total_extant / count;
    let avg_s = total_speciations / count;
    let avg_d = total_duplications / count;
    let avg_t = total_transfers / count;
    let avg_l = total_losses / count;

    println!("=== Results Summary ===");
    println!();
    println!("Simulation Statistics:");
    println!("  Total simulations: {}", count);
    println!("  Average events per tree: {}", avg_events);
    println!("  Range: {} - {} events", min_events, max_events);
    println!("  Average extant genes: {}", avg_extant);
    println!("  Range: {} - {} extant genes", min_extant, max_extant);
    println!();

    println!("Event Breakdown (averages):");
    println!("  Speciations:  {} ({:.1}%)", avg_s, 100.0 * avg_s as f64 / avg_events as f64);
    println!("  Duplications: {} ({:.1}%)", avg_d, 100.0 * avg_d as f64 / avg_events as f64);
    println!("  Transfers:    {} ({:.1}%)", avg_t, 100.0 * avg_t as f64 / avg_events as f64);
    println!("  Losses:       {} ({:.1}%)", avg_l, 100.0 * avg_l as f64 / avg_events as f64);
    println!();

    let total_time = phase1_time + phase2_time + phase3_time + phase3b_time + phase4_time;

    println!("=== Timing Breakdown ===");
    println!();
    println!("Phase 1 - Species tree generation:  {:8.3}s ({:5.1}%)",
             phase1_time.as_secs_f64(),
             100.0 * phase1_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Phase 2 - DTL simulations:           {:8.3}s ({:5.1}%)",
             phase2_time.as_secs_f64(),
             100.0 * phase2_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Phase 3 - XML export:                {:8.3}s ({:5.1}%)",
             phase3_time.as_secs_f64(),
             100.0 * phase3_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Phase 3b - Newick export:            {:8.3}s ({:5.1}%)",
             phase3b_time.as_secs_f64(),
             100.0 * phase3b_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Phase 4 - CSV export:                {:8.3}s ({:5.1}%)",
             phase4_time.as_secs_f64(),
             100.0 * phase4_time.as_secs_f64() / total_time.as_secs_f64());
    println!("         {}", "-".repeat(40));
    println!("Total time:                          {:8.3}s (100.0%)", total_time.as_secs_f64());
    println!();

    println!("Performance Metrics:");
    println!("  DTL simulation rate:     {:.1} trees/second", throughput);
    println!("  XML export rate:         {:.1} files/second", count as f64 / phase3_time.as_secs_f64());
    println!("  Newick export rate:      {:.1} files/second", count as f64 / phase3b_time.as_secs_f64());
    println!("  CSV export rate:         {:.0} events/second", total_events as f64 / phase4_time.as_secs_f64());
    println!("  Total events processed:  {}", total_events);
    println!("  Total extant genes:      {}", total_extant);
    println!();

    println!("Output files:");
    println!("  XML files:    dtl_gene_trees/ ({} files)", count);
    println!("  Newick files: dtl_gene_trees_newick/ ({} files)", count);
    println!("  CSV file:     dtl_all_events.csv ({} events)", total_events);
}
