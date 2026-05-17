use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use gpurec_backtrack::{sample_recphyloxml, BacktrackInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut samples = 1usize;
    let mut output_dir: Option<PathBuf> = None;
    let mut seed_override: Option<u64> = None;
    let mut positionals: Vec<PathBuf> = Vec::new();

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        match text.as_ref() {
            "--samples" => {
                let value = args.next().ok_or("--samples requires a value")?;
                samples = value.to_string_lossy().parse()?;
            }
            "--output-dir" => {
                let value = args.next().ok_or("--output-dir requires a value")?;
                output_dir = Some(PathBuf::from(value));
            }
            "--seed" => {
                let value = args.next().ok_or("--seed requires a value")?;
                seed_override = Some(value.to_string_lossy().parse()?);
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => positionals.push(PathBuf::from(arg)),
        }
    }

    if samples == 0 {
        return Err("--samples must be positive".into());
    }
    if positionals.len() > 2 {
        print_usage();
        return Err("too many positional arguments".into());
    }

    let input_path = positionals.first().cloned();
    let output_path = positionals.get(1).cloned();
    if samples > 1 && output_path.is_some() {
        return Err("multi-sample mode writes to --output-dir, not a single output file".into());
    }

    let input_json = match input_path {
        Some(path) => fs::read_to_string(path)?,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };
    let mut input: BacktrackInput = serde_json::from_str(&input_json)?;
    if let Some(seed) = seed_override {
        input.seed = Some(seed);
    }

    if samples == 1 && output_dir.is_none() {
        let xml = sample_recphyloxml(&input)?;
        match output_path {
            Some(path) => fs::write(path, xml)?,
            None => print!("{xml}"),
        }
        return Ok(());
    }

    let dir = output_dir.ok_or("--output-dir is required in multi-sample mode")?;
    fs::create_dir_all(&dir)?;
    let base_seed = input.seed.unwrap_or(0);
    for sample_idx in 0..samples {
        let mut sample_input = input.clone();
        sample_input.seed = Some(base_seed + sample_idx as u64);
        let xml = sample_recphyloxml(&sample_input)?;
        fs::write(dir.join(format!("sample_{sample_idx}.xml")), xml)?;
    }

    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage: gpurec-backtrack [--samples N --output-dir DIR --seed SEED] [input.json] [output.xml]"
    );
}
