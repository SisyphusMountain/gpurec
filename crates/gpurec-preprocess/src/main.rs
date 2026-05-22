use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use gpurec_preprocess::{preprocess_request, write_binary_output, PreprocessRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args
        .iter()
        .any(|arg| arg == &PathBuf::from("--help") || arg == &PathBuf::from("-h"))
    {
        print_usage();
        return Ok(());
    }
    let mut discard_output = false;
    let mut binary_output = false;
    let mut input_path = None;
    for arg in args {
        if arg == PathBuf::from("--discard-output") {
            discard_output = true;
        } else if arg == PathBuf::from("--binary-output") {
            binary_output = true;
        } else if input_path.is_none() {
            input_path = Some(arg);
        } else {
            return Err(
                "usage: gpurec-preprocess [--discard-output] [--binary-output] [request.json]"
                    .into(),
            );
        }
    }

    let input = if let Some(path) = input_path {
        fs::read_to_string(path)?
    } else {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        input
    };
    let request: PreprocessRequest = serde_json::from_str(&input)?;
    let output = preprocess_request(&request)?;
    if discard_output {
        return Ok(());
    }
    if binary_output {
        write_binary_output(&output, std::io::stdout())?;
    } else {
        serde_json::to_writer(std::io::stdout(), &output)?;
        println!();
    }
    Ok(())
}

fn print_usage() {
    eprintln!("usage: gpurec-preprocess [--discard-output] [--binary-output] [request.json]");
}
