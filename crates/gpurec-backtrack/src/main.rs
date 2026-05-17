use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use gpurec_backtrack::{sample_recphyloxml, BacktrackInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let input_path = args.next().map(PathBuf::from);
    let output_path = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err("usage: gpurec-backtrack [input.json] [output.xml]".into());
    }

    let input_json = match input_path {
        Some(path) => fs::read_to_string(path)?,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };
    let input: BacktrackInput = serde_json::from_str(&input_json)?;
    let xml = sample_recphyloxml(&input)?;

    match output_path {
        Some(path) => fs::write(path, xml)?,
        None => print!("{xml}"),
    }
    Ok(())
}
