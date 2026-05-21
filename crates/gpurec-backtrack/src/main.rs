use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use gpurec_backtrack::{sample_recphyloxml, sample_recphyloxmls, BacktrackInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(CliError::Help) => {
            print_usage();
            return Ok(());
        }
        Err(CliError::Message(message)) => return Err(message.into()),
    };

    let input_path = args.positionals.first().cloned();
    let output_path = args.positionals.get(1).cloned();

    let input_json = match input_path {
        Some(path) => fs::read_to_string(path)?,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };
    let mut input: BacktrackInput = serde_json::from_str(&input_json)?;
    if let Some(seed) = args.seed_override {
        input.seed = Some(seed);
    }
    if let Some(max_events) = args.max_events_override {
        input.max_events = Some(max_events);
    }

    if args.samples == 1 && args.output_dir.is_none() {
        let xml = sample_recphyloxml(&input)?;
        match output_path {
            Some(path) => fs::write(path, xml)?,
            None => print!("{xml}"),
        }
        return Ok(());
    }

    let dir = args
        .output_dir
        .ok_or("--output-dir is required in multi-sample mode")?;
    fs::create_dir_all(&dir)?;
    let base_seed = input.seed.unwrap_or(0);
    for (sample_idx, xml) in sample_recphyloxmls(&input, args.samples, base_seed)?
        .into_iter()
        .enumerate()
    {
        fs::write(dir.join(format!("sample_{sample_idx}.xml")), xml)?;
    }

    Ok(())
}

#[derive(Debug, PartialEq)]
struct CliArgs {
    samples: usize,
    output_dir: Option<PathBuf>,
    seed_override: Option<u64>,
    max_events_override: Option<usize>,
    positionals: Vec<PathBuf>,
}

#[derive(Debug, PartialEq)]
enum CliError {
    Help,
    Message(String),
}

fn parse_args<I>(args: I) -> Result<CliArgs, CliError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut parsed = CliArgs {
        samples: 1,
        output_dir: None,
        seed_override: None,
        max_events_override: None,
        positionals: Vec::new(),
    };

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        match text.as_ref() {
            "--samples" => {
                let value = next_option_value(&mut args, "--samples")?;
                parsed.samples = parse_option_value(&value, "--samples")?;
            }
            "--output-dir" => {
                let value = next_option_value(&mut args, "--output-dir")?;
                parsed.output_dir = Some(PathBuf::from(value));
            }
            "--seed" => {
                let value = next_option_value(&mut args, "--seed")?;
                parsed.seed_override = Some(parse_option_value(&value, "--seed")?);
            }
            "--max-events" => {
                let value = next_option_value(&mut args, "--max-events")?;
                parsed.max_events_override = Some(parse_option_value(&value, "--max-events")?);
            }
            "--help" | "-h" => {
                return Err(CliError::Help);
            }
            _ if text.starts_with('-') => {
                return Err(CliError::Message(format!("unknown option: {text}")));
            }
            _ => parsed.positionals.push(PathBuf::from(arg)),
        }
    }

    if parsed.samples == 0 {
        return Err(CliError::Message("--samples must be positive".to_string()));
    }
    if parsed.max_events_override == Some(0) {
        return Err(CliError::Message(
            "--max-events must be positive".to_string(),
        ));
    }
    if parsed.positionals.len() > 2 {
        print_usage();
        return Err(CliError::Message(
            "too many positional arguments".to_string(),
        ));
    }
    if parsed.output_dir.is_some() && parsed.positionals.len() == 2 {
        return Err(CliError::Message(
            "--output-dir writes samples to a directory, not a single output file".to_string(),
        ));
    }

    Ok(parsed)
}

fn next_option_value<I>(args: &mut I, option: &str) -> Result<OsString, CliError>
where
    I: Iterator<Item = OsString>,
{
    args.next()
        .ok_or_else(|| CliError::Message(format!("{option} requires a value")))
}

fn parse_option_value<T>(value: &OsString, option: &str) -> Result<T, CliError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .to_string_lossy()
        .parse()
        .map_err(|exc| CliError::Message(format!("{option} has invalid value: {exc}")))
}

fn print_usage() {
    eprintln!(
        "usage: gpurec-backtrack [--samples N --output-dir DIR --seed SEED --max-events N] [input.json] [output.xml]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(values: &[&str]) -> Result<CliArgs, CliError> {
        parse_args(values.iter().map(OsString::from))
    }

    #[test]
    fn parses_max_events_override() {
        let parsed = parse(&[
            "--samples",
            "2",
            "--output-dir",
            "out",
            "--seed",
            "17",
            "--max-events",
            "99",
            "input.json",
        ])
        .unwrap();

        assert_eq!(parsed.samples, 2);
        assert_eq!(parsed.output_dir, Some(PathBuf::from("out")));
        assert_eq!(parsed.seed_override, Some(17));
        assert_eq!(parsed.max_events_override, Some(99));
        assert_eq!(parsed.positionals, vec![PathBuf::from("input.json")]);
    }

    #[test]
    fn rejects_unknown_options() {
        let err = parse(&["--sampls", "2"]).unwrap_err();

        assert_eq!(
            err,
            CliError::Message("unknown option: --sampls".to_string())
        );
    }

    #[test]
    fn missing_option_value_names_option() {
        let samples = parse(&["--samples"]).unwrap_err();
        let seed = parse(&["--seed"]).unwrap_err();
        let max_events = parse(&["--max-events"]).unwrap_err();

        assert_eq!(
            samples,
            CliError::Message("--samples requires a value".to_string())
        );
        assert_eq!(
            seed,
            CliError::Message("--seed requires a value".to_string())
        );
        assert_eq!(
            max_events,
            CliError::Message("--max-events requires a value".to_string())
        );
    }

    #[test]
    fn rejects_zero_max_events() {
        let err = parse(&["--max-events", "0"]).unwrap_err();

        assert_eq!(
            err,
            CliError::Message("--max-events must be positive".to_string())
        );
    }

    #[test]
    fn rejects_output_file_when_output_dir_is_set() {
        let err = parse(&[
            "--samples",
            "1",
            "--output-dir",
            "out",
            "input.json",
            "ignored.xml",
        ])
        .unwrap_err();

        assert_eq!(
            err,
            CliError::Message(
                "--output-dir writes samples to a directory, not a single output file".to_string()
            )
        );
    }
}
