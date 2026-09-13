mod args;
mod error;
mod output;

use std::{
    error::Error,
    io::{self, Write},
    process::ExitCode,
};

use clap::Parser;
use mlx_lm::{Model, Prompt};

use args::{Cli, Command, GenerationFormat, Source};
use error::CliError;
use output::{encode_event, model_info, write_event, write_info, OutputEvent};

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = error.exit_code() as u8;
            let _ = write!(io::stderr().lock(), "{error}");
            return ExitCode::from(code);
        }
    };
    match run(cli, &mut io::stdout().lock(), &mut io::stderr().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.is_broken_pipe() => ExitCode::SUCCESS,
        Err(error) => {
            let mut stderr = io::stderr().lock();
            let _ = writeln!(stderr, "mlx-lm: {error}");
            let mut source = error.source();
            while let Some(error) = source {
                let _ = writeln!(stderr, "  caused by: {error}");
                source = error.source();
            }
            ExitCode::FAILURE
        }
    }
}

fn load(source: &Source) -> Result<Model, CliError> {
    if let Some(path) = &source.model {
        return Model::from_dir(path).map_err(CliError::Load);
    }
    #[cfg(feature = "hf-hub")]
    if let Some(repo) = &source.repo {
        return Model::from_hub(repo, source.hub_options()).map_err(CliError::Hub);
    }
    unreachable!("clap requires exactly one model source")
}

fn run(cli: Cli, stdout: &mut impl Write, stderr: &mut impl Write) -> Result<(), CliError> {
    match cli.command {
        Command::Info(args) => {
            let model = load(&args.source)?;
            write_info(stdout, &model_info(&model), args.format)
        }
        Command::Generate(args) => {
            let options = args.options();
            let mut model = load(&args.source)?;
            options
                .sampling
                .validate(model.config().dimensions.vocabulary_size)
                .map_err(mlx_lm::GenerationError::from)?;
            let generation = model.generate(Prompt::Text(&args.prompt), options)?;
            for event in generation {
                let event = encode_event(event?)?;
                write_event(stdout, &event, args.format)?;
                if args.format == GenerationFormat::Text {
                    if let OutputEvent::Prefill { processed, total } = event {
                        writeln!(stderr, "Prefill: {processed}/{total}")?;
                    }
                }
            }
            Ok(())
        }
    }
}
