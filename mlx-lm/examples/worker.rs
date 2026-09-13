use std::{
    error::Error,
    fmt,
    io::{self, Write},
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
};

use mlx_lm::{GenerationEvent, GenerationOptions, Model, Prompt};
use mlx_rs::{with_device, Device};

#[derive(Debug)]
struct WorkerError {
    operation: &'static str,
    message: String,
}

impl WorkerError {
    fn new(operation: &'static str, error: impl fmt::Display) -> Self {
        Self {
            operation,
            message: error.to_string(),
        }
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.operation, self.message)
    }
}

impl Error for WorkerError {}

struct Request {
    prompt: String,
    options: GenerationOptions,
    response: SyncSender<Result<GenerationEvent, WorkerError>>,
}

fn serve(model: &mut Model, request: Request) -> Result<(), WorkerError> {
    let mut cache = model
        .new_cache(request.options.cache.clone())
        .map_err(|error| WorkerError::new("create cache", error))?;
    let generation = model
        .generate_with_cache(Prompt::Text(&request.prompt), request.options, &mut cache)
        .map_err(|error| WorkerError::new("start generation", error))?;
    for event in generation {
        let event = event.map_err(|error| WorkerError::new("generate", error));
        if request.response.send(event).is_err() {
            // Receiver loss drops the iterator here, on its owning thread.
            break;
        }
    }
    Ok(())
}

fn worker(
    checkpoint: PathBuf,
    requests: Receiver<Request>,
    ready: SyncSender<Result<(), WorkerError>>,
) {
    with_device(Device::gpu(), || {
        let mut model = match Model::from_dir(checkpoint) {
            Ok(model) => model,
            Err(error) => {
                let _ = ready.send(Err(WorkerError::new("load checkpoint", error)));
                return;
            }
        };
        if ready.send(Ok(())).is_err() {
            return;
        }
        for request in requests {
            let response = request.response.clone();
            if let Err(error) = serve(&mut model, request) {
                let _ = response.send(Err(error));
            }
        }
    });
}

fn assert_send<T: Send>() {}

fn main() -> Result<(), Box<dyn Error>> {
    assert_send::<GenerationOptions>();
    assert_send::<GenerationEvent>();
    assert_send::<WorkerError>();
    assert_send::<Request>();

    let mut args = std::env::args_os().skip(1);
    let checkpoint = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: cargo run -p mlx-lm --example worker -- <checkpoint-directory>")?;
    if args.next().is_some() {
        return Err("expected one checkpoint-directory argument".into());
    }

    let (requests, incoming) = mpsc::sync_channel(1);
    let (ready, started) = mpsc::sync_channel(0);
    let worker = thread::spawn(move || worker(checkpoint, incoming, ready));
    let result = (|| -> Result<(), Box<dyn Error>> {
        started.recv()??;
        for cancel in [false, true] {
            let (response, events) = mpsc::sync_channel(1);
            requests.send(Request {
                prompt: "hello the small fox runs over green hill".to_owned(),
                options: GenerationOptions::default(),
                response,
            })?;
            for event in &events {
                if let GenerationEvent::Token { text, .. } = event? {
                    if cancel {
                        break;
                    }
                    print!("{text}");
                    io::stdout().flush()?;
                }
            }
            drop(events);
        }
        println!();
        Ok(())
    })();
    // All response receivers must be gone before join can wait for a blocked sender.
    drop(started);
    drop(requests);
    let joined = worker.join();
    result?;
    joined.map_err(|_| "model worker panicked")?;
    Ok(())
}
