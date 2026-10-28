use std::env;
use std::path::PathBuf;

mod api_baseline;
#[path = "../../mlx-sys/src/bindgen_config.rs"]
mod bindgen_config;
mod checkpoints;
mod config_report;
mod fingerprint;
mod linked_exports;
mod verify_bump;
mod verify_ffi;
mod verify_ledger;
mod verify_lm_features;
mod verify_oracle_boundary;

fn get_repo_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "api-baseline") {
        std::process::exit(api_baseline::run(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "fingerprint") {
        std::process::exit(fingerprint::run_fingerprint(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "fingerprint-delta") {
        std::process::exit(fingerprint::run_delta(&args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "verify-lm-features") {
        std::process::exit(verify_lm_features::run(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "verify-ledger") {
        std::process::exit(verify_ledger::run(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "verify-ffi") {
        std::process::exit(verify_ffi::run(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "verify-bump") {
        std::process::exit(verify_bump::run(&get_repo_root(), &args[2..]));
    }
    if args
        .get(1)
        .is_some_and(|arg| arg == "verify-oracle-boundary")
    {
        std::process::exit(verify_oracle_boundary::run(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "config-report") {
        std::process::exit(config_report::run(&get_repo_root(), &args[2..]));
    }
    if args.get(1).is_some_and(|arg| arg == "linked-exports") {
        std::process::exit(linked_exports::run(&get_repo_root(), &args[2..]));
    }
    eprintln!("usage: cargo xtask <api-baseline|fingerprint|fingerprint-delta|verify-lm-features|verify-ledger|verify-ffi|verify-bump|verify-oracle-boundary|config-report|linked-exports> [options]");
    std::process::exit(2);
}
