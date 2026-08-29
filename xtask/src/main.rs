use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

mod api_baseline;
mod bindgen_config;
mod fingerprint;
mod verify_bump;
mod verify_ffi;
mod verify_ledger;
mod verify_oracle_boundary;

fn get_repo_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
}

fn get_current_tag(mlx_c_dir: &Path) -> String {
    let output = Command::new("git")
        .args(["describe", "--tags"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to get current tag");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn get_latest_tag(mlx_c_dir: &Path) -> String {
    // Fetch tags first
    Command::new("git")
        .args(["fetch", "--tags", "--quiet"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to fetch tags");

    let output = Command::new("git")
        .args(["rev-list", "--tags", "--max-count=1"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to get latest tag commit");

    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();

    let output = Command::new("git")
        .args(["describe", "--tags", &commit])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to get tag name");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn checkout_tag(mlx_c_dir: &Path, tag: &str) {
    Command::new("git")
        .args(["checkout", tag, "--quiet"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to checkout tag");
}

fn generate_bindings(root_dir: &Path) -> String {
    let mlx_c_dir = root_dir.join("mlx-sys/src/mlx-c");

    let bindings = bindgen::Builder::default()
        .header(mlx_c_dir.join("mlx/c/mlx.h").to_str().unwrap())
        .header(mlx_c_dir.join("mlx/c/linalg.h").to_str().unwrap())
        .header(mlx_c_dir.join("mlx/c/error.h").to_str().unwrap())
        .header(mlx_c_dir.join("mlx/c/transforms_impl.h").to_str().unwrap())
        .clang_arg(format!("-I{}", mlx_c_dir.to_str().unwrap()))
        .generate()
        .expect("Unable to generate bindings");

    bindings.to_string()
}

/// Represents a parsed function signature
#[derive(Debug, Clone)]
struct FunctionSig {
    name: String,
    full_signature: String,
}

/// Extract function signatures from bindings
/// Looks for patterns like: pub fn function_name(...) -> ReturnType;
fn extract_functions(bindings: &str) -> HashMap<String, FunctionSig> {
    let mut functions = HashMap::new();
    let mut current_fn: Option<(String, String)> = None;

    for line in bindings.lines() {
        let trimmed = line.trim();

        // Look for function declarations
        if trimmed.contains("pub fn ") {
            // Extract function name
            if let Some(start) = trimmed.find("pub fn ") {
                let after_fn = &trimmed[start + 7..];
                if let Some(paren) = after_fn.find('(') {
                    let fn_name = after_fn[..paren].trim().to_string();
                    // Start collecting the signature
                    current_fn = Some((fn_name, line.to_string()));
                }
            }
        } else if let Some((ref _name, ref mut sig)) = current_fn {
            // Continue collecting multi-line signature
            sig.push('\n');
            sig.push_str(line);
        }

        // Check if signature is complete (ends with semicolon or has opening brace for body)
        if let Some((name, sig)) = current_fn.take() {
            if trimmed.ends_with(';')
                || trimmed.contains(") {")
                || trimmed.contains(") ->") && trimmed.ends_with(';')
            {
                functions.insert(
                    name.clone(),
                    FunctionSig {
                        name,
                        full_signature: sig,
                    },
                );
            } else {
                // Keep collecting
                current_fn = Some((name, sig));
            }
        }
    }

    functions
}

/// Represents a parsed struct
#[derive(Debug, Clone)]
struct StructDef {
    name: String,
    full_definition: String,
}

/// Extract struct definitions from bindings
fn extract_structs(bindings: &str) -> HashMap<String, StructDef> {
    let mut structs = HashMap::new();
    let mut current_struct: Option<(String, String, i32)> = None; // (name, content, brace_depth)

    for line in bindings.lines() {
        let trimmed = line.trim();

        // Look for struct declarations
        if let Some(after_struct) = trimmed.strip_prefix("pub struct ") {
            if let Some(name_end) = after_struct.find([' ', '{', ';']) {
                let struct_name = after_struct[..name_end].trim().to_string();
                let initial_depth = if trimmed.contains('{') { 1 } else { 0 };
                current_struct = Some((struct_name, line.to_string(), initial_depth));

                // Handle single-line structs
                if trimmed.ends_with(';') || (trimmed.contains('{') && trimmed.contains('}')) {
                    if let Some((name, content, _)) = current_struct.take() {
                        structs.insert(
                            name.clone(),
                            StructDef {
                                name,
                                full_definition: content,
                            },
                        );
                    }
                }
                continue;
            }
        }

        // Continue collecting multi-line struct
        if let Some((ref name, ref mut content, ref mut depth)) = current_struct {
            content.push('\n');
            content.push_str(line);

            *depth += trimmed.matches('{').count() as i32;
            *depth -= trimmed.matches('}').count() as i32;

            if *depth <= 0 {
                let name = name.clone();
                let content = content.clone();
                current_struct = None;
                structs.insert(
                    name.clone(),
                    StructDef {
                        name,
                        full_definition: content,
                    },
                );
            }
        }
    }

    structs
}

/// Compare two sets of functions and return (added, removed, modified)
fn compare_functions(
    old: &HashMap<String, FunctionSig>,
    new: &HashMap<String, FunctionSig>,
) -> (
    Vec<FunctionSig>,
    Vec<FunctionSig>,
    Vec<(FunctionSig, FunctionSig)>,
) {
    let old_names: HashSet<_> = old.keys().collect();
    let new_names: HashSet<_> = new.keys().collect();

    let added: Vec<FunctionSig> = new_names
        .difference(&old_names)
        .filter_map(|name| new.get(*name).cloned())
        .collect();

    let removed: Vec<FunctionSig> = old_names
        .difference(&new_names)
        .filter_map(|name| old.get(*name).cloned())
        .collect();

    let modified: Vec<(FunctionSig, FunctionSig)> = old_names
        .intersection(&new_names)
        .filter_map(|name| {
            let old_fn = old.get(*name)?;
            let new_fn = new.get(*name)?;
            // Normalize whitespace for comparison
            let old_normalized: String = old_fn.full_signature.split_whitespace().collect();
            let new_normalized: String = new_fn.full_signature.split_whitespace().collect();
            if old_normalized != new_normalized {
                Some((old_fn.clone(), new_fn.clone()))
            } else {
                None
            }
        })
        .collect();

    (added, removed, modified)
}

/// Compare two sets of structs and return (added, removed, modified)
fn compare_structs(
    old: &HashMap<String, StructDef>,
    new: &HashMap<String, StructDef>,
) -> (Vec<StructDef>, Vec<StructDef>, Vec<(StructDef, StructDef)>) {
    let old_names: HashSet<_> = old.keys().collect();
    let new_names: HashSet<_> = new.keys().collect();

    let added: Vec<StructDef> = new_names
        .difference(&old_names)
        .filter_map(|name| new.get(*name).cloned())
        .collect();

    let removed: Vec<StructDef> = old_names
        .difference(&new_names)
        .filter_map(|name| old.get(*name).cloned())
        .collect();

    let modified: Vec<(StructDef, StructDef)> = old_names
        .intersection(&new_names)
        .filter_map(|name| {
            let old_struct = old.get(*name)?;
            let new_struct = new.get(*name)?;
            let old_normalized: String = old_struct.full_definition.split_whitespace().collect();
            let new_normalized: String = new_struct.full_definition.split_whitespace().collect();
            if old_normalized != new_normalized {
                Some((old_struct.clone(), new_struct.clone()))
            } else {
                None
            }
        })
        .collect();

    (added, removed, modified)
}

fn print_diff(old: &str, new: &str, current_tag: &str, target_tag: &str, root_dir: &Path) {
    // Extract and compare functions
    let old_functions = extract_functions(old);
    let new_functions = extract_functions(new);
    let (added_fns, removed_fns, modified_fns) = compare_functions(&old_functions, &new_functions);

    // Extract and compare structs
    let old_structs = extract_structs(old);
    let new_structs = extract_structs(new);
    let (added_structs, removed_structs, modified_structs) =
        compare_structs(&old_structs, &new_structs);

    // Build output
    let mut output = String::new();
    output.push_str(&format!(
        "=== Bindings Diff ({} -> {}) ===\n\n",
        current_tag, target_tag
    ));

    output.push_str(&format!(
        "Functions: +{} added, -{} removed, ~{} modified\n",
        added_fns.len(),
        removed_fns.len(),
        modified_fns.len()
    ));
    output.push_str(&format!(
        "Structs:   +{} added, -{} removed, ~{} modified\n\n",
        added_structs.len(),
        removed_structs.len(),
        modified_structs.len()
    ));

    // === FUNCTIONS ===
    if !added_fns.is_empty() {
        output.push_str("=== Added Functions ===\n");
        for f in &added_fns {
            output.push_str(&format!("+ {}\n", f.name));
        }
        output.push('\n');
    }

    if !removed_fns.is_empty() {
        output.push_str("=== Removed Functions ===\n");
        for f in &removed_fns {
            output.push_str(&format!("- {}\n", f.name));
        }
        output.push('\n');
    }

    if !modified_fns.is_empty() {
        output.push_str("=== Modified Functions (signature changed) ===\n");
        for (old_f, new_f) in &modified_fns {
            output.push_str(&format!("~ {}\n", old_f.name));
            output.push_str("  OLD:\n");
            for line in old_f.full_signature.lines() {
                output.push_str(&format!("    {}\n", line));
            }
            output.push_str("  NEW:\n");
            for line in new_f.full_signature.lines() {
                output.push_str(&format!("    {}\n", line));
            }
            output.push('\n');
        }
    }

    // === STRUCTS ===
    if !added_structs.is_empty() {
        output.push_str("=== Added Structs ===\n");
        for s in &added_structs {
            output.push_str(&format!("+ {}\n", s.name));
        }
        output.push('\n');
    }

    if !removed_structs.is_empty() {
        output.push_str("=== Removed Structs ===\n");
        for s in &removed_structs {
            output.push_str(&format!("- {}\n", s.name));
        }
        output.push('\n');
    }

    if !modified_structs.is_empty() {
        output.push_str("=== Modified Structs ===\n");
        for (old_s, new_s) in &modified_structs {
            output.push_str(&format!("~ {}\n", old_s.name));
            output.push_str("  OLD:\n");
            for line in old_s.full_definition.lines().take(10) {
                output.push_str(&format!("    {}\n", line));
            }
            output.push_str("  NEW:\n");
            for line in new_s.full_definition.lines().take(10) {
                output.push_str(&format!("    {}\n", line));
            }
            output.push('\n');
        }
    }

    // Print to console with colors
    println!(
        "\n\x1b[32mFunctions:\x1b[0m +{} added, -{} removed, ~{} modified",
        added_fns.len(),
        removed_fns.len(),
        modified_fns.len()
    );
    println!(
        "\x1b[32mStructs:\x1b[0m   +{} added, -{} removed, ~{} modified\n",
        added_structs.len(),
        removed_structs.len(),
        modified_structs.len()
    );

    if !added_fns.is_empty() {
        println!("\x1b[32m=== Added Functions ===\x1b[0m");
        for f in added_fns.iter().take(30) {
            println!("\x1b[32m+ {}\x1b[0m", f.name);
        }
        if added_fns.len() > 30 {
            println!("... and {} more", added_fns.len() - 30);
        }
        println!();
    }

    if !removed_fns.is_empty() {
        println!("\x1b[31m=== Removed Functions ===\x1b[0m");
        for f in removed_fns.iter().take(30) {
            println!("\x1b[31m- {}\x1b[0m", f.name);
        }
        if removed_fns.len() > 30 {
            println!("... and {} more", removed_fns.len() - 30);
        }
        println!();
    }

    if !modified_fns.is_empty() {
        println!("\x1b[33m=== Modified Functions (signature changed) ===\x1b[0m");
        for (old_f, new_f) in modified_fns.iter().take(20) {
            println!("\x1b[33m~ {}\x1b[0m", old_f.name);
            println!("  \x1b[31mOLD:\x1b[0m");
            for line in old_f.full_signature.lines().take(5) {
                println!("    {}", line);
            }
            println!("  \x1b[32mNEW:\x1b[0m");
            for line in new_f.full_signature.lines().take(5) {
                println!("    {}", line);
            }
            println!();
        }
        if modified_fns.len() > 20 {
            println!(
                "... and {} more modified functions",
                modified_fns.len() - 20
            );
        }
    }

    if !added_structs.is_empty() {
        println!("\x1b[32m=== Added Structs ===\x1b[0m");
        for s in &added_structs {
            println!("\x1b[32m+ {}\x1b[0m", s.name);
        }
        println!();
    }

    if !removed_structs.is_empty() {
        println!("\x1b[31m=== Removed Structs ===\x1b[0m");
        for s in &removed_structs {
            println!("\x1b[31m- {}\x1b[0m", s.name);
        }
        println!();
    }

    // Save full diff to file
    let diff_file = root_dir.join(format!("mlx-c-diff-{}-to-{}.txt", current_tag, target_tag));
    if let Err(e) = std::fs::write(&diff_file, &output) {
        eprintln!("Warning: Failed to write diff file: {}", e);
    } else {
        println!(
            "\n\x1b[33mFull diff saved to:\x1b[0m {}",
            diff_file.display()
        );
    }
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
    let target_tag = args.get(1).cloned();

    let root_dir = get_repo_root();
    let mlx_c_dir = root_dir.join("mlx-sys/src/mlx-c");

    println!("\x1b[33mChecking for mlx-c updates...\x1b[0m\n");

    let current_tag = get_current_tag(&mlx_c_dir);
    let latest_tag = get_latest_tag(&mlx_c_dir);
    let target_tag = target_tag.unwrap_or(latest_tag.clone());

    println!("Current version: \x1b[32m{}\x1b[0m", current_tag);
    println!("Latest version:  \x1b[32m{}\x1b[0m", latest_tag);
    if target_tag != latest_tag {
        println!("Target version:  \x1b[32m{}\x1b[0m", target_tag);
    }
    println!();

    if current_tag == target_tag {
        println!("\x1b[32mAlready up to date!\x1b[0m");
        return;
    }

    println!("\x1b[33mGenerating bindings for {}...\x1b[0m", current_tag);
    let current_bindings = generate_bindings(&root_dir);

    println!("\x1b[33mChecking out {}...\x1b[0m", target_tag);
    checkout_tag(&mlx_c_dir, &target_tag);

    println!("\x1b[33mGenerating bindings for {}...\x1b[0m", target_tag);
    let target_bindings = generate_bindings(&root_dir);

    // Restore original
    println!("\x1b[33mRestoring {}...\x1b[0m", current_tag);
    checkout_tag(&mlx_c_dir, &current_tag);

    println!(
        "\n\x1b[33m=== Bindings Diff ({} -> {}) ===\x1b[0m",
        current_tag, target_tag
    );

    if current_bindings == target_bindings {
        println!("\n\x1b[32mNo API changes detected!\x1b[0m");
    } else {
        print_diff(
            &current_bindings,
            &target_bindings,
            &current_tag,
            &target_tag,
            &root_dir,
        );
    }

    println!("\n\x1b[33mTo update, run:\x1b[0m");
    println!("  cd mlx-sys/src/mlx-c && git checkout {}", target_tag);
}
