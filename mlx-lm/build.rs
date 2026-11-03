use std::{env, fs, path::Path, process::Command};

fn main() {
    let database = env::var("DEP_MLX_COMPILE_COMMANDS").expect("MLX compilation database");
    let commands: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(&database).expect("Read MLX compiler settings"))
            .expect("Parse MLX compiler settings");
    let command = commands
        .iter()
        .find(|entry| {
            entry["file"]
                .as_str()
                .is_some_and(|file| Path::new(file).ends_with("mlx/array.cpp"))
        })
        .expect("MLX array.cpp compiler settings");
    let args = shlex::split(command["command"].as_str().expect("MLX compiler command"))
        .expect("Split MLX compiler command");
    let mut compiler = Command::new(&args[0]);
    compiler.current_dir(command["directory"].as_str().expect("MLX build directory"));
    let mut flags = args[1..].iter();
    while let Some(flag) = flags.next() {
        if flag == "-o" {
            flags.next().expect("MLX compiler output argument");
        } else if flag != "-c" && Some(flag.as_str()) != command["file"].as_str() {
            compiler.arg(flag);
        }
    }
    for include in env::split_paths(&env::var_os("DEP_MLX_INCLUDE").expect("MLX include paths")) {
        compiler.arg("-I").arg(include);
    }
    if env::var("DEP_MLX_METAL").as_deref() == Ok("true") {
        compiler.arg("-DMLX_LM_METAL");
    }
    let source =
        Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/cache/kv_append.cpp");
    let object = Path::new(&env::var("OUT_DIR").unwrap()).join("kv_append.o");
    let status = compiler
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("Compile K/V append with MLX compiler settings");
    assert!(status.success(), "K/V append compilation failed");
    cc::Build::new()
        .cpp(true)
        .object(object)
        .compile("mlx_lm_kv_append");
    println!("cargo:rerun-if-changed=src/cache/kv_append.cpp");
    println!("cargo:rerun-if-changed={database}");
}
