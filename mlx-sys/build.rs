extern crate cmake;

use cmake::Config;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[path = "src/bindgen_config.rs"]
mod bindgen_config;

#[path = "src/native_link.rs"]
mod native_link;

#[cfg(feature = "metal")]
fn mlx_c_key(mlx_c_root: &Path) -> String {
    if mlx_c_root.join(".git").exists() {
        let output = Command::new("git")
            .arg("-C")
            .arg(mlx_c_root)
            .args(["rev-parse", "HEAD"])
            .output();
        if let Ok(output) = output {
            if output.status.success() {
                let commit = String::from_utf8_lossy(&output.stdout);
                let commit = commit.trim();
                if commit.len() >= 12 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return commit[..12].to_owned();
                }
            }
        }
    }

    let mut files = bindgen_config::discover_headers(mlx_c_root)
        .expect("Unable to discover mlx-c headers for the metallib key");
    files.push(mlx_c_root.join("CMakeLists.txt"));
    files.sort();

    let mut hash = 0xcbf29ce484222325_u64;
    for file in files {
        let relative = file.strip_prefix(mlx_c_root).unwrap_or(&file);
        for byte in relative.to_string_lossy().bytes().chain([0]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for byte in fs::read(&file).expect("Unable to hash mlx-c source for the metallib key") {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")[..12].to_owned()
}

#[cfg(feature = "metal")]
fn configure_metallib(config: &mut Config, mlx_c_root: &Path) {
    let hook = Path::new("metallib.cmake")
        .canonicalize()
        .expect("Unable to locate the metallib configuration");
    config.define("CMAKE_PROJECT_mlx_INCLUDE", hook);
    if let Some(path) = env::var_os("MLX_RS_METAL_PATH") {
        config.define("MLX_RS_METAL_BASE", "");
        config.define("MLX_METAL_PATH", path);
        return;
    }

    let home =
        env::var_os("HOME").expect("HOME must be set when MLX_RS_METAL_PATH is not provided");
    // A JIT metallib cannot supply the kernels needed by a non-JIT build.
    let key = if cfg!(feature = "metal-jit") {
        format!("{}-jit", mlx_c_key(mlx_c_root))
    } else {
        mlx_c_key(mlx_c_root)
    };
    config.define(
        "MLX_RS_METAL_BASE",
        PathBuf::from(home).join(".mlx").join("lib").join(key),
    );
}

fn system_sdk_enabled() -> bool {
    // Cargo tracks option_env! without adding rerun directives to the default build.
    option_env!("MLX_SYS_SYSTEM_SDK") == Some("1")
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
}

fn native_out_dir() -> PathBuf {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    if system_sdk_enabled() {
        out.join("system-sdk")
    } else {
        out
    }
}

fn cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    cache.lines().find_map(|line| {
        let (name, entry) = line.split_once(':')?;
        let (_, value) = entry.split_once('=')?;
        (name == key).then_some(value)
    })
}

fn host_macos_version() -> String {
    xcode_path(
        Command::new("/usr/bin/sw_vers").arg("-productVersion"),
        "cannot determine the host macOS version",
    )
    .expect("Unable to select the macOS deployment target")
}

fn deployment_target() -> String {
    env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .filter(|target| !target.is_empty())
        .unwrap_or_else(host_macos_version)
}

fn xcode_path(command: &mut Command, missing: &str) -> Result<String, String> {
    let output = command.output().map_err(|_| missing.to_owned())?;
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || path.is_empty() {
        return Err(missing.to_owned());
    }
    Ok(path)
}

fn configure_macos(config: &mut Config) -> Result<(), String> {
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    let developer = xcode_path(
        Command::new("/usr/bin/xcode-select")
            .arg("-p")
            .env_remove("DEVELOPER_DIR"),
        "system Xcode developer directory is missing; select an Xcode installation with xcode-select",
    )?;
    let query = |args: &[&str], missing: &str| {
        xcode_path(
            Command::new("/usr/bin/xcrun")
                .args(["--sdk", "macosx"])
                .args(args)
                .env("DEVELOPER_DIR", &developer)
                .env_remove("SDKROOT"),
            missing,
        )
    };
    let sdk = query(
        &["--show-sdk-path"],
        "macOS SDK is missing from system Xcode",
    )?;
    let cc = query(&["--find", "clang"], "clang is missing from system Xcode")?;
    let cxx = query(
        &["--find", "clang++"],
        "clang++ is missing from system Xcode",
    )?;
    #[cfg(feature = "metal")]
    query(
        &["metal", "--version"],
        "Metal compiler is missing; select a full Xcode installation with the Metal toolchain",
    )?;
    let mut paths = vec![PathBuf::from("/usr/bin")];
    paths.extend(env::split_paths(&env::var_os("PATH").unwrap_or_default()));
    config.env("PATH", env::join_paths(paths).map_err(|e| e.to_string())?);
    config.env("DEVELOPER_DIR", developer);
    config.env("SDKROOT", &sdk);
    if Path::new(&sdk).join("SDKSettings.json").is_file() {
        println!("cargo:rerun-if-changed={sdk}/SDKSettings.json");
    }
    let out = native_out_dir();
    let build = out.join("build");
    let target = deployment_target();
    if let Ok(cache) = fs::read_to_string(build.join("CMakeCache.txt")) {
        if [
            ("CMAKE_C_COMPILER", &cc),
            ("CMAKE_CXX_COMPILER", &cxx),
            ("CMAKE_OSX_SYSROOT", &sdk),
            ("CMAKE_OSX_DEPLOYMENT_TARGET", &target),
        ]
        .iter()
        .any(|(key, value)| cache_value(&cache, key) != Some(value.as_str()))
        {
            // CMake drops -D options on compiler changes; Metal AIR rules ignore target changes.
            fs::remove_dir_all(&build)
                .map_err(|e| format!("cannot reset {}: {e}", build.display()))?;
        }
    }
    config.out_dir(out);
    config.define("CMAKE_OSX_SYSROOT", &sdk);
    // Dependents replay compile_commands.json outside CMake's environment.
    config.define("CMAKE_C_COMPILER", &cc);
    config.define("CMAKE_CXX_COMPILER", &cxx);
    config.define("CMAKE_OSX_DEPLOYMENT_TARGET", target);

    // CMake owns the SDK and deployment target; cc's defaults come from Nix.
    config.no_default_flags(true);
    let mut c = cc::Build::new();
    c.compiler(cc);
    let mut cpp = cc::Build::new();
    cpp.compiler(cxx);
    config.init_c_cfg(c).init_cxx_cfg(cpp);
    Ok(())
}

fn build_and_link_mlx_c() {
    let mlx_c_root = Path::new("src/mlx-c");
    let mut config = Config::new(mlx_c_root);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define("CMAKE_EXPORT_COMPILE_COMMANDS", "ON");

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");
    if system_sdk_enabled() {
        if let Err(error) = configure_macos(&mut config) {
            eprintln!("mlx-sys: cannot configure the system SDK: {error}");
            std::process::exit(1);
        }
    }

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    // With `MLX_SYS_PREBUILT`, mlx-c is built against an installed MLX instead (`link_prebuilt_mlx`).
    println!("cargo:rerun-if-env-changed=MLX_SYS_PREBUILT");
    if let Some(prefix) = env::var_os("MLX_SYS_PREBUILT") {
        link_prebuilt_mlx(config, Path::new(&prefix), mlx_c_root);
        return;
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
        #[cfg(feature = "metal-jit")]
        config.define("MLX_METAL_JIT", "ON");
        configure_metallib(&mut config, mlx_c_root);
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    publish_native_sources(&dst, mlx_c_root);

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    // mlx's GGUF io depends on the vendored gguflib archive, which cmake leaves
    // in the mlx build tree instead of installing next to libmlx.
    println!(
        "cargo:rustc-link-search=native={}/build/_deps/mlx-build/mlx/io",
        dst.display()
    );
    for library in native_link::libraries(&dst.join("build")) {
        if library == "jaccl" {
            println!(
                "cargo:rustc-link-search=native={}/build/jaccl",
                dst.display()
            );
        }
        println!("cargo:rustc-link-lib=static={library}");
    }

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        let directory = fs::read_to_string(dst.join("build/mlx-metal-path.txt"))
            .expect("Unable to read the configured metallib directory");
        let metallib = Path::new(&directory).join("mlx.metallib");
        if !metallib.exists() {
            println!(
                "cargo:warning=mlx.metallib was not created at {}; Metal operations may fail at runtime",
                metallib.display()
            );
        }
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = native_link::find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

/// Dependents read this metadata as `DEP_MLX_*`; the README documents the contract.
fn publish_native_sources(dst: &Path, mlx_c_root: &Path) {
    let deps = dst.join("build").join("_deps");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include = [
        Some(deps.join("mlx-src")),
        cfg!(feature = "metal").then(|| deps.join("metal_cpp-src")),
        Some(manifest.join(mlx_c_root)),
    ];
    let include = env::join_paths(include.into_iter().flatten())
        .expect("MLX include directories must not contain the path-list separator");
    println!("cargo::metadata=include={}", include.display());
    println!("cargo::metadata=defines=MLX_STATIC");
    println!(
        "cargo::metadata=compile_commands={}",
        dst.join("build/compile_commands.json").display()
    );
    println!("cargo::metadata=metal={}", cfg!(feature = "metal"));
    if let Some(version) = pinned_mlx_version(mlx_c_root) {
        println!("cargo::metadata=version={version}");
    }
}

/// The MLX version mlx-c's CMake configuration pins (`GIT_TAG v<version>`), when it can be read.
fn pinned_mlx_version(mlx_c_root: &Path) -> Option<String> {
    let cmake = fs::read_to_string(mlx_c_root.join("CMakeLists.txt")).ok()?;
    cmake.lines().find_map(|line| {
        let tag = line.trim().strip_prefix("GIT_TAG v")?;
        tag.split(|c: char| c == ')' || c.is_whitespace())
            .next()
            .map(str::to_owned)
    })
}

/// Builds mlx-c against the installed MLX at `prefix` (`MLX_SYS_PREBUILT`) and links MLX's shared
/// library, instead of building MLX from source, which compiles its Metal kernels ahead of time
/// with Xcode's Metal compiler (the Command Line Tools do not include it). The prefix holds
/// `lib/libmlx.dylib` with `lib/mlx.metallib` beside it, `include/` and `share/cmake/MLX/`, as the
/// `mlx/` directory of the `mlx-metal` wheel does.
fn link_prebuilt_mlx(mut config: Config, prefix: &Path, mlx_c_root: &Path) {
    let lib_dir = prefix.join("lib");
    for file in ["libmlx.dylib", "mlx.metallib"] {
        assert!(
            lib_dir.join(file).is_file(),
            "MLX_SYS_PREBUILT: no {file} in {}",
            lib_dir.display()
        );
    }
    let cmake_dir = prefix.join("share").join("cmake").join("MLX");
    let version = prebuilt_version(&cmake_dir);
    if let (Some(found), Some(pinned)) = (&version, pinned_mlx_version(mlx_c_root)) {
        if *found != pinned {
            println!(
                "cargo:warning=MLX_SYS_PREBUILT holds MLX {found}, but mlx-c is written against MLX {pinned}; mlx-c may not compile against it"
            );
        }
    }
    config.define("MLX_C_USE_SYSTEM_MLX", "ON");
    config.define("MLX_DIR", &cmake_dir);
    let dst = config.build();

    // The same metadata as a build from source (`publish_native_sources`), for this MLX: its
    // installed headers, and no definitions, since it is a shared library (no `MLX_STATIC`).
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include = [
        Some(prefix.join("include")),
        cfg!(feature = "metal").then(|| prefix.join("include").join("metal_cpp")),
        Some(manifest.join(mlx_c_root)),
    ];
    let include = env::join_paths(include.into_iter().flatten())
        .expect("MLX include directories must not contain the path-list separator");
    println!("cargo::metadata=include={}", include.display());
    println!("cargo::metadata=defines=");
    if let Some(version) = version {
        println!("cargo::metadata=version={version}");
    }
    // libmlx.dylib's install name is `@rpath/libmlx.dylib`, and MLX loads `mlx.metallib` from
    // beside it. A build script cannot add an rpath to another package's binaries, so the
    // directory goes to dependents, as `DEP_MLX_LIB_DIR`.
    println!("cargo::metadata=lib_dir={}", lib_dir.display());

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");
    if let Some(clang_rt_path) = native_link::find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

/// The version of the MLX CMake package in `cmake_dir` (`PACKAGE_VERSION` in
/// `MLXConfigVersion.cmake`), when it can be read.
fn prebuilt_version(cmake_dir: &Path) -> Option<String> {
    let text = fs::read_to_string(cmake_dir.join("MLXConfigVersion.cmake")).ok()?;
    text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("set(PACKAGE_VERSION \"")?;
        rest.split('"').next().map(str::to_owned)
    })
}

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_RS_METAL_PATH");
    println!("cargo:rerun-if-changed=native-libraries.txt");
    println!("cargo:rerun-if-changed=metallib.cmake");
    println!("cargo:rerun-if-changed=src/native_link.rs");
    build_and_link_mlx_c();

    let mlx_c_root = PathBuf::from("src/mlx-c");
    let headers =
        bindgen_config::discover_headers(&mlx_c_root).expect("Unable to discover headers");
    for header in &headers {
        let relative =
            bindgen_config::relative_header(&mlx_c_root, header).expect("Unable to record header");
        println!(
            "cargo:rerun-if-changed={}",
            mlx_c_root.join(relative).display()
        );
    }
    let bindings = bindgen_config::builder(&mlx_c_root, &headers)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
