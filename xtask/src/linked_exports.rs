use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[path = "../../mlx-sys/src/native_link.rs"]
mod native_link;

fn output(command: &mut Command) -> Result<String, String> {
    let result = command.output().map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!(
            "{command:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    String::from_utf8(result.stdout).map_err(|e| e.to_string())
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn symbol_set(value: &Value) -> Result<BTreeSet<String>, String> {
    let entries = value.as_array().ok_or("missing symbol inventory")?;
    let symbols = entries
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "invalid symbol".to_owned())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if symbols.len() != entries.len() {
        return Err("duplicate symbol".into());
    }
    Ok(symbols)
}

fn validate_symbols(planned: &BTreeSet<String>, found: &BTreeSet<String>) -> Result<(), String> {
    if planned.is_empty() {
        return Err("empty planned symbol inventory".into());
    }
    let missing: Vec<_> = planned.difference(found).collect();
    if !missing.is_empty() {
        return Err(format!("missing: {missing:?}"));
    }
    Ok(())
}

fn parse_nm(text: &str) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    let mut c = BTreeSet::new();
    let mut cpp = BTreeSet::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.ends_with(':'))
    {
        // -j emits names only; Mach-O adds exactly one underscore to C identifiers.
        let symbol = line
            .strip_prefix('_')
            .ok_or_else(|| format!("unexpected nm output: {line}"))?;
        if symbol.starts_with("_Z") {
            cpp.insert(symbol.to_owned());
        } else {
            c.insert(symbol.to_owned());
        }
    }
    Ok((c, cpp))
}

fn scan(path: &Path) -> Result<Value, String> {
    let text = output(
        Command::new("/usr/bin/nm")
            .args(["-arch", "arm64", "-g", "-U", "-j"])
            .arg(path),
    )?;
    let (c, cpp) = parse_nm(&text)?;
    Ok(
        json!({"path": path, "sha256": crate::checkpoints::hash(path)?,
        "architecture": "arm64", "c_symbols": c, "cpp_mangled_count": cpp.len()}),
    )
}

fn archive_paths(build: &Path) -> Result<Vec<PathBuf>, String> {
    let paths: Vec<_> = read(&build.join("linked-archives.txt"))?
        .lines()
        .map(PathBuf::from)
        .collect();
    let expected: Vec<_> = native_link::libraries(build)
        .map(|name| format!("lib{name}.a"))
        .collect();
    if paths.len() != expected.len()
        || paths
            .iter()
            .zip(expected)
            .any(|(path, name)| path.file_name().and_then(|s| s.to_str()) != Some(&name))
    {
        return Err("native archives differ from mlx-sys link contract".into());
    }
    let directory = build.canonicalize().map_err(|e| e.to_string())?;
    for path in &paths {
        if !path
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(&directory)
        {
            return Err(format!(
                "archive outside selected build: {}",
                path.display()
            ));
        }
    }
    Ok(paths)
}

fn toolchain_paths() -> Vec<PathBuf> {
    native_link::find_clang_rt_path()
        .map(|runtime| Path::new(&runtime).join("libclang_rt.osx.a"))
        .into_iter()
        .collect()
}

fn toolchain_archives(paths: &[PathBuf]) -> Result<Vec<Value>, String> {
    paths
        .iter()
        .map(|path| Ok(json!({"path": path, "sha256": crate::checkpoints::hash(path)?})))
        .collect()
}

fn canary(
    root: &Path,
    build: &Path,
    planned: &BTreeSet<String>,
    archives: &[PathBuf],
    metal: bool,
) -> Result<Value, String> {
    let (headers, _) = crate::fingerprint::planned_exports(root)?;
    let mut source = headers
        .iter()
        .map(|h| format!("#include \"{h}\"\n"))
        .collect::<String>();
    source.push_str("#include <mlx/version.h>\n#include <cstdio>\n#include <cstring>\n");
    source.push_str("using function = void (*)();\nfunction volatile symbols[] = {\n");
    for symbol in planned {
        source.push_str(&format!("reinterpret_cast<function>(&{symbol}),\n"));
    }
    source.push_str("};\nint main() {\nfor (auto &symbol : symbols) if (!symbol) return 1;\nmlx_string value = mlx_string_new();\nif (mlx_version(&value)) return 2;\nconst char *c = mlx_string_data(value);\nauto cpp = mlx::core::version();\nstd::printf(\"%s\\n%s\\n\", c, cpp);\nint result = std::strcmp(c, cpp);\nmlx_string_free(value);\nreturn result != 0;\n}\n");
    let src = build.join("linked-export-canary.cpp");
    let obj = build.join("linked-export-canary.o");
    let exe = build.join("linked-export-canary");
    fs::write(&src, source).map_err(|e| e.to_string())?;
    let mut compile = Command::new("/usr/bin/c++");
    compile.args(["-std=c++17", "-O0", "-c"]);
    for include in read(&build.join("linked-includes.txt"))?.lines() {
        compile.arg("-I").arg(include);
    }
    compile.arg(&src).arg("-o").arg(&obj);
    output(&mut compile)?;
    let undefined = output(
        Command::new("/usr/bin/nm")
            .args(["-arch", "arm64", "-u", "-j"])
            .arg(&obj),
    )?;
    let (references, _) = parse_nm(&undefined)?;
    if !planned.is_subset(&references) {
        return Err("canary object did not retain every planned reference".into());
    }
    let mut link = Command::new("/usr/bin/c++");
    link.arg(&obj).args(archives).args([
        "-lc++",
        "-lobjc",
        "-framework",
        "Foundation",
        "-framework",
        "Accelerate",
    ]);
    if metal {
        link.args(["-framework", "Metal"]);
    }
    link.arg("-o").arg(&exe);
    output(&mut link)?;
    let stdout = output(&mut Command::new(&exe))?;
    let versions: Vec<_> = stdout.lines().collect();
    let identity = crate::config_report::identity(root)?;
    if versions
        != [identity["nested_mlx_version"]
            .as_str()
            .ok_or("missing version")?; 2]
    {
        return Err(format!("runtime version mismatch: {stdout}"));
    }
    Ok(
        json!({"verdict": "pass", "compile_command": format!("{compile:?}"),
        "link_command": format!("{link:?}"), "path": exe, "sha256": crate::checkpoints::hash(&exe)?,
        "retained_references": planned, "mlx_c_runtime_version": versions[0], "mlx_runtime_version": versions[1]}),
    )
}

fn collect(root: &Path, build: &Path) -> Result<Value, String> {
    let started = Instant::now();
    let identity = crate::config_report::identity(root)?;
    let dirty = !output(
        Command::new("git")
            .args(["status", "--porcelain", "--untracked-files=normal"])
            .current_dir(root),
    )?
    .trim()
    .is_empty();
    if dirty {
        return Err("dirty linked-export checkout".into());
    }
    let (_, planned) = crate::fingerprint::planned_exports(root)?;
    let paths = archive_paths(build)?;
    let archives = paths
        .iter()
        .map(|p| scan(p))
        .collect::<Result<Vec<_>, _>>()?;
    let found = archives
        .iter()
        .map(|a| symbol_set(&a["c_symbols"]))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
    let toolchain_paths = toolchain_paths();
    let toolchain_archives = toolchain_archives(&toolchain_paths)?;
    let linked_paths = paths.into_iter().chain(toolchain_paths).collect::<Vec<_>>();
    let configuration = configuration(build)?;
    let provenance = build_provenance(root, build)?;
    let canary = match canary(
        root,
        build,
        &planned,
        &linked_paths,
        configuration["MLX_BUILD_METAL"] == "ON",
    ) {
        Ok(value) => value,
        Err(error) => json!({"verdict": "fail", "error": error}),
    };
    let mut report = json!({"schema_version": 1, "identity": identity, "dirty": dirty, "build_directory": build,
        "configuration": configuration, "build_provenance": provenance, "archives": archives, "planned": planned,
        "toolchain_archives": toolchain_archives,
        "planned_count": planned.len(), "found_count": found.len(),
        "missing": planned.difference(&found).collect::<Vec<_>>(),
        "canary": canary, "elapsed_seconds": started.elapsed().as_secs_f64(), "verdict": "fail"});
    match validate_symbols(&planned, &found) {
        Ok(()) if report["canary"]["verdict"] == "pass" => report["verdict"] = "pass".into(),
        Ok(()) => report["error"] = report["canary"]["error"].clone(),
        Err(error) => report["error"] = error.into(),
    }
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    Ok(report)
}

fn cache(build: &Path) -> Result<BTreeMap<String, String>, String> {
    Ok(read(&build.join("CMakeCache.txt"))?
        .lines()
        .filter_map(|line| {
            if line.starts_with(['#', '/']) {
                return None;
            }
            let (key, rest) = line.split_once(':')?;
            let (_, value) = rest.split_once('=')?;
            Some((key.to_owned(), value.to_owned()))
        })
        .collect())
}

fn configuration(build: &Path) -> Result<Value, String> {
    let cache = cache(build)?;
    let mut config = serde_json::Map::new();
    for key in [
        "CMAKE_BUILD_TYPE",
        "MLX_BUILD_METAL",
        "MLX_BUILD_ACCELERATE",
        "MLX_METAL_JIT",
    ] {
        config.insert(
            key.into(),
            cache
                .get(key)
                .ok_or_else(|| format!("missing {key}"))?
                .clone()
                .into(),
        );
    }
    Ok(config.into())
}

fn build_provenance(root: &Path, build: &Path) -> Result<Value, String> {
    let cache = cache(build)?;
    let directory = |key: &str| -> Result<PathBuf, String> {
        Path::new(cache.get(key).ok_or_else(|| format!("missing {key}"))?)
            .canonicalize()
            .map_err(|e| e.to_string())
    };
    let native = root
        .join("mlx-sys/src/mlx-c")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let home = directory("CMAKE_HOME_DIRECTORY")?;
    if home != native {
        return Err("native build belongs to another source checkout".into());
    }
    let mlx = directory("mlx_SOURCE_DIR")?;
    let git = |source: &Path, args: &[&str]| -> Result<String, String> {
        output(Command::new("git").args(args).current_dir(source)).map(|s| s.trim().to_owned())
    };
    for source in [&native, &mlx] {
        if !git(
            source,
            &["status", "--porcelain", "--untracked-files=normal"],
        )?
        .is_empty()
        {
            return Err("dirty native build source".into());
        }
    }
    let identity = crate::config_report::identity(root)?;
    let tag = format!(
        "refs/tags/v{}^{{commit}}",
        identity["nested_mlx_version"]
            .as_str()
            .ok_or("missing MLX version")?
    );
    let mlx_commit = git(&mlx, &["rev-parse", "HEAD"])?;
    if mlx_commit != git(&mlx, &["rev-parse", &tag])? {
        return Err("native build source differs from pinned MLX tag".into());
    }
    Ok(
        json!({"mlx_c_source": native, "mlx_source": mlx, "mlx_commit": mlx_commit,
        "cmake_cache_sha256": crate::checkpoints::hash(&build.join("CMakeCache.txt"))?}),
    )
}

fn cargo_build(root: &Path, messages: &Path) -> Result<PathBuf, String> {
    let records = read(messages)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let builds: Vec<_> = records
        .iter()
        .filter(|record| {
            record["reason"] == "build-script-executed"
                && record["linked_libs"]
                    .as_array()
                    .is_some_and(|libs| libs.contains(&json!("static=mlxc")))
        })
        .collect();
    if builds.len() != 1 {
        return Err("expected exactly one mlx-sys build-script artifact".into());
    }
    let record = builds[0];
    let build = Path::new(record["out_dir"].as_str().ok_or("missing Cargo out_dir")?).join("build");
    let search: Vec<_> = record["linked_paths"]
        .as_array()
        .ok_or("missing Cargo library search paths")?
        .iter()
        .filter_map(|p| p.as_str())
        .map(|p| p.strip_prefix("native=").unwrap_or(p))
        .collect();
    let libraries: Vec<_> = record["linked_libs"]
        .as_array()
        .ok_or("missing Cargo libraries")?
        .iter()
        .filter_map(|l| l.as_str()?.strip_prefix("static="))
        .collect();
    let mut expected: Vec<_> = native_link::libraries(&build).collect();
    if native_link::find_clang_rt_path().is_some() {
        expected.push("clang_rt.osx");
    }
    if libraries != expected {
        return Err("Cargo static libraries differ from link contract".into());
    }
    let archives = libraries
        .iter()
        .map(|name| {
            search
                .iter()
                .map(|p| Path::new(p).join(format!("lib{name}.a")))
                .find(|p| p.is_file())
                .ok_or_else(|| format!("cannot resolve Cargo archive {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(runtime) = native_link::find_clang_rt_path() {
        if archives.last() != Some(&Path::new(&runtime).join("libclang_rt.osx.a")) {
            return Err("Cargo compiler runtime differs from native canary".into());
        }
    }
    let manifest = archives
        .iter()
        .take(native_link::libraries(&build).count())
        .map(|p| format!("{}\n", p.display()))
        .collect::<String>();
    fs::write(build.join("linked-archives.txt"), manifest).map_err(|e| e.to_string())?;
    fs::write(
        build.join("linked-includes.txt"),
        format!(
            "{}\n{}\n{}\n",
            root.join("mlx-sys/src/mlx-c").display(),
            build.join("_deps/mlx-src").display(),
            build.join("_deps/mlx-build").display()
        ),
    )
    .map_err(|e| e.to_string())?;
    Ok(build)
}

pub(crate) fn run(root: &Path, args: &[String]) -> i32 {
    let result = match args {
        [build_flag, build, report_flag, report]
            if build_flag == "--cargo-output" && report_flag == "--report" =>
        {
            let build = cargo_build(root, Path::new(build));
            let collected = build.and_then(|build| collect(root, &build));
            let value =
                collected.unwrap_or_else(|error| json!({"verdict": "fail", "error": error}));
            (|| {
                for archive in value["archives"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .chain(value["toolchain_archives"].as_array().into_iter().flatten())
                {
                    println!("{} sha256={}", archive["path"], archive["sha256"]);
                }
                println!("link/version canary: {}", value["canary"]["verdict"]);
                fs::write(
                    report,
                    format!("{}\n", serde_json::to_string_pretty(&value).unwrap()),
                )
                .map_err(|e| e.to_string())?;
                if value["verdict"] == "pass" {
                    Ok(())
                } else {
                    Err(value["error"].to_string())
                }
            })()
        }
        _ => Err("usage: linked-exports --cargo-output <cargo-json> --report <file>".into()),
    };
    if let Err(error) = result {
        eprintln!("linked exports: {error}");
        1
    } else {
        0
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn symbols(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn toolchain_provenance_hashes_without_scanning_symbols() {
        let archive = tempfile::NamedTempFile::new().unwrap();
        fs::write(archive.path(), b"not a symbol archive").unwrap();
        let paths = vec![archive.path().to_owned()];
        let original = toolchain_archives(&paths).unwrap();
        assert_eq!(original[0].as_object().unwrap().len(), 2);
        assert_eq!(original[0]["path"], json!(archive.path()));
        fs::write(archive.path(), b"changed runtime").unwrap();
        assert_ne!(original, toolchain_archives(&paths).unwrap());
    }

    #[test]
    fn requires_jaccl_archive_only_when_built() {
        let build = tempfile::tempdir().unwrap();
        let mut manifest = String::new();
        for name in ["mlx", "mlxc", "gguflib"] {
            let archive = build.path().join(format!("lib{name}.a"));
            fs::write(&archive, b"archive").unwrap();
            manifest.push_str(&format!("{}\n", archive.display()));
        }
        let manifest_path = build.path().join("linked-archives.txt");
        fs::write(&manifest_path, &manifest).unwrap();
        assert!(archive_paths(build.path()).is_ok());

        let jaccl = build.path().join("jaccl/libjaccl.a");
        fs::create_dir(jaccl.parent().unwrap()).unwrap();
        fs::write(&jaccl, b"archive").unwrap();
        assert!(archive_paths(build.path())
            .unwrap_err()
            .contains("link contract"));

        manifest.push_str(&format!("{}\n", jaccl.display()));
        fs::write(&manifest_path, manifest).unwrap();
        assert!(archive_paths(build.path()).unwrap().contains(&jaccl));

        fs::remove_file(jaccl).unwrap();
        assert!(archive_paths(build.path())
            .unwrap_err()
            .contains("link contract"));
    }

    #[test]
    fn rejects_cross_build_archives() {
        let build = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let mut manifest = String::new();
        for name in ["mlx", "mlxc", "gguflib"] {
            let archive = other.path().join(format!("lib{name}.a"));
            fs::write(&archive, b"archive").unwrap();
            manifest.push_str(&format!("{}\n", archive.display()));
        }
        fs::write(build.path().join("linked-archives.txt"), manifest).unwrap();
        assert!(archive_paths(build.path()).unwrap_err().contains("outside"));
    }

    #[test]
    fn normalizes_macho_without_hiding_internal_or_weak_symbols() {
        let (c, cpp) = parse_nm(
            "/tmp/lib.a(member.o):\n_mlx_add\n__mlx_private\n__ZN3mlx4core4evalEv\n_third_party\n",
        )
        .unwrap();
        assert_eq!(c, symbols(&["mlx_add", "_mlx_private", "third_party"]));
        assert_eq!(cpp, symbols(&["_ZN3mlx4core4evalEv"]));
        assert!(parse_nm("undecorated").is_err());
    }

    #[test]
    fn scans_defined_external_symbols_from_a_real_archive() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("symbols.cpp");
        let object = dir.path().join("symbols.o");
        let archive = dir.path().join("libsymbols.a");
        fs::write(&source, "extern \"C\" int mlx_probe() { return 1; }\nextern \"C\" __attribute__((weak)) int extra() { return 2; }\nint cpp_only() { return 3; }\nextern \"C\" int absent();\nint caller() { return absent(); }\n").unwrap();
        output(
            Command::new("/usr/bin/c++")
                .arg("-c")
                .arg(&source)
                .arg("-o")
                .arg(&object),
        )
        .unwrap();
        output(
            Command::new("/usr/bin/ar")
                .arg("rcs")
                .arg(&archive)
                .arg(&object),
        )
        .unwrap();
        let report = scan(&archive).unwrap();
        assert_eq!(
            symbol_set(&report["c_symbols"]).unwrap(),
            symbols(&["mlx_probe", "extra"])
        );
        assert_eq!(report["cpp_mangled_count"], 2);
    }

    #[test]
    fn rejects_missing_planned_symbol() {
        assert!(
            validate_symbols(&symbols(&["mlx_add", "mlx_eval"]), &symbols(&["mlx_add"]))
                .unwrap_err()
                .contains("missing")
        );
    }

    #[test]
    fn rejects_empty_plan() {
        assert!(validate_symbols(&symbols(&[]), &symbols(&["mlx_add"])).is_err());
    }

    #[test]
    fn accepts_unplanned_exports() {
        assert!(
            validate_symbols(&symbols(&["mlx_add"]), &symbols(&["mlx_add", "internal"])).is_ok()
        );
    }
}
