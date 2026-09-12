use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const EXCLUDED_HEADERS: &[&str] = &[];

pub fn discover_headers(mlx_c_root: &Path) -> io::Result<Vec<PathBuf>> {
    let header_dir = mlx_c_root.join("mlx/c");
    let mut headers = fs::read_dir(&header_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    headers.retain(|path| {
        path.extension() == Some(OsStr::new("h"))
            && path
                .file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| !EXCLUDED_HEADERS.contains(&name))
    });
    headers.sort();
    Ok(headers)
}

pub fn relative_header(mlx_c_root: &Path, header: &Path) -> io::Result<String> {
    header
        .strip_prefix(mlx_c_root)
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .map_err(io::Error::other)
}

pub fn builder(mlx_c_root: &Path, headers: &[PathBuf]) -> bindgen::Builder {
    headers.iter().fold(
        bindgen::Builder::default()
            .rust_target("1.73.0".parse().expect("rust-version"))
            .clang_arg(format!("-I{}", mlx_c_root.display())),
        |builder, header| builder.header(header.to_string_lossy()),
    )
}
